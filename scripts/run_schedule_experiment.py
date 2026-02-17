"""
Training schedule experiment: find optimal TD->SL learning rate schedule.

Uses single NN (80 hidden, 196 Tesauro inputs) to test different combinations of:
- TD game count and learning rate schedules
- SL learning rate schedules and epoch counts

TD phases run in parallel (CPU). SL phases run sequentially (GPU).

Usage:
    python run_schedule_experiment.py                    # run all experiments
    python run_schedule_experiment.py --skip-td          # reuse TD weights, SL only
    python run_schedule_experiment.py --td-only          # run TD only, skip SL
    python run_schedule_experiment.py --experiments A B C # run specific experiments
"""

import os
import sys
import time
import json
import argparse
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')
LOGS_DIR = os.path.join(project_dir, 'logs')
RESULTS_DIR = os.path.join(project_dir, 'experiments', 'schedule')

N_HIDDEN = 80
N_INPUTS = 196
BASE_SEED = 42

# ============================================================
# Experiment definitions
# ============================================================
# Each experiment: {td_phases: [(games, alpha), ...], sl_phases: [(epochs, alpha), ...]}

EXPERIMENTS = {
    # --- Vary TD duration ---
    'A': {
        'desc': 'TD 10k@0.1 -> SL 3-phase',
        'td_phases': [(10000, 0.1)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'B': {
        'desc': 'TD 25k@0.1 -> SL 3-phase',
        'td_phases': [(25000, 0.1)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'C': {
        'desc': 'TD 50k@0.1 -> SL 3-phase',
        'td_phases': [(50000, 0.1)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'D': {
        'desc': 'TD 100k@0.1 -> SL 3-phase',
        'td_phases': [(100000, 0.1)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'E': {
        'desc': 'TD 200k@0.1 -> SL 3-phase',
        'td_phases': [(200000, 0.1)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    # --- Two-phase TD ---
    'F': {
        'desc': 'TD 50k@0.1+50k@0.02 -> SL 3-phase',
        'td_phases': [(50000, 0.1), (50000, 0.02)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    # --- Alternative SL schedules ---
    'G': {
        'desc': 'TD 50k@0.1 -> SL halved alphas',
        'td_phases': [(50000, 0.1)],
        'sl_phases': [(200, 10.0), (200, 3.0), (200, 1.0)],
    },
    'H': {
        'desc': 'TD 50k@0.1 -> SL single long phase',
        'td_phases': [(50000, 0.1)],
        'sl_phases': [(400, 10.0)],
    },
    # --- Long TD training (full production schedule) ---
    'I': {
        'desc': 'TD 200k@0.1+1M@0.02 -> SL 3-phase',
        'td_phases': [(200000, 0.1), (1000000, 0.02)],
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
}


def model_prefix(exp_id):
    return f'sched_{exp_id}'


def td_weights_path(exp_id):
    return os.path.join(MODELS_DIR, f'{model_prefix(exp_id)}.weights')


def sl_weights_path(exp_id):
    return os.path.join(MODELS_DIR, f'sl_{model_prefix(exp_id)}.weights')


def sl_best_weights_path(exp_id):
    return os.path.join(MODELS_DIR, f'sl_{model_prefix(exp_id)}.weights.best')


def run_td_worker(args):
    """Run TD training for one experiment in a subprocess."""
    exp_id, td_phases, seed = args
    prefix = model_prefix(exp_id)

    # Build a script that runs the TD phases
    lines = [
        'import os, sys, time',
        'if sys.platform == "win32":',
        f'    cuda_bin = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"',
        '    if os.path.isdir(cuda_bin): os.add_dll_directory(cuda_bin)',
        f'    bd = r"{build_dir}"',
        '    if os.path.isdir(bd): os.add_dll_directory(bd)',
        f'sys.path.insert(0, r"{build_dir}")',
        f'sys.path.insert(0, r"{os.path.join(project_dir, "python")}")',
        'import bgbot_cpp',
        'from bgsage.data import load_benchmark_file',
        f'bm = load_benchmark_file(r"{os.path.join(DATA_DIR, "contact.bm")}", step=10)',
        f'print("[TD {exp_id}] Starting", flush=True)',
        't0 = time.time()',
    ]

    resume = ''
    for i, (games, alpha) in enumerate(td_phases):
        phase_name = f'{prefix}_p{i}' if len(td_phases) > 1 else prefix
        lines.append(
            f'result = bgbot_cpp.td_train('
            f'n_games={games}, alpha={alpha}, n_hidden={N_HIDDEN}, '
            f'eps=0.1, seed={seed}, benchmark_interval=10000, '
            f'model_name="{phase_name}", '
            f'models_dir=r"{MODELS_DIR}", '
            f'resume_from=r"{resume}", '
            f'scenarios=bm)'
        )
        if len(td_phases) > 1 and i < len(td_phases) - 1:
            # Set resume path for next phase
            resume = os.path.join(MODELS_DIR, f'{phase_name}.weights')
            lines.append(f'print("[TD {exp_id}] Phase {i} done", flush=True)')

    # If multi-phase, copy final weights to the canonical path
    if len(td_phases) > 1:
        final_phase = f'{prefix}_p{len(td_phases)-1}'
        final_path = os.path.join(MODELS_DIR, f'{final_phase}.weights')
        canonical = td_weights_path(exp_id)
        lines.append(f'import shutil')
        lines.append(f'shutil.copy(r"{final_path}", r"{canonical}")')

    lines.append(f'print(f"[TD {exp_id}] Done in {{time.time()-t0:.1f}}s", flush=True)')

    script = '\n'.join(lines)
    fd, script_path = tempfile.mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(script)
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True, text=True, timeout=7200)
        return exp_id, result.stdout, result.stderr
    finally:
        os.unlink(script_path)


def run_sl_phase(exp_id, weights_path, epochs, alpha, benchmark, boards, targets):
    """Run one SL phase. Returns result dict."""
    save_path = sl_weights_path(exp_id)

    result = bgbot_cpp.cuda_supervised_train(
        boards, targets,
        weights_path=weights_path,
        n_hidden=N_HIDDEN,
        n_inputs=N_INPUTS,
        alpha=alpha,
        epochs=epochs,
        batch_size=128,
        seed=BASE_SEED,
        print_interval=10,
        save_path=save_path,
        benchmark_scenarios=benchmark,
    )
    return result


def score_full_benchmark(weights_path):
    """Score against full contact.bm (all scenarios, not downsampled)."""
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    scenarios = load_benchmark_file(bm_path)
    result = bgbot_cpp.score_benchmarks_nn(scenarios, weights_path, N_HIDDEN, N_INPUTS)
    return result.score()


def check_backgammon_rate(weights_path, n_games=2000):
    """Play vs PubEval and return backgammon fraction (both sides)."""
    stats = bgbot_cpp.play_games_nn_vs_pubeval(
        weights_path, N_HIDDEN, n_games=n_games, seed=BASE_SEED)
    total = stats.n_games
    backgammons = stats.p1_backgammons + stats.p2_backgammons
    return backgammons / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Training Schedule Experiment')
    parser.add_argument('--skip-td', action='store_true',
                        help='Skip TD phase, reuse existing weights')
    parser.add_argument('--td-only', action='store_true',
                        help='Run TD only, skip SL')
    parser.add_argument('--experiments', nargs='+', default=None,
                        help='Run specific experiments (e.g., A B C)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    exp_ids = args.experiments if args.experiments else sorted(EXPERIMENTS.keys())

    # Validate experiment IDs
    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f'ERROR: Unknown experiment ID: {eid}')
            print(f'Available: {", ".join(sorted(EXPERIMENTS.keys()))}')
            sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'=== Training Schedule Experiment ===')
    print(f'Experiments: {" ".join(exp_ids)}')
    print(f'Network: {N_HIDDEN}h, {N_INPUTS} inputs (single NN, Tesauro)')
    print()

    for eid in exp_ids:
        exp = EXPERIMENTS[eid]
        td_desc = ' -> '.join(f'{g//1000}k@{a}' for g, a in exp['td_phases'])
        sl_desc = ' -> '.join(f'{e}ep@{a}' for e, a in exp['sl_phases'])
        print(f'  {eid}: TD [{td_desc}] -> SL [{sl_desc}]')
    print()

    results = {}

    # ========== Phase 1: TD Training (parallel) ==========
    if not args.skip_td:
        print('\n--- Phase 1: TD Self-Play (parallel) ---', flush=True)
        t0 = time.time()

        workers = []
        for i, eid in enumerate(exp_ids):
            exp = EXPERIMENTS[eid]
            workers.append((eid, exp['td_phases'], BASE_SEED + i * 100))

        with ProcessPoolExecutor(max_workers=len(workers)) as executor:
            futures = {executor.submit(run_td_worker, w): w[0] for w in workers}
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    _, stdout, stderr = future.result()
                    if stdout:
                        # Print last few lines (benchmark scores)
                        lines = stdout.strip().split('\n')
                        for line in lines:
                            print(f'  [{eid}] {line}', flush=True)
                    if stderr:
                        print(f'  [{eid}] STDERR: {stderr[:200]}', flush=True)
                except Exception as e:
                    print(f'  [{eid}] ERROR: {e}', flush=True)

        td_elapsed = time.time() - t0
        print(f'\nAll TD runs complete in {td_elapsed:.1f}s', flush=True)

    # Score TD results and check backgammon rates
    print('\n--- TD Results ---', flush=True)
    print(f'{"Exp":>4s}  {"TD Score":>10s}  {"BG Rate":>8s}  {"Status":>10s}')
    print(f'{"---":>4s}  {"--------":>10s}  {"-------":>8s}  {"------":>10s}')

    for eid in exp_ids:
        wp = td_weights_path(eid)
        if not os.path.exists(wp):
            print(f'{eid:>4s}  {"N/A":>10s}  {"N/A":>8s}  {"MISSING":>10s}')
            results[eid] = {'td_score': None, 'bg_rate': None, 'status': 'td_missing'}
            continue

        td_score = score_full_benchmark(wp)
        bg_rate = check_backgammon_rate(wp)
        status = 'OK' if bg_rate < 0.02 else 'HIGH_BG'
        print(f'{eid:>4s}  {td_score:10.2f}  {bg_rate:8.1%}  {status:>10s}')
        results[eid] = {
            'td_score': td_score,
            'bg_rate': bg_rate,
            'td_status': status,
        }

    if args.td_only:
        save_results(results, timestamp)
        return

    # ========== Phase 2: SL Training (sequential) ==========
    print('\n--- Phase 2: Supervised Learning (sequential, GPU) ---', flush=True)

    # Load training data once (shared across all experiments)
    import numpy as np
    print('Loading training data...', flush=True)
    t0 = time.time()
    boards1, targets1 = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    boards2, targets2 = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    boards = np.concatenate([boards1, boards2])
    targets = np.concatenate([targets1, targets2])
    print(f'  Loaded {len(boards)} positions in {time.time()-t0:.1f}s', flush=True)

    # Load benchmark once
    benchmark = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'), step=10)
    print(f'  Loaded {len(benchmark)} benchmark scenarios', flush=True)
    print()

    for eid in exp_ids:
        exp = EXPERIMENTS[eid]
        wp = td_weights_path(eid)
        if not os.path.exists(wp):
            print(f'[SL {eid}] Skipping (no TD weights)', flush=True)
            continue

        print(f'\n=== SL Experiment {eid}: {exp["desc"]} ===', flush=True)

        current_weights = wp
        sl_results = []

        for phase_idx, (epochs, alpha) in enumerate(exp['sl_phases']):
            phase_num = phase_idx + 1
            print(f'\n[SL {eid}] Phase {phase_num}: {epochs} epochs @ alpha={alpha}',
                  flush=True)
            print(f'  Starting from: {current_weights}', flush=True)

            t0 = time.time()
            result = run_sl_phase(eid, current_weights, epochs, alpha,
                                  benchmark, boards, targets)
            sl_elapsed = time.time() - t0

            phase_result = {
                'phase': phase_num,
                'epochs': epochs,
                'alpha': alpha,
                'best_score': result['best_score'],
                'best_epoch': result['best_epoch'],
                'time': sl_elapsed,
            }
            sl_results.append(phase_result)

            print(f'[SL {eid}] Phase {phase_num}: best={result["best_score"]:.2f} '
                  f'@ epoch {result["best_epoch"]} ({sl_elapsed:.1f}s)', flush=True)

            # Use best weights from this phase as starting point for next
            best_path = sl_best_weights_path(eid)
            if os.path.exists(best_path):
                current_weights = best_path

        # Final full benchmark score
        final_path = sl_best_weights_path(eid)
        if os.path.exists(final_path):
            final_score = score_full_benchmark(final_path)
            final_bg = check_backgammon_rate(final_path)
            print(f'[SL {eid}] Final full benchmark: {final_score:.2f}, '
                  f'BG rate: {final_bg:.1%}', flush=True)
            results[eid]['sl_phases'] = sl_results
            results[eid]['final_score'] = final_score
            results[eid]['final_bg_rate'] = final_bg

    # ========== Summary ==========
    print('\n' + '=' * 70)
    print('TRAINING SCHEDULE EXPERIMENT SUMMARY')
    print('=' * 70)
    print(f'{"Exp":>4s}  {"TD Score":>10s}  {"BG%":>6s}  ', end='')
    print(f'{"SL P1":>8s}  {"SL P2":>8s}  {"SL P3":>8s}  {"Final":>8s}')
    print(f'{"---":>4s}  {"--------":>10s}  {"----":>6s}  ', end='')
    print(f'{"-----":>8s}  {"-----":>8s}  {"-----":>8s}  {"-----":>8s}')

    for eid in exp_ids:
        r = results.get(eid, {})
        td = f'{r["td_score"]:.2f}' if r.get('td_score') is not None else '-'
        bg = f'{r["bg_rate"]:.1%}' if r.get('bg_rate') is not None else '-'

        sl_scores = []
        for phase in r.get('sl_phases', []):
            sl_scores.append(f'{phase["best_score"]:.2f}')
        while len(sl_scores) < 3:
            sl_scores.append('-')

        final = f'{r["final_score"]:.2f}' if r.get('final_score') is not None else '-'

        print(f'{eid:>4s}  {td:>10s}  {bg:>6s}  '
              f'{sl_scores[0]:>8s}  {sl_scores[1]:>8s}  {sl_scores[2]:>8s}  {final:>8s}')

    print('=' * 70)
    print()

    save_results(results, timestamp)


def save_results(results, timestamp):
    """Save results to JSON."""
    results_path = os.path.join(RESULTS_DIR, f'schedule_{timestamp}.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Add experiment descriptions
    output = {}
    for eid, r in results.items():
        exp = EXPERIMENTS.get(eid, {})
        output[eid] = {
            'description': exp.get('desc', ''),
            'td_phases': exp.get('td_phases', []),
            'sl_phases_config': exp.get('sl_phases', []),
            **r,
        }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved to: {results_path}')


if __name__ == '__main__':
    main()
