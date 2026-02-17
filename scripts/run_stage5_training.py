"""
Stage 5 Training: Larger hidden layers (200h purerace, 400h contact NNs).

Same architecture and training pipeline as Stage 4, but with:
  - PureRace: 200 hidden (was 120), 196 Tesauro inputs
  - Racing/Attacking/Priming/Anchoring: 400 hidden (was 250), 244 extended inputs

Uses optimal per-NN game plan weights from gpw experiment:
  gpw=5.0 for racing/attacking/priming, gpw=1.5 for anchoring.

Training pipeline:
  TD: 200k@0.1 + 1M@0.02 (from scratch)
  SL: Racing/Attacking/Priming: 100ep@a=20 -> 200ep@a=10 -> 200ep@a=3.1 -> 2000ep@a=1.0
      Anchoring:                200ep@a=20 -> 200ep@a=6.3 -> 1000ep@a=2.0
      PureRace:                 200ep@a=20 -> 500ep@a=6.3 -> 500ep@a=2.0

Usage:
    python python/run_stage5_training.py                  # Full pipeline: TD + SL + score
    python python/run_stage5_training.py --sl-only        # Skip TD, use existing TD weights
    python python/run_stage5_training.py --score-only     # Just score existing weights
    python python/run_stage5_training.py --nn racing anchoring  # Train only specific NNs
"""

import os
import sys
import json
import time
import shutil
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))

# Use isolated copy of .pyd so the main build/ is not locked by this process.
isolated_build_dir = os.path.join(project_dir, 'experiments', 'gpw_sensitivity', 'isolated_build')
build_dir = isolated_build_dir if os.path.isdir(isolated_build_dir) else os.path.join(project_dir, 'build')

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

# Stage 5 network config — larger hidden layers
N_INPUTS = 244
N_HIDDEN = 400            # was 250 in Stage 4
N_HIDDEN_PURERACE = 200   # was 120 in Stage 4
N_INPUTS_PURERACE = 196

MODEL_PREFIX = 'sl_s5'    # Stage 5
TD_MODEL_NAME = 'td_s5'   # TD weight prefix

GAMEPLAN_IDS = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}

# Per-NN training configs (same schedules as Stage 4)
CONFIGS = {
    'purerace': {
        'gpw': 1.0,
        'phases': [(200, 20.0), (200, 6.3), (1000, 2.0)],
    },
    'racing': {
        'gpw': 5.0,
        'phases': [(100, 20.0), (200, 10.0), (200, 3.1), (2000, 1.0)],
    },
    'attacking': {
        'gpw': 5.0,
        'phases': [(100, 20.0), (200, 10.0), (200, 3.1), (2000, 1.0)],
    },
    'priming': {
        'gpw': 5.0,
        'phases': [(100, 20.0), (200, 10.0), (200, 3.1), (2000, 1.0)],
    },
    'anchoring': {
        'gpw': 1.5,
        'phases': [(200, 20.0), (200, 6.3), (1000, 2.0)],
    },
}


def run_td_training():
    """Run TD self-play training: 200k@0.1 + 1M@0.02."""

    print(f'\n{"="*60}')
    print(f'  TD SELF-PLAY TRAINING (Stage 5)')
    print(f'  PureRace: {N_HIDDEN_PURERACE}h, {N_INPUTS_PURERACE} inputs')
    print(f'  Contact NNs: {N_HIDDEN}h, {N_INPUTS} inputs')
    print(f'{"="*60}\n')

    # Load benchmark scenarios for progress tracking
    benchmark_sets = {}
    for bm_name in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if os.path.exists(bm_path):
            print(f'Loading {bm_name} benchmark (step=10)...')
            t0 = time.time()
            ss = load_benchmark_file(bm_path, step=10)
            print(f'  Loaded {len(ss)} scenarios in {time.time()-t0:.1f}s')
            benchmark_sets[bm_name] = ss
        else:
            benchmark_sets[bm_name] = None
    print()

    # Phase 1: 200k @ alpha=0.1
    print(f'=== TD Phase 1: 200k @ alpha=0.1 ===')
    print(f'  PureRace:  {N_HIDDEN_PURERACE} hidden, {N_INPUTS_PURERACE} inputs')
    print(f'  Racing:    {N_HIDDEN} hidden, {N_INPUTS} inputs')
    print(f'  Attacking: {N_HIDDEN} hidden, {N_INPUTS} inputs')
    print(f'  Priming:   {N_HIDDEN} hidden, {N_INPUTS} inputs')
    print(f'  Anchoring: {N_HIDDEN} hidden, {N_INPUTS} inputs')
    print(flush=True)

    result1 = bgbot_cpp.td_train_gameplan(
        n_games=200000,
        alpha=0.1,
        n_hidden_purerace=N_HIDDEN_PURERACE,
        n_hidden_racing=N_HIDDEN,
        n_hidden_attacking=N_HIDDEN,
        n_hidden_priming=N_HIDDEN,
        n_hidden_anchoring=N_HIDDEN,
        eps=0.1,
        seed=42,
        benchmark_interval=10000,
        model_name=TD_MODEL_NAME,
        models_dir=MODELS_DIR,
        purerace_benchmark=benchmark_sets.get('purerace'),
        attacking_benchmark=benchmark_sets.get('attacking'),
        priming_benchmark=benchmark_sets.get('priming'),
        anchoring_benchmark=benchmark_sets.get('anchoring'),
        race_benchmark=benchmark_sets.get('racing'),
    )
    print(f'Phase 1 done: {result1.games_played} games in {result1.total_seconds:.1f}s')

    # Phase 2: 1M @ alpha=0.02 (resume from Phase 1)
    print()
    td_phase2_name = f'{TD_MODEL_NAME}_1200k'
    print(f'=== TD Phase 2: 1M @ alpha=0.02 ===')
    print(flush=True)

    result2 = bgbot_cpp.td_train_gameplan(
        n_games=1000000,
        alpha=0.02,
        n_hidden_purerace=N_HIDDEN_PURERACE,
        n_hidden_racing=N_HIDDEN,
        n_hidden_attacking=N_HIDDEN,
        n_hidden_priming=N_HIDDEN,
        n_hidden_anchoring=N_HIDDEN,
        eps=0.1,
        seed=42,
        benchmark_interval=10000,
        model_name=td_phase2_name,
        models_dir=MODELS_DIR,
        resume_purerace=os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_purerace.weights'),
        resume_racing=os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_racing.weights'),
        resume_attacking=os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_attacking.weights'),
        resume_priming=os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_priming.weights'),
        resume_anchoring=os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_anchoring.weights'),
        purerace_benchmark=benchmark_sets.get('purerace'),
        attacking_benchmark=benchmark_sets.get('attacking'),
        priming_benchmark=benchmark_sets.get('priming'),
        anchoring_benchmark=benchmark_sets.get('anchoring'),
        race_benchmark=benchmark_sets.get('racing'),
    )
    print(f'Phase 2 done: {result2.games_played} games in {result2.total_seconds:.1f}s')
    total_td_time = result1.total_seconds + result2.total_seconds
    print(f'Total TD time: {total_td_time:.1f}s ({total_td_time/60:.1f}m)')

    return td_phase2_name


def load_training_data():
    """Load contact+crashed training data."""
    print('Loading training data...')
    t0 = time.time()
    boards_c, targets_c = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    print(f'  contact-train-data: {len(boards_c)} positions ({time.time()-t0:.1f}s)')
    t0 = time.time()
    boards_r, targets_r = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    print(f'  crashed-train-data: {len(boards_r)} positions ({time.time()-t0:.1f}s)')
    boards = np.concatenate([boards_c, boards_r], axis=0)
    targets = np.concatenate([targets_c, targets_r], axis=0)
    print(f'  Total: {len(boards)} positions')
    return boards, targets


def load_purerace_training_data():
    """Load purerace training data."""
    print('Loading purerace training data...')
    t0 = time.time()
    boards, targets = load_gnubg_training_data(os.path.join(DATA_DIR, 'purerace-train-data'))
    print(f'  purerace-train-data: {len(boards)} positions ({time.time()-t0:.1f}s)')
    return boards, targets


def load_benchmarks():
    """Load per-plan benchmarks (step=10 for training progress)."""
    benchmarks = {}
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if os.path.exists(bm_path):
            benchmarks[bm_type] = load_benchmark_file(bm_path, step=10)
    return benchmarks


def classify_game_plans(boards):
    """Classify all boards by game plan."""
    print('Classifying game plans...')
    t0 = time.time()
    gp_ids = bgbot_cpp.classify_game_plans_batch(boards)
    for nn_type, gp_id in GAMEPLAN_IDS.items():
        n_match = int(np.sum(gp_ids == gp_id))
        print(f'  {nn_type}: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
    print(f'  Classified in {time.time()-t0:.1f}s')
    return gp_ids


def train_one_nn(nn_type, config, boards, targets, gp_ids, benchmark_scenarios,
                 td_model_name):
    """Train a single NN through all SL phases."""
    gpw = config['gpw']
    phases = config['phases']

    if nn_type == 'purerace':
        td_weights = os.path.join(MODELS_DIR, f'{td_model_name}_purerace.weights')
        n_hidden = N_HIDDEN_PURERACE
        n_inputs = N_INPUTS_PURERACE
    else:
        td_weights = os.path.join(MODELS_DIR, f'{td_model_name}_{nn_type}.weights')
        n_hidden = N_HIDDEN
        n_inputs = N_INPUTS

    save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{nn_type}.weights')

    if not os.path.exists(td_weights):
        print(f'  ERROR: TD weights not found: {td_weights}')
        return None

    # Build sample weights
    sample_weights = None
    n_match = len(boards)
    if nn_type != 'purerace' and gpw != 1.0:
        target_gp = GAMEPLAN_IDS[nn_type]
        sample_weights = np.ones(len(boards), dtype=np.float32)
        sample_weights[gp_ids == target_gp] = gpw
        n_match = int(np.sum(gp_ids == target_gp))

    total_epochs = sum(ep for ep, _ in phases)
    phase_desc = ' -> '.join(f'{ep}ep@a={a}' for ep, a in phases)
    print(f'\n{"="*60}')
    print(f'  {nn_type.upper()} ({n_hidden}h, {n_inputs} inputs, gpw={gpw}, {total_epochs} total epochs)')
    print(f'  Schedule: {phase_desc}')
    print(f'  TD weights: {td_weights}')
    print(f'  Save path:  {save_path}')
    if nn_type != 'purerace':
        print(f'  Matching positions: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
    else:
        print(f'  Training positions: {len(boards)}')
    print(f'{"="*60}')

    current_weights = td_weights
    best_score = float('inf')
    best_epoch_total = 0
    total_time = 0.0
    epoch_offset = 0
    phase_results = []

    for phase_idx, (epochs, alpha) in enumerate(phases):
        phase_label = f'Phase {phase_idx+1}/{len(phases)}'
        print(f'\n  {phase_label}: {epochs} epochs @ alpha={alpha}')

        print_interval = min(10, max(1, epochs // 20))

        result = bgbot_cpp.cuda_supervised_train(
            boards=boards,
            targets=targets,
            weights_path=current_weights,
            n_hidden=n_hidden,
            n_inputs=n_inputs,
            alpha=alpha,
            epochs=epochs,
            batch_size=128,
            seed=42,
            print_interval=print_interval,
            save_path=save_path,
            benchmark_scenarios=benchmark_scenarios,
            sample_weights=sample_weights,
        )

        phase_best = result['best_score']
        phase_best_epoch = result['best_epoch']
        phase_time = result['total_seconds']
        total_time += phase_time

        if phase_best < best_score:
            best_score = phase_best
            best_epoch_total = epoch_offset + phase_best_epoch

        phase_results.append({
            'phase': phase_idx + 1,
            'epochs': epochs,
            'alpha': alpha,
            'best_score': phase_best,
            'best_epoch': phase_best_epoch,
            'time': phase_time,
        })
        print(f'  {phase_label} done: best={phase_best:.2f} (epoch {phase_best_epoch}), time={phase_time:.1f}s')

        # Next phase resumes from best of this phase
        current_weights = save_path + '.best'
        epoch_offset += epochs

    print(f'\n  FINAL: {nn_type} {n_hidden}h gpw={gpw} -> best={best_score:.2f} (epoch {best_epoch_total}), total time={total_time:.1f}s')
    return {
        'best_score': best_score,
        'best_epoch': best_epoch_total,
        'total_time': total_time,
        'phases': phase_results,
    }


def score_stage5():
    """Score the Stage 5 trained models."""
    purerace_w = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_purerace.weights.best')
    racing_w = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_racing.weights.best')
    attacking_w = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_attacking.weights.best')
    priming_w = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_priming.weights.best')
    anchoring_w = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_anchoring.weights.best')

    for label, path in [('purerace', purerace_w), ('racing', racing_w),
                         ('attacking', attacking_w), ('priming', priming_w),
                         ('anchoring', anchoring_w)]:
        if not os.path.exists(path):
            print(f'  Missing: {path}')
            return None

    scores = {}

    print(f'\n{"="*60}')
    print(f'  STAGE 5 BENCHMARKS')
    print(f'{"="*60}\n')
    print(f'  PureRace:  {purerace_w} ({N_HIDDEN_PURERACE}h)')
    print(f'  Racing:    {racing_w} ({N_HIDDEN}h)')
    print(f'  Attacking: {attacking_w} ({N_HIDDEN}h)')
    print(f'  Priming:   {priming_w} ({N_HIDDEN}h)')
    print(f'  Anchoring: {anchoring_w} ({N_HIDDEN}h)')
    print()

    # Per-plan benchmarks
    print('--- Game Plan benchmarks (full) ---')
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN)
        t_score = time.time() - t0
        scores[bm_type] = result.score()
        print(f'  {bm_type:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # Old-style benchmarks
    print()
    print('--- Old-style benchmarks (for comparison) ---')
    for bm_name in ['contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN)
        t_score = time.time() - t0
        scores[bm_name] = result.score()
        print(f'  {bm_name:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # vs PubEval
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_5nn_vs_pubeval(
        purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN,
        n_games=10000, seed=42)
    t_games = time.time() - t0
    scores['vs_pubeval'] = stats.avg_ppg()
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')

    # Self-play
    print()
    print('=== Self-play outcome distribution (10k games) ===')
    t0 = time.time()
    ss = bgbot_cpp.play_games_5nn_vs_self(
        purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN,
        n_games=10000, seed=42)
    t_sp = time.time() - t0
    total = ss.n_games
    singles = ss.p1_wins + ss.p2_wins
    gammons = ss.p1_gammons + ss.p2_gammons
    backgammons = ss.p1_backgammons + ss.p2_backgammons
    print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
          f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
          f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
          f'({total} games in {t_sp:.1f}s)')

    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage 5 Training (200h purerace, 400h contact NNs)')
    parser.add_argument('--score-only', action='store_true',
                        help='Skip training, just score existing weight files')
    parser.add_argument('--sl-only', action='store_true',
                        help='Skip TD training, use existing TD weights')
    parser.add_argument('--nn', type=str, nargs='+', default=None,
                        help='Train only specific NNs (e.g., --nn racing anchoring purerace)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not args.score_only:
        if not bgbot_cpp.cuda_available():
            print('ERROR: CUDA not available')
            sys.exit(1)
        print('CUDA GPU detected')
        print()

        # TD training
        td_model_name = f'{TD_MODEL_NAME}_1200k'
        if not args.sl_only:
            td_model_name = run_td_training()
        else:
            print(f'Skipping TD training, using existing weights: {td_model_name}')
            # Verify TD weights exist
            for nn_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
                td_path = os.path.join(MODELS_DIR, f'{td_model_name}_{nn_type}.weights')
                if not os.path.exists(td_path):
                    print(f'  ERROR: TD weights not found: {td_path}')
                    sys.exit(1)
            print('  All TD weights found.')
            print()

        # Determine which NNs to train
        nn_types = args.nn if args.nn else ['purerace', 'racing', 'attacking', 'priming', 'anchoring']

        # Load data
        benchmarks = load_benchmarks()
        all_results = {}

        # Train purerace separately (different data)
        if 'purerace' in nn_types:
            pr_boards, pr_targets = load_purerace_training_data()
            pr_benchmark = benchmarks.get('purerace')
            result = train_one_nn('purerace', CONFIGS['purerace'],
                                  pr_boards, pr_targets, None, pr_benchmark,
                                  td_model_name)
            if result:
                all_results['purerace'] = result

        # Train contact NNs (share data)
        contact_types = [t for t in nn_types if t != 'purerace']
        if contact_types:
            boards, targets = load_training_data()
            gp_ids = classify_game_plans(boards)

            for nn_type in contact_types:
                if nn_type not in CONFIGS:
                    print(f'Unknown NN type: {nn_type}')
                    continue
                config = CONFIGS[nn_type]
                bm_scenarios = benchmarks.get(nn_type)
                result = train_one_nn(nn_type, config, boards, targets, gp_ids,
                                      bm_scenarios, td_model_name)
                if result:
                    all_results[nn_type] = result

        # Print training summary
        print(f'\n{"="*60}')
        print(f'  TRAINING SUMMARY')
        print(f'{"="*60}\n')
        for nn_type, result in all_results.items():
            config = CONFIGS[nn_type]
            n_h = N_HIDDEN_PURERACE if nn_type == 'purerace' else N_HIDDEN
            print(f'  {nn_type:10s}: best={result["best_score"]:.2f} (epoch {result["best_epoch"]}), '
                  f'{n_h}h, gpw={config["gpw"]}, time={result["total_time"]:.0f}s')
        total_time = sum(r['total_time'] for r in all_results.values())
        print(f'\n  Total SL training time: {total_time:.0f}s ({total_time/60:.1f}m)')

    # Score
    scores = score_stage5()
    if scores:
        # Save results
        results_dir = os.path.join(project_dir, 'experiments', 'stage5')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    main()
