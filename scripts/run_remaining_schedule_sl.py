"""Run SL training for remaining schedule experiments (D, E, F, G, H).
Each experiment runs 3 SL phases sequentially (GPU), scoring on full benchmark after."""

import os
import sys
import time
import json
import numpy as np

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

N_HIDDEN = 80
N_INPUTS = 196
BASE_SEED = 42

# Experiments to run and their SL configs
EXPERIMENTS = {
    'D': {
        'td_weights': os.path.join(MODELS_DIR, 'sched_D.weights'),
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'E': {
        'td_weights': os.path.join(MODELS_DIR, 'sched_E.weights'),
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'F': {
        'td_weights': os.path.join(MODELS_DIR, 'sched_F.weights'),
        'sl_phases': [(200, 20.0), (200, 6.3), (200, 2.0)],
    },
    'G': {
        'td_weights': os.path.join(MODELS_DIR, 'sched_G.weights'),
        'sl_phases': [(200, 10.0), (200, 3.0), (200, 1.0)],
    },
    'H': {
        'td_weights': os.path.join(MODELS_DIR, 'sched_H.weights'),
        'sl_phases': [(400, 10.0)],
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS.keys()))
    args = parser.parse_args()

    exp_ids = args.experiments

    # Load training data
    print('Loading training data...', flush=True)
    t0 = time.time()
    boards1, targets1 = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    boards2, targets2 = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    boards = np.concatenate([boards1, boards2])
    targets = np.concatenate([targets1, targets2])
    print(f'  Loaded {len(boards)} positions in {time.time()-t0:.1f}s', flush=True)

    # Load benchmark (1/10 for during-training scoring)
    bm_sub = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'), step=10)
    # Full benchmark for final scoring
    bm_full = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'))
    print(f'  Benchmark: {len(bm_sub)} (subsample), {len(bm_full)} (full)', flush=True)

    all_results = {}

    for eid in exp_ids:
        if eid not in EXPERIMENTS:
            print(f'Unknown experiment: {eid}', flush=True)
            continue

        exp = EXPERIMENTS[eid]
        td_weights = exp['td_weights']

        if not os.path.exists(td_weights):
            print(f'\n[{eid}] TD weights missing: {td_weights}', flush=True)
            continue

        sl_desc = ' -> '.join(f'{e}ep@{a}' for e, a in exp['sl_phases'])
        print(f'\n{"="*60}', flush=True)
        print(f'Experiment {eid}: SL [{sl_desc}]', flush=True)
        print(f'{"="*60}', flush=True)

        current_weights = td_weights
        save_path = os.path.join(MODELS_DIR, f'sl_sched_{eid}.weights')
        best_path = os.path.join(MODELS_DIR, f'sl_sched_{eid}.weights.best')
        phase_results = []
        total_sl_time = 0

        for phase_idx, (epochs, alpha) in enumerate(exp['sl_phases']):
            phase_num = phase_idx + 1
            print(f'\n[{eid}] Phase {phase_num}: {epochs} epochs @ alpha={alpha}', flush=True)
            print(f'  Starting from: {os.path.basename(current_weights)}', flush=True)

            t0 = time.time()
            result = bgbot_cpp.cuda_supervised_train(
                boards, targets,
                weights_path=current_weights,
                n_hidden=N_HIDDEN,
                n_inputs=N_INPUTS,
                alpha=alpha,
                epochs=epochs,
                batch_size=128,
                seed=BASE_SEED,
                print_interval=10,
                save_path=save_path,
                benchmark_scenarios=bm_sub,
            )
            elapsed = time.time() - t0
            total_sl_time += elapsed

            phase_results.append({
                'phase': phase_num,
                'epochs': epochs,
                'alpha': alpha,
                'best_score': result['best_score'],
                'best_epoch': result['best_epoch'],
                'time': elapsed,
            })

            print(f'[{eid}] Phase {phase_num}: best={result["best_score"]:.2f} '
                  f'@ epoch {result["best_epoch"]} ({elapsed:.1f}s)', flush=True)

            # Use best weights for next phase
            if os.path.exists(best_path):
                current_weights = best_path

        # Final full benchmark score
        if os.path.exists(best_path):
            final_result = bgbot_cpp.score_benchmarks_nn(bm_full, best_path, N_HIDDEN, N_INPUTS)
            final_score = final_result.score()
            print(f'\n[{eid}] Final full benchmark: {final_score:.2f} '
                  f'(total SL time: {total_sl_time:.1f}s)', flush=True)

            all_results[eid] = {
                'sl_phases': phase_results,
                'final_score': final_score,
                'total_sl_time': total_sl_time,
            }

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'{"Exp":>4s}  {"P1 Best":>8s}  {"P2 Best":>8s}  {"P3 Best":>8s}  '
          f'{"Final":>8s}  {"SL Time":>8s}')
    print(f'{"---":>4s}  {"-------":>8s}  {"-------":>8s}  {"-------":>8s}  '
          f'{"-----":>8s}  {"-------":>8s}')

    for eid in exp_ids:
        r = all_results.get(eid)
        if not r:
            print(f'{eid:>4s}  {"SKIP":>8s}')
            continue

        scores = [f'{p["best_score"]:.2f}' for p in r['sl_phases']]
        while len(scores) < 3:
            scores.append('-')

        final_str = f'{r["final_score"]:.2f}'
        print(f'{eid:>4s}  {scores[0]:>8s}  {scores[1]:>8s}  {scores[2]:>8s}  '
              f'{final_str:>8s}  {r["total_sl_time"]:.0f}s')

    # Save results
    results_path = os.path.join(project_dir, 'experiments', 'schedule',
                                f'sl_remaining_{time.strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    main()
