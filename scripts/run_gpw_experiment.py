"""
Experiment: Game Plan Weight Sensitivity

Tests different --gameplan-weight values for SL training of the 4 contact game plan NNs
(racing/attacking/priming/anchoring). PureRace is excluded since it has its own training data.

All runs start from TD weights (td_gp_244_1200k_*.weights), NOT from existing SL best.
Uses shortened SL schedule: 200 epochs @ alpha=20, 100 @ 6.3, 50 @ 2.0.

Tested weights: 1.0, 1.5, 3.0, 5.0
(Our current best used weight=2.0 with longer schedule: 200@20, 500@6.3, 500@2.0)

Usage:
    python python/run_gpw_experiment.py              # Run all weights
    python python/run_gpw_experiment.py --weights 1 3 # Run specific weights
    python python/run_gpw_experiment.py --score-only  # Just score existing results
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))

# Use isolated copy of .pyd so the main build/ is not locked by this process.
# The other session can rebuild freely while this experiment runs.
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
EXPERIMENT_DIR = os.path.join(project_dir, 'experiments', 'gpw_sensitivity')

# Network config
NN_TYPES = ['racing', 'attacking', 'priming', 'anchoring']
GAMEPLAN_IDS = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}
N_INPUTS = 244
N_HIDDEN = 250

# SL schedule (shortened)
SL_PHASES = [
    (200, 20.0),   # Phase 1: 200 epochs @ alpha=20
    (100, 6.3),    # Phase 2: 100 epochs @ alpha=6.3
    (50,  2.0),    # Phase 3: 50 epochs @ alpha=2.0
]

# PureRace weights (fixed, not retrained)
PURERACE_WEIGHTS = os.path.join(MODELS_DIR, 'sl_purerace.weights.best')
N_HIDDEN_PURERACE = 120


def model_prefix(gpw):
    """Model name prefix for a given gameplan weight, e.g. 'sl_gpw1.0'."""
    return f'sl_gpw{gpw:.1f}'


def load_training_data():
    """Load contact+crashed training data (shared across all experiments)."""
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


def load_benchmarks():
    """Load per-plan and overall benchmarks."""
    benchmarks = {}
    # Per-plan benchmarks (for progress during training, use step=10 for speed)
    for bm_type in NN_TYPES:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if os.path.exists(bm_path):
            benchmarks[bm_type] = load_benchmark_file(bm_path, step=10)
    return benchmarks


def classify_game_plans(boards):
    """Classify all boards by game plan for sample weighting."""
    print('Classifying game plans...')
    t0 = time.time()
    gp_ids = bgbot_cpp.classify_game_plans_batch(boards)
    for nn_type, gp_id in GAMEPLAN_IDS.items():
        n_match = int(np.sum(gp_ids == gp_id))
        print(f'  {nn_type}: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
    print(f'  Classified in {time.time()-t0:.1f}s')
    return gp_ids


def train_one_nn(nn_type, gpw, boards, targets, gp_ids, benchmark_scenarios):
    """Train a single NN through all 3 SL phases, starting from TD weights."""
    prefix = model_prefix(gpw)
    td_weights = os.path.join(MODELS_DIR, f'td_gp_244_1200k_{nn_type}.weights')
    save_path = os.path.join(MODELS_DIR, f'{prefix}_{nn_type}.weights')

    if not os.path.exists(td_weights):
        print(f'  ERROR: TD weights not found: {td_weights}')
        return None

    # Build sample weights
    target_gp = GAMEPLAN_IDS[nn_type]
    sample_weights = np.ones(len(boards), dtype=np.float32)
    if gpw != 1.0:
        sample_weights[gp_ids == target_gp] = gpw

    print(f'\n  --- {nn_type} (gpw={gpw}) ---')
    print(f'  TD weights: {td_weights}')
    print(f'  Save path:  {save_path}')

    current_weights = td_weights
    best_score = float('inf')
    best_epoch_total = 0
    total_time = 0.0
    epoch_offset = 0

    for phase_idx, (epochs, alpha) in enumerate(SL_PHASES):
        phase_label = f'Phase {phase_idx+1}'
        print(f'  {phase_label}: {epochs} epochs @ alpha={alpha}')

        result = bgbot_cpp.cuda_supervised_train(
            boards=boards,
            targets=targets,
            weights_path=current_weights,
            n_hidden=N_HIDDEN,
            n_inputs=N_INPUTS,
            alpha=alpha,
            epochs=epochs,
            batch_size=128,
            seed=42,
            print_interval=50,  # Print every 50 epochs (brief)
            save_path=save_path,
            benchmark_scenarios=benchmark_scenarios,
            sample_weights=sample_weights if gpw != 1.0 else None,
        )

        phase_best = result['best_score']
        phase_best_epoch = result['best_epoch']
        phase_time = result['total_seconds']
        total_time += phase_time

        if phase_best < best_score:
            best_score = phase_best
            best_epoch_total = epoch_offset + phase_best_epoch

        print(f'  {phase_label} done: best={phase_best:.2f} (epoch {phase_best_epoch}), time={phase_time:.1f}s')

        # Next phase resumes from best of this phase
        current_weights = save_path + '.best'
        epoch_offset += epochs

    print(f'  FINAL: {nn_type} gpw={gpw} -> best={best_score:.2f} (epoch {best_epoch_total}), total time={total_time:.1f}s')
    return {
        'best_score': best_score,
        'best_epoch': best_epoch_total,
        'total_time': total_time,
    }


def train_all_for_weight(gpw, boards, targets, gp_ids, benchmarks):
    """Train all 4 contact NNs for a given gameplan weight."""
    print(f'\n{"="*60}')
    print(f'  GAMEPLAN WEIGHT = {gpw}')
    print(f'{"="*60}')

    results = {}
    for nn_type in NN_TYPES:
        bm_scenarios = benchmarks.get(nn_type)
        result = train_one_nn(nn_type, gpw, boards, targets, gp_ids, bm_scenarios)
        results[nn_type] = result

    return results


def score_configuration(gpw):
    """Score a trained configuration using the 5-NN benchmark (purerace from best + 4 new)."""
    prefix = model_prefix(gpw)
    purerace_w = PURERACE_WEIGHTS
    racing_w = os.path.join(MODELS_DIR, f'{prefix}_racing.weights.best')
    attacking_w = os.path.join(MODELS_DIR, f'{prefix}_attacking.weights.best')
    priming_w = os.path.join(MODELS_DIR, f'{prefix}_priming.weights.best')
    anchoring_w = os.path.join(MODELS_DIR, f'{prefix}_anchoring.weights.best')

    # Check all files exist
    for path in [purerace_w, racing_w, attacking_w, priming_w, anchoring_w]:
        if not os.path.exists(path):
            print(f'  Missing: {path}')
            return None

    scores = {}

    # Per-plan benchmarks
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            continue
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN)
        scores[bm_type] = result.score()

    # Contact, crashed, and race overall
    for bm_name in ['contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            continue
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN)
        scores[bm_name] = result.score()

    # vs PubEval
    stats = bgbot_cpp.play_games_5nn_vs_pubeval(
        purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN,
        n_games=10000, seed=42)
    scores['vs_pubeval'] = stats.avg_ppg()

    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Game Plan Weight Sensitivity Experiment')
    parser.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.5, 3.0, 5.0],
                        help='Gameplan weights to test (default: 1.0 1.5 3.0 5.0)')
    parser.add_argument('--score-only', action='store_true',
                        help='Skip training, just score existing weight files')
    args = parser.parse_args()

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_results = {}

    if not args.score_only:
        # Verify CUDA
        if not bgbot_cpp.cuda_available():
            print('ERROR: CUDA not available')
            sys.exit(1)
        print('CUDA GPU detected')

        # Load data once (shared across all experiments)
        boards, targets = load_training_data()
        gp_ids = classify_game_plans(boards)
        benchmarks = load_benchmarks()

        # Train each weight configuration
        for gpw in args.weights:
            train_results = train_all_for_weight(gpw, boards, targets, gp_ids, benchmarks)
            all_results[f'gpw_{gpw}'] = {'training': {k: v for k, v in train_results.items()}}

    # Score all configurations
    print(f'\n{"="*60}')
    print('  SCORING ALL CONFIGURATIONS')
    print(f'{"="*60}\n')

    for gpw in args.weights:
        print(f'\n--- gpw={gpw} ---')
        scores = score_configuration(gpw)
        if scores:
            key = f'gpw_{gpw}'
            if key not in all_results:
                all_results[key] = {}
            all_results[key]['scores'] = scores

            for name, score in scores.items():
                if name == 'vs_pubeval':
                    print(f'  {name:12s}: {score:+.3f}')
                else:
                    print(f'  {name:12s}: {score:8.2f}')

    # Also score the current best (gpw=2.0 with full schedule) for reference
    print(f'\n--- Reference: current best (sl_*.weights.best, gpw=2.0 full schedule) ---')
    ref_scores = {}
    purerace_w = PURERACE_WEIGHTS
    racing_w = os.path.join(MODELS_DIR, 'sl_racing.weights.best')
    attacking_w = os.path.join(MODELS_DIR, 'sl_attacking.weights.best')
    priming_w = os.path.join(MODELS_DIR, 'sl_priming.weights.best')
    anchoring_w = os.path.join(MODELS_DIR, 'sl_anchoring.weights.best')

    for bm_name in ['contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if os.path.exists(bm_path):
            scenarios = load_benchmark_file(bm_path)
            result = bgbot_cpp.score_benchmarks_5nn(
                scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN)
            ref_scores[bm_name] = result.score()

    stats = bgbot_cpp.play_games_5nn_vs_pubeval(
        purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        N_HIDDEN_PURERACE, N_HIDDEN, N_HIDDEN, N_HIDDEN, N_HIDDEN,
        n_games=10000, seed=42)
    ref_scores['vs_pubeval'] = stats.avg_ppg()

    for name, score in ref_scores.items():
        if name == 'vs_pubeval':
            print(f'  {name:12s}: {score:+.3f}')
        else:
            print(f'  {name:12s}: {score:8.2f}')

    all_results['reference_gpw2_full'] = {'scores': ref_scores}

    # Print comparison table
    print(f'\n{"="*60}')
    print('  COMPARISON TABLE')
    print(f'{"="*60}\n')

    weights_tested = args.weights
    header = f'{"Metric":<14s}'
    for gpw in weights_tested:
        header += f'  gpw={gpw:<4s}'
    header += '  ref(2.0)'
    print(header)
    print('-' * len(header))

    for metric in ['racing', 'attacking', 'priming', 'anchoring', 'contact', 'crashed', 'race', 'vs_pubeval']:
        row = f'{metric:<14s}'
        for gpw in weights_tested:
            key = f'gpw_{gpw}'
            if key in all_results and 'scores' in all_results[key]:
                val = all_results[key]['scores'].get(metric)
                if val is not None:
                    if metric == 'vs_pubeval':
                        row += f'  {val:+.3f} '
                    else:
                        row += f'  {val:6.2f} '
                else:
                    row += f'  {"N/A":>7s}'
            else:
                row += f'  {"N/A":>7s}'
        # Reference
        if metric in ref_scores:
            val = ref_scores[metric]
            if metric == 'vs_pubeval':
                row += f'  {val:+.3f}'
            else:
                row += f'  {val:6.2f}'
        print(row)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(EXPERIMENT_DIR, f'results_{timestamp}.json')
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    main()
