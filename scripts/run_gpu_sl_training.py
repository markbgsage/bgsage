"""
Run GPU-accelerated supervised learning training for backgammon neural networks.

Uses CUDA/cuBLAS for forward pass, backprop, and weight updates on GPU.
Benchmarking is done on CPU by copying weights back each epoch.

Usage:
    python run_gpu_sl_training.py --type contact --epochs 1000 --alpha 1.0
    python run_gpu_sl_training.py --type crashed --epochs 500 --alpha 1.0
    python run_gpu_sl_training.py --type race --epochs 100 --alpha 0.1
    python run_gpu_sl_training.py --type all --epochs 200 --alpha 1.0
    python run_gpu_sl_training.py --type contact --resume models/sl_contact.weights.best
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    # Add CUDA DLL directory
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

# Default hidden sizes per network type
DEFAULT_HIDDEN = {
    'contact': 250,
    'crashed': 250,
    'race': 80,
    'purerace': 120,
    'racing': 250,
    'attacking': 250,
    'priming': 250,
    'anchoring': 250,
}


def find_best_weights(net_type):
    """Find the best available weights for a given network type, preferring SL > TD."""
    candidates = [
        os.path.join(MODELS_DIR, f'sl_{net_type}.weights.best'),
        os.path.join(MODELS_DIR, f'sl_{net_type}.weights'),
    ]
    # Add TD weight candidates based on type
    if net_type in ('contact', 'crashed', 'race'):
        candidates += [
            os.path.join(MODELS_DIR, f'td_multi_350k_{net_type}.weights'),
            os.path.join(MODELS_DIR, f'td_multi_{net_type}.weights'),
        ]
    else:
        # Game plan types: look for td weights (newest naming first)
        candidates += [
            os.path.join(MODELS_DIR, f'td_gp_244_final_{net_type}.weights'),
            os.path.join(MODELS_DIR, f'td_gp_244_1200k_{net_type}.weights'),
            os.path.join(MODELS_DIR, f'td_gameplan_1200k_{net_type}.weights'),
            os.path.join(MODELS_DIR, f'td_gameplan_{net_type}.weights'),
        ]
        # Also check .best variants from TD training
        candidates.append(os.path.join(MODELS_DIR, f'td_gp_244_final_{net_type}.weights.best'))
        candidates.append(os.path.join(MODELS_DIR, f'td_gp_244_1200k_{net_type}.weights.best'))
        candidates.append(os.path.join(MODELS_DIR, f'td_gameplan_1200k_{net_type}.weights.best'))
        candidates.append(os.path.join(MODELS_DIR, f'td_gameplan_{net_type}.weights.best'))
    for path in candidates:
        if os.path.exists(path):
            return path
    return ''


def get_hidden_size(net_type, args):
    """Get hidden size for a network type, using type-specific defaults."""
    if args.hidden is not None:
        return args.hidden
    return DEFAULT_HIDDEN.get(net_type, 120)


def train_network(net_type, args):
    """Train a single network type (contact, crashed, or race) using GPU."""

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Configuration per network type
    if net_type == 'single':
        # Single NN with 196 Tesauro inputs, all training data combined
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
            os.path.join(DATA_DIR, 'crashed-train-data'),
            os.path.join(DATA_DIR, 'race-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'contact.bm')
        n_inputs = 196  # TESAURO_INPUTS
        default_alpha = 1.0
        default_epochs = 200
        benchmark_step = 10
    elif net_type == 'contact':
        # Contact NN with 214 extended inputs, trained on contact data only
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'contact.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS
        default_alpha = 1.0
        default_epochs = 1000
        benchmark_step = 10
    elif net_type == 'crashed':
        # Crashed NN with 214 extended inputs, trained on crashed data only
        train_data_paths = [
            os.path.join(DATA_DIR, 'crashed-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'crashed.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS
        default_alpha = 1.0
        default_epochs = 500
        benchmark_step = 10
    elif net_type == 'race':
        train_data_paths = [os.path.join(DATA_DIR, 'race-train-data')]
        benchmark_path = os.path.join(DATA_DIR, 'race.bm')
        n_inputs = 196  # TESAURO_INPUTS
        default_alpha = 0.1
        default_epochs = 100
        benchmark_step = 10
    elif net_type == 'purerace':
        train_data_paths = [os.path.join(DATA_DIR, 'purerace-train-data')]
        benchmark_path = os.path.join(DATA_DIR, 'purerace.bm')
        n_inputs = 196  # TESAURO_INPUTS
        default_alpha = 0.1
        default_epochs = 100
        benchmark_step = 10
    elif net_type == 'racing':
        # All contact game plan NNs train on full contact+crashed data.
        # Only the benchmark differs per game plan.
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
            os.path.join(DATA_DIR, 'crashed-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'racing.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS (racing has contact)
        default_alpha = 1.0
        default_epochs = 500
        benchmark_step = 10
    elif net_type == 'attacking':
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
            os.path.join(DATA_DIR, 'crashed-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'attacking.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS
        default_alpha = 1.0
        default_epochs = 500
        benchmark_step = 10
    elif net_type == 'priming':
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
            os.path.join(DATA_DIR, 'crashed-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'priming.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS
        default_alpha = 1.0
        default_epochs = 500
        benchmark_step = 10
    elif net_type == 'anchoring':
        train_data_paths = [
            os.path.join(DATA_DIR, 'contact-train-data'),
            os.path.join(DATA_DIR, 'crashed-train-data'),
        ]
        benchmark_path = os.path.join(DATA_DIR, 'anchoring.bm')
        n_inputs = 244  # EXTENDED_CONTACT_INPUTS
        default_alpha = 1.0
        default_epochs = 500
        benchmark_step = 10
    else:
        raise ValueError(f"Unknown network type: {net_type}")

    alpha = args.alpha if args.alpha is not None else default_alpha
    epochs = args.epochs if args.epochs is not None else default_epochs
    n_hidden = get_hidden_size(net_type, args)

    # Determine starting weights
    weights_path = ''
    if args.resume:
        if args.resume.lower() == 'none':
            weights_path = ''  # Force random initialization
        else:
            weights_path = args.resume
    else:
        weights_path = find_best_weights(net_type)

    print(f'=== Training {net_type} network (GPU) ===')
    for p in train_data_paths:
        print(f'  Data:     {p}')
    print(f'  Inputs:   {n_inputs}')
    print(f'  Hidden:   {n_hidden}')
    print(f'  Alpha:    {alpha}')
    print(f'  Epochs:   {epochs}')
    print(f'  Batch:    {args.batch_size}')
    print(f'  Seed:     {args.seed}')
    if net_type in ('racing', 'attacking', 'priming', 'anchoring') and args.gameplan_weight != 1.0:
        print(f'  GP Weight: {args.gameplan_weight}')
    if weights_path:
        print(f'  Starting from: {weights_path}')
    else:
        print(f'  Starting from: random weights')
    print()

    # Load training data (combine multiple files if needed)
    import numpy as np
    print(f'Loading training data...')
    t0 = time.time()
    all_boards = []
    all_targets = []
    for path in train_data_paths:
        boards, targets = load_gnubg_training_data(path)
        print(f'  Loaded {len(boards)} positions from {os.path.basename(path)} in {time.time()-t0:.1f}s')
        all_boards.append(boards)
        all_targets.append(targets)
        t0 = time.time()
    boards = np.concatenate(all_boards, axis=0) if len(all_boards) > 1 else all_boards[0]
    targets = np.concatenate(all_targets, axis=0) if len(all_targets) > 1 else all_targets[0]
    print(f'  Total: {len(boards)} positions')

    # Build per-sample weights for game-plan-specific training
    gameplan_types = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}
    sample_weights = None
    if net_type in gameplan_types and args.gameplan_weight != 1.0:
        target_gp = gameplan_types[net_type]
        print(f'Classifying game plans for sample weighting (weight={args.gameplan_weight})...')
        t0 = time.time()
        gp_ids = bgbot_cpp.classify_game_plans_batch(boards)
        n_match = int(np.sum(gp_ids == target_gp))
        n_total = len(boards)
        sample_weights = np.ones(n_total, dtype=np.float32)
        sample_weights[gp_ids == target_gp] = args.gameplan_weight
        print(f'  {n_match}/{n_total} positions match {net_type} ({100*n_match/n_total:.1f}%)')
        print(f'  Classified in {time.time()-t0:.1f}s')

    # Load benchmark scenarios for progress tracking
    scenarios = None
    if os.path.exists(benchmark_path):
        print(f'Loading benchmark (step={benchmark_step})...')
        t0 = time.time()
        scenarios = load_benchmark_file(benchmark_path, step=benchmark_step)
        print(f'  Loaded {len(scenarios)} scenarios in {time.time()-t0:.1f}s')
    print()

    save_path = os.path.join(MODELS_DIR, f'sl_{net_type}.weights')

    # Run GPU supervised training
    print(f'--- Starting GPU SL: {net_type} ---', flush=True)
    result = bgbot_cpp.cuda_supervised_train(
        boards=boards,
        targets=targets,
        weights_path=weights_path,
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        alpha=alpha,
        epochs=epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        print_interval=args.print_interval,
        save_path=save_path,
        benchmark_scenarios=scenarios,
        sample_weights=sample_weights,
        label=net_type,
    )

    print()
    print(f'{net_type.capitalize()} training complete:')
    print(f'  Best score: {result["best_score"]:.2f}')
    print(f'  Best epoch: {result["best_epoch"]}')
    print(f'  Time: {result["total_seconds"]:.1f}s')
    print(f'  Weights: {save_path}')
    print(f'  Best weights: {save_path}.best')

    return result


def run_final_benchmarks(args):
    """Run full benchmarks on trained models."""

    if args.type in ('purerace', 'racing', 'attacking', 'priming', 'anchoring', 'gameplan'):
        # 5-NN game plan mode
        purerace_w = find_best_weights('purerace')
        racing_w = find_best_weights('racing')
        attacking_w = find_best_weights('attacking')
        priming_w = find_best_weights('priming')
        anchoring_w = find_best_weights('anchoring')

        missing = []
        if not purerace_w: missing.append('purerace')
        if not racing_w: missing.append('racing')
        if not attacking_w: missing.append('attacking')
        if not priming_w: missing.append('priming')
        if not anchoring_w: missing.append('anchoring')
        if missing:
            print(f'Cannot run final benchmarks: missing weights for {", ".join(missing)}')
            return

        n_h_purerace = get_hidden_size('purerace', args)
        n_h_racing = get_hidden_size('racing', args)
        n_h_attacking = get_hidden_size('attacking', args)
        n_h_priming = get_hidden_size('priming', args)
        n_h_anchoring = get_hidden_size('anchoring', args)

        print()
        print('=== Final Benchmark Scores (5-NN Game Plan Strategy) ===')
        print(f'  PureRace:  {purerace_w} ({n_h_purerace}h)')
        print(f'  Racing:    {racing_w} ({n_h_racing}h)')
        print(f'  Attacking: {attacking_w} ({n_h_attacking}h)')
        print(f'  Priming:   {priming_w} ({n_h_priming}h)')
        print(f'  Anchoring: {anchoring_w} ({n_h_anchoring}h)')
        print()

        print('--- Game Plan benchmarks (step=1) ---')
        for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
            bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
            if not os.path.exists(bm_path):
                print(f'  {bm_type:10s}: (not found)')
                continue
            t0 = time.time()
            scenarios = load_benchmark_file(bm_path)
            t_load = time.time() - t0
            t0 = time.time()
            result = bgbot_cpp.score_benchmarks_5nn(
                scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring)
            t_score = time.time() - t0
            print(f'  {bm_type:10s}: {result.score():8.2f}  '
                  f'({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')

        # Old-style benchmarks for comparison
        print()
        print('--- Old-style benchmarks (step=1, for comparison) ---')
        for bm_name in ['contact', 'race']:
            bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
            if not os.path.exists(bm_path):
                print(f'  {bm_name:10s}: (not found)')
                continue
            t0 = time.time()
            scenarios = load_benchmark_file(bm_path)
            t_load = time.time() - t0
            t0 = time.time()
            result = bgbot_cpp.score_benchmarks_5nn(
                scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring)
            t_score = time.time() - t0
            print(f'  {bm_name:10s}: {result.score():8.2f}  '
                  f'({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')

        # vs PubEval
        print()
        print('=== vs PubEval (10k games) ===')
        t0 = time.time()
        stats = bgbot_cpp.play_games_5nn_vs_pubeval(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring,
            n_games=10000, seed=args.seed)
        t_games = time.time() - t0
        print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
        print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
        print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')

        # Self-play outcome distribution
        print()
        print('=== Self-play outcome distribution (1k games) ===')
        t0 = time.time()
        ss = bgbot_cpp.play_games_5nn_vs_self(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring,
            n_games=1000, seed=args.seed)
        t_sp = time.time() - t0
        total = ss.n_games
        singles = ss.p1_wins + ss.p2_wins
        gammons = ss.p1_gammons + ss.p2_gammons
        backgammons = ss.p1_backgammons + ss.p2_backgammons
        print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
              f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
              f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
              f'({total} games in {t_sp:.1f}s)')
        return

    if args.type == 'single':
        # Single-NN mode: use same weights for all benchmarks
        weights = os.path.join(MODELS_DIR, 'sl_single.weights.best')
        if not os.path.exists(weights):
            weights = os.path.join(MODELS_DIR, 'sl_single.weights')
        if not os.path.exists(weights):
            print('Cannot run final benchmarks: missing single weights')
            return

        n_hidden = get_hidden_size('single', args)
        print()
        print(f'=== Final Benchmark Scores (Single NN, {n_hidden} hidden) ===')
        print(f'  Weights: {weights}')
        print()

        for bm_type in ['contact', 'race']:
            bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
            if not os.path.exists(bm_path):
                print(f'  {bm_type:8s}: (benchmark file not found)')
                continue

            t0 = time.time()
            scenarios = load_benchmark_file(bm_path)
            t_load = time.time() - t0

            t0 = time.time()
            result = bgbot_cpp.score_benchmarks_nn(scenarios, weights, n_hidden)
            t_score = time.time() - t0

            print(f'  {bm_type:8s}: {result.score():8.2f}  '
                  f'({result.count} scenarios, '
                  f'load {t_load:.1f}s + score {t_score:.1f}s)')

        # Play vs PubEval with single NN
        print()
        print('=== vs PubEval (10k games) ===')
        t0 = time.time()
        stats = bgbot_cpp.play_games_nn_vs_pubeval(
            weights, n_hidden, n_games=10000, seed=args.seed)
        t_games = time.time() - t0
        print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
        print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
        print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')
        return

    # 3-NN mode: find best weights for each network
    contact_weights = find_best_weights('contact')
    crashed_weights = find_best_weights('crashed')
    race_weights = find_best_weights('race')

    if not contact_weights or not crashed_weights or not race_weights:
        missing = []
        if not contact_weights: missing.append('contact')
        if not crashed_weights: missing.append('crashed')
        if not race_weights: missing.append('race')
        print(f'Cannot run final benchmarks: missing weights for {", ".join(missing)}')
        return

    n_hidden_contact = get_hidden_size('contact', args)
    n_hidden_crashed = get_hidden_size('crashed', args)
    n_hidden_race = get_hidden_size('race', args)

    print()
    print('=== Final Benchmark Scores (3-NN Strategy) ===')
    print(f'  Contact: {contact_weights} ({n_hidden_contact}h)')
    print(f'  Crashed: {crashed_weights} ({n_hidden_crashed}h)')
    print(f'  Race:    {race_weights} ({n_hidden_race}h)')
    print()

    for bm_type in ['contact', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            print(f'  {bm_type:8s}: (benchmark file not found)')
            continue

        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        t_load = time.time() - t0

        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_3nn(
            scenarios, contact_weights, crashed_weights, race_weights,
            n_hidden_contact, n_hidden_crashed, n_hidden_race)
        t_score = time.time() - t0

        print(f'  {bm_type:8s}: {result.score():8.2f}  '
              f'({result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    # Play vs PubEval with 3-NN
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_3nn_vs_pubeval(
        contact_weights, crashed_weights, race_weights,
        n_hidden_contact, n_hidden_crashed, n_hidden_race,
        n_games=10000, seed=args.seed)
    t_games = time.time() - t0
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')


def main():
    parser = argparse.ArgumentParser(description='GPU Supervised Learning Training')
    parser.add_argument('--type', type=str, default='contact',
                        choices=['contact', 'crashed', 'race', 'single', 'all', 'both',
                                 'purerace', 'racing', 'attacking', 'priming', 'anchoring', 'gameplan'],
                        help='Network type to train. '
                             '"all" trains contact, crashed, race sequentially. '
                             '"gameplan" trains purerace, racing, attacking, priming, anchoring sequentially. '
                             '"single" uses 196 Tesauro inputs with all data combined.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: auto per type)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Learning rate (default: auto per type)')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Hidden layer size (default: per type - contact=120, crashed=120, race=80)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for GPU training (default: 128)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to .weights file to resume from')
    parser.add_argument('--print-interval', type=int, default=1,
                        help='Print/benchmark every N epochs (default: 1)')
    parser.add_argument('--gameplan-weight', type=float, default=1.0,
                        help='Extra weight for positions matching the game plan (default: 1.0 = no extra weight). '
                             'E.g., --gameplan-weight 3 means matching positions get 3x the gradient. '
                             'Only affects racing/attacking/priming/anchoring types.')
    parser.add_argument('--skip-benchmarks', action='store_true',
                        help='Skip final full benchmarks')
    parser.add_argument('--log-dir', type=str, default='',
                        help='Directory to save log files (default: logs/)')
    args = parser.parse_args()

    # Setup logging to file
    log_dir = args.log_dir if args.log_dir else os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    net_type_label = args.type
    alpha_label = args.alpha if args.alpha is not None else 'auto'
    log_filename = f'sl_{net_type_label}_a{alpha_label}_b{args.batch_size}_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)

    # Tee ALL output (including C++ std::cout) to both console and log file.
    # We duplicate the OS-level stdout fd so C++ output is also captured.
    log_file = open(log_path, 'w')
    original_stdout_fd = os.dup(1)  # save original stdout fd

    # Create a pipe: C++ writes to pipe_w, we read from pipe_r
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 1)  # redirect fd 1 (stdout) to pipe write end
    os.close(pipe_w)    # close extra copy

    import threading

    def tee_thread():
        """Read from pipe and write to both original stdout and log file."""
        with os.fdopen(pipe_r, 'r', errors='replace') as reader:
            for line in reader:
                os.write(original_stdout_fd, line.encode())
                log_file.write(line)
                log_file.flush()

    tee = threading.Thread(target=tee_thread, daemon=True)
    tee.start()

    # Also redirect Python's sys.stdout to the new fd 1
    sys.stdout = os.fdopen(os.dup(1), 'w')

    print(f'Log file: {log_path}')
    print()

    # Verify CUDA is available
    if not bgbot_cpp.cuda_available():
        print('ERROR: CUDA is not available. Cannot run GPU training.')
        sys.exit(1)
    print('CUDA GPU detected')
    print()

    if args.type in ('both', 'all'):
        # Train all 3 networks sequentially
        train_network('contact', args)
        print()
        saved_resume = args.resume
        args.resume = ''
        train_network('crashed', args)
        print()
        train_network('race', args)
        args.resume = saved_resume
    elif args.type == 'gameplan':
        # Train all 5 game plan networks sequentially
        train_network('purerace', args)
        print()
        saved_resume = args.resume
        args.resume = ''
        train_network('racing', args)
        print()
        train_network('attacking', args)
        print()
        train_network('priming', args)
        print()
        train_network('anchoring', args)
        args.resume = saved_resume
    else:
        train_network(args.type, args)

    if not args.skip_benchmarks:
        run_final_benchmarks(args)

    # Close log file - restore original stdout
    sys.stdout.close()           # closes the pipe write end (fd 1)
    tee.join(timeout=5)          # wait for tee thread to drain pipe and exit
    os.dup2(original_stdout_fd, 1)  # restore fd 1 to original stdout
    os.close(original_stdout_fd)
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    main()
