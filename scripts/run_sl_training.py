"""
Run supervised learning training for backgammon neural networks.

Trains separate contact and race networks on GNUbg training databases.
Starts from TD-trained weights by default, or from specified weights.

Usage:
    python run_sl_training.py --type contact --epochs 500 --alpha 0.03
    python run_sl_training.py --type race --epochs 20 --alpha 0.001
    python run_sl_training.py --type both
    python run_sl_training.py --type contact --resume models/td_multi_350k_contact.weights
"""

import os
import sys
import time
import argparse

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32' and os.path.isdir(build_dir):
    os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')


def find_td_weights(net_type):
    """Find the best available TD-trained weights for a given network type."""
    # Look for multi-network TD weights first (preferred)
    candidates = [
        os.path.join(MODELS_DIR, f'td_multi_350k_{net_type}.weights'),
        os.path.join(MODELS_DIR, f'td_multi_{net_type}.weights'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ''


def train_network(net_type, args):
    """Train a single network type (contact or race)."""

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Configuration per network type
    if net_type == 'contact':
        train_data_path = os.path.join(DATA_DIR, 'contact-train-data')
        benchmark_path = os.path.join(DATA_DIR, 'contact.bm')
        n_inputs = 214  # EXTENDED_CONTACT_INPUTS
        default_alpha = 0.03
        default_epochs = 500
        benchmark_step = 100  # subsample for speed during training
    elif net_type == 'race':
        train_data_path = os.path.join(DATA_DIR, 'race-train-data')
        benchmark_path = os.path.join(DATA_DIR, 'race.bm')
        n_inputs = 196  # TESAURO_INPUTS
        default_alpha = 0.001
        default_epochs = 20
        benchmark_step = 50
    else:
        raise ValueError(f"Unknown network type: {net_type}")

    alpha = args.alpha if args.alpha is not None else default_alpha
    epochs = args.epochs if args.epochs is not None else default_epochs
    n_hidden = args.hidden

    # Determine starting weights
    weights_path = ''
    if args.resume:
        weights_path = args.resume
    else:
        # Auto-find TD-trained weights
        weights_path = find_td_weights(net_type)

    print(f'=== Training {net_type} network ===')
    print(f'  Data:     {train_data_path}')
    print(f'  Inputs:   {n_inputs}')
    print(f'  Hidden:   {n_hidden}')
    print(f'  Alpha:    {alpha}')
    print(f'  Epochs:   {epochs}')
    print(f'  Batch:    {args.batch_size}')
    print(f'  Seed:     {args.seed}')
    if weights_path:
        print(f'  Starting from: {weights_path}')
    else:
        print(f'  Starting from: random weights')
    print()

    # Load training data
    print(f'Loading training data...')
    t0 = time.time()
    boards, targets = load_gnubg_training_data(train_data_path)
    print(f'  Loaded {len(boards)} positions in {time.time()-t0:.1f}s')

    # Load benchmark scenarios for progress tracking
    scenarios = None
    if os.path.exists(benchmark_path):
        print(f'Loading benchmark (step={benchmark_step})...')
        t0 = time.time()
        scenarios = load_benchmark_file(benchmark_path, step=benchmark_step)
        print(f'  Loaded {len(scenarios)} scenarios in {time.time()-t0:.1f}s')
    print()

    save_path = os.path.join(MODELS_DIR, f'sl_{net_type}.weights')

    # Run supervised training
    result = bgbot_cpp.supervised_train(
        boards=boards,
        targets=targets,
        weights_path=weights_path,
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        alpha=alpha,
        epochs=epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        print_interval=1,
        save_path=save_path,
        benchmark_scenarios=scenarios,
    )

    print()
    print(f'{net_type.capitalize()} training complete:')
    print(f'  Best score: {result["best_score"]:.2f}')
    print(f'  Time: {result["total_seconds"]:.1f}s')
    print(f'  Weights: {save_path}')
    print(f'  Best weights: {save_path}.best')

    return result


def run_final_benchmarks(args):
    """Run full benchmarks on trained models using MultiNNStrategy."""
    contact_weights = os.path.join(MODELS_DIR, 'sl_contact.weights.best')
    race_weights = os.path.join(MODELS_DIR, 'sl_race.weights.best')

    # Fall back to non-best weights
    if not os.path.exists(contact_weights):
        contact_weights = os.path.join(MODELS_DIR, 'sl_contact.weights')
    if not os.path.exists(race_weights):
        race_weights = os.path.join(MODELS_DIR, 'sl_race.weights')

    # Fall back to TD weights
    if not os.path.exists(contact_weights):
        contact_weights = find_td_weights('contact')
    if not os.path.exists(race_weights):
        race_weights = find_td_weights('race')

    if not contact_weights or not race_weights:
        print('Cannot run final benchmarks: missing weights')
        return

    print()
    print('=== Final Benchmark Scores (MultiNNStrategy) ===')
    print(f'  Contact weights: {contact_weights}')
    print(f'  Race weights:    {race_weights}')
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
        result = bgbot_cpp.score_benchmarks_multi_nn(
            scenarios, contact_weights, race_weights, args.hidden)
        t_score = time.time() - t0

        print(f'  {bm_type:8s}: {result.score():8.2f}  '
              f'({result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    # Play vs PubEval
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_multi_nn_vs_pubeval(
        contact_weights, race_weights, args.hidden,
        n_games=10000, seed=args.seed)
    t_games = time.time() - t0
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')


def main():
    parser = argparse.ArgumentParser(description='Supervised Learning Training')
    parser.add_argument('--type', type=str, default='both',
                        choices=['contact', 'race', 'both'],
                        help='Network type to train (default: both)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: auto per type)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Learning rate (default: auto per type)')
    parser.add_argument('--hidden', type=int, default=120,
                        help='Hidden layer size (default: 120)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for SGD (default: 1 = online)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to .weights file to resume from')
    parser.add_argument('--skip-benchmarks', action='store_true',
                        help='Skip final full benchmarks')
    args = parser.parse_args()

    if args.type == 'both':
        train_network('contact', args)
        print()
        # Reset resume for second network
        saved_resume = args.resume
        args.resume = ''
        train_network('race', args)
        args.resume = saved_resume
    else:
        train_network(args.type, args)

    if not args.skip_benchmarks:
        run_final_benchmarks(args)


if __name__ == '__main__':
    main()
