"""
Run multi-network TD(0) self-play training.

Creates two neural networks from scratch:
  - Contact (214 inputs, extended encoding) - also used for crashed positions
  - Race (196 inputs, Tesauro encoding)

During self-play, each position is classified and the appropriate network is updated.

Usage:
    python run_td_multi_training.py [--games 100000] [--alpha 0.1] [--name td_multi]
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
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')


def main():
    parser = argparse.ArgumentParser(description='Multi-network TD(0) training')
    parser.add_argument('--games', type=int, default=100000,
                        help='Number of self-play games (default: 100000)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--hidden', type=int, default=120,
                        help='Hidden layer size (default: 120)')
    parser.add_argument('--name', type=str, default='td_multi',
                        help='Model name for saving (default: td_multi)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--benchmark-interval', type=int, default=5000,
                        help='Benchmark every N games (default: 5000)')
    parser.add_argument('--benchmark-step', type=int, default=100,
                        help='Subsample benchmarks: every Nth scenario (default: 100)')
    parser.add_argument('--resume-contact', type=str, default='',
                        help='Path to contact .weights file to resume from')
    parser.add_argument('--resume-race', type=str, default='',
                        help='Path to race .weights file to resume from')
    args = parser.parse_args()

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load contact benchmark for progress tracking
    contact_bm_path = os.path.join(DATA_DIR, 'contact.bm')
    contact_ss = None
    if os.path.exists(contact_bm_path):
        print(f'Loading contact benchmarks (step={args.benchmark_step})...')
        t0 = time.time()
        contact_ss = load_benchmark_file(contact_bm_path, step=args.benchmark_step)
        print(f'  Loaded {len(contact_ss)} scenarios in {time.time()-t0:.1f}s')
        print()
    else:
        print(f'  WARNING: {contact_bm_path} not found, skipping benchmarks')
        print()

    # Run training
    print(f'=== Multi-Network TD Training ===')
    print(f'  Games:    {args.games}')
    print(f'  Alpha:    {args.alpha}')
    print(f'  Hidden:   {args.hidden}')
    print(f'  Seed:     {args.seed}')
    print(f'  Name:     {args.name}')
    print(f'  Networks: contact (214-input) + race (196-input)')
    if args.resume_contact:
        print(f'  Resume contact: {args.resume_contact}')
    if args.resume_race:
        print(f'  Resume race:    {args.resume_race}')
    print(flush=True)

    result = bgbot_cpp.td_train_multi(
        n_games=args.games,
        alpha=args.alpha,
        n_hidden=args.hidden,
        seed=args.seed,
        benchmark_interval=args.benchmark_interval,
        model_name=args.name,
        models_dir=MODELS_DIR,
        resume_contact=args.resume_contact,
        resume_race=args.resume_race,
        contact_benchmark=contact_ss,
    )

    print()
    print(f'Training complete: {result.games_played} games in '
          f'{result.total_seconds:.1f}s')

    # Print history summary
    if result.history:
        print()
        print('Training progress:')
        print(f'  {"Game":>8}  {"Contact ER":>12}  {"Time (s)":>10}')
        print(f'  {"----":>8}  {"----------":>12}  {"--------":>10}')
        for entry in result.history:
            print(f'  {entry.game_number:>8}  '
                  f'{entry.contact_score:>12.2f}  '
                  f'{entry.elapsed_seconds:>10.1f}')

    # Run full benchmarks on final models
    contact_path = os.path.join(MODELS_DIR, f'{args.name}_contact.weights')
    race_path = os.path.join(MODELS_DIR, f'{args.name}_race.weights')

    if os.path.exists(contact_path) and os.path.exists(race_path):
        print()
        print(f'=== Final Benchmark Scores (MultiNNStrategy) ===')
        for bm_type in ['contact', 'race']:
            bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
            if not os.path.exists(bm_path):
                print(f'  {bm_type:8s}: (file not found)')
                continue
            t0 = time.time()
            bm_scenarios = load_benchmark_file(bm_path)
            t_load = time.time() - t0
            t0 = time.time()
            bm_result = bgbot_cpp.score_benchmarks_multi_nn(
                bm_scenarios, contact_path, race_path, args.hidden)
            t_score = time.time() - t0
            print(f'  {bm_type:8s}: {bm_result.score():8.2f}  '
                  f'({bm_result.count} scenarios, '
                  f'load {t_load:.1f}s + score {t_score:.1f}s)')

        # Play vs PubEval
        print()
        print('=== vs PubEval (10k games) ===')
        t0 = time.time()
        stats = bgbot_cpp.play_games_multi_nn_vs_pubeval(
            contact_path, race_path, args.hidden, n_games=10000, seed=args.seed)
        t_games = time.time() - t0
        print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
        print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
        print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')


if __name__ == '__main__':
    main()
