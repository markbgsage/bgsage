"""
Run TD(0) self-play training for backgammon neural networks (CPU).

Trains 3 separate NNs (contact, crashed, race) via multi-network TD learning.
Supports multi-phase training with decreasing alpha.

Usage:
    python run_td_training.py --games 100000 --alpha 0.1
    python run_td_training.py --games 100000 --alpha 0.1 --games2 250000 --alpha2 0.02
    python run_td_training.py --resume-contact models/td_multi_contact.weights
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
    # Add CUDA DLL directory (needed even for CPU code since .pyd links CUDA)
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')


def run_td_phase(args, phase, n_games, alpha, model_name,
                 resume_contact='', resume_crashed='', resume_race='',
                 benchmark_scenarios=None):
    """Run one phase of TD training."""

    print(f'=== TD Phase {phase}: {n_games//1000}k @ alpha={alpha} ===')
    print(f'  Contact: {args.hidden_contact} hidden, 214 inputs')
    print(f'  Crashed: {args.hidden_crashed} hidden, 214 inputs')
    print(f'  Race:    {args.hidden_race} hidden, 196 inputs')
    print(f'  Alpha:   {alpha}')
    print(f'  Games:   {n_games}')
    print(f'  Seed:    {args.seed}')
    if resume_contact:
        print(f'  Resume contact: {resume_contact}')
    if resume_crashed:
        print(f'  Resume crashed: {resume_crashed}')
    if resume_race:
        print(f'  Resume race: {resume_race}')
    print(flush=True)

    result = bgbot_cpp.td_train_multi(
        n_games=n_games,
        alpha=alpha,
        n_hidden_contact=args.hidden_contact,
        n_hidden_crashed=args.hidden_crashed,
        n_hidden_race=args.hidden_race,
        eps=args.eps,
        seed=args.seed,
        benchmark_interval=args.benchmark_interval,
        model_name=model_name,
        models_dir=MODELS_DIR,
        resume_contact=resume_contact,
        resume_crashed=resume_crashed,
        resume_race=resume_race,
        contact_benchmark=benchmark_scenarios,
    )

    print(f'Phase {phase} done: {result.games_played} games in {result.total_seconds:.1f}s')
    return result


def run_final_benchmarks(args, model_name):
    """Run full benchmarks on the trained 3-NN model."""

    contact_weights = os.path.join(MODELS_DIR, f'{model_name}_contact.weights')
    crashed_weights = os.path.join(MODELS_DIR, f'{model_name}_crashed.weights')
    race_weights = os.path.join(MODELS_DIR, f'{model_name}_race.weights')

    missing = []
    if not os.path.exists(contact_weights): missing.append('contact')
    if not os.path.exists(crashed_weights): missing.append('crashed')
    if not os.path.exists(race_weights): missing.append('race')
    if missing:
        print(f'Cannot run final benchmarks: missing weights for {", ".join(missing)}')
        return

    print()
    print(f'=== Final Benchmark Scores (3-NN) ===')
    print(f'  Contact: {contact_weights} ({args.hidden_contact}h)')
    print(f'  Crashed: {crashed_weights} ({args.hidden_crashed}h)')
    print(f'  Race:    {race_weights} ({args.hidden_race}h)')
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
            args.hidden_contact, args.hidden_crashed, args.hidden_race)
        t_score = time.time() - t0

        print(f'  {bm_type:8s}: {result.score():8.2f}  '
              f'({result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    # Play vs PubEval
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_3nn_vs_pubeval(
        contact_weights, crashed_weights, race_weights,
        args.hidden_contact, args.hidden_crashed, args.hidden_race,
        n_games=10000, seed=args.seed)
    t_games = time.time() - t0
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')


def main():
    parser = argparse.ArgumentParser(description='TD Self-Play Training (CPU, 3-NN)')
    parser.add_argument('--games', type=int, default=100000,
                        help='Phase 1 games (default: 100000)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Phase 1 learning rate (default: 0.1)')
    parser.add_argument('--games2', type=int, default=0,
                        help='Phase 2 games (default: 0 = skip phase 2)')
    parser.add_argument('--alpha2', type=float, default=0.02,
                        help='Phase 2 learning rate (default: 0.02)')
    parser.add_argument('--hidden-contact', type=int, default=120,
                        help='Contact NN hidden nodes (default: 120)')
    parser.add_argument('--hidden-crashed', type=int, default=120,
                        help='Crashed NN hidden nodes (default: 120)')
    parser.add_argument('--hidden-race', type=int, default=80,
                        help='Race NN hidden nodes (default: 80)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Weight init scale (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--benchmark-interval', type=int, default=10000,
                        help='Benchmark every N games (default: 10000)')
    parser.add_argument('--model-name', type=str, default='td_multi',
                        help='Model name prefix (default: td_multi)')
    parser.add_argument('--resume-contact', type=str, default='',
                        help='Resume contact from weights file')
    parser.add_argument('--resume-crashed', type=str, default='',
                        help='Resume crashed from weights file')
    parser.add_argument('--resume-race', type=str, default='',
                        help='Resume race from weights file')
    parser.add_argument('--benchmark-step', type=int, default=10,
                        help='Load every Nth benchmark scenario (default: 10)')
    parser.add_argument('--skip-benchmarks', action='store_true',
                        help='Skip final full benchmarks')
    parser.add_argument('--log-dir', type=str, default='',
                        help='Directory for log files (default: logs/)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Setup logging to file
    log_dir = args.log_dir if args.log_dir else os.path.join(project_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_games = args.games + args.games2
    log_filename = f'td_{total_games//1000}k_a{args.alpha}_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)

    # Tee ALL output (including C++ std::cout) to both console and log file
    log_file = open(log_path, 'w')
    original_stdout_fd = os.dup(1)

    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 1)
    os.close(pipe_w)

    import threading

    def tee_thread():
        with os.fdopen(pipe_r, 'r', errors='replace') as reader:
            for line in reader:
                os.write(original_stdout_fd, line.encode())
                log_file.write(line)
                log_file.flush()

    tee = threading.Thread(target=tee_thread, daemon=True)
    tee.start()

    sys.stdout = os.fdopen(os.dup(1), 'w')

    print(f'Log file: {log_path}')
    print()

    # Load benchmark scenarios for progress tracking
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    benchmark_scenarios = None
    if os.path.exists(bm_path):
        print(f'Loading benchmark (step={args.benchmark_step})...')
        t0 = time.time()
        benchmark_scenarios = load_benchmark_file(bm_path, step=args.benchmark_step)
        print(f'  Loaded {len(benchmark_scenarios)} scenarios in {time.time()-t0:.1f}s')
    print()

    # Phase 1
    result1 = run_td_phase(
        args, phase=1,
        n_games=args.games,
        alpha=args.alpha,
        model_name=args.model_name,
        resume_contact=args.resume_contact,
        resume_crashed=args.resume_crashed,
        resume_race=args.resume_race,
        benchmark_scenarios=benchmark_scenarios,
    )

    final_model_name = args.model_name

    # Phase 2 (optional)
    if args.games2 > 0:
        print()
        contact_path = os.path.join(MODELS_DIR, f'{args.model_name}_contact.weights')
        crashed_path = os.path.join(MODELS_DIR, f'{args.model_name}_crashed.weights')
        race_path = os.path.join(MODELS_DIR, f'{args.model_name}_race.weights')

        # Update model name for phase 2 total
        total_k = (args.games + args.games2) // 1000
        final_model_name = f'{args.model_name}_{total_k}k'

        result2 = run_td_phase(
            args, phase=2,
            n_games=args.games2,
            alpha=args.alpha2,
            model_name=final_model_name,
            resume_contact=contact_path,
            resume_crashed=crashed_path,
            resume_race=race_path,
            benchmark_scenarios=benchmark_scenarios,
        )

    # Print final weights locations
    print()
    print(f'Final weights:')
    print(f'  Contact: {os.path.join(MODELS_DIR, f"{final_model_name}_contact.weights")}')
    print(f'  Crashed: {os.path.join(MODELS_DIR, f"{final_model_name}_crashed.weights")}')
    print(f'  Race:    {os.path.join(MODELS_DIR, f"{final_model_name}_race.weights")}')

    # Final full benchmarks
    if not args.skip_benchmarks:
        run_final_benchmarks(args, final_model_name)

    # Restore stdout
    sys.stdout.close()
    tee.join(timeout=5)
    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    main()
