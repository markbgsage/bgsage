"""
Run TD(0) self-play training for 5-NN game plan strategy (CPU).

Trains 5 separate NNs (purerace, racing, attacking, priming, anchoring) via game plan TD learning.
Supports multi-phase training with decreasing alpha.

Usage:
    python run_td_gameplan_training.py --games 200000 --alpha 0.1
    python run_td_gameplan_training.py --games 200000 --alpha 0.1 --games2 1000000 --alpha2 0.02
"""

import os
import sys
import time
import argparse
from datetime import datetime

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
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')


def run_td_phase(args, phase, n_games, alpha, model_name,
                 resume_purerace='', resume_racing='',
                 resume_attacking='', resume_priming='',
                 resume_anchoring='',
                 purerace_ss=None, attacking_ss=None, priming_ss=None,
                 anchoring_ss=None, race_ss=None):
    """Run one phase of game plan TD training."""

    print(f'=== TD Phase {phase}: {n_games//1000}k @ alpha={alpha} ===')
    print(f'  PureRace:  {args.hidden_purerace} hidden, 196 inputs')
    print(f'  Racing:    {args.hidden_racing} hidden, 244 inputs')
    print(f'  Attacking: {args.hidden_attacking} hidden, 244 inputs')
    print(f'  Priming:   {args.hidden_priming} hidden, 244 inputs')
    print(f'  Anchoring: {args.hidden_anchoring} hidden, 244 inputs')
    print(f'  Alpha:     {alpha}')
    print(f'  Games:     {n_games}')
    print(f'  Seed:      {args.seed}')
    if resume_purerace:  print(f'  Resume purerace:  {resume_purerace}')
    if resume_racing:    print(f'  Resume racing:    {resume_racing}')
    if resume_attacking: print(f'  Resume attacking: {resume_attacking}')
    if resume_priming:   print(f'  Resume priming:   {resume_priming}')
    if resume_anchoring: print(f'  Resume anchoring: {resume_anchoring}')
    print(flush=True)

    result = bgbot_cpp.td_train_gameplan(
        n_games=n_games,
        alpha=alpha,
        n_hidden_purerace=args.hidden_purerace,
        n_hidden_racing=args.hidden_racing,
        n_hidden_attacking=args.hidden_attacking,
        n_hidden_priming=args.hidden_priming,
        n_hidden_anchoring=args.hidden_anchoring,
        eps=args.eps,
        seed=args.seed,
        benchmark_interval=args.benchmark_interval,
        model_name=model_name,
        models_dir=MODELS_DIR,
        resume_purerace=resume_purerace,
        resume_racing=resume_racing,
        resume_attacking=resume_attacking,
        resume_priming=resume_priming,
        resume_anchoring=resume_anchoring,
        purerace_benchmark=purerace_ss,
        attacking_benchmark=attacking_ss,
        priming_benchmark=priming_ss,
        anchoring_benchmark=anchoring_ss,
        race_benchmark=race_ss,
    )

    print(f'Phase {phase} done: {result.games_played} games in {result.total_seconds:.1f}s')
    return result


def run_final_benchmarks(args, model_name):
    """Run full benchmarks on the trained 5-NN model."""

    purerace_w  = os.path.join(MODELS_DIR, f'{model_name}_purerace.weights')
    racing_w    = os.path.join(MODELS_DIR, f'{model_name}_racing.weights')
    attacking_w = os.path.join(MODELS_DIR, f'{model_name}_attacking.weights')
    priming_w   = os.path.join(MODELS_DIR, f'{model_name}_priming.weights')
    anchoring_w = os.path.join(MODELS_DIR, f'{model_name}_anchoring.weights')

    missing = []
    if not os.path.exists(purerace_w):  missing.append('purerace')
    if not os.path.exists(racing_w):    missing.append('racing')
    if not os.path.exists(attacking_w): missing.append('attacking')
    if not os.path.exists(priming_w):   missing.append('priming')
    if not os.path.exists(anchoring_w): missing.append('anchoring')
    if missing:
        print(f'Cannot run final benchmarks: missing weights for {", ".join(missing)}')
        return

    print()
    print(f'=== Final Benchmark Scores (5-NN Game Plan) ===')
    print(f'  PureRace:  {purerace_w} ({args.hidden_purerace}h)')
    print(f'  Racing:    {racing_w} ({args.hidden_racing}h)')
    print(f'  Attacking: {attacking_w} ({args.hidden_attacking}h)')
    print(f'  Priming:   {priming_w} ({args.hidden_priming}h)')
    print(f'  Anchoring: {anchoring_w} ({args.hidden_anchoring}h)')
    print()

    # Game plan benchmarks (step=1)
    print('--- Game Plan benchmarks (step=1) ---')
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            print(f'  {bm_type:10s}: (benchmark file not found)')
            continue

        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        t_load = time.time() - t0

        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            args.hidden_purerace, args.hidden_racing, args.hidden_attacking,
            args.hidden_priming, args.hidden_anchoring)
        t_score = time.time() - t0

        print(f'  {bm_type:10s}: {result.score():8.2f}  '
              f'({result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    # Old-style benchmarks (step=1) for comparison
    print()
    print('--- Old-style benchmarks (step=1, for comparison) ---')
    for bm_name in ['contact', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            print(f'  {bm_name:10s}: (benchmark file not found)')
            continue

        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        t_load = time.time() - t0

        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_5nn(
            scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            args.hidden_purerace, args.hidden_racing, args.hidden_attacking,
            args.hidden_priming, args.hidden_anchoring)
        t_score = time.time() - t0

        print(f'  {bm_name:10s}: {result.score():8.2f}  '
              f'({result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    # Play vs PubEval
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_5nn_vs_pubeval(
        purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        args.hidden_purerace, args.hidden_racing, args.hidden_attacking,
        args.hidden_priming, args.hidden_anchoring,
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
        args.hidden_purerace, args.hidden_racing, args.hidden_attacking,
        args.hidden_priming, args.hidden_anchoring,
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


def main():
    parser = argparse.ArgumentParser(description='TD Self-Play Training (CPU, 5-NN Game Plan)')
    parser.add_argument('--games', type=int, default=200000,
                        help='Phase 1 games (default: 200000)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Phase 1 learning rate (default: 0.1)')
    parser.add_argument('--games2', type=int, default=0,
                        help='Phase 2 games (default: 0 = skip phase 2)')
    parser.add_argument('--alpha2', type=float, default=0.02,
                        help='Phase 2 learning rate (default: 0.02)')
    parser.add_argument('--hidden-purerace', type=int, default=120,
                        help='PureRace NN hidden nodes (default: 120)')
    parser.add_argument('--hidden-racing', type=int, default=250,
                        help='Racing NN hidden nodes (default: 250)')
    parser.add_argument('--hidden-attacking', type=int, default=250,
                        help='Attacking NN hidden nodes (default: 250)')
    parser.add_argument('--hidden-priming', type=int, default=250,
                        help='Priming NN hidden nodes (default: 250)')
    parser.add_argument('--hidden-anchoring', type=int, default=250,
                        help='Anchoring NN hidden nodes (default: 250)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Weight init scale (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (default: 42)')
    parser.add_argument('--benchmark-interval', type=int, default=10000,
                        help='Benchmark every N games (default: 10000)')
    parser.add_argument('--model-name', type=str, default='td_gameplan',
                        help='Model name prefix (default: td_gameplan)')
    parser.add_argument('--resume-purerace', type=str, default='',
                        help='Resume purerace from weights file')
    parser.add_argument('--resume-racing', type=str, default='',
                        help='Resume racing from weights file')
    parser.add_argument('--resume-attacking', type=str, default='',
                        help='Resume attacking from weights file')
    parser.add_argument('--resume-priming', type=str, default='',
                        help='Resume priming from weights file')
    parser.add_argument('--resume-anchoring', type=str, default='',
                        help='Resume anchoring from weights file')
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
    log_filename = f'td_gp_{total_games//1000}k_a{args.alpha}_{timestamp}.log'
    log_path = os.path.join(log_dir, log_filename)

    # Tee ALL output to both console and log file
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
    benchmark_sets = {}
    for bm_name in ['purerace', 'attacking', 'priming', 'anchoring', 'racing']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if os.path.exists(bm_path):
            print(f'Loading {bm_name} benchmark (step={args.benchmark_step})...')
            t0 = time.time()
            ss = load_benchmark_file(bm_path, step=args.benchmark_step)
            print(f'  Loaded {len(ss)} scenarios in {time.time()-t0:.1f}s')
            benchmark_sets[bm_name] = ss
        else:
            benchmark_sets[bm_name] = None
    print()

    # Phase 1
    result1 = run_td_phase(
        args, phase=1,
        n_games=args.games,
        alpha=args.alpha,
        model_name=args.model_name,
        resume_purerace=args.resume_purerace,
        resume_racing=args.resume_racing,
        resume_attacking=args.resume_attacking,
        resume_priming=args.resume_priming,
        resume_anchoring=args.resume_anchoring,
        purerace_ss=benchmark_sets.get('purerace'),
        attacking_ss=benchmark_sets.get('attacking'),
        priming_ss=benchmark_sets.get('priming'),
        anchoring_ss=benchmark_sets.get('anchoring'),
        race_ss=benchmark_sets.get('racing'),
    )

    final_model_name = args.model_name

    # Phase 2 (optional)
    if args.games2 > 0:
        print()
        purerace_path  = os.path.join(MODELS_DIR, f'{args.model_name}_purerace.weights')
        racing_path    = os.path.join(MODELS_DIR, f'{args.model_name}_racing.weights')
        attacking_path = os.path.join(MODELS_DIR, f'{args.model_name}_attacking.weights')
        priming_path   = os.path.join(MODELS_DIR, f'{args.model_name}_priming.weights')
        anchoring_path = os.path.join(MODELS_DIR, f'{args.model_name}_anchoring.weights')

        total_k = (args.games + args.games2) // 1000
        final_model_name = f'{args.model_name}_{total_k}k'

        result2 = run_td_phase(
            args, phase=2,
            n_games=args.games2,
            alpha=args.alpha2,
            model_name=final_model_name,
            resume_purerace=purerace_path,
            resume_racing=racing_path,
            resume_attacking=attacking_path,
            resume_priming=priming_path,
            resume_anchoring=anchoring_path,
            purerace_ss=benchmark_sets.get('purerace'),
            attacking_ss=benchmark_sets.get('attacking'),
            priming_ss=benchmark_sets.get('priming'),
            anchoring_ss=benchmark_sets.get('anchoring'),
            race_ss=benchmark_sets.get('racing'),
        )

    # Print final weights locations
    print()
    print(f'Final weights:')
    print(f'  PureRace:  {os.path.join(MODELS_DIR, f"{final_model_name}_purerace.weights")}')
    print(f'  Racing:    {os.path.join(MODELS_DIR, f"{final_model_name}_racing.weights")}')
    print(f'  Attacking: {os.path.join(MODELS_DIR, f"{final_model_name}_attacking.weights")}')
    print(f'  Priming:   {os.path.join(MODELS_DIR, f"{final_model_name}_priming.weights")}')
    print(f'  Anchoring: {os.path.join(MODELS_DIR, f"{final_model_name}_anchoring.weights")}')

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
