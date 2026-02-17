"""Full benchmark suite for the 5-NN game plan strategy.

Runs game plan benchmarks, old-style benchmarks, vs PubEval, and self-play
outcome distribution. Supports 0-ply (direct NN) and N-ply (multi-ply search).

Usage:
  python bgsage/scripts/run_full_benchmark.py                     # 0-ply, production model
  python bgsage/scripts/run_full_benchmark.py --model stage3      # 0-ply, specific model
  python bgsage/scripts/run_full_benchmark.py --ply 1             # 1-ply
  python bgsage/scripts/run_full_benchmark.py --ply 2 --scenarios 500  # 2-ply, subsample
"""

import os
import sys
import time
import argparse

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
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')

GAME_PLAN_BENCHMARKS = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
OLD_STYLE_BENCHMARKS = ['contact', 'crashed', 'race']


def score_benchmark(scenarios, w, n_plies, multipy, n_threads):
    """Score a benchmark at the given ply level. Returns (result, elapsed)."""
    t0 = time.perf_counter()
    if n_plies == 0:
        result = bgbot_cpp.score_benchmarks_5nn(scenarios, *w.weight_args)
    else:
        result = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy, n_threads)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def load_scenarios(bm_name, max_scenarios):
    """Load benchmark scenarios, optionally subsampling."""
    bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
    if not os.path.exists(bm_path):
        return None, 0, 0
    scenarios_full = load_benchmark_file(bm_path)
    n_total = scenarios_full.size()
    if max_scenarios > 0 and max_scenarios < n_total:
        step = max(1, n_total // max_scenarios)
        scenarios = load_benchmark_file(bm_path, step=step)
        return scenarios, scenarios.size(), n_total
    return scenarios_full, n_total, n_total


def run_benchmarks(w, n_plies, max_scenarios, n_threads):
    """Run game plan and old-style benchmarks."""
    multipy = None
    if n_plies > 0:
        multipy = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=n_plies)

    for label, bm_list in [('Game Plan benchmarks', GAME_PLAN_BENCHMARKS),
                            ('Old-style benchmarks', OLD_STYLE_BENCHMARKS)]:
        print(f'--- {label} ({n_plies}-ply) ---')
        for bm_name in bm_list:
            scenarios, n, n_total = load_scenarios(bm_name, max_scenarios)
            if scenarios is None:
                print(f'  {bm_name:10s}: (not found)')
                continue

            result, elapsed = score_benchmark(scenarios, w, n_plies, multipy, n_threads)

            count_str = f'{n}' if n == n_total else f'{n}/{n_total}'
            print(f'  {bm_name:10s}: {result.score():8.2f}  ({count_str} scenarios, {elapsed:.1f}s)')

            if n_plies > 0:
                multipy.clear_cache()

        print()


def run_vs_pubeval(w, n_games):
    """Play games against PubEval."""
    print(f'=== vs PubEval ({n_games // 1000}k games) ===')
    t0 = time.perf_counter()
    stats = bgbot_cpp.play_games_5nn_vs_pubeval(*w.weight_args, n_games=n_games, seed=42)
    elapsed = time.perf_counter() - t0
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {elapsed:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')
    print()


def run_self_play(w, n_games):
    """Run self-play to measure outcome distribution."""
    print(f'=== Self-play outcome distribution ({n_games // 1000}k games) ===')
    t0 = time.perf_counter()
    ss = bgbot_cpp.play_games_5nn_vs_self(*w.weight_args, n_games=n_games, seed=42)
    elapsed = time.perf_counter() - t0
    total = ss.n_games
    singles = ss.p1_wins + ss.p2_wins
    gammons = ss.p1_gammons + ss.p2_gammons
    backgammons = ss.p1_backgammons + ss.p2_backgammons
    print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
          f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
          f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
          f'({total} games in {elapsed:.1f}s)')
    print()


def main():
    parser = argparse.ArgumentParser(description='Full benchmark suite')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--ply', type=int, default=0, help='Number of plies (default: 0)')
    parser.add_argument('--scenarios', type=int, default=0, help='Max scenarios per benchmark (0=all)')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads for multi-ply (0=auto)')
    parser.add_argument('--games', type=int, default=10000, help='Games for PubEval/self-play (default: 10000)')
    args = parser.parse_args()

    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary(f'{args.model} ({args.ply}-ply)')

    run_benchmarks(w, args.ply, args.scenarios, args.threads)
    run_vs_pubeval(w, args.games)
    run_self_play(w, args.games)


if __name__ == '__main__':
    main()
