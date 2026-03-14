"""Benchmark cubeful cube decision at 2-ply through 4-ply.

Uses a specific contact position to measure timing and equity values.
Run before and after algorithm changes to compare.
"""

import sys
import os
import time
import statistics

os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from bgsage import BgBotAnalyzer

BOARD = [0, 0, 1, 2, 2, -2, 3, 0, 3, 0, 0, 0, -4, 4, 0, 0, 0, -3, 0, -3, -1, -2, 0, 0, 0, 0]
N_RUNS = 5


def benchmark_ply(ply_str, n_runs=N_RUNS):
    """Run cube_action n_runs times and return median time + last result."""
    analyzer = BgBotAnalyzer(eval_level=ply_str, cubeful=True)

    # Warmup
    analyzer.cube_action(BOARD, cube_value=1, cube_owner='centered')

    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = analyzer.cube_action(BOARD, cube_value=1, cube_owner='centered')
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = statistics.median(times)
    return median_time, result


def main():
    print(f"Position: {BOARD}")
    print(f"Cube: centered, value=1")
    print(f"Runs per ply: {N_RUNS} (median time)")
    print()

    for ply_str in ['2ply', '3ply', '4ply', 'rollout']:
        print(f"--- {ply_str} ---")
        median_time, r = benchmark_ply(ply_str, n_runs=2 if ply_str == 'rollout' else N_RUNS)

        print(f"  Probs: win={r.probs.win:.5f} gw={r.probs.gammon_win:.5f} "
              f"bw={r.probs.backgammon_win:.5f} gl={r.probs.gammon_loss:.5f} "
              f"bl={r.probs.backgammon_loss:.5f}")
        print(f"  Cubeless eq: {r.cubeless_equity:.5f}")
        print(f"  ND:  {r.equity_nd:+.5f}")
        print(f"  DT:  {r.equity_dt:+.5f}")
        print(f"  DP:  {r.equity_dp:+.5f}")
        print(f"  Decision: {r.optimal_action}")
        print(f"  Median time: {median_time:.4f}s")
        print()


if __name__ == '__main__':
    main()
