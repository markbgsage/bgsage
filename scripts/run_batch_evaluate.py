#!/usr/bin/env python3
"""Demo script for batch position evaluation.

Usage:
    python bgsage/scripts/run_batch_evaluate.py [--model stage5] [--level 0ply]
"""

import argparse
import sys
import os
import time

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

from bgsage.batch import batch_evaluate
from bgsage.weights import WeightConfig


STARTING_BOARD = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

# A few diverse positions for testing
TEST_POSITIONS = [
    # Starting position
    {"board": STARTING_BOARD, "cube_value": 1, "cube_owner": "centered"},
    # Race position
    {"board": [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2, -4, -3, -3, 0, 0],
     "cube_value": 1, "cube_owner": "centered"},
    # Contact position with cube at 2, player owns
    {"board": [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0],
     "cube_value": 2, "cube_owner": "player"},
    # Near-bearoff
    {"board": [0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, -3, 0, 0],
     "cube_value": 1, "cube_owner": "centered"},
    # Blitz position (opponent on bar)
    {"board": [1, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, -4, 3, 0, 0, 0, -2, 0, -4, -2, 0, -2, 0, 0, 1],
     "cube_value": 1, "cube_owner": "centered"},
]


def main():
    parser = argparse.ArgumentParser(description="Batch position evaluation demo")
    WeightConfig.add_model_arg(parser)
    parser.add_argument("--level", type=str, default="0ply",
                        choices=["0ply", "1ply", "2ply", "3ply"],
                        help="Evaluation level (default: 0ply)")
    parser.add_argument("--n-threads", type=int, default=0,
                        help="Thread count (0 = auto)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat positions N times for benchmarking")
    args = parser.parse_args()

    weights = WeightConfig.from_args(args)
    weights.validate()
    weights.print_summary(f"Model: {args.model}")

    positions = TEST_POSITIONS * args.repeat
    print(f"Evaluating {len(positions)} positions at {args.level} "
          f"with {args.n_threads or 'auto'} threads...\n")

    t0 = time.perf_counter()
    results = batch_evaluate(
        positions,
        eval_level=args.level,
        weights=weights,
        n_threads=args.n_threads,
    )
    elapsed = time.perf_counter() - t0

    print(f"{'Pos':>4s}  {'CL Equity':>10s}  {'CF Equity':>10s}  "
          f"{'P(win)':>8s}  {'P(gw)':>7s}  {'P(gl)':>7s}  "
          f"{'Cube':>12s}")
    print("-" * 78)

    for i, r in enumerate(results):
        print(f"{i:4d}  {r.cubeless_equity:+10.4f}  {r.cubeful_equity:+10.4f}  "
              f"{r.probs.win:8.3%}  {r.probs.gammon_win:7.3%}  "
              f"{r.probs.gammon_loss:7.3%}  {r.optimal_action:>12s}")

    print(f"\nElapsed: {elapsed:.3f}s  ({len(positions)/elapsed:.0f} pos/s)")


if __name__ == "__main__":
    main()
