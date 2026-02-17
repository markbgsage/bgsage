"""
Test variance reduction effectiveness.

Grab a representative position, rollout it with different seeds (VR on vs off),
compare the std dev of the resulting equity/probability estimates.

Usage:
  python python/test_vr_effectiveness.py [--n-seeds 30] [--trials 36]
"""

import os
import sys
import time
import argparse
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp

MODELS_DIR = os.path.join(project_dir, 'models')
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250


def get_best_weights():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    paths = {}
    for t in types:
        path = os.path.join(MODELS_DIR, f'sl_{t}.weights.best')
        if os.path.exists(path):
            paths[t] = path
        else:
            return None
    return paths


def weight_args(w):
    return (w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'])


def stats(values):
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser(description='Test VR effectiveness')
    parser.add_argument('--n-seeds', type=int, default=30,
                        help='Number of different seeds (default: 30)')
    parser.add_argument('--trials', type=int, default=36,
                        help='Trials per rollout (default: 36)')
    parser.add_argument('--trunc', type=int, default=7,
                        help='Truncation depth (default: 7)')
    parser.add_argument('--threads', type=int, default=1,
                        help='CPU threads (default: 1 for reproducibility)')
    args = parser.parse_args()

    weights = get_best_weights()
    if weights is None:
        print("Cannot find weight files.")
        sys.exit(1)

    # Use the starting board as a representative position
    # Board layout: [0]=p2bar, [1-24]=points, [25]=p1bar
    board = [0,
             -2, 0, 0, 0, 0, 5,
             0, 3, 0, 0, 0, -5,
             5, 0, 0, 0, -3, 0,
             -5, 0, 0, 0, 0, 2,
             0]

    print(f"VR effectiveness test: {args.n_seeds} seeds x {args.trials} trials, trunc={args.trunc}")
    print(f"Position: starting board")
    print()

    vr_equities = []
    novr_equities = []

    print(f"{'Seed':>6}  {'VR equity':>12}  {'NoVR equity':>12}")
    print(f"{'----':>6}  {'---------':>12}  {'-----------':>12}")

    t0 = time.perf_counter()
    for i in range(args.n_seeds):
        seed = 100 + i * 97

        # With VR
        rollout_vr = bgbot_cpp.create_rollout_5nn(
            *weight_args(weights), NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_trials=args.trials, truncation_depth=args.trunc,
            decision_ply=0, vr_ply=0,
            n_threads=args.threads, seed=seed)
        r_vr = rollout_vr.rollout_position(board)

        # Without VR
        rollout_novr = bgbot_cpp.create_rollout_5nn(
            *weight_args(weights), NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_trials=args.trials, truncation_depth=args.trunc,
            decision_ply=0, vr_ply=-1,
            n_threads=args.threads, seed=seed)
        r_novr = rollout_novr.rollout_position(board)

        vr_equities.append(r_vr.equity)
        novr_equities.append(r_novr.equity)

        print(f"{seed:>6}  {r_vr.equity:>12.6f}  {r_novr.equity:>12.6f}")

    elapsed = time.perf_counter() - t0

    vr_mean, vr_std = stats(vr_equities)
    novr_mean, novr_std = stats(novr_equities)

    print()
    print("=" * 50)
    print(f"  {'':>12}  {'Mean':>12}  {'Std Dev':>12}")
    print(f"  {'With VR':>12}  {vr_mean:>12.6f}  {vr_std:>12.6f}")
    print(f"  {'Without VR':>12}  {novr_mean:>12.6f}  {novr_std:>12.6f}")
    if novr_std > 0 and vr_std > 0:
        ratio = novr_std / vr_std
        print(f"\n  VR reduces std dev by {ratio:.1f}x "
              f"(equivalent to {ratio**2:.0f}x more trials)")
    print(f"  Mean shift: {vr_mean - novr_mean:+.6f} "
          f"(should be ~0 if VR is unbiased)")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
