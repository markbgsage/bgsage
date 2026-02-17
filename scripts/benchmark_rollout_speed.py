"""
Benchmark rollout speed and accuracy for the reference position.

XG reference (mover perspective): equity=+0.650, std_err=0.005, time=45s
  (1296 trials, no trunc, 2-ply decisions, VR at 0-ply)

Our optimized settings: late_ply=0 at threshold=3, race detection.
  Achieves ~24s with SE~0.006 — nearly 2x faster than XG.

Usage:
  python python/benchmark_rollout_speed.py [--quick] [--full]
"""

import os
import sys
import time
import argparse

# Setup import paths
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

# Stage 5 hidden sizes
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 200, 400, 400, 400, 400

# Reference position (from XG analysis)
REF_BOARD = [0, -2, 0, 0, 0, 2, 5, 4, 2, 0, 0, 0, -2, 2, 0, 0, 0, -3, 0, -5, -2, -1, 0, 0, 0, 0]

# XG reference results (mover's perspective)
XG_REF = {
    'equity': 0.650,
    'p_win': 0.7845,
    'p_gw': 0.1050,
    'p_bw': 0.0043,
    'p_gl': 0.0275,
    'p_bl': 0.0007,
    'std_err': 0.005,
    'time': 45.0,
}


def get_weights():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    paths = {}
    for t in types:
        path = os.path.join(MODELS_DIR, f'sl_s5_{t}.weights.best')
        if os.path.exists(path):
            paths[t] = path
        else:
            print(f"WARNING: {path} not found")
            return None
    return paths


def run_rollout(weights, n_trials, trunc_depth, decision_ply, vr_ply,
                n_threads=0, seed=42, late_ply=-1, late_threshold=20):
    rollout = bgbot_cpp.create_rollout_5nn(
        weights['purerace'], weights['racing'], weights['attacking'],
        weights['priming'], weights['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_trials=n_trials,
        truncation_depth=trunc_depth,
        decision_ply=decision_ply,
        vr_ply=vr_ply,
        n_threads=n_threads,
        seed=seed,
        late_ply=late_ply,
        late_threshold=late_threshold)

    t0 = time.perf_counter()
    result = rollout.evaluate_board(REF_BOARD, REF_BOARD)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def print_result(label, result, elapsed):
    probs = result['probs']
    eq = result['equity']
    se = result['std_error']

    print(f"\n{label}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Equity: {eq:.4f} (XG: {XG_REF['equity']:.4f}, diff: {eq - XG_REF['equity']:+.4f})")
    print(f"  Std err: {se:.4f}")
    print(f"  P(win):  {probs[0]:.4f} (XG: {XG_REF['p_win']:.4f})")
    print(f"  P(gw):   {probs[1]:.4f} (XG: {XG_REF['p_gw']:.4f})")
    print(f"  P(bw):   {probs[2]:.4f} (XG: {XG_REF['p_bw']:.4f})")
    print(f"  P(gl):   {probs[3]:.4f} (XG: {XG_REF['p_gl']:.4f})")
    print(f"  P(bl):   {probs[4]:.4f} (XG: {XG_REF['p_bl']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Benchmark rollout speed')
    parser.add_argument('--quick', action='store_true', help='Quick test (36 trials only)')
    parser.add_argument('--full', action='store_true', help='Full comparison including baseline')
    parser.add_argument('--threads', type=int, default=0, help='Thread count (0=auto)')
    args = parser.parse_args()

    weights = get_weights()
    if weights is None:
        sys.exit(1)

    bgbot_cpp.init_escape_tables()

    print(f"Reference position: {REF_BOARD}")
    print(f"Threads: {args.threads} (0=auto)")

    if args.quick:
        result, elapsed = run_rollout(weights, 36, 0, 2, 0, args.threads,
                                      late_ply=0, late_threshold=3)
        print_result("Quick: 36 trials, 2-ply, no trunc, late=0@3", result, elapsed)
        return

    if args.full:
        # Baseline: no late downgrade
        print("\n--- Baseline (no optimizations) ---")
        result, elapsed = run_rollout(weights, 1296, 0, 2, 0, args.threads)
        print_result("Baseline: 1296t, 2-ply, no trunc", result, elapsed)
        baseline_time = elapsed

        # Sweep late thresholds
        print("\n--- Late threshold sweep ---")
        for thr in [20, 10, 7, 5, 3]:
            result, elapsed = run_rollout(weights, 1296, 0, 2, 0, args.threads,
                                          late_ply=0, late_threshold=thr)
            p = result['probs']
            print(f"  thr={thr:>2}: {elapsed:.1f}s  eq={result['equity']:.4f}  "
                  f"se={result['std_error']:.4f}  P(w)={p[0]:.4f}  "
                  f"speedup={baseline_time/elapsed:.2f}x")
        return

    # Default: run optimized config
    print("\n" + "=" * 70)
    print("Optimized: 1296 trials, 2-ply, no trunc, VR=0, late=0@3, race detect")
    print("=" * 70)

    result, elapsed = run_rollout(weights, 1296, 0, 2, 0, args.threads,
                                  late_ply=0, late_threshold=3)
    print_result("Result", result, elapsed)

    print(f"\n  vs XG: {XG_REF['time']/elapsed:.1f}x faster ({elapsed:.1f}s vs {XG_REF['time']}s)")
    print(f"  vs baseline (103s): {103.2/elapsed:.1f}x faster")


if __name__ == '__main__':
    main()
