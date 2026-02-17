"""
Multi-ply benchmark script.

Tests:
  1. 0-ply passthrough: MultiPlyStrategy(base, 0) must produce identical scores to base directly
  2. 1-ply benchmark: score contact/race benchmarks at 1-ply depth
  3. Timing comparison between 0-ply and 1-ply

Usage:
  python python/run_multipy_benchmark.py [--ply N] [--scenarios N] [--threads N]
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
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')

# Hidden sizes for Stage 3 (244 inputs, 250h contact plans)
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250


def get_best_weights():
    """Return paths to best SL weights for 5-NN strategy."""
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    paths = {}
    for t in types:
        path = os.path.join(MODELS_DIR, f'sl_{t}.weights.best')
        if os.path.exists(path):
            paths[t] = path
        else:
            print(f"WARNING: {path} not found")
            return None
    return paths


def run_0ply_passthrough_test(weights, n_threads=0):
    """Verify that MultiPlyStrategy at 0-ply matches the base GamePlanStrategy exactly."""
    print("=" * 60)
    print("TEST: 0-ply passthrough (should match base strategy exactly)")
    print("=" * 60)

    # Create 0-ply MultiPlyStrategy wrapping the same weights
    multipy_0 = bgbot_cpp.create_multipy_5nn(
        weights['purerace'], weights['racing'],
        weights['attacking'], weights['priming'], weights['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=0)

    benchmarks = ['contact', 'purerace']
    for bm_name in benchmarks:
        bm_file = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_file):
            print(f"  Skipping {bm_name} (file not found)")
            continue

        scenarios = load_benchmark_file(bm_file)
        n = scenarios.size()
        print(f"\n  {bm_name}: {n} scenarios")

        # Base strategy (direct score_benchmarks_5nn)
        t0 = time.perf_counter()
        result_base = bgbot_cpp.score_benchmarks_5nn(
            scenarios,
            weights['purerace'], weights['racing'],
            weights['attacking'], weights['priming'], weights['anchoring'],
            NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)
        t_base = time.perf_counter() - t0

        # MultiPly 0-ply
        t0 = time.perf_counter()
        result_mp0 = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy_0, n_threads)
        t_mp0 = time.perf_counter() - t0

        base_score = result_base.score()
        mp0_score = result_mp0.score()
        diff = abs(base_score - mp0_score)

        status = "PASS" if diff < 0.01 else "FAIL"
        print(f"    Base:     {base_score:.2f}  ({t_base:.2f}s)")
        print(f"    MultiPly: {mp0_score:.2f}  ({t_mp0:.2f}s)")
        print(f"    Diff:     {diff:.4f}  [{status}]")

    print()


def run_nply_benchmark(weights, n_plies=1, max_scenarios=0, n_threads=0, benchmarks=None):
    """Score benchmarks at N-ply depth and compare with 0-ply."""
    print("=" * 60)
    print(f"BENCHMARK: {n_plies}-ply scoring")
    print("=" * 60)

    # Create N-ply strategy
    multipy = bgbot_cpp.create_multipy_5nn(
        weights['purerace'], weights['racing'],
        weights['attacking'], weights['priming'], weights['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=n_plies)

    if benchmarks is None:
        benchmarks = ['contact', 'purerace']
    for bm_name in benchmarks:
        bm_file = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_file):
            print(f"  Skipping {bm_name} (file not found)")
            continue

        scenarios_full = load_benchmark_file(bm_file)
        n_total = scenarios_full.size()

        if max_scenarios > 0 and max_scenarios < n_total:
            step = max(1, n_total // max_scenarios)
            scenarios = load_benchmark_file(bm_file, step=step)
            n = scenarios.size()
            print(f"\n  {bm_name}: {n}/{n_total} scenarios (step={step})")
        else:
            scenarios = scenarios_full
            n = n_total
            print(f"\n  {bm_name}: {n} scenarios")

        # 0-ply baseline
        t0 = time.perf_counter()
        result_0ply = bgbot_cpp.score_benchmarks_5nn(
            scenarios,
            weights['purerace'], weights['racing'],
            weights['attacking'], weights['priming'], weights['anchoring'],
            NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)
        t_0ply = time.perf_counter() - t0

        # N-ply
        t0 = time.perf_counter()
        result_nply = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy, n_threads)
        t_nply = time.perf_counter() - t0

        score_0 = result_0ply.score()
        score_n = result_nply.score()
        improvement = score_0 - score_n

        print(f"    0-ply:    {score_0:.2f}  ({t_0ply:.2f}s)")
        print(f"    {n_plies}-ply:    {score_n:.2f}  ({t_nply:.2f}s)")
        print(f"    Improvement: {improvement:+.2f}")
        print(f"    Slowdown: {t_nply / max(t_0ply, 0.001):.1f}x")
        if n_plies >= 1:
            cache_sz = multipy.cache_size()
            cache_h = multipy.cache_hits()
            cache_m = multipy.cache_misses()
            cache_total = cache_h + cache_m
            hit_rate = cache_h / max(cache_total, 1) * 100
            print(f"    Cache: {cache_sz} entries, {cache_h} hits / {cache_total} lookups ({hit_rate:.1f}% hit rate)")
            multipy.clear_cache()

    print()


def main():
    parser = argparse.ArgumentParser(description='Multi-ply benchmark')
    parser.add_argument('--ply', type=int, default=1, help='Number of plies (default: 1)')
    parser.add_argument('--scenarios', type=int, default=0, help='Max scenarios (0=all)')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads (0=auto)')
    parser.add_argument('--skip-0ply-test', action='store_true', help='Skip 0-ply passthrough test')
    parser.add_argument('--benchmarks', type=str, default='contact,purerace',
                        help='Comma-separated list of benchmarks (default: contact,purerace)')
    args = parser.parse_args()

    weights = get_best_weights()
    if weights is None:
        print("Cannot find all weight files. Exiting.")
        sys.exit(1)

    print(f"Weight files found:")
    for t, p in weights.items():
        print(f"  {t}: {p}")
    print()

    if not args.skip_0ply_test:
        run_0ply_passthrough_test(weights, n_threads=args.threads)

    benchmark_list = [b.strip() for b in args.benchmarks.split(',')]
    run_nply_benchmark(weights, n_plies=args.ply, max_scenarios=args.scenarios,
                       n_threads=args.threads, benchmarks=benchmark_list)


if __name__ == '__main__':
    main()
