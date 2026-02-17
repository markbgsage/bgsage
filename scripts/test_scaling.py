"""
Test parallel scaling of the C++ benchmark scorer.
Loads the contact benchmark once, then scores it with varying thread counts
to measure speedup.
"""

import os
import sys
import time

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


def test_scaling():
    n_hw = os.cpu_count() or 1
    print(f'Hardware concurrency: {n_hw} threads (likely {n_hw // 2} physical cores)')
    print()

    # Load contact benchmark (largest)
    filepath = os.path.join(DATA_DIR, 'contact.bm')
    print(f'Loading {filepath}...')
    t0 = time.time()
    scenarios = load_benchmark_file(filepath)
    t_load = time.time() - t0
    print(f'Loaded {len(scenarios)} scenarios in {t_load:.1f}s')
    print()

    weights = bgbot_cpp.PubEvalWeights.TESAURO

    # Warmup run
    _ = bgbot_cpp.score_benchmarks_pubeval(scenarios, weights, n_threads=1)

    # Test a range of thread counts
    thread_counts = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    thread_counts = [t for t in thread_counts if t <= n_hw]

    n_trials = 3  # take the best of N runs

    print(f'{"Threads":>8s}  {"Time (ms)":>10s}  {"Speedup":>8s}  {"Efficiency":>10s}  {"~1/N?":>8s}  {"Score":>8s}')
    print('-' * 66)

    baseline_time = None
    results = []

    for n_threads in thread_counts:
        best_time = float('inf')
        score = None

        for _ in range(n_trials):
            t0 = time.perf_counter()
            result = bgbot_cpp.score_benchmarks_pubeval(scenarios, weights, n_threads=n_threads)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            if elapsed < best_time:
                best_time = elapsed
                score = result.score()

        if baseline_time is None:
            baseline_time = best_time

        speedup = baseline_time / best_time
        efficiency = speedup / n_threads * 100
        ideal = baseline_time / n_threads  # ideal 1/N scaling

        results.append((n_threads, best_time, speedup, efficiency, score))
        print(f'{n_threads:>8d}  {best_time:>10.1f}  {speedup:>8.2f}x  {efficiency:>9.1f}%  {ideal:>7.0f}ms  {score:>8.2f}')

    print()

    # Check scores are all identical
    scores = set(round(r[4], 2) for r in results)
    if len(scores) == 1:
        print(f'All scores identical: {scores.pop()} (deterministic: OK)')
    else:
        print(f'WARNING: Scores differ across thread counts: {scores}')

    # Identify sweet spot
    best_idx = min(range(len(results)), key=lambda i: results[i][1])
    best_t, best_ms = results[best_idx][0], results[best_idx][1]
    print(f'Best time: {best_ms:.1f}ms at {best_t} threads ({results[best_idx][2]:.2f}x speedup)')
    print()

    # Physical core saturation analysis
    n_physical = n_hw // 2
    phys_results = [r for r in results if r[0] == n_physical]
    if phys_results:
        pt, pms, psp, peff, _ = phys_results[0]
        print(f'At {n_physical} physical cores: {pms:.1f}ms, {psp:.2f}x speedup, {peff:.1f}% efficiency')
        above_phys = [r for r in results if r[0] > n_physical]
        if above_phys:
            ht, hms, hsp, _, _ = above_phys[0]
            ht_gain = (pms - hms) / pms * 100
            print(f'HT benefit ({ht} threads vs {n_physical}): {ht_gain:+.1f}% '
                  f'({"marginal" if abs(ht_gain) < 10 else "significant"})')

    print()

    # Full benchmark with auto thread count
    print('=== Full benchmark (auto threads, all 3 types) ===')
    print()
    total_ms = 0
    for bm_type in ['contact', 'crashed', 'race']:
        filepath = os.path.join(DATA_DIR, f'{bm_type}.bm')
        scenarios_bm = load_benchmark_file(filepath)
        t0 = time.perf_counter()
        result = bgbot_cpp.score_benchmarks_pubeval(scenarios_bm, weights, n_threads=0)
        elapsed = (time.perf_counter() - t0) * 1000
        total_ms += elapsed
        print(f'  {bm_type:8s}: {result.score():8.2f}  ({result.count} scenarios in {elapsed:.0f}ms)')
    print(f'\n  Total C++ scoring time: {total_ms:.0f}ms')


if __name__ == '__main__':
    test_scaling()
