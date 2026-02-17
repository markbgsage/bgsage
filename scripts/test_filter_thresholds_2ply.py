"""
Test filter thresholds at 2-ply depth on subsets of benchmarks.
Uses 500 scenarios (sampled) to keep timing manageable.
"""

import os
import sys
import time

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

NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250
N_THREADS = 24
N_SCENARIOS = 500


def main():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    wpaths = {t: os.path.join(MODELS_DIR, f'sl_{t}.weights.best') for t in types}

    # Load sampled scenarios
    contact_full = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'))
    purerace_full = load_benchmark_file(os.path.join(DATA_DIR, 'purerace.bm'))

    n_c = contact_full.size()
    n_r = purerace_full.size()
    step_c = max(1, n_c // N_SCENARIOS)
    step_r = max(1, n_r // N_SCENARIOS)

    contact_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'), step=step_c)
    purerace_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'purerace.bm'), step=step_r)

    print(f"Contact: {contact_scenarios.size()}/{n_c} scenarios (step={step_c})")
    print(f"PureRace: {purerace_scenarios.size()}/{n_r} scenarios (step={step_r})")
    print(f"Threads: {N_THREADS}")

    # 0-ply baselines on same subsets
    result_0ply_c = bgbot_cpp.score_benchmarks_5nn(
        contact_scenarios,
        wpaths['purerace'], wpaths['racing'],
        wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)
    result_0ply_r = bgbot_cpp.score_benchmarks_5nn(
        purerace_scenarios,
        wpaths['purerace'], wpaths['racing'],
        wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)

    base_c = result_0ply_c.score()
    base_r = result_0ply_r.score()
    print(f"0-ply baselines: contact={base_c:.2f}, purerace={base_r:.2f}")
    print()

    configs = [
        (2,  0.02, "ULTRA2(2, 0.02)"),
        (3,  0.03, "ULTRA3(3, 0.03)"),
        (4,  0.04, "TIGHT (4, 0.04)"),
        (5,  0.08, "TINY  (5, 0.08)"),
        (8,  0.12, "NARROW(8, 0.12)"),
        (8,  0.16, "NORMAL(8, 0.16)"),
    ]

    print(f"{'Config':<24} {'Contact':>10} {'Impr':>7} {'Time':>8}  {'PureRace':>10} {'Impr':>7} {'Time':>8}  {'Cache%':>7}")
    print("-" * 100)

    for max_moves, threshold, label in configs:
        multipy = bgbot_cpp.create_multipy_5nn(
            wpaths['purerace'], wpaths['racing'],
            wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
            NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_plies=2,
            filter_max_moves=max_moves,
            filter_threshold=threshold)

        # Contact
        t0 = time.perf_counter()
        result_c = bgbot_cpp.score_benchmarks_multipy(contact_scenarios, multipy, N_THREADS)
        t_c = time.perf_counter() - t0
        cache_h = multipy.cache_hits()
        cache_m = multipy.cache_misses()
        cache_total = cache_h + cache_m
        hit_rate_c = cache_h / max(cache_total, 1) * 100
        multipy.clear_cache()

        # PureRace
        t0 = time.perf_counter()
        result_r = bgbot_cpp.score_benchmarks_multipy(purerace_scenarios, multipy, N_THREADS)
        t_r = time.perf_counter() - t0
        cache_h2 = multipy.cache_hits()
        cache_m2 = multipy.cache_misses()
        cache_total2 = cache_h2 + cache_m2
        hit_rate_r = cache_h2 / max(cache_total2, 1) * 100
        multipy.clear_cache()

        sc = result_c.score()
        sr = result_r.score()
        ic = base_c - sc
        ir = base_r - sr

        print(f"{label:<24} {sc:>10.2f} {ic:>+7.2f} {t_c:>7.1f}s  {sr:>10.2f} {ir:>+7.2f} {t_r:>7.1f}s  {hit_rate_c:>5.1f}%/{hit_rate_r:.1f}%")


if __name__ == '__main__':
    main()
