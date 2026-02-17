"""
Test filter thresholds on the full contact benchmark for definitive results.
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


def main():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    wpaths = {t: os.path.join(MODELS_DIR, f'sl_{t}.weights.best') for t in types}

    # Load full scenarios
    contact_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'))
    purerace_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'purerace.bm'))

    # 0-ply baselines
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

    print(f"0-ply baselines: contact={result_0ply_c.score():.2f} ({contact_scenarios.size()} scenarios)")
    print(f"                 purerace={result_0ply_r.score():.2f} ({purerace_scenarios.size()} scenarios)")
    print(f"Threads: {N_THREADS}")
    print()

    configs = [
        (4,  0.04, "TIGHT (4, 0.04)"),
        (5,  0.08, "TINY  (5, 0.08)"),
        (8,  0.12, "NARROW(8, 0.12)"),
        (8,  0.16, "NORMAL(8, 0.16)"),
        (12, 0.24, "WIDE (12, 0.24)"),
        (16, 0.32, "LARGE (16, 0.32)"),
    ]

    print(f"{'Config':<24} {'Contact':>10} {'Impr':>7} {'Time':>8} {'PureRace':>10} {'Impr':>7} {'Time':>8}")
    print("-" * 85)

    for max_moves, threshold, label in configs:
        multipy = bgbot_cpp.create_multipy_5nn(
            wpaths['purerace'], wpaths['racing'],
            wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
            NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_plies=1,
            filter_max_moves=max_moves,
            filter_threshold=threshold)

        # Contact
        t0 = time.perf_counter()
        result_c = bgbot_cpp.score_benchmarks_multipy(contact_scenarios, multipy, N_THREADS)
        t_c = time.perf_counter() - t0
        multipy.clear_cache()

        # PureRace
        t0 = time.perf_counter()
        result_r = bgbot_cpp.score_benchmarks_multipy(purerace_scenarios, multipy, N_THREADS)
        t_r = time.perf_counter() - t0
        multipy.clear_cache()

        sc = result_c.score()
        sr = result_r.score()
        ic = result_0ply_c.score() - sc
        ir = result_0ply_r.score() - sr

        print(f"{label:<24} {sc:>10.2f} {ic:>+7.2f} {t_c:>7.1f}s {sr:>10.2f} {ir:>+7.2f} {t_r:>7.1f}s")


if __name__ == '__main__':
    main()
