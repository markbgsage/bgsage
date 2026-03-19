"""Run full Contact benchmark at 1-4 ply and truncated rollout levels (1T, 2T).

Scores contact.bm at multiple evaluation levels. N-ply (1-4) run in-process;
rollout levels run in subprocesses to avoid OOM from thread-local cache
accumulation.

Rollout levels use subsamples because they crash on >5k scenarios (1T) or
>1k scenarios (2T) due to thread-local PosCache accumulation.

Usage:
  python scripts/run_contact_benchmark.py
  python scripts/run_contact_benchmark.py --threads 16
"""

import os
import sys
import time
import subprocess
import json
import argparse
import gc

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
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')


def run_rollout_subprocess(label, rollout_args, w, step, threads):
    """Run a rollout benchmark in a subprocess. Returns (er, elapsed, count)."""
    helper = os.path.join(script_dir, '_run_rollout_contact.py')
    cmd = [sys.executable, helper] + rollout_args + [
        '--threads', str(threads), '--step', str(step)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                            cwd=project_dir)
    if result.returncode != 0:
        print(f"  {label:<25} CRASHED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f"    {line}")
        return None, None, 0
    data = json.loads(result.stdout.strip())
    return data['er'], data['elapsed'], data['count']


def main():
    parser = argparse.ArgumentParser(description='Contact benchmark at all evaluation levels')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--threads', type=int, default=16, help='Threads (default: 16)')
    args = parser.parse_args()

    bgbot_cpp.init_escape_tables()
    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary(f'Contact benchmark: {args.model}')

    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    scenarios = load_benchmark_file(contact_file)
    n = scenarios.size()
    print(f"Contact scenarios: {n}")
    print()

    results = []

    # 1-ply
    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_5nn(scenarios, *w.weight_args)
    elapsed = time.perf_counter() - t0
    print(f"  {'1-ply':<25} ER={r.score():8.2f}  ({n} scenarios, {elapsed:.1f}s)")
    results.append(('1-ply', r.score(), elapsed, n))

    # 2-ply through 4-ply
    for ply in [2, 3, 4]:
        multipy = bgbot_cpp.create_multipy_5nn(
            *w.weight_args, n_plies=ply,
            parallel_evaluate=True, parallel_threads=args.threads)
        multipy.clear_cache()
        t0 = time.perf_counter()
        r = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy, 1)
        elapsed = time.perf_counter() - t0
        print(f"  {f'{ply}-ply':<25} ER={r.score():8.2f}  ({n} scenarios, {elapsed:.1f}s)")
        results.append((f'{ply}-ply', r.score(), elapsed, n))
        del multipy
        gc.collect()

    # Truncated rollout levels in subprocesses
    # 1T can handle ~5k scenarios; 2T only ~1k before crashing
    rollout_configs = [
        ('1T (XG Roller)',
         ['--n-trials', '42', '--truncation-depth', '5', '--decision-ply', '1'],
         20),  # step=20 → ~5k scenarios
        ('2T (XG Roller+)',
         ['--n-trials', '360', '--truncation-depth', '7', '--decision-ply', '2',
          '--late-ply', '1', '--late-threshold', '2'],
         100),  # step=100 → ~1k scenarios
    ]

    for label, rollout_args, step in rollout_configs:
        print(f"  Running {label} in subprocess (step={step})...")
        er, elapsed, count = run_rollout_subprocess(label, rollout_args, w, step, args.threads)
        if er is not None:
            print(f"  {label:<25} ER={er:8.2f}  ({count} scenarios*, {elapsed:.1f}s)")
            results.append((label, er, elapsed, count))
        else:
            results.append((label, float('nan'), 0, 0))

    # Summary
    print()
    print("=" * 70)
    print(f"Contact Benchmark Summary ({args.model}, contact.bm)")
    print("=" * 70)
    print(f"  {'Strategy':<25} {'ER':>8}  {'Scenarios':>10}  {'Time':>10}")
    print(f"  {'-'*25} {'-'*8}  {'-'*10}  {'-'*10}")
    for name, er, t, cnt in results:
        star = '*' if cnt < n else ''
        print(f"  {name:<25} {er:>8.2f}  {cnt:>9}{star}  {t:>9.1f}s")
    if any(cnt < n for _, _, _, cnt in results):
        print(f"\n  * Subsample — rollout levels crash on full dataset due to memory.")
    print()


if __name__ == '__main__':
    main()
