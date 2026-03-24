"""Full contact.bm benchmark at a given evaluation level, parallelized by scenario.

Each worker thread gets a batch of scenarios and evaluates them single-threaded.
This maximizes CPU utilization across all cores.

Usage:
  python bgsage/scripts/run_contact_benchmark.py --ply 2
  python bgsage/scripts/run_contact_benchmark.py --ply 3 --threads 32
  python bgsage/scripts/run_contact_benchmark.py --ply 4 --model stage7
  python bgsage/scripts/run_contact_benchmark.py --ply 1 --model stage5

Supports both 5-NN (stage3-6) and 17-NN pair (stage7) models automatically.
"""

import os
import sys
import time
import argparse

# Setup import paths — find the main project root (contains build/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir
while project_dir != os.path.dirname(project_dir):
    if os.path.isdir(os.path.join(project_dir, 'build')):
        break
    project_dir = os.path.dirname(project_dir)
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    build_msvc = os.path.join(project_dir, 'build_msvc')
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
    if os.path.isdir(build_msvc):
        os.add_dll_directory(build_msvc)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')

SHARED_PAIRS = {'prim_prim': 'prim_anch', 'anch_prim': 'prim_anch', 'anch_anch': 'prim_anch'}


def get_s7_weights():
    """Build S7 17-NN weight paths and hidden sizes."""
    pair_names = bgbot_cpp.game_plan_pair_names()
    weight_paths, hidden_sizes = [], []
    for name in pair_names:
        canonical = SHARED_PAIRS.get(name, name)
        if name == 'purerace':
            weight_paths.append(os.path.join(MODELS_DIR, 'sl_s7_purerace.weights.best'))
            hidden_sizes.append(100)
        else:
            weight_paths.append(os.path.join(MODELS_DIR, f'sl_s7_{canonical}.weights.best'))
            hidden_sizes.append(300)
    return weight_paths, hidden_sizes


def main():
    parser = argparse.ArgumentParser(
        description='Full contact.bm benchmark at a given ply, parallelized by scenario')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--ply', type=int, default=1, help='Ply level (1-4, default: 1)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Worker threads (0=auto, uses all CPUs)')
    parser.add_argument('--include-crashed', action='store_true',
                        help='Also score crashed.bm')
    parser.add_argument('--include-race', action='store_true',
                        help='Also score race.bm')
    args = parser.parse_args()

    bgbot_cpp.init_escape_tables()

    is_pair = (args.model == 'stage7')

    n_threads = args.threads
    if n_threads <= 0:
        n_threads = os.cpu_count() or 1

    print(f"Model: {args.model}")
    print(f"Level: {args.ply}-ply")
    print(f"Threads: {n_threads} (scenario-level parallelism, single-threaded evals)")
    print(f"Strategy: {'17-NN pair' if is_pair else '5-NN plan'}")
    print()

    # Build benchmark list
    benchmarks = [('contact', os.path.join(DATA_DIR, 'contact.bm'))]
    if args.include_crashed:
        benchmarks.append(('crashed', os.path.join(DATA_DIR, 'crashed.bm')))
    if args.include_race:
        benchmarks.append(('race', os.path.join(DATA_DIR, 'race.bm')))

    # Create scoring function
    if is_pair:
        weight_paths, hidden_sizes = get_s7_weights()
        for p in weight_paths:
            if not os.path.exists(p):
                print(f"MISSING: {p}")
                return

        if args.ply == 1:
            def score_fn(ss):
                return bgbot_cpp.score_benchmarks_pair(ss, weight_paths, hidden_sizes, n_threads)
        else:
            strat = bgbot_cpp.create_multipy_pair(
                weight_paths, hidden_sizes,
                n_plies=args.ply,
                parallel_evaluate=False)

            def score_fn(ss):
                strat.clear_cache()
                return bgbot_cpp.score_benchmarks_multipy(ss, strat, n_threads)
    else:
        w = WeightConfig.from_args(args)
        w.validate()

        if args.ply == 1:
            def score_fn(ss):
                return bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args, n_threads)
        else:
            strat = bgbot_cpp.create_multipy_5nn(
                *w.weight_args, n_plies=args.ply,
                parallel_evaluate=False)

            def score_fn(ss):
                strat.clear_cache()
                return bgbot_cpp.score_benchmarks_multipy(ss, strat, n_threads)

    # Run benchmarks
    for bm_name, bm_path in benchmarks:
        print(f"Loading {bm_name}.bm...", flush=True)
        scenarios = load_benchmark_file(bm_path)
        n = scenarios.size()
        print(f"  {n} scenarios", flush=True)

        print(f"Scoring at {args.ply}-ply...", flush=True)
        t0 = time.perf_counter()
        result = score_fn(scenarios)
        elapsed = time.perf_counter() - t0

        er = result.score()
        print(f"  {bm_name} ER: {er:.2f}  ({n} scenarios, {elapsed:.1f}s)")
        print(f"  Rate: {n / elapsed:.0f} scenarios/sec")
        print()


if __name__ == '__main__':
    main()
