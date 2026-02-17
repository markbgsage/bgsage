"""
Rollout benchmark: find top-100 worst 0-ply errors, compare with 1-ply, 2-ply, and rollout.

Usage:
  python bgsage/scripts/run_rollout_benchmark.py [--top N] [--threads N] [--skip-2ply]
  python bgsage/scripts/run_rollout_benchmark.py --model stage3  # specific model
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
from bgsage.data import load_benchmark_file, load_benchmark_scenarios_by_indices
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'data')


def main():
    parser = argparse.ArgumentParser(description='Rollout benchmark: top-N worst 0-ply errors')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--top', type=int, default=100, help='Number of worst scenarios (default: 100)')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads (0=auto)')
    parser.add_argument('--skip-2ply', action='store_true', help='Skip 2-ply and 3-ply')
    parser.add_argument('--skip-rollout-1p', action='store_true', help='Skip 1-ply rollout')
    parser.add_argument('--skip-rollout-2p', action='store_true', help='Skip 2-ply rollout')
    args = parser.parse_args()

    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary(f'Model: {args.model}')

    bgbot_cpp.init_escape_tables()

    # Step 1: Load and score all scenarios at 0-ply
    print("Loading benchmarks...")
    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

    scenarios_contact = load_benchmark_file(contact_file)
    n_contact = scenarios_contact.size()
    scenarios_crashed = load_benchmark_file(crashed_file)
    n_crashed = scenarios_crashed.size()
    total = n_contact + n_crashed
    print(f"  contact: {n_contact}, crashed: {n_crashed}, total: {total}")
    print()

    # Helper to get weight paths as a tuple (for functions needing positional args)
    wt = (w.purerace, w.racing, w.attacking, w.priming, w.anchoring)
    ht = w.hidden_sizes

    print("Scoring all scenarios at 0-ply...")
    t0 = time.perf_counter()
    errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_contact, *wt, *ht)
    errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_crashed, *wt, *ht)
    t_0ply_full = time.perf_counter() - t0
    print(f"  Done in {t_0ply_full:.1f}s")

    # Combine and sort
    all_errors = []
    for i, err in enumerate(errors_contact):
        all_errors.append((err, 'contact', i))
    for i, err in enumerate(errors_crashed):
        all_errors.append((err, 'crashed', i))
    all_errors.sort(key=lambda x: -x[0])

    overall_er = sum(e[0] for e in all_errors) / total * 1000
    print(f"  Overall 0-ply ER: {overall_er:.2f}")

    # Step 2: Extract top-N indices
    top_n = min(args.top, total)
    top_errors = all_errors[:top_n]
    top_mean_0ply = sum(e[0] for e in top_errors) / top_n * 1000

    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])

    print(f"\nTop {top_n} worst 0-ply errors:")
    print(f"  Mean error: {top_mean_0ply:.2f} millipips")
    print(f"  Range: [{top_errors[-1][0]*1000:.1f}, {top_errors[0][0]*1000:.1f}]")
    print(f"  From contact: {len(contact_indices)}, from crashed: {len(crashed_indices)}")
    print()

    # Step 3: Build sub-ScenarioSets for the top-N scenarios
    print("Loading top-N scenarios as subsets...")
    top_contact_ss = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                      if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed_ss = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                      if crashed_indices else bgbot_cpp.ScenarioSet())
    print(f"  Contact subset: {top_contact_ss.size()}, Crashed subset: {top_crashed_ss.size()}")
    print()

    # Helper to score a subset and get combined ER
    def score_subset(label, score_fn):
        t0 = time.perf_counter()
        total_err = 0.0
        total_count = 0
        if top_contact_ss.size() > 0:
            r = score_fn(top_contact_ss)
            total_err += r.total_error
            total_count += r.count
        if top_crashed_ss.size() > 0:
            r = score_fn(top_crashed_ss)
            total_err += r.total_error
            total_count += r.count
        elapsed = time.perf_counter() - t0
        mean_err = (total_err / total_count * 1000) if total_count > 0 else 0
        print(f"  {label}: {mean_err:.2f} ({elapsed:.1f}s)")
        return mean_err, elapsed

    # Step 4: Score with each strategy
    results = []

    # 0-ply (subset verification)
    def score_0ply(ss):
        return bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args)
    er_0ply, t_0ply = score_subset('0-ply', score_0ply)
    results.append(('0-ply', er_0ply, t_0ply))

    # 1-ply
    multipy_1 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=1)

    def score_1ply(ss):
        multipy_1.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, multipy_1, args.threads)
    er_1ply, t_1ply = score_subset('1-ply', score_1ply)
    results.append(('1-ply', er_1ply, t_1ply))

    # 2-ply
    if not args.skip_2ply:
        multipy_2 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=2)

        def score_2ply(ss):
            multipy_2.clear_cache()
            return bgbot_cpp.score_benchmarks_multipy(ss, multipy_2, args.threads)
        er_2ply, t_2ply = score_subset('2-ply', score_2ply)
        results.append(('2-ply', er_2ply, t_2ply))

    # 3-ply
    if not args.skip_2ply:
        multipy_3 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=3)

        def score_3ply(ss):
            multipy_3.clear_cache()
            return bgbot_cpp.score_benchmarks_multipy(ss, multipy_3, args.threads)
        er_3ply, t_3ply = score_subset('3-ply', score_3ply)
        results.append(('3-ply', er_3ply, t_3ply))

    # Rollout: 1-ply decisions, 1296 trials, no trunc, VR at 0-ply, late=0@3
    if not args.skip_rollout_1p:
        rollout_1p_1296 = bgbot_cpp.create_rollout_5nn(
            *w.weight_args,
            n_trials=1296, truncation_depth=0,
            decision_ply=1, vr_ply=0,
            n_threads=args.threads,
            late_ply=0, late_threshold=3)

        def score_rollout_1p_1296(ss):
            return bgbot_cpp.score_benchmarks_rollout(ss, rollout_1p_1296, 1)
        er_ro, t_ro = score_subset('Rollout(dp=1, 1296t, late=0@3)', score_rollout_1p_1296)
        results.append(('Rollout(dp=1, 1296t, late=0@3)', er_ro, t_ro))

    # Rollout: 2-ply decisions, 1296 trials, no trunc, VR at 0-ply, late=0@3
    if not args.skip_rollout_2p:
        rollout_2p_1296 = bgbot_cpp.create_rollout_5nn(
            *w.weight_args,
            n_trials=1296, truncation_depth=0,
            decision_ply=2, vr_ply=0,
            n_threads=args.threads,
            late_ply=0, late_threshold=3)

        def score_rollout_2p_1296(ss):
            return bgbot_cpp.score_benchmarks_rollout(ss, rollout_2p_1296, 1)
        er_ro2, t_ro2 = score_subset('Rollout(dp=2, 1296t, late=0@3)', score_rollout_2p_1296)
        results.append(('Rollout(dp=2, 1296t, late=0@3)', er_ro2, t_ro2))

    # Step 5: Summary table
    print()
    print("=" * 75)
    print(f"Summary: top {top_n} worst 0-ply scenarios from {total} total")
    print("=" * 75)
    print(f"  {'Strategy':<45} {'ER':>8} {'Time':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for name, er, t in results:
        print(f"  {name:<45} {er:>8.2f} {t:>9.1f}s")
    print()


if __name__ == '__main__':
    main()
