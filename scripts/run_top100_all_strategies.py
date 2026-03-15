"""
Top-100 worst 1-ply positions scored at all strategy levels.
Includes 1-4 ply and XG Roller / Roller+ / Roller++ truncated rollouts.

Benchmark scoring is cubeless checker play evaluation. Rollout strategies
use the unified trial function (run_trial_unified with n_branches=0),
which skips all cubeful overhead -- equivalent to dedicated cubeless code.
"""

import os
import sys
import time

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

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')


def main():
    bgbot_cpp.init_escape_tables()

    w = WeightConfig.default()
    w.validate()
    w.print_summary('Model: stage5')

    # Step 1: Load and score all scenarios at 1-ply
    print("Loading benchmarks...")
    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

    scenarios_contact = load_benchmark_file(contact_file)
    scenarios_crashed = load_benchmark_file(crashed_file)
    n_contact = scenarios_contact.size()
    n_crashed = scenarios_crashed.size()
    total = n_contact + n_crashed
    print(f"  contact: {n_contact}, crashed: {n_crashed}, total: {total}")

    print("Scoring all scenarios at 1-ply...")
    t0 = time.perf_counter()
    errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_contact, w.purerace, w.racing, w.attacking, w.priming, w.anchoring,
        *w.hidden_sizes)
    errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_crashed, w.purerace, w.racing, w.attacking, w.priming, w.anchoring,
        *w.hidden_sizes)
    t_1ply_full = time.perf_counter() - t0
    print(f"  Done in {t_1ply_full:.1f}s")

    # Combine and sort
    all_errors = []
    for i, err in enumerate(errors_contact):
        all_errors.append((err, 'contact', i))
    for i, err in enumerate(errors_crashed):
        all_errors.append((err, 'crashed', i))
    all_errors.sort(key=lambda x: -x[0])

    overall_er = sum(e[0] for e in all_errors) / total * 1000
    print(f"  Overall 1-ply ER: {overall_er:.2f}")

    # Step 2: Extract top-100
    top_n = 100
    top_errors = all_errors[:top_n]
    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])
    print(f"\nTop {top_n} worst: contact={len(contact_indices)}, crashed={len(crashed_indices)}")
    top_mean_1ply = sum(e[0] for e in top_errors) / top_n * 1000
    print(f"  Mean 1-ply error: {top_mean_1ply:.2f}")

    # Build subsets
    top_contact_ss = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                      if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed_ss = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                      if crashed_indices else bgbot_cpp.ScenarioSet())
    print(f"  Contact subset: {top_contact_ss.size()}, Crashed subset: {top_crashed_ss.size()}")

    # Helper to score a subset
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
        print(f"  {label:<50} {mean_err:>8.2f}  {elapsed:>9.1f}s")
        return mean_err, elapsed

    print()
    print(f"{'Strategy':<52} {'ER':>8}  {'Time':>9}")
    print(f"{'-'*52} {'-'*8}  {'-'*9}")

    results = []

    # 1-ply
    def score_1ply(ss):
        return bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args)
    er, t = score_subset('1-ply', score_1ply)
    results.append(('1-ply', er, t))

    # 2-ply
    multipy_2 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=2)
    def score_2ply(ss):
        multipy_2.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, multipy_2, 0)
    er, t = score_subset('2-ply', score_2ply)
    results.append(('2-ply', er, t))

    # 3-ply
    multipy_3 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=3)
    def score_3ply(ss):
        multipy_3.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, multipy_3, 0)
    er, t = score_subset('3-ply', score_3ply)
    results.append(('3-ply', er, t))

    # 4-ply
    multipy_4 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=4)
    def score_4ply(ss):
        multipy_4.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, multipy_4, 0)
    er, t = score_subset('4-ply', score_4ply)
    results.append(('4-ply', er, t))

    # XG Roller: 42 trials, trunc=5, dp=1
    roller = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=42, truncation_depth=5,
        decision_ply=1, n_threads=0)
    def score_roller(ss):
        return bgbot_cpp.score_benchmarks_rollout(ss, roller, 1)
    er, t = score_subset('XG Roller (42t, trunc=5, dp=1)', score_roller)
    results.append(('XG Roller', er, t))

    # XG Roller+: 360 trials, trunc=7, dp=2, late=1@2
    roller_plus = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=360, truncation_depth=7,
        decision_ply=2, n_threads=0,
        late_ply=1, late_threshold=2)
    def score_roller_plus(ss):
        return bgbot_cpp.score_benchmarks_rollout(ss, roller_plus, 1)
    er, t = score_subset('XG Roller+ (360t, trunc=7, dp=2, late=1@2)', score_roller_plus)
    results.append(('XG Roller+', er, t))

    # XG Roller++ Checker: 360 trials, trunc=5, dp=3, late=2@2
    roller_pp = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=360, truncation_depth=5,
        decision_ply=3, n_threads=0,
        late_ply=2, late_threshold=2)
    def score_roller_pp(ss):
        return bgbot_cpp.score_benchmarks_rollout(ss, roller_pp, 1)
    er, t = score_subset('XG Roller++ (360t, trunc=5, dp=3, late=2@2)', score_roller_pp)
    results.append(('XG Roller++', er, t))

    # Summary
    print()
    print("=" * 75)
    print(f"Summary: top {top_n} worst 1-ply scenarios from {total} total")
    print("=" * 75)
    print(f"  {'Strategy':<50} {'ER':>8}  {'Time':>9}")
    print(f"  {'-'*50} {'-'*8}  {'-'*9}")
    for name, er, t in results:
        print(f"  {name:<50} {er:>8.2f}  {t:>8.1f}s")
    print()


if __name__ == '__main__':
    main()
