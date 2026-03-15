"""
Benchmark XG Roller-style truncated rollouts on top-100 worst 1-ply scenarios.

Compares: 1-ply, 2-ply, 3-ply, 4-ply, XGRoller, XGRoller+, XGRoller++ Checker, XGRoller++ Cube.

Usage:
  python bgsage/scripts/run_xgroller_benchmark.py [--top N] [--threads N]
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


def log(msg=""):
    print(msg, flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='XG Roller benchmark: top-N worst 1-ply errors')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--top', type=int, default=100, help='Number of worst scenarios (default: 100)')
    parser.add_argument('--threads', type=int, default=0, help='CPU threads (0=auto)')
    parser.add_argument('--skip-4ply', action='store_true', help='Skip 4-ply')
    parser.add_argument('--skip-xgrpp', action='store_true', help='Skip XGRoller++ (very slow)')
    args = parser.parse_args()

    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary(f'Model: {args.model}')

    bgbot_cpp.init_escape_tables()

    # Step 1: Load and score all scenarios at 1-ply
    log("Loading benchmarks...")
    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

    scenarios_contact = load_benchmark_file(contact_file)
    n_contact = scenarios_contact.size()
    scenarios_crashed = load_benchmark_file(crashed_file)
    n_crashed = scenarios_crashed.size()
    total = n_contact + n_crashed
    log(f"  contact: {n_contact}, crashed: {n_crashed}, total: {total}")

    log("Scoring all at 1-ply...")
    t0 = time.perf_counter()
    errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_contact,
        w.purerace, w.racing, w.attacking, w.priming, w.anchoring,
        *w.hidden_sizes)
    errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_crashed,
        w.purerace, w.racing, w.attacking, w.priming, w.anchoring,
        *w.hidden_sizes)
    log(f"  Done in {time.perf_counter()-t0:.1f}s")

    all_errors = []
    for i, err in enumerate(errors_contact):
        all_errors.append((err, 'contact', i))
    for i, err in enumerate(errors_crashed):
        all_errors.append((err, 'crashed', i))
    all_errors.sort(key=lambda x: -x[0])

    top_n = min(args.top, total)
    top_errors = all_errors[:top_n]
    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])

    top_contact = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                   if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                   if crashed_indices else bgbot_cpp.ScenarioSet())
    log(f"Top-{top_n} loaded: {top_contact.size()} contact, {top_crashed.size()} crashed")
    log()

    def score_subset(label, score_fn):
        log(f"  Scoring {label}...")
        t0 = time.perf_counter()
        total_err = 0.0
        total_count = 0
        if top_contact.size() > 0:
            r = score_fn(top_contact)
            total_err += r.total_error
            total_count += r.count
        if top_crashed.size() > 0:
            r = score_fn(top_crashed)
            total_err += r.total_error
            total_count += r.count
        elapsed = time.perf_counter() - t0
        mean_err = (total_err / total_count * 1000) if total_count > 0 else 0
        log(f"  {label:<50} {mean_err:>8.2f}  {elapsed:>9.1f}s")
        return mean_err, elapsed

    hdr = f"  {'Strategy':<50} {'ER':>8}  {'Time':>9}"
    sep = f"  {'-'*50} {'-'*8}  {'-'*9}"
    log(f"Scoring top-{top_n} worst 1-ply scenarios:")
    log(hdr)
    log(sep)
    results = []

    # 1-ply baseline
    def score_1ply(ss):
        return bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args)
    er, t = score_subset('1-ply', score_1ply)
    results.append(('1-ply', er, t))

    # 2-ply
    mp2 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=2)
    def score_2ply(ss):
        mp2.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, mp2, args.threads)
    er, t = score_subset('2-ply', score_2ply)
    results.append(('2-ply', er, t))

    # 3-ply
    mp3 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=3)
    def score_3ply(ss):
        mp3.clear_cache()
        return bgbot_cpp.score_benchmarks_multipy(ss, mp3, args.threads)
    er, t = score_subset('3-ply', score_3ply)
    results.append(('3-ply', er, t))

    # 4-ply
    if not args.skip_4ply:
        mp4 = bgbot_cpp.create_multipy_5nn(*w.weight_args, n_plies=4)
        def score_4ply(ss):
            mp4.clear_cache()
            return bgbot_cpp.score_benchmarks_multipy(ss, mp4, args.threads)
        er, t = score_subset('4-ply', score_4ply)
        results.append(('4-ply', er, t))

    # XGRoller: 42 trials, trunc=5, dp=1
    ro_xgr = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=42, truncation_depth=5, decision_ply=1,
        n_threads=args.threads, seed=42)
    def score_xgr(ss):
        return bgbot_cpp.score_benchmarks_rollout(ss, ro_xgr, 1)
    er, t = score_subset('XGRoller (42t, trunc=5, dp=1)', score_xgr)
    results.append(('XGRoller', er, t))

    # XGRoller+: 360 trials, trunc=7, dp=2, late=1@2
    ro_xgrp = bgbot_cpp.create_rollout_5nn(
        *w.weight_args,
        n_trials=360, truncation_depth=7, decision_ply=2,
        late_ply=1, late_threshold=2,
        n_threads=args.threads, seed=42)
    def score_xgrp(ss):
        return bgbot_cpp.score_benchmarks_rollout(ss, ro_xgrp, 1)
    er, t = score_subset('XGRoller+ (360t, trunc=7, dp=2, lp=1@2)', score_xgrp)
    results.append(('XGRoller+', er, t))

    # XGRoller++ Checker: 360 trials, trunc=5, dp=3, late=2@2
    if not args.skip_xgrpp:
        ro_xgrpp_c = bgbot_cpp.create_rollout_5nn(
            *w.weight_args,
            n_trials=360, truncation_depth=5, decision_ply=3,
            late_ply=2, late_threshold=2,
            n_threads=args.threads, seed=42)
        def score_xgrpp_c(ss):
            return bgbot_cpp.score_benchmarks_rollout(ss, ro_xgrpp_c, 1)
        er, t = score_subset('XGRoller++ Chk (360t, trunc=5, dp=3, lp=2@2)', score_xgrpp_c)
        results.append(('XGRoller++ Checker', er, t))

    # XGRoller++ Cube: 360 trials, trunc=7, dp=3, late=2@2
    if not args.skip_xgrpp:
        ro_xgrpp_q = bgbot_cpp.create_rollout_5nn(
            *w.weight_args,
            n_trials=360, truncation_depth=7, decision_ply=3,
            late_ply=2, late_threshold=2,
            n_threads=args.threads, seed=42)
        def score_xgrpp_q(ss):
            return bgbot_cpp.score_benchmarks_rollout(ss, ro_xgrpp_q, 1)
        er, t = score_subset('XGRoller++ Cube (360t, trunc=7, dp=3, lp=2@2)', score_xgrpp_q)
        results.append(('XGRoller++ Cube', er, t))

    # Summary
    log()
    log("=" * 75)
    log(f"Summary: top-{top_n} worst 1-ply scenarios from {total} total")
    log("=" * 75)
    log(hdr)
    log(sep)
    for name, er, t in results:
        log(f"  {name:<50} {er:>8.2f}  {t:>9.1f}s")
    log()


if __name__ == '__main__':
    main()
