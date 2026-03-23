"""
Run all benchmarks for Stage 5 Small and Hybrid evaluators.

Benchmarks:
  1. Stage 5 Small (all small): Contact benchmark 1-3 ply (full), 4-ply + truncated (subsample)
  2. Stage 5 Small: Top-100 rollout benchmark (1-4 ply + truncated)
  3. Hybrid (small filter + full leaf): Contact benchmark 1-3 ply (full)
  4. Hybrid: Top-100 rollout benchmark (1-4 ply + truncated)

Usage:
    python scripts/run_stage5small_benchmarks.py                # All benchmarks
    python scripts/run_stage5small_benchmarks.py --small-only   # Only Stage 5 Small
    python scripts/run_stage5small_benchmarks.py --hybrid-only  # Only Hybrid
    python scripts/run_stage5small_benchmarks.py --top100-only  # Only top-100
    python scripts/run_stage5small_benchmarks.py --contact-only # Only contact
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
_candidate = os.path.dirname(os.path.dirname(script_dir))
if not os.path.isdir(os.path.join(_candidate, 'build')):
    _candidate = r'C:\Users\mghig\Dropbox\agents\bgbot'
project_dir = _candidate
build_dirs = [
    os.path.join(project_dir, 'build_msvc'),
    os.path.join(project_dir, 'build_cpu_build'),
    os.path.join(project_dir, 'build'),
]

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    for d in build_dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)

for d in reversed(build_dirs):
    if os.path.isdir(d):
        sys.path.insert(0, d)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_benchmark_scenarios_by_indices
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
THREADS = 16


def get_weights(model_name):
    w = WeightConfig.from_model(model_name)
    w.validate()
    return w


def score_1ply(label, scenarios, w):
    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_5nn(scenarios, *w.weight_args)
    elapsed = time.perf_counter() - t0
    print(f"  {label:<30} ER={r.score():8.2f}  ({r.count} scenarios, {elapsed:.1f}s)")
    return r.score(), elapsed


def score_nply(label, scenarios, w, ply, threads=THREADS):
    multipy = bgbot_cpp.create_multipy_5nn(
        *w.weight_args, n_plies=ply,
        parallel_evaluate=True, parallel_threads=threads)
    multipy.clear_cache()
    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy, 1)
    elapsed = time.perf_counter() - t0
    print(f"  {label:<30} ER={r.score():8.2f}  ({r.count} scenarios, {elapsed:.1f}s)")
    del multipy
    gc.collect()
    return r.score(), elapsed


def score_nply_hybrid(label, scenarios, w_leaf, w_filter, ply, threads=THREADS):
    multipy = bgbot_cpp.create_multipy_hybrid_5nn(
        *w_leaf.weight_args,
        *w_filter.weight_args,
        n_plies=ply,
        parallel_evaluate=True, parallel_threads=threads)
    multipy.clear_cache()
    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_multipy(scenarios, multipy, 1)
    elapsed = time.perf_counter() - t0
    print(f"  {label:<30} ER={r.score():8.2f}  ({r.count} scenarios, {elapsed:.1f}s)")
    del multipy
    gc.collect()
    return r.score(), elapsed


def run_rollout_subprocess(label, w, create_kwargs, contact_indices, crashed_indices,
                           hybrid_w_filter=None):
    """Run a single rollout level in a subprocess."""
    if hybrid_w_filter:
        create_fn = 'create_rollout_hybrid_5nn'
        create_args = f'*{repr(w.weight_args)}, *{repr(hybrid_w_filter.weight_args)}'
    else:
        create_fn = 'create_rollout_5nn'
        create_args = f'*{repr(w.weight_args)}'

    code = f'''
import os, sys, json, time
for d in {repr([os.path.abspath(d) for d in build_dirs if os.path.isdir(d)])}:
    sys.path.insert(0, d)
    if sys.platform == 'win32':
        os.add_dll_directory(d)
sys.path.insert(0, {repr(os.path.join(project_dir, 'bgsage', 'python'))})
if sys.platform == 'win32':
    cuda_x64 = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)

import bgbot_cpp
from bgsage.data import load_benchmark_scenarios_by_indices

bgbot_cpp.init_escape_tables()

contact_file = {repr(os.path.join(DATA_DIR, 'contact.bm'))}
crashed_file = {repr(os.path.join(DATA_DIR, 'crashed.bm'))}
ci = {repr(contact_indices)}
ki = {repr(crashed_indices)}

tc = load_benchmark_scenarios_by_indices(contact_file, ci) if ci else bgbot_cpp.ScenarioSet()
tk = load_benchmark_scenarios_by_indices(crashed_file, ki) if ki else bgbot_cpp.ScenarioSet()

strat = bgbot_cpp.{create_fn}({create_args}, **{repr(create_kwargs)})

t0 = time.perf_counter()
total_err = 0.0
total_count = 0
if tc.size() > 0:
    r = bgbot_cpp.score_benchmarks_rollout(tc, strat, 1)
    total_err += r.total_error
    total_count += r.count
if tk.size() > 0:
    r = bgbot_cpp.score_benchmarks_rollout(tk, strat, 1)
    total_err += r.total_error
    total_count += r.count
elapsed = time.perf_counter() - t0
er = total_err / total_count * 1000 if total_count > 0 else 0
print(json.dumps({{"er": er, "elapsed": elapsed}}))
'''
    result = subprocess.run(
        [sys.executable, '-u', '-c', code],
        capture_output=True, text=True, timeout=600,
        cwd=project_dir
    )
    if result.returncode != 0:
        print(f"  {label:<50} CRASHED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f"    {line}")
        return None, None

    data = json.loads(result.stdout.strip())
    return data['er'], data['elapsed']


def run_top100_benchmark(label_prefix, w, threads=THREADS, hybrid_w_filter=None):
    """Run top-100 worst positions benchmark."""
    bgbot_cpp.init_escape_tables()

    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

    # Use Stage 5 (full) for identifying top-100 worst positions (consistent comparison)
    w_ref = WeightConfig.from_model('stage5')
    scenarios_contact = load_benchmark_file(contact_file)
    scenarios_crashed = load_benchmark_file(crashed_file)

    print(f"\n{'='*70}")
    print(f"  Top-100 Benchmark: {label_prefix}")
    print(f"{'='*70}")
    print(f"  Identifying top-100 worst 1-ply positions (using Stage 5 reference)...")

    errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_contact, *w_ref.weight_args[:5], *w_ref.hidden_sizes)
    errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
        scenarios_crashed, *w_ref.weight_args[:5], *w_ref.hidden_sizes)

    all_errors = []
    for i, err in enumerate(errors_contact):
        all_errors.append((err, 'contact', i))
    for i, err in enumerate(errors_crashed):
        all_errors.append((err, 'crashed', i))
    all_errors.sort(key=lambda x: -x[0])

    top_n = 100
    top_errors = all_errors[:top_n]
    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])
    print(f"  Top {top_n}: contact={len(contact_indices)}, crashed={len(crashed_indices)}")

    top_contact_ss = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                      if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed_ss = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                      if crashed_indices else bgbot_cpp.ScenarioSet())

    results = []

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
        return mean_err, elapsed

    # 1-ply
    er, t = score_subset(f'{label_prefix} 1-ply',
        lambda ss: bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args))
    print(f"  {'1-ply':<50} {er:>8.2f}  {t:>9.1f}s")
    results.append(('1-ply', er, t))

    # N-ply (2-4)
    for ply in [2, 3, 4]:
        if hybrid_w_filter:
            multipy = bgbot_cpp.create_multipy_hybrid_5nn(
                *w.weight_args, *hybrid_w_filter.weight_args,
                n_plies=ply, parallel_evaluate=True, parallel_threads=threads)
        else:
            multipy = bgbot_cpp.create_multipy_5nn(
                *w.weight_args, n_plies=ply,
                parallel_evaluate=True, parallel_threads=threads)
        er, t = score_subset(f'{label_prefix} {ply}-ply',
            lambda ss: (multipy.clear_cache(), bgbot_cpp.score_benchmarks_multipy(ss, multipy, 1))[1])
        print(f"  {f'{ply}-ply':<50} {er:>8.2f}  {t:>9.1f}s")
        results.append((f'{ply}-ply', er, t))
        del multipy
        gc.collect()

    # Rollout levels
    rollout_configs = [
        ('XG Roller (42t, trunc=5, dp=1)',
         dict(n_trials=42, truncation_depth=5, decision_ply=1, n_threads=threads)),
        ('XG Roller+ (360t, trunc=7, dp=2, late=1@2)',
         dict(n_trials=360, truncation_depth=7, decision_ply=2, n_threads=threads,
              late_ply=1, late_threshold=2)),
        ('XG Roller++ (360t, trunc=5, dp=3, late=2@2)',
         dict(n_trials=360, truncation_depth=5, decision_ply=3, n_threads=threads,
              late_ply=2, late_threshold=2)),
    ]

    for name, kwargs in rollout_configs:
        er, t = run_rollout_subprocess(name, w, kwargs, contact_indices, crashed_indices,
                                        hybrid_w_filter=hybrid_w_filter)
        if er is not None:
            print(f"  {name:<50} {er:>8.2f}  {t:>9.1f}s")
            results.append((name, er, t))
        else:
            results.append((name, float('nan'), 0))

    return results


def main():
    parser = argparse.ArgumentParser(description='Stage 5 Small + Hybrid benchmarks')
    parser.add_argument('--small-only', action='store_true', help='Only run Stage 5 Small benchmarks')
    parser.add_argument('--hybrid-only', action='store_true', help='Only run Hybrid benchmarks')
    parser.add_argument('--top100-only', action='store_true', help='Only run top-100 benchmarks')
    parser.add_argument('--contact-only', action='store_true', help='Only run contact benchmarks')
    parser.add_argument('--threads', type=int, default=THREADS, help='Threads (default: 16)')
    args = parser.parse_args()

    run_small = not args.hybrid_only
    run_hybrid = not args.small_only
    run_top100 = not args.contact_only
    run_contact = not args.top100_only

    bgbot_cpp.init_escape_tables()

    w_small = get_weights('stage5small')
    w_full = get_weights('stage5')

    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    contact_scenarios = load_benchmark_file(contact_file)
    n_contact = contact_scenarios.size()

    all_results = {}

    # --- Stage 5 Small benchmarks ---
    if run_small:
        if run_contact:
            print(f"\n{'='*70}")
            print(f"  Contact Benchmark: Stage 5 Small (100h/200h)")
            print(f"{'='*70}")

            # 1-3 ply full
            small_contact = []
            er, t = score_1ply('S5Small 1-ply', contact_scenarios, w_small)
            small_contact.append(('1-ply', er, t, n_contact))

            for ply in [2, 3]:
                er, t = score_nply(f'S5Small {ply}-ply', contact_scenarios, w_small, ply, args.threads)
                small_contact.append((f'{ply}-ply', er, t, n_contact))

            # 4-ply + truncated on subsample (same as Stage 5 full benchmark)
            contact_sub = load_benchmark_file(contact_file, step=100)
            n_sub = contact_sub.size()
            print(f"\n  Subsample ({n_sub} scenarios, step=100):")

            for ply in [1, 2, 3, 4]:
                if ply == 1:
                    er, t = score_1ply(f'S5Small {ply}-ply (sub)', contact_sub, w_small)
                else:
                    er, t = score_nply(f'S5Small {ply}-ply (sub)', contact_sub, w_small, ply, args.threads)
                small_contact.append((f'{ply}-ply (sub)', er, t, n_sub))

            all_results['small_contact'] = small_contact

        if run_top100:
            results = run_top100_benchmark('S5Small', w_small, args.threads)
            all_results['small_top100'] = results

    # --- Hybrid benchmarks ---
    if run_hybrid:
        if run_contact:
            print(f"\n{'='*70}")
            print(f"  Contact Benchmark: Hybrid (S5Small filter + S5 leaf)")
            print(f"{'='*70}")

            hybrid_contact = []
            # 1-ply: hybrid has no effect at 1-ply (leaf-only)
            er, t = score_1ply('Hybrid 1-ply (=S5 full)', contact_scenarios, w_full)
            hybrid_contact.append(('1-ply', er, t, n_contact))

            for ply in [2, 3]:
                er, t = score_nply_hybrid(f'Hybrid {ply}-ply', contact_scenarios,
                                          w_full, w_small, ply, args.threads)
                hybrid_contact.append((f'{ply}-ply', er, t, n_contact))

            all_results['hybrid_contact'] = hybrid_contact

        if run_top100:
            results = run_top100_benchmark('Hybrid', w_full, args.threads,
                                            hybrid_w_filter=w_small)
            all_results['hybrid_top100'] = results

    # Save results
    results_dir = os.path.join(project_dir, 'experiments', 'stage5small')
    os.makedirs(results_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(results_dir, f'benchmark_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
