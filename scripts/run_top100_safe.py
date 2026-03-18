"""
Top-100 worst 1-ply positions scored at all strategy levels.
Runs each rollout level in a separate subprocess to avoid segfaults
from thread-local storage accumulation across strategies.
"""

import argparse
import os
import sys
import time
import subprocess
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir is the main bgbot directory (parent of the bgsage submodule)
# Handle both normal repo layout and worktree layout
_candidate = os.path.dirname(os.path.dirname(script_dir))  # normal: bgsage/../ = bgbot
if not os.path.isdir(os.path.join(_candidate, 'build')):
    # Worktree: try the main repo
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
    for build_dir in build_dirs:
        if os.path.isdir(build_dir):
            os.add_dll_directory(build_dir)

for build_dir in reversed(build_dirs):
    if os.path.isdir(build_dir):
        sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_benchmark_scenarios_by_indices
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')


def score_subset(label, score_fn, top_contact_ss, top_crashed_ss):
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


def run_rollout_subprocess(level_name, create_kwargs, w, contact_indices, crashed_indices, threads):
    """Run a single rollout level in a subprocess to avoid segfaults."""
    code = f'''
import os, sys, json, time
sys.path.insert(0, {repr(os.path.join(project_dir, 'build'))})
sys.path.insert(0, {repr(os.path.join(project_dir, 'bgsage', 'python'))})
if sys.platform == 'win32':
    for d in {repr([os.path.abspath(d) for d in build_dirs if os.path.isdir(d)])}:
        os.add_dll_directory(d)
    cuda_x64 = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)

import bgbot_cpp
from bgsage.data import load_benchmark_scenarios_by_indices

bgbot_cpp.init_escape_tables()
weight_args = {repr(w.weight_args)}

contact_file = {repr(os.path.join(DATA_DIR, 'contact.bm'))}
crashed_file = {repr(os.path.join(DATA_DIR, 'crashed.bm'))}
ci = {repr(contact_indices)}
ki = {repr(crashed_indices)}

tc = load_benchmark_scenarios_by_indices(contact_file, ci) if ci else bgbot_cpp.ScenarioSet()
tk = load_benchmark_scenarios_by_indices(crashed_file, ki) if ki else bgbot_cpp.ScenarioSet()

import sys as _sys
print(f'Creating rollout...', file=_sys.stderr, flush=True)
strat = bgbot_cpp.create_rollout_5nn(*weight_args, **{repr(create_kwargs)})
print(f'Rollout created. contact={{tc.size()}}, crashed={{tk.size()}}', file=_sys.stderr, flush=True)

t0 = time.perf_counter()
total_err = 0.0
total_count = 0
if tc.size() > 0:
    print(f'Scoring contact...', file=_sys.stderr, flush=True)
    r = bgbot_cpp.score_benchmarks_rollout(tc, strat, 1)
    total_err += r.total_error
    total_count += r.count
    print(f'Contact done: {{r.count}} scenarios', file=_sys.stderr, flush=True)
if tk.size() > 0:
    print(f'Scoring crashed...', file=_sys.stderr, flush=True)
    r = bgbot_cpp.score_benchmarks_rollout(tk, strat, 1)
    total_err += r.total_error
    total_count += r.count
    print(f'Crashed done: {{r.count}} scenarios', file=_sys.stderr, flush=True)
elapsed = time.perf_counter() - t0
er = total_err / total_count * 1000 if total_count > 0 else 0
print(json.dumps({{"er": er, "elapsed": elapsed}}))
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, timeout=600,
        cwd=project_dir
    )
    if result.returncode != 0:
        print(f"  {level_name:<50} CRASHED (exit {result.returncode})")
        if result.stdout:
            print(f"    stdout: {result.stdout[:300]}")
        if result.stderr:
            print(f"    stderr: {result.stderr[:300]}")
        return None, None

    data = json.loads(result.stdout.strip())
    return data['er'], data['elapsed']


def main():
    parser = argparse.ArgumentParser(
        description='Top-100 worst 1-ply positions scored at 1-4 ply and truncated rollout levels.'
    )
    WeightConfig.add_model_arg(parser)
    parser.add_argument(
        '--threads', type=int, default=16,
        help='Threads used inside multi-ply / rollout evaluations (default=16)'
    )
    args = parser.parse_args()
    rollout_threads = args.threads

    bgbot_cpp.init_escape_tables()

    w = WeightConfig.from_args(args)
    w.validate()
    w.print_summary(f'Model: {args.model}')

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
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

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

    # Build subsets
    top_contact_ss = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                      if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed_ss = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                      if crashed_indices else bgbot_cpp.ScenarioSet())

    print()
    print(f"{'Strategy':<52} {'ER':>8}  {'Time':>9}")
    print(f"{'-'*52} {'-'*8}  {'-'*9}")

    results = []

    # 1-ply
    er, t = score_subset('1-ply',
        lambda ss: bgbot_cpp.score_benchmarks_5nn(ss, *w.weight_args),
        top_contact_ss, top_crashed_ss)
    print(f"  {'1-ply':<50} {er:>8.2f}  {t:>9.1f}s")
    results.append(('1-ply', er, t))

    # N-ply evaluations: create, score, then delete to free thread-local caches.
    # Thread-local PosCache is 64MB per thread, so keeping multiple multipy
    # strategies alive simultaneously can exhaust memory.
    for ply in [2, 3, 4]:
        multipy = bgbot_cpp.create_multipy_5nn(
            *w.weight_args, n_plies=ply,
            parallel_evaluate=True, parallel_threads=args.threads)
        er, t = score_subset(f'{ply}-ply',
            lambda ss: (multipy.clear_cache(), bgbot_cpp.score_benchmarks_multipy(ss, multipy, 1))[1],
            top_contact_ss, top_crashed_ss)
        print(f"  {f'{ply}-ply':<50} {er:>8.2f}  {t:>9.1f}s")
        results.append((f'{ply}-ply', er, t))
        del multipy
        import gc; gc.collect()

    # Rollout levels: run each in subprocess to avoid segfault
    rollout_configs = [
        ('XG Roller (42t, trunc=5, dp=1)',
         dict(n_trials=42, truncation_depth=5, decision_ply=1, n_threads=rollout_threads)),
        ('XG Roller+ (360t, trunc=7, dp=2, late=1@2)',
         dict(n_trials=360, truncation_depth=7, decision_ply=2, n_threads=rollout_threads,
              late_ply=1, late_threshold=2)),
        ('XG Roller++ (360t, trunc=5, dp=3, late=2@2)',
         dict(n_trials=360, truncation_depth=5, decision_ply=3, n_threads=rollout_threads,
              late_ply=2, late_threshold=2)),
    ]

    for name, kwargs in rollout_configs:
        er, t = run_rollout_subprocess(name, kwargs, w, contact_indices, crashed_indices, rollout_threads)
        if er is not None:
            print(f"  {name:<50} {er:>8.2f}  {t:>9.1f}s")
            results.append((name, er, t))
        else:
            results.append((name, float('nan'), 0))

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
