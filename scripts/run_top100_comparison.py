"""
Run top-100 benchmark for Stage 5, Stage 5 Small, and Hybrid models.
All three use the same top-100 positions (identified by Stage 5 1-ply errors).
Runs serially for fair wall-clock timing comparison.
"""

import gc
import json
import os
import subprocess
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
if not os.path.isdir(os.path.join(project_dir, 'build')):
    project_dir = r'C:\Users\mghig\Dropbox\agents\bgbot'

build_dirs = [
    os.path.join(project_dir, 'build_msvc'),
    os.path.join(project_dir, 'build'),
]

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
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

bgbot_cpp.init_escape_tables()

# Identify top-100 using Stage 5
w_s5 = WeightConfig.from_model('stage5')
w_s5s = WeightConfig.from_model('stage5small')

print('Loading benchmarks and identifying top-100...', flush=True)
contact_file = os.path.join(DATA_DIR, 'contact.bm')
crashed_file = os.path.join(DATA_DIR, 'crashed.bm')
sc = load_benchmark_file(contact_file)
sk = load_benchmark_file(crashed_file)

ec = bgbot_cpp.score_benchmarks_per_scenario_5nn(sc, *w_s5.weight_args[:5], *w_s5.hidden_sizes)
ek = bgbot_cpp.score_benchmarks_per_scenario_5nn(sk, *w_s5.weight_args[:5], *w_s5.hidden_sizes)

all_errors = [(e, 'contact', i) for i, e in enumerate(ec)] + \
             [(e, 'crashed', i) for i, e in enumerate(ek)]
all_errors.sort(key=lambda x: -x[0])
top100 = all_errors[:100]
ci = sorted([e[2] for e in top100 if e[1] == 'contact'])
ki = sorted([e[2] for e in top100 if e[1] == 'crashed'])
print(f'Top 100: contact={len(ci)}, crashed={len(ki)}', flush=True)

tc = load_benchmark_scenarios_by_indices(contact_file, ci) if ci else bgbot_cpp.ScenarioSet()
tk = load_benchmark_scenarios_by_indices(crashed_file, ki) if ki else bgbot_cpp.ScenarioSet()

del sc, sk, ec, ek, all_errors
gc.collect()


def score_top100(label, score_fn):
    t0 = time.perf_counter()
    total_err, total_count = 0.0, 0
    if tc.size() > 0:
        r = score_fn(tc)
        total_err += r.total_error
        total_count += r.count
    if tk.size() > 0:
        r = score_fn(tk)
        total_err += r.total_error
        total_count += r.count
    elapsed = time.perf_counter() - t0
    er = total_err / total_count * 1000 if total_count > 0 else 0
    print(f'  {label:<55} {er:>8.2f}  {elapsed:>8.1f}s', flush=True)
    return er, elapsed


def run_rollout_subprocess(label, model_name, kwargs, ci, ki,
                           hybrid_filter_model=None):
    """Run rollout in subprocess to avoid OOM from cache accumulation."""
    abs_build_dirs = [os.path.abspath(d) for d in build_dirs if os.path.isdir(d)]
    abs_python_path = os.path.join(os.path.abspath(project_dir), 'bgsage', 'python')
    abs_contact = os.path.join(os.path.abspath(DATA_DIR), 'contact.bm')
    abs_crashed = os.path.join(os.path.abspath(DATA_DIR), 'crashed.bm')

    code_lines = [
        'import os, sys, json, time',
        f'for d in {repr(abs_build_dirs)}:',
        '    sys.path.insert(0, d)',
        '    if sys.platform == "win32": os.add_dll_directory(d)',
        f'sys.path.insert(0, {repr(abs_python_path)})',
        'if sys.platform == "win32":',
        '    cuda = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/x64"',
        '    if os.path.isdir(cuda): os.add_dll_directory(cuda)',
        'import bgbot_cpp',
        'from bgsage.data import load_benchmark_scenarios_by_indices',
        'from bgsage.weights import WeightConfig',
        'bgbot_cpp.init_escape_tables()',
        f'tc = load_benchmark_scenarios_by_indices({repr(abs_contact)}, {repr(ci)}) if {repr(ci)} else bgbot_cpp.ScenarioSet()',
        f'tk = load_benchmark_scenarios_by_indices({repr(abs_crashed)}, {repr(ki)}) if {repr(ki)} else bgbot_cpp.ScenarioSet()',
        f'w = WeightConfig.from_model({repr(model_name)})',
    ]

    if hybrid_filter_model:
        code_lines += [
            f'wf = WeightConfig.from_model({repr(hybrid_filter_model)})',
            f'strat = bgbot_cpp.create_rollout_hybrid_5nn(*w.weight_args, *wf.weight_args, **{repr(kwargs)})',
        ]
    else:
        code_lines += [
            f'strat = bgbot_cpp.create_rollout_5nn(*w.weight_args, **{repr(kwargs)})',
        ]

    code_lines += [
        't0 = time.perf_counter()',
        'te, tn = 0.0, 0',
        'if tc.size() > 0:',
        '    r = bgbot_cpp.score_benchmarks_rollout(tc, strat, 1)',
        '    te += r.total_error; tn += r.count',
        'if tk.size() > 0:',
        '    r = bgbot_cpp.score_benchmarks_rollout(tk, strat, 1)',
        '    te += r.total_error; tn += r.count',
        'elapsed = time.perf_counter() - t0',
        'er = te / tn * 1000 if tn > 0 else 0',
        'print(json.dumps({"er": er, "elapsed": elapsed}))',
    ]

    code = '\n'.join(code_lines)
    result = subprocess.run(
        [sys.executable, '-u', '-c', code],
        capture_output=True, text=True, timeout=600,
        cwd=project_dir
    )
    if result.returncode != 0:
        print(f'  {label:<55} CRASHED (exit {result.returncode})', flush=True)
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-3:]:
                print(f'    {line}')
        return None, None
    data = json.loads(result.stdout.strip())
    print(f'  {label:<55} {data["er"]:>8.2f}  {data["elapsed"]:>8.1f}s', flush=True)
    return data['er'], data['elapsed']


# ============================================================
# Run all three models serially for fair timing comparison
# ============================================================

rollout_config = (
    'XG Roller++ (360t,trunc=5,dp=3,late=2@2)',
    dict(n_trials=360, truncation_depth=5, decision_ply=3, n_threads=THREADS,
         late_ply=2, late_threshold=2),
)

configs = [
    ('Stage 5 (200h/400h)', 'stage5', w_s5, None, None),
    ('S5 Small (100h/200h)', 'stage5small', w_s5s, None, None),
    ('Hybrid (S5S filter + S5 leaf)', 'stage5', w_s5, w_s5s, 'stage5small'),
]

for model_label, model_name, w, w_filter, filter_model_name in configs:
    print(f"\n{'='*70}")
    print(f'  Top-100 Benchmark: {model_label}')
    print(f"{'='*70}", flush=True)

    # 4-ply
    if w_filter:
        mp = bgbot_cpp.create_multipy_hybrid_5nn(
            *w.weight_args, *w_filter.weight_args,
            n_plies=4, parallel_evaluate=True, parallel_threads=THREADS)
    else:
        mp = bgbot_cpp.create_multipy_5nn(
            *w.weight_args, n_plies=4,
            parallel_evaluate=True, parallel_threads=THREADS)
    score_top100('4-ply',
                 lambda ss, m=mp: (m.clear_cache(),
                                   bgbot_cpp.score_benchmarks_multipy(ss, m, 1))[1])
    del mp
    gc.collect()

    # XG Roller++
    name, kwargs = rollout_config
    run_rollout_subprocess(name, model_name, kwargs, ci, ki,
                           hybrid_filter_model=filter_model_name)

print(f"\n{'='*70}")
print('  ALL DONE')
print(f"{'='*70}", flush=True)
