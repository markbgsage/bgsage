# SPDX-License-Identifier: MPL-2.0
"""Stage 9 benchmarks matching what Stage 8 reports.

Sections:
  1. Standard 1-ply benchmarks (per-plan, contact, crashed, race, vs PubEval)
  2. Pair-filtered ER: S9 vs S5 vs S8
  3. Multi-ply contact (1, 2, 3 ply full, 4 ply subsample)
  4. Top-100 worst-1-ply N-ply (1-4 ply) + XG Roller / Roller+ / Roller++

Self-play (S8 reports S/G/B%) is unavailable for S9 — no 19-NN
play_games_*_vs_self binding exists. It is omitted from the S9 report.

Run from anywhere:
    python bgsage/scripts/run_s9_benchmarks.py
"""
import os
import sys
import json
import time
import gc
import subprocess
import numpy as np
from datetime import datetime

# Path setup — handles both normal repo and worktree layout (matches run_s8_training.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
bgsage_dir = os.path.dirname(script_dir)

_parts = os.path.normpath(bgsage_dir).replace('\\', '/').split('/')
if '.claude' in _parts:
    _idx = _parts.index('.claude')
    bgsage_dir = '/'.join(_parts[:_idx])
project_dir = os.path.dirname(bgsage_dir)
build_dirs = [os.path.join(project_dir, 'build_msvc'), os.path.join(project_dir, 'build')]

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
sys.path.insert(0, os.path.join(bgsage_dir, 'python'))

import bgbot_cpp
from bgsage.data import (load_benchmark_file, load_benchmark_scenarios_by_indices,
                          board_from_gnubg_position_string)
from bgsage.weights import WeightConfig, WeightConfigPair

DATA_DIR = os.path.join(bgsage_dir, 'data')
MODELS_DIR = os.path.join(bgsage_dir, 'models')
N_THREADS = 32                # User has 32-core desktop
N_THREADS_MULTIPY = 32        # Multi-ply N-ply on full contact.bm
N_THREADS_PAIR_HIGH_PLY = 4   # Pair strategy segfaults at 8+ threads, 3-ply decision (top-100 + rollouts)
CHECKPOINT_PATH = None        # Set in main()


def save_checkpoint(results):
    """Save current results dict to checkpoint path."""
    if CHECKPOINT_PATH:
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Pair names + canonical map (same as S8)
# ---------------------------------------------------------------------------
PAIR_NAMES = [
    'purerace',
    'race_race', 'race_att', 'race_prim', 'race_anch',
    'att_race', 'att_att', 'att_prim', 'att_anch',
    'prim_race', 'prim_att', 'prim_prim', 'prim_anch',
    'anch_race', 'anch_att', 'anch_prim', 'anch_anch',
]
CANONICAL_MAP = list(range(17))
CANONICAL_MAP[11] = 12
CANONICAL_MAP[15] = 12
CANONICAL_MAP[16] = 12

SHARED_PAIRS = {
    ('priming', 'priming'), ('priming', 'anchoring'),
    ('anchoring', 'priming'), ('anchoring', 'anchoring'),
}
SHARED_CANONICAL = ('priming', 'anchoring')

ALL_CONTACT_PAIRS = [
    (p, o) for p in ['racing', 'attacking', 'priming', 'anchoring']
    for o in ['racing', 'attacking', 'priming', 'anchoring']
]
CANONICAL_CONTACT_PAIRS = []
_seen = False
for pair in ALL_CONTACT_PAIRS:
    if pair in SHARED_PAIRS:
        if not _seen:
            CANONICAL_CONTACT_PAIRS.append(SHARED_CANONICAL)
            _seen = True
    else:
        CANONICAL_CONTACT_PAIRS.append(pair)

GP_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}


def abbrev(name):
    return {'racing': 'race', 'attacking': 'att', 'priming': 'prim', 'anchoring': 'anch'}[name]


def pair_name(p, o):
    return f'{abbrev(p)}_{abbrev(o)}'


def flip_boards_numpy(boards):
    flipped = np.zeros_like(boards)
    flipped[:, 0] = boards[:, 25]
    flipped[:, 25] = boards[:, 0]
    for i in range(1, 25):
        flipped[:, i] = -boards[:, 25 - i]
    return flipped


def load_bm_data():
    bm_data = {}
    for bm_name in ['racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        with open(bm_path) as f:
            lines = [l for l in f if l.startswith('m ')]
        boards = np.array(
            [board_from_gnubg_position_string(l.split()[1]) for l in lines],
            dtype=np.int32)
        flipped = flip_boards_numpy(boards)
        opp_gps = bgbot_cpp.classify_game_plans_batch(flipped)
        opp_indices = {}
        for gp_id in [1, 2, 3, 4]:
            opp_indices[GP_NAMES[gp_id]] = list(np.where(opp_gps == gp_id)[0])
        bm_data[bm_name] = {'lines': lines, 'opp_indices': opp_indices}
    return bm_data


def _add_scenario(ss, line):
    bits = line.split()
    start_board = board_from_gnubg_position_string(bits[1])
    die1, die2 = int(bits[2]), int(bits[3])
    ranked_boards, ranked_errors = [], []
    i = 4
    while i < len(bits):
        ranked_boards.append(board_from_gnubg_position_string(bits[i]))
        ranked_errors.append(float(bits[i + 1]) if i + 1 < len(bits) else 0.0)
        i += 2
    ss.add(start_board, die1, die2, ranked_boards, ranked_errors)


def build_pair_benchmark(p, o, bm_data):
    ss = bgbot_cpp.ScenarioSet()
    if (p, o) == SHARED_CANONICAL:
        for sp, so in SHARED_PAIRS:
            indices = bm_data[sp]['opp_indices'].get(so, [])
            lines = bm_data[sp]['lines']
            for idx in indices:
                _add_scenario(ss, lines[idx])
    else:
        indices = bm_data[p]['opp_indices'].get(o, [])
        lines = bm_data[p]['lines']
        for idx in indices:
            _add_scenario(ss, lines[idx])
    return ss


def log_section(title):
    print(f'\n{"="*70}', flush=True)
    print(f'  {title}', flush=True)
    print(f'{"="*70}', flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    global CHECKPOINT_PATH
    t_start = time.time()
    bgbot_cpp.init_escape_tables()

    w9 = WeightConfigPair.from_model('stage9')
    w9.validate()

    # Set up checkpoint path before any work
    out_dir = os.path.join(project_dir, 'experiments', 'stage9')
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    CHECKPOINT_PATH = os.path.join(out_dir, f's9_benchmarks_{timestamp}_checkpoint.json')

    log_section('STAGE 9 BENCHMARKS  (32 threads where safe)')
    print(f'  19-NN paths: {len(w9.paths)}, hiddens: {w9.hiddens[0]} + {w9.hiddens[1]}x{len(w9.hiddens)-1}')
    print(f'  Models dir: {MODELS_DIR}')

    results = {}

    # =====================================================================
    # 1. Standard 1-ply benchmarks
    # =====================================================================
    log_section('1. STANDARD 1-PLY BENCHMARKS')
    standard = {}
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring',
                     'contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_stage9(scenarios, w9.paths, w9.hiddens, N_THREADS)
        elapsed = time.time() - t0
        standard[bm_type] = result.score()
        print(f'  {bm_type:10s}: {result.score():8.2f}  ({result.count} scenarios, {elapsed:.1f}s)', flush=True)

    # vs PubEval via 1-ply MultiPlyStrategy wrapper
    print(f'\n  vs PubEval (10k games):', flush=True)
    t0 = time.time()
    pe_wrap = bgbot_cpp.create_multipy_stage9(
        w9.paths, w9.hiddens, n_plies=1, parallel_evaluate=False)
    pe_stats = bgbot_cpp.play_games_multipy_vs_pubeval(pe_wrap, n_games=10000, seed=42,
                                                      n_threads=N_THREADS)
    standard['vs_pubeval'] = pe_stats.avg_ppg()
    print(f'    PPG: {pe_stats.avg_ppg():+.3f}  ({pe_stats.n_games} games in {time.time()-t0:.1f}s)',
          flush=True)
    del pe_wrap
    gc.collect()

    results['standard'] = standard
    save_checkpoint(results)

    # =====================================================================
    # 2. Pair-filtered: S9 vs S5 vs S8
    # =====================================================================
    log_section('2. PAIR-FILTERED BENCHMARKS: S9 vs S5 vs S8')
    bm_data = load_bm_data()
    w5 = WeightConfig.from_model('stage5')
    w5.validate()
    w8 = WeightConfigPair.from_model('stage8')
    w8.validate()

    print(f"  {'NN':16s} {'Count':>7s} {'S9 ER':>7s} {'S5 ER':>7s} {'S8 ER':>7s} "
          f"{'S9-S5':>7s} {'S9-S8':>7s}", flush=True)
    print(f"  {'-'*16} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}", flush=True)

    pair_results = {}
    total_s9_w = total_s5_w = total_s8_w = 0.0
    total_freq = 0

    for p, o in CANONICAL_CONTACT_PAIRS:
        name = pair_name(p, o)
        pair_bm = build_pair_benchmark(p, o, bm_data)
        if pair_bm.size() == 0:
            continue
        r9 = bgbot_cpp.score_benchmarks_stage9(pair_bm, w9.paths, w9.hiddens, N_THREADS)
        s9_er = r9.score()
        r5 = bgbot_cpp.score_benchmarks_5nn(pair_bm, *w5.weight_args)
        s5_er = r5.score()
        r8 = bgbot_cpp.score_benchmarks_pair(pair_bm, w8.paths, w8.hiddens)
        s8_er = r8.score()

        freq = pair_bm.size()
        total_freq += freq
        total_s9_w += s9_er * freq
        total_s5_w += s5_er * freq
        total_s8_w += s8_er * freq

        print(f"  {name:16s} {freq:7d} {s9_er:7.2f} {s5_er:7.2f} {s8_er:7.2f} "
              f"{s9_er-s5_er:+7.2f} {s9_er-s8_er:+7.2f}", flush=True)
        pair_results[name] = {'count': freq, 's9_er': s9_er, 's5_er': s5_er, 's8_er': s8_er}

    avg_s9 = total_s9_w / total_freq
    avg_s5 = total_s5_w / total_freq
    avg_s8 = total_s8_w / total_freq
    print(f"  {'Weighted avg':16s} {total_freq:7d} {avg_s9:7.2f} {avg_s5:7.2f} {avg_s8:7.2f} "
          f"{avg_s9-avg_s5:+7.2f} {avg_s9-avg_s8:+7.2f}", flush=True)
    pair_results['_weighted_avg'] = {
        'count': total_freq, 's9_er': avg_s9, 's5_er': avg_s5, 's8_er': avg_s8,
    }
    results['pair_filtered'] = pair_results
    save_checkpoint(results)

    # =====================================================================
    # 3. Multi-ply contact benchmarks
    # =====================================================================
    log_section(f'3. MULTI-PLY CONTACT BENCHMARKS  (n_threads={N_THREADS})')

    contact_path = os.path.join(DATA_DIR, 'contact.bm')
    full_scenarios = load_benchmark_file(contact_path)
    n_total = full_scenarios.size()
    print(f'  contact.bm: {n_total} scenarios', flush=True)

    multipy_results = {}
    results['multipy_contact'] = multipy_results

    # 1-ply
    t0 = time.time()
    r = bgbot_cpp.score_benchmarks_stage9(full_scenarios, w9.paths, w9.hiddens, N_THREADS)
    multipy_results['1-ply'] = {'er': r.score(), 'time': time.time() - t0, 'n': n_total}
    print(f"  1-ply: {r.score():.2f}  ({time.time()-t0:.1f}s)", flush=True)
    save_checkpoint(results)

    # 2-ply
    t0 = time.time()
    multipy = bgbot_cpp.create_multipy_stage9(
        w9.paths, w9.hiddens, n_plies=2,
        parallel_evaluate=True, parallel_threads=N_THREADS_MULTIPY)
    r = bgbot_cpp.score_benchmarks_multipy(full_scenarios, multipy, 1)
    multipy_results['2-ply'] = {'er': r.score(), 'time': time.time() - t0, 'n': n_total}
    print(f"  2-ply: {r.score():.2f}  ({time.time()-t0:.1f}s)", flush=True)
    del multipy
    gc.collect()
    save_checkpoint(results)

    # 3-ply
    t0 = time.time()
    multipy = bgbot_cpp.create_multipy_stage9(
        w9.paths, w9.hiddens, n_plies=3,
        parallel_evaluate=True, parallel_threads=N_THREADS_MULTIPY)
    r = bgbot_cpp.score_benchmarks_multipy(full_scenarios, multipy, 1)
    multipy_results['3-ply'] = {'er': r.score(), 'time': time.time() - t0, 'n': n_total}
    print(f"  3-ply: {r.score():.2f}  ({time.time()-t0:.1f}s)", flush=True)
    del multipy
    gc.collect()
    save_checkpoint(results)

    # 4-ply: subsample (S8 used n_total/5000)
    step = max(1, n_total // 5000)
    sub_scenarios = load_benchmark_file(contact_path, step=step)
    n_sub = sub_scenarios.size()
    t0 = time.time()
    multipy = bgbot_cpp.create_multipy_stage9(
        w9.paths, w9.hiddens, n_plies=4,
        parallel_evaluate=True, parallel_threads=N_THREADS_MULTIPY)
    r = bgbot_cpp.score_benchmarks_multipy(sub_scenarios, multipy, 1)
    multipy_results['4-ply'] = {'er': r.score(), 'time': time.time() - t0,
                                  'n': n_sub, 'step': step}
    print(f"  4-ply subsample (step={step}, {n_sub}/{n_total}): "
          f"{r.score():.2f}  ({time.time()-t0:.1f}s)", flush=True)
    del multipy
    gc.collect()
    save_checkpoint(results)

    # =====================================================================
    # 4. Top-100 worst 1-ply scenarios benchmark
    # =====================================================================
    log_section('4. TOP-100 WORST-1-PLY BENCHMARK')

    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')
    scenarios_contact = load_benchmark_file(contact_file)
    scenarios_crashed = load_benchmark_file(crashed_file)
    n_contact = scenarios_contact.size()
    n_crashed = scenarios_crashed.size()
    total = n_contact + n_crashed
    print(f'  contact: {n_contact}, crashed: {n_crashed}, total: {total}', flush=True)

    # Per-scenario 1-ply scoring: no native _stage9 per-scenario binding, so we
    # replicate the C++ score_slice_per_scenario logic in Python using a 1-ply
    # MultiPlyStrategy wrapper. ~30s for 207k scenarios in benchmarking.
    from bgsage.board import flip_board

    strat_1ply = bgbot_cpp.create_multipy_stage9(
        w9.paths, w9.hiddens, n_plies=1, parallel_evaluate=False)

    def per_scenario_errors(bm_file):
        with open(bm_file) as f:
            lines = [l for l in f if l.startswith('m ')]
        errs = np.zeros(len(lines), dtype=np.float64)
        for s, line in enumerate(lines):
            bits = line.split()
            start = board_from_gnubg_position_string(bits[1])
            d1, d2 = int(bits[2]), int(bits[3])
            ranked_boards = []
            ranked_errs = []
            i = 4
            while i < len(bits):
                ranked_boards.append(tuple(board_from_gnubg_position_string(bits[i])))
                ranked_errs.append(float(bits[i + 1]) if i + 1 < len(bits) else 0.0)
                i += 2
            candidates = bgbot_cpp.possible_moves(start, d1, d2)
            if not candidates:
                continue
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                idx = strat_1ply.best_move_index(candidates, start)
                chosen = candidates[idx]
            chosen_flipped = tuple(flip_board(chosen))
            err = 0.0
            found = False
            for k, rb in enumerate(ranked_boards):
                if rb == chosen_flipped:
                    err = 0.0 if k == 0 else ranked_errs[k]
                    found = True
                    break
            if not found and len(ranked_errs) > 1:
                err = ranked_errs[-1]
            errs[s] = err
        return errs

    print('  Computing per-scenario 1-ply errors via multipy(1) wrapper...', flush=True)
    t0_psc = time.time()
    errors_contact = per_scenario_errors(contact_file)
    print(f'    contact: {len(errors_contact)} scored in {time.time()-t0_psc:.1f}s', flush=True)
    t0b = time.time()
    errors_crashed = per_scenario_errors(crashed_file)
    print(f'    crashed: {len(errors_crashed)} scored in {time.time()-t0b:.1f}s', flush=True)
    del strat_1ply
    gc.collect()

    # Combine and find top-100 worst
    all_errors = [(e, 'contact', i) for i, e in enumerate(errors_contact)] + \
                 [(e, 'crashed', i) for i, e in enumerate(errors_crashed)]
    all_errors.sort(key=lambda x: -x[0])
    overall_er = sum(e[0] for e in all_errors) / total * 1000
    print(f'  Overall 1-ply ER (recomputed): {overall_er:.2f}', flush=True)

    top_n = 100
    top_errors = all_errors[:top_n]
    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])
    print(f'  Top {top_n}: contact={len(contact_indices)}, crashed={len(crashed_indices)}', flush=True)

    top_contact_ss = (load_benchmark_scenarios_by_indices(contact_file, contact_indices)
                      if contact_indices else bgbot_cpp.ScenarioSet())
    top_crashed_ss = (load_benchmark_scenarios_by_indices(crashed_file, crashed_indices)
                      if crashed_indices else bgbot_cpp.ScenarioSet())

    top100_results = []

    def score_subset(score_fn):
        t0a = time.perf_counter()
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
        elapsed = time.perf_counter() - t0a
        mean_err = (total_err / total_count * 1000) if total_count > 0 else 0
        return mean_err, elapsed

    print()
    print(f"  {'Strategy':<50} {'ER':>8}  {'Time':>9}", flush=True)
    print(f"  {'-'*50} {'-'*8}  {'-'*9}", flush=True)

    # 1-ply
    er, t = score_subset(
        lambda ss: bgbot_cpp.score_benchmarks_stage9(ss, w9.paths, w9.hiddens, N_THREADS))
    print(f"  {'1-ply':<50} {er:>8.2f}  {t:>9.1f}s", flush=True)
    top100_results.append(('1-ply', er, t))

    # N-ply: 4 threads to avoid pair-strategy segfault at 16+ threads with 3-ply
    for ply in [2, 3, 4]:
        multipy = bgbot_cpp.create_multipy_stage9(
            w9.paths, w9.hiddens, n_plies=ply,
            parallel_evaluate=True, parallel_threads=N_THREADS_PAIR_HIGH_PLY)
        er, t = score_subset(
            lambda ss: (multipy.clear_cache(),
                        bgbot_cpp.score_benchmarks_multipy(ss, multipy, 1))[1])
        print(f"  {f'{ply}-ply':<50} {er:>8.2f}  {t:>9.1f}s", flush=True)
        top100_results.append((f'{ply}-ply', er, t))
        del multipy
        gc.collect()
        results['top100'] = [(n, e, t) for n, e, t in top100_results]
        save_checkpoint(results)

    # Rollout levels in subprocesses (same as S8)
    rollout_configs = [
        ('XG Roller (42t, trunc=5, dp=1)',
         dict(n_trials=42, truncation_depth=5, decision_ply=1, n_threads=N_THREADS_PAIR_HIGH_PLY)),
        ('XG Roller+ (360t, trunc=7, dp=2, late=1@2)',
         dict(n_trials=360, truncation_depth=7, decision_ply=2,
              n_threads=N_THREADS_PAIR_HIGH_PLY,
              late_ply=1, late_threshold=2)),
        ('XG Roller++ (360t, trunc=5, dp=3, late=2@2)',
         dict(n_trials=360, truncation_depth=5, decision_ply=3,
              n_threads=N_THREADS_PAIR_HIGH_PLY,
              late_ply=2, late_threshold=2)),
    ]
    for name, kwargs in rollout_configs:
        er, t = _run_rollout_subprocess(name, kwargs, w9.paths, w9.hiddens,
                                         contact_indices, crashed_indices,
                                         build_dirs, project_dir, bgsage_dir)
        if er is not None:
            print(f"  {name:<50} {er:>8.2f}  {t:>9.1f}s", flush=True)
            top100_results.append((name, er, t))
        else:
            print(f"  {name:<50} CRASHED", flush=True)
            top100_results.append((name, float('nan'), 0))
        results['top100'] = [(n, e, t) for n, e, t in top100_results]
        save_checkpoint(results)

    # =====================================================================
    # Save final results
    # =====================================================================
    final_path = os.path.join(out_dir, f's9_benchmarks_{timestamp}_final.json')
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nFinal results saved: {final_path}', flush=True)

    total_elapsed = time.time() - t_start
    print(f'\nTotal benchmark time: {total_elapsed/60:.1f} min', flush=True)


def _run_rollout_subprocess(level_name, create_kwargs, weight_paths, hidden_sizes,
                              contact_indices, crashed_indices, build_dirs,
                              project_dir, bgsage_dir):
    """Run a single rollout level in a subprocess (stage9 strategy)."""
    code = f'''
import os, sys, json, time
if sys.platform == 'win32':
    for d in {repr([os.path.abspath(d) for d in build_dirs if os.path.isdir(d)])}:
        os.add_dll_directory(d)
    cuda_x64 = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
for d in {repr([os.path.abspath(d) for d in reversed(build_dirs) if os.path.isdir(d)])}:
    sys.path.insert(0, d)
sys.path.insert(0, {repr(os.path.join(bgsage_dir, 'python'))})

import bgbot_cpp
from bgsage.data import load_benchmark_scenarios_by_indices
bgbot_cpp.init_escape_tables()

weight_paths = {repr(weight_paths)}
hidden_sizes = {repr(hidden_sizes)}

DATA_DIR = {repr(os.path.join(bgsage_dir, 'data'))}
contact_file = os.path.join(DATA_DIR, 'contact.bm')
crashed_file = os.path.join(DATA_DIR, 'crashed.bm')
ci = {repr(contact_indices)}
ki = {repr(crashed_indices)}

tc = load_benchmark_scenarios_by_indices(contact_file, ci) if ci else bgbot_cpp.ScenarioSet()
tk = load_benchmark_scenarios_by_indices(crashed_file, ki) if ki else bgbot_cpp.ScenarioSet()

strat = bgbot_cpp.create_rollout_stage9(weight_paths, hidden_sizes, **{repr(create_kwargs)})

t0 = time.perf_counter()
total_err = 0.0
total_count = 0
for label, indices, bm_file in [('contact', ci, contact_file), ('crashed', ki, crashed_file)]:
    if not indices:
        continue
    for i, idx in enumerate(indices):
        ss = load_benchmark_scenarios_by_indices(bm_file, [idx])
        r = bgbot_cpp.score_benchmarks_rollout(ss, strat, 1)
        total_err += r.total_error
        total_count += r.count
        strat.clear_internal_caches()
elapsed = time.perf_counter() - t0
er = total_err / total_count * 1000 if total_count > 0 else 0
print(json.dumps({{"er": er, "elapsed": elapsed}}))
'''
    try:
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, timeout=3600,
            cwd=project_dir
        )
        if result.returncode != 0:
            print(f"  {level_name}: CRASHED (exit {result.returncode})", flush=True)
            if result.stderr:
                print(f"    stderr: {result.stderr[:300]}", flush=True)
            return None, None
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return data['er'], data['elapsed']
    except Exception as e:
        print(f"  {level_name}: ERROR {e}", flush=True)
        return None, None


if __name__ == '__main__':
    main()
