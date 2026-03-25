"""
Stage 8 Training: 17-NN pair strategy (100h purerace, 400h contact).

Same architecture as Stage 7 (17-NN pair strategy with NN sharing) but with 400h
contact NNs (matching Stage 5 hidden size) instead of S7's 300h. PureRace weights
are copied from Stage 7 (same 100h architecture).

Full pipeline:
  1. Copy PureRace weights from S7
  2. TD: 300k@0.1 + 1.5M@0.02 (same schedule as S7)
  3. GPW scan: For each of 13 canonical contact NNs, try gpw in [2, 5, 7, 10, 12]
     with SL phases 1-2 (100ep@20 + 200ep@10). Pick gpw minimizing pair-filtered ER.
  4. SL phases 3-4: 200ep@3.1 + 500ep@1.0 with optimal gpw, starting from scan weights.
  5. Copy canonical weights to aliases
  6. Score S5 pair-filtered comparison
  7. Run benchmarks (1-ply contact/race, 2-4 ply contact, top-100)

Progress updates print every 10 minutes with estimated time remaining.

Usage:
    python bgsage/scripts/run_s8_training.py                    # Full pipeline
    python bgsage/scripts/run_s8_training.py --sl-only          # Skip TD
    python bgsage/scripts/run_s8_training.py --phase34-only     # Skip TD+scan, phases 3-4 only
    python bgsage/scripts/run_s8_training.py --score-only       # Score existing weights
    python bgsage/scripts/run_s8_training.py --benchmark-only   # Multi-ply + top-100 benchmarks
    python bgsage/scripts/run_s8_training.py --nn race_race     # Train specific NN(s)
"""

import os
import sys
import json
import time
import shutil
import threading
import subprocess
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup (handles both normal repo and worktree layout)
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
bgsage_dir = os.path.dirname(script_dir)

# Resolve main project root: handles worktree layout
_parts = os.path.normpath(bgsage_dir).replace('\\', '/').split('/')
if '.claude' in _parts:
    _idx = _parts.index('.claude')
    bgsage_dir = '/'.join(_parts[:_idx])
project_dir = os.path.dirname(bgsage_dir)

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
from bgsage.data import (load_benchmark_file, load_gnubg_training_data,
                          load_benchmark_scenarios_by_indices,
                          board_from_gnubg_position_string)
from bgsage.weights import WeightConfig

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')

# ---------------------------------------------------------------------------
# Stage 8 configuration
# ---------------------------------------------------------------------------
N_INPUTS = 244
N_HIDDEN = 400               # 400h contact NNs (same as S5)
N_HIDDEN_PURERACE = 100      # 100h purerace (same as S7)
N_INPUTS_PURERACE = 196

MODEL_PREFIX = 'sl_s8'
TD_MODEL_NAME = 'td_s8'

# 17 pair names (index 0 = purerace, 1-16 = contact pairs)
PAIR_NAMES = [
    'purerace',
    'race_race', 'race_att', 'race_prim', 'race_anch',
    'att_race', 'att_att', 'att_prim', 'att_anch',
    'prim_race', 'prim_att', 'prim_prim', 'prim_anch',
    'anch_race', 'anch_att', 'anch_prim', 'anch_anch',
]

PLAYER_PLAN_IDS = {}
OPP_PLAN_IDS = {}
for i, name in enumerate(PAIR_NAMES[1:], start=1):
    p, o = name.split('_')
    plan_map = {'race': 1, 'att': 2, 'prim': 3, 'anch': 4}
    PLAYER_PLAN_IDS[name] = plan_map[p]
    OPP_PLAN_IDS[name] = plan_map[o]

GP_IDS = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}
GP_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}
PLAN_BM_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}

# NN sharing: same as S7
CANONICAL_MAP = list(range(17))
CANONICAL_MAP[11] = 12   # prim_prim -> prim_anch
CANONICAL_MAP[15] = 12   # anch_prim -> prim_anch
CANONICAL_MAP[16] = 12   # anch_anch -> prim_anch

CANONICAL_CONTACT_NAMES = [
    name for i, name in enumerate(PAIR_NAMES[1:], start=1)
    if CANONICAL_MAP[i] == i
]

HIDDEN_SIZES = [N_HIDDEN_PURERACE] + [N_HIDDEN] * 16

# SL schedule
CONTACT_PHASES = [(100, 20.0), (200, 10.0), (200, 3.1), (500, 1.0)]
SCAN_PHASES = [(100, 20.0), (200, 10.0)]       # Phases 1+2 for GPW scan
REMAINING_PHASES = [(200, 3.1), (500, 1.0)]     # Phases 3+4 after scan
PURERACE_PHASES = [(200, 20.0), (200, 6.3), (1000, 2.0)]

# GPW candidates
GPW_CANDIDATES = [2.0, 5.0, 7.0, 10.0, 12.0]

# Divergence threshold: if best_score exceeds this after a phase, skip remaining phases
DIVERGENCE_THRESHOLD = 40.0

# Shared pairs (same as S7)
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


# ---------------------------------------------------------------------------
# Progress tracking with 10-minute updates
# ---------------------------------------------------------------------------
class ProgressTracker:
    """Background thread that prints progress every 10 minutes."""

    # Estimated relative durations (will be calibrated after TD Phase 1)
    PHASE_WEIGHTS = {
        'td_phase1': 1.0,     # baseline
        'td_phase2': 5.0,     # 5x more games
        'gpw_scan': 4.0,      # 13 NNs × 5 gpw × 300 epochs
        'sl_phase34': 3.0,    # 13 NNs × 700 epochs
        'scoring': 0.3,       # fast
        'benchmarks': 1.0,    # multi-ply can be slow
    }

    def __init__(self, interval=600):
        self.interval = interval
        self.start_time = time.time()
        self.phase = 'initializing'
        self.detail = ''
        self.completed = []  # list of (phase_name, duration_seconds)
        self._lock = threading.Lock()
        self._timer = None
        self._stopped = False

    def set_phase(self, name, detail=''):
        with self._lock:
            self.phase = name
            self.detail = detail
            print(f'\n>>> Phase: {name} {detail}', flush=True)

    def update_detail(self, detail):
        with self._lock:
            self.detail = detail

    def complete_phase(self, name, duration):
        with self._lock:
            self.completed.append((name, duration))

    def start(self):
        self._schedule()

    def stop(self):
        self._stopped = True
        if self._timer:
            self._timer.cancel()

    def _schedule(self):
        if self._stopped:
            return
        self._timer = threading.Timer(self.interval, self._tick)
        self._timer.daemon = True
        self._timer.start()

    def _tick(self):
        if self._stopped:
            return
        self._print_status()
        self._schedule()

    def _print_status(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            elapsed_h = elapsed / 3600

            # Estimate remaining
            completed_weighted = 0
            for name, dur in self.completed:
                completed_weighted += self.PHASE_WEIGHTS.get(name, 1.0)

            total_weighted = sum(self.PHASE_WEIGHTS.values())
            current_weight = self.PHASE_WEIGHTS.get(self.phase, 1.0)

            if self.completed:
                # Calibrate: actual_time / weighted_units = time_per_unit
                completed_time = sum(d for _, d in self.completed)
                if completed_weighted > 0:
                    time_per_unit = completed_time / completed_weighted
                    remaining_weighted = total_weighted - completed_weighted
                    est_remaining = remaining_weighted * time_per_unit
                    est_total = elapsed + est_remaining
                    est_remaining_h = est_remaining / 3600
                else:
                    est_remaining_h = None
            else:
                est_remaining_h = None

            print(f'\n{"="*65}', flush=True)
            print(f'  PROGRESS UPDATE  ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
            print(f'  Elapsed: {elapsed_h:.1f}h')
            print(f'  Current: {self.phase}')
            if self.detail:
                print(f'  Detail:  {self.detail}')
            if self.completed:
                print(f'  Completed:')
                for name, dur in self.completed:
                    print(f'    {name:20s}: {dur/3600:.1f}h')
            if est_remaining_h is not None:
                print(f'  Est. remaining: {est_remaining_h:.1f}h')
            print(f'{"="*65}\n', flush=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
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


def classify_pairs(boards):
    """Classify all boards by (player, opponent) pair.
    Returns pair_ids array (-1=purerace, 0-15=contact pair index)."""
    print('Classifying game plan pairs...')
    t0 = time.time()
    player_gps = bgbot_cpp.classify_game_plans_batch(boards)
    flipped = flip_boards_numpy(boards)
    opp_gps = bgbot_cpp.classify_game_plans_batch(flipped)

    pair_ids = np.full(len(boards), -1, dtype=np.int32)
    contact_mask = player_gps > 0
    opp_adjusted = opp_gps.copy()
    opp_adjusted[opp_adjusted == 0] = 1
    raw_pair_ids = (player_gps[contact_mask] - 1) * 4 + (opp_adjusted[contact_mask] - 1)
    canonical_contact = np.array([CANONICAL_MAP[1 + pid] - 1 for pid in range(16)], dtype=np.int32)
    pair_ids[contact_mask] = canonical_contact[raw_pair_ids]

    n = len(boards)
    n_pr = int(np.sum(~contact_mask))
    print(f'  PureRace: {n_pr}/{n} ({100*n_pr/n:.1f}%)')
    for i, name in enumerate(PAIR_NAMES[1:]):
        cnt = int(np.sum(pair_ids == i))
        if cnt > 0:
            print(f'  {name:12s}: {cnt:7d}/{n} ({100*cnt/n:.1f}%)')
    print(f'  Classified in {time.time()-t0:.1f}s')
    return pair_ids


def build_sample_weights(pair_ids, target_pair_idx, gpw):
    """Build sample weights: gpw for matching pairs, 1.0 for rest."""
    weights = np.ones(len(pair_ids), dtype=np.float32)
    weights[pair_ids == target_pair_idx] = gpw
    return weights


# ---------------------------------------------------------------------------
# Pair-filtered benchmarks
# ---------------------------------------------------------------------------
def load_bm_data():
    """Parse all .bm files and classify opponent plans."""
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
        print(f'  {bm_name}.bm: {len(lines)} scenarios')
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
    """Build a ScenarioSet filtered to the (player, opponent) pair."""
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_training_data():
    print('Loading training data...')
    t0 = time.time()
    boards_c, targets_c = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    print(f'  contact: {len(boards_c)} positions ({time.time()-t0:.1f}s)')
    t0 = time.time()
    boards_r, targets_r = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    print(f'  crashed: {len(boards_r)} positions ({time.time()-t0:.1f}s)')
    boards = np.concatenate([boards_c, boards_r], axis=0)
    targets = np.concatenate([targets_c, targets_r], axis=0)
    print(f'  Total: {len(boards)} positions')
    return boards, targets


def load_benchmarks():
    benchmarks = {}
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if os.path.exists(bm_path):
            benchmarks[bm_type] = load_benchmark_file(bm_path, step=10)
    return benchmarks


# ---------------------------------------------------------------------------
# TD Training
# ---------------------------------------------------------------------------
def run_td_training(tracker):
    """Run TD self-play: 300k@0.1 + 1.5M@0.02 (same schedule as S7)."""

    print(f'\n{"="*65}')
    print(f'  TD SELF-PLAY TRAINING (Stage 8 — 17-NN Pair, 400h contact)')
    print(f'  PureRace: {N_HIDDEN_PURERACE}h, Contact: {N_HIDDEN}h')
    print(f'{"="*65}\n')

    # Load benchmark scenarios for TD progress tracking
    benchmark_sets = {}
    for bm_name in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if os.path.exists(bm_path):
            print(f'Loading {bm_name} benchmark (step=10)...')
            t0 = time.time()
            ss = load_benchmark_file(bm_path, step=10)
            print(f'  Loaded {len(ss)} scenarios in {time.time()-t0:.1f}s')
            benchmark_sets[bm_name] = ss

    benchmarks = [None] * 17
    benchmarks[0] = benchmark_sets.get('purerace')
    for plan_bm_name, idx in [('racing', 1), ('attacking', 5), ('anchoring', 13)]:
        if CANONICAL_MAP[idx] == idx:
            benchmarks[idx] = benchmark_sets.get(plan_bm_name)

    # Phase 1: 300k @ alpha=0.1
    tracker.set_phase('td_phase1', '300k games @ alpha=0.1')
    print(f'=== TD Phase 1: 300k @ alpha=0.1 ===', flush=True)

    t0 = time.time()
    result1 = bgbot_cpp.td_train_gameplan_pair(
        n_games=300000,
        alpha=0.1,
        hidden_sizes=HIDDEN_SIZES,
        eps=0.1,
        seed=42,
        benchmark_interval=10000,
        model_name=TD_MODEL_NAME,
        models_dir=MODELS_DIR,
        resume_paths=[],
        benchmarks=benchmarks,
    )
    td1_time = time.time() - t0
    print(f'Phase 1 done: {result1.games_played} games in {td1_time:.1f}s ({td1_time/3600:.1f}h)')
    tracker.complete_phase('td_phase1', td1_time)

    # Phase 2: 1.5M @ alpha=0.02 (resume from Phase 1)
    td_phase2_name = f'{TD_MODEL_NAME}_1800k'
    tracker.set_phase('td_phase2', '1.5M games @ alpha=0.02')
    print(f'\n=== TD Phase 2: 1.5M @ alpha=0.02 ===', flush=True)

    resume_paths = [
        os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_{name}.weights')
        for name in PAIR_NAMES
    ]

    t0 = time.time()
    result2 = bgbot_cpp.td_train_gameplan_pair(
        n_games=1500000,
        alpha=0.02,
        hidden_sizes=HIDDEN_SIZES,
        eps=0.1,
        seed=42,
        benchmark_interval=10000,
        model_name=td_phase2_name,
        models_dir=MODELS_DIR,
        resume_paths=resume_paths,
        benchmarks=benchmarks,
    )
    td2_time = time.time() - t0
    print(f'Phase 2 done: {result2.games_played} games in {td2_time:.1f}s ({td2_time/3600:.1f}h)')
    print(f'Total TD time: {(td1_time+td2_time)/3600:.1f}h')
    tracker.complete_phase('td_phase2', td2_time)

    # Copy prim_anch weights to aliases
    canonical_name = 'prim_anch'
    canonical_src = os.path.join(MODELS_DIR, f'{td_phase2_name}_{canonical_name}.weights')
    for alias_name in ['prim_prim', 'anch_prim', 'anch_anch']:
        alias_dst = os.path.join(MODELS_DIR, f'{td_phase2_name}_{alias_name}.weights')
        if os.path.exists(canonical_src):
            shutil.copy2(canonical_src, alias_dst)
            print(f'  Copied TD weights: {canonical_name} -> {alias_name}')

    return td_phase2_name


# ---------------------------------------------------------------------------
# GPW Scan (Phases 1-2 with pair-filtered benchmarks)
# ---------------------------------------------------------------------------
def run_gpw_scan(boards, targets, pair_ids, bm_data, td_model_name, tracker,
                 nn_filter=None):
    """Run GPW scan for all canonical contact NNs.

    For each NN, trains phases 1+2 (300 epochs) at each gpw candidate,
    using pair-filtered benchmarks to select the best gpw.
    Returns dict mapping nn_name -> optimal_gpw.
    """
    tracker.set_phase('gpw_scan', f'{len(GPW_CANDIDATES)} candidates × {len(CANONICAL_CONTACT_NAMES)} NNs')

    scan_dir = os.path.join(MODELS_DIR, 's8_gpw_scan')
    os.makedirs(scan_dir, exist_ok=True)

    print(f'\n{"="*65}')
    print(f'  GPW SCAN (Stage 8): {GPW_CANDIDATES}')
    print(f'  Schedule: {" -> ".join(f"{e}ep@a={a}" for e, a in SCAN_PHASES)}')
    print(f'  Pair-filtered benchmarks for scoring')
    print(f'{"="*65}\n')

    nns_to_scan = CANONICAL_CONTACT_NAMES
    if nn_filter:
        nns_to_scan = [n for n in nns_to_scan if n in nn_filter]

    optimal_gpw = {}
    t0_total = time.time()

    for nn_idx, nn_name in enumerate(nns_to_scan):
        pair_idx = PAIR_NAMES.index(nn_name) - 1  # 0-based contact pair index

        # Determine (player, opponent) for pair-filtered benchmark
        p_abbr, o_abbr = nn_name.split('_')
        p_full = {'race': 'racing', 'att': 'attacking', 'prim': 'priming', 'anch': 'anchoring'}[p_abbr]
        o_full = {'race': 'racing', 'att': 'attacking', 'prim': 'priming', 'anch': 'anchoring'}[o_abbr]

        # Build pair-filtered benchmark
        pair_bm = build_pair_benchmark(p_full, o_full, bm_data)

        td_weights = os.path.join(MODELS_DIR, f'{td_model_name}_{nn_name}.weights')
        n_match = int(np.sum(pair_ids == pair_idx))

        tracker.update_detail(f'NN {nn_idx+1}/{len(nns_to_scan)}: {nn_name}')

        print(f'\n--- [{nn_idx+1}/{len(nns_to_scan)}] {nn_name.upper()} '
              f'(match={n_match}, pair_bm={pair_bm.size()}) ---')

        best_gpw = 2.0
        best_score = float('inf')
        results = []

        for gpw in GPW_CANDIDATES:
            sample_weights = build_sample_weights(pair_ids, pair_idx, gpw)
            save_path = os.path.join(scan_dir, f'{nn_name}_gpw{gpw:.1f}.weights')

            current_weights = td_weights
            phase_best_score = float('inf')
            diverged = False

            for phase_idx, (epochs, alpha) in enumerate(SCAN_PHASES):
                print_interval = max(1, epochs // 10)
                result = bgbot_cpp.cuda_supervised_train(
                    boards=boards,
                    targets=targets,
                    weights_path=current_weights,
                    n_hidden=N_HIDDEN,
                    n_inputs=N_INPUTS,
                    alpha=alpha,
                    epochs=epochs,
                    batch_size=128,
                    seed=42,
                    print_interval=print_interval,
                    save_path=save_path,
                    benchmark_scenarios=pair_bm,
                    sample_weights=sample_weights,
                    label=f'{nn_name} gpw={gpw}',
                )
                if result['best_score'] < phase_best_score:
                    phase_best_score = result['best_score']
                current_weights = save_path + '.best'

                # Early stop if diverged
                if result['best_score'] > DIVERGENCE_THRESHOLD:
                    print(f'  DIVERGED: {nn_name} gpw={gpw} score={result["best_score"]:.2f} '
                          f'> {DIVERGENCE_THRESHOLD} — skipping remaining phases')
                    diverged = True
                    break

            results.append((gpw, phase_best_score))
            if phase_best_score < best_score:
                best_score = phase_best_score
                best_gpw = gpw

        optimal_gpw[nn_name] = best_gpw
        print(f'\n  {nn_name} GPW scan results:')
        for gpw, score in results:
            marker = ' <-- BEST' if gpw == best_gpw else ''
            print(f'    gpw={gpw:5.1f}: pair ER={score:.2f}{marker}')

    elapsed = time.time() - t0_total
    tracker.complete_phase('gpw_scan', elapsed)

    print(f'\n{"="*65}')
    print(f'  GPW SCAN COMPLETE ({elapsed/3600:.1f}h)')
    print(f'{"="*65}')
    for name, gpw in optimal_gpw.items():
        print(f'  {name:20s}: {gpw}')

    # Save results
    results_path = os.path.join(scan_dir, 'optimal_gpw.json')
    with open(results_path, 'w') as f:
        json.dump(optimal_gpw, f, indent=2)
    print(f'\n  Saved to: {results_path}')

    return optimal_gpw


# ---------------------------------------------------------------------------
# SL Phases 3-4 (resume from GPW scan weights)
# ---------------------------------------------------------------------------
def run_sl_phase34(boards, targets, pair_ids, bm_data, optimal_gpw, tracker,
                   nn_filter=None):
    """Run SL phases 3+4 for each NN, starting from GPW scan's best weights."""

    tracker.set_phase('sl_phase34', 'Phases 3+4 for all contact NNs')
    scan_dir = os.path.join(MODELS_DIR, 's8_gpw_scan')

    phase_desc = ' -> '.join(f'{e}ep@a={a}' for e, a in REMAINING_PHASES)
    total_epochs = sum(e for e, _ in REMAINING_PHASES)

    print(f'\n{"="*65}')
    print(f'  SL PHASES 3-4 (Stage 8)')
    print(f'  Schedule: {phase_desc}')
    print(f'{"="*65}\n')

    nns_to_train = CANONICAL_CONTACT_NAMES
    if nn_filter:
        nns_to_train = [n for n in nns_to_train if n in nn_filter]

    all_results = {}
    t0_total = time.time()

    for nn_idx, nn_name in enumerate(nns_to_train):
        gpw = optimal_gpw.get(nn_name, 5.0)
        pair_idx = PAIR_NAMES.index(nn_name) - 1

        # Starting weights: optimal gpw .best from scan
        start_weights = os.path.join(scan_dir, f'{nn_name}_gpw{gpw:.1f}.weights.best')
        save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{nn_name}.weights')

        # Pair-filtered benchmark
        p_abbr, o_abbr = nn_name.split('_')
        p_full = {'race': 'racing', 'att': 'attacking', 'prim': 'priming', 'anch': 'anchoring'}[p_abbr]
        o_full = {'race': 'racing', 'att': 'attacking', 'prim': 'priming', 'anch': 'anchoring'}[o_abbr]
        pair_bm = build_pair_benchmark(p_full, o_full, bm_data)

        # Sample weights
        n_match = int(np.sum(pair_ids == pair_idx))
        sample_weights = None
        if gpw != 1.0:
            sample_weights = build_sample_weights(pair_ids, pair_idx, gpw)

        tracker.update_detail(f'NN {nn_idx+1}/{len(nns_to_train)}: {nn_name} (gpw={gpw})')

        print(f'{"="*65}')
        print(f'  [{nn_idx+1}/{len(nns_to_train)}] {nn_name.upper()} '
              f'({N_HIDDEN}h, gpw={gpw}, {total_epochs} epochs)')
        print(f'  Phases 3-4: {phase_desc}')
        print(f'  Resume from: {start_weights}')
        print(f'  Pair benchmark: {pair_bm.size()} scenarios')
        print(f'  Match: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
        print(f'{"="*65}')

        if not os.path.exists(start_weights):
            print(f'  ERROR: Starting weights not found: {start_weights}')
            continue

        current_weights = start_weights
        best_score = float('inf')
        best_epoch_total = 0
        total_time = 0.0
        epoch_offset = 300  # phases 1+2 done in scan

        for phase_idx, (epochs, alpha) in enumerate(REMAINING_PHASES):
            phase_num = phase_idx + 3
            print_interval = min(10, max(1, epochs // 20))

            result = bgbot_cpp.cuda_supervised_train(
                boards=boards,
                targets=targets,
                weights_path=current_weights,
                n_hidden=N_HIDDEN,
                n_inputs=N_INPUTS,
                alpha=alpha,
                epochs=epochs,
                batch_size=128,
                seed=42,
                print_interval=print_interval,
                save_path=save_path,
                benchmark_scenarios=pair_bm,
                sample_weights=sample_weights,
                label=nn_name,
            )

            phase_best = result['best_score']
            phase_best_epoch = result['best_epoch']
            phase_time = result['total_seconds']
            total_time += phase_time

            if phase_best < best_score:
                best_score = phase_best
                best_epoch_total = epoch_offset + phase_best_epoch

            print(f'  Phase {phase_num}/4: best={phase_best:.2f} '
                  f'(epoch {phase_best_epoch}), time={phase_time:.1f}s')

            # Early stop if diverged
            if phase_best > DIVERGENCE_THRESHOLD:
                print(f'  DIVERGED: {nn_name} score={phase_best:.2f} '
                      f'> {DIVERGENCE_THRESHOLD} — skipping remaining phases')
                break

            current_weights = save_path + '.best'
            epoch_offset += epochs

        all_results[nn_name] = {
            'best_score': best_score,
            'best_epoch': best_epoch_total,
            'total_time': total_time,
            'gpw': gpw,
        }
        print(f'\n  FINAL: {nn_name} -> pair ER={best_score:.2f} '
              f'(epoch {best_epoch_total}), gpw={gpw}, time={total_time:.0f}s\n')

    # Copy shared weights to aliases
    shared_name = pair_name(*SHARED_CANONICAL)
    shared_best = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{shared_name}.weights.best')
    if os.path.exists(shared_best):
        for sp, so in SHARED_PAIRS:
            alias = pair_name(sp, so)
            if alias != shared_name:
                dst = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{alias}.weights.best')
                shutil.copy2(shared_best, dst)
                print(f'  Copied {shared_name} -> {alias}')

    elapsed = time.time() - t0_total
    tracker.complete_phase('sl_phase34', elapsed)

    print(f'\n{"="*65}')
    print(f'  SL PHASES 3-4 COMPLETE ({elapsed/3600:.1f}h)')
    print(f'{"="*65}\n')
    for name, r in all_results.items():
        print(f'  {name:20s}: pair ER={r["best_score"]:.2f} (epoch {r["best_epoch"]}), '
              f'gpw={r["gpw"]}, time={r["total_time"]:.0f}s')

    return all_results


# ---------------------------------------------------------------------------
# S5 fallback: replace any S8 NN that's worse than S5 on pair-filtered benchmark
# ---------------------------------------------------------------------------
def apply_s5_fallback(bm_data, tracker):
    """For any S8 NN with worse pair-filtered ER than S5, copy S5 weights.

    The S5 model uses the player's game plan to select the NN, so for a pair
    (racing, attacking), S5 uses the Racing NN. We copy that into the S8 pair
    slot so that no pair-filtered benchmark regresses vs S5.
    """
    tracker.set_phase('s5_fallback', 'Replacing worse-than-S5 NNs')

    w5 = WeightConfig.from_model('stage5')
    try:
        w5.validate()
    except FileNotFoundError as e:
        print(f'  S5 weights not found: {e} — skipping fallback')
        return {}

    # Map player plan abbreviation to S5 weight file
    s5_plan_weights = {
        'race': w5.weight_paths['racing'],
        'att': w5.weight_paths['attacking'],
        'prim': w5.weight_paths['priming'],
        'anch': w5.weight_paths['anchoring'],
    }

    weight_paths_s8 = build_s8_weight_paths()
    replacements = {}

    print(f'\n{"="*65}')
    print(f'  S5 FALLBACK CHECK')
    print(f'{"="*65}\n')
    print(f'  {"NN":20s} {"S8 ER":>7s} {"S5 ER":>7s} {"Action":s}')
    print(f'  {"-"*20} {"-"*7} {"-"*7} {"-"*20}')

    for p, o in CANONICAL_CONTACT_PAIRS:
        name = pair_name(p, o)
        pair_bm = build_pair_benchmark(p, o, bm_data)
        if pair_bm.size() == 0:
            continue

        # Score S8
        r8 = bgbot_cpp.score_benchmarks_pair(pair_bm, weight_paths_s8, HIDDEN_SIZES)
        s8_er = r8.score()

        # Score S5
        r5 = bgbot_cpp.score_benchmarks_5nn(pair_bm, *w5.weight_args)
        s5_er = r5.score()

        if s8_er > s5_er:
            # S8 is worse — copy S5 weights
            player_plan = name.split('_')[0]
            s5_src = s5_plan_weights[player_plan]
            s8_dst = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{name}.weights.best')
            shutil.copy2(s5_src, s8_dst)
            replacements[name] = {'s8_er': s8_er, 's5_er': s5_er, 'source': f'S5 {player_plan}'}
            action = f'REPLACED with S5 {player_plan}'

            # If this is the shared canonical, also copy to aliases
            if (p, o) == SHARED_CANONICAL:
                for sp, so in SHARED_PAIRS:
                    alias = pair_name(sp, so)
                    if alias != name:
                        alias_dst = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{alias}.weights.best')
                        shutil.copy2(s5_src, alias_dst)
        else:
            action = 'kept (S8 better)'

        print(f'  {name:20s} {s8_er:7.2f} {s5_er:7.2f} {action}')

    if replacements:
        print(f'\n  Replaced {len(replacements)} NN(s) with S5 weights')
    else:
        print(f'\n  All S8 NNs beat S5 — no replacements needed')

    return replacements


# ---------------------------------------------------------------------------
# Scoring: S8 pair-filtered + S5 comparison
# ---------------------------------------------------------------------------
def build_s8_weight_paths():
    """Build the 17-element weight path and hidden size lists for S8."""
    weight_paths = []
    for i, name in enumerate(PAIR_NAMES):
        canonical_idx = CANONICAL_MAP[i]
        canonical_name = PAIR_NAMES[canonical_idx]
        weight_paths.append(os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{canonical_name}.weights.best'))
    return weight_paths


def score_pair_filtered(bm_data, tracker):
    """Score S8 and S5 on pair-filtered benchmarks for comparison."""

    tracker.set_phase('scoring', 'Pair-filtered S8 vs S5 comparison')

    weight_paths = build_s8_weight_paths()

    # Check weights exist
    for i, path in enumerate(weight_paths):
        if CANONICAL_MAP[i] == i and not os.path.exists(path):
            print(f'  Missing: {path}')
            return None

    # Load S5 weights
    w5 = WeightConfig.from_model('stage5')
    try:
        w5.validate()
    except FileNotFoundError as e:
        print(f'  S5 weights not found: {e}')
        w5 = None

    print(f'\n{"="*65}')
    print(f'  PAIR-FILTERED BENCHMARKS: S8 vs S5')
    print(f'{"="*65}\n')
    print(f'  {"NN":20s} {"Freq":>6s} {"S8 Pair ER":>11s} {"S5 Plan ER":>11s} {"Delta":>7s}')
    print(f'  {"-"*20} {"-"*6} {"-"*11} {"-"*11} {"-"*7}')

    results = {}
    total_s8_weighted = 0.0
    total_s5_weighted = 0.0
    total_freq = 0.0

    for p, o in CANONICAL_CONTACT_PAIRS:
        name = pair_name(p, o)
        pair_bm = build_pair_benchmark(p, o, bm_data)
        if pair_bm.size() == 0:
            continue

        # S8 pair scoring
        r8 = bgbot_cpp.score_benchmarks_pair(pair_bm, weight_paths, HIDDEN_SIZES)
        s8_er = r8.score()

        # S5 5-NN scoring on same subset
        s5_er = None
        if w5:
            r5 = bgbot_cpp.score_benchmarks_5nn(pair_bm, *w5.weight_args)
            s5_er = r5.score()

        # Frequency (use pair_bm size as proxy)
        freq = pair_bm.size()
        total_freq += freq
        total_s8_weighted += s8_er * freq
        if s5_er is not None:
            total_s5_weighted += s5_er * freq

        delta = f'{s8_er - s5_er:+.2f}' if s5_er is not None else '—'
        s5_str = f'{s5_er:.2f}' if s5_er is not None else '—'

        print(f'  {name:20s} {freq:6d} {s8_er:11.2f} {s5_str:>11s} {delta:>7s}')

        results[name] = {'s8_er': s8_er, 's5_er': s5_er, 'count': freq}

    if total_freq > 0:
        avg_s8 = total_s8_weighted / total_freq
        avg_s5 = total_s5_weighted / total_freq if w5 else None
        avg_delta = f'{avg_s8 - avg_s5:+.2f}' if avg_s5 is not None else '—'
        avg_s5_str = f'{avg_s5:.2f}' if avg_s5 is not None else '—'
        print(f'  {"Weighted avg":20s} {"":6s} {avg_s8:11.2f} {avg_s5_str:>11s} {avg_delta:>7s}')

    return results


def score_standard_benchmarks():
    """Run standard 1-ply benchmarks (contact, race, per-plan, vs PubEval)."""
    weight_paths = build_s8_weight_paths()

    print(f'\n{"="*65}')
    print(f'  STAGE 8 STANDARD BENCHMARKS (1-ply)')
    print(f'{"="*65}\n')

    scores = {}

    # Per-plan benchmarks
    print('--- Game Plan benchmarks ---')
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, HIDDEN_SIZES)
        t_score = time.time() - t0
        scores[bm_type] = result.score()
        print(f'  {bm_type:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # Old-style benchmarks
    print('\n--- Old-style benchmarks ---')
    for bm_name in ['contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, HIDDEN_SIZES)
        t_score = time.time() - t0
        scores[bm_name] = result.score()
        print(f'  {bm_name:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # vs PubEval
    print(f'\n=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_pair_vs_pubeval(weight_paths, HIDDEN_SIZES, n_games=10000, seed=42)
    t_games = time.time() - t0
    scores['vs_pubeval'] = stats.avg_ppg()
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')

    # Self-play
    print(f'\n=== Self-play (10k games) ===')
    t0 = time.time()
    ss = bgbot_cpp.play_games_pair_vs_self(weight_paths, HIDDEN_SIZES, n_games=10000, seed=42)
    t_sp = time.time() - t0
    total = ss.n_games
    singles = ss.p1_wins + ss.p2_wins
    gammons = ss.p1_gammons + ss.p2_gammons
    backgammons = ss.p1_backgammons + ss.p2_backgammons
    print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
          f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
          f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
          f'({total} games in {t_sp:.1f}s)')

    return scores


# ---------------------------------------------------------------------------
# Multi-ply benchmarks (2-ply, 3-ply, 4-ply contact ER)
# ---------------------------------------------------------------------------
def run_multiply_benchmarks(tracker):
    """Run contact benchmark at 2-ply, 3-ply, 4-ply."""
    tracker.set_phase('benchmarks', 'Multi-ply contact benchmarks')

    weight_paths = build_s8_weight_paths()

    print(f'\n{"="*65}')
    print(f'  MULTI-PLY CONTACT BENCHMARKS (Stage 8)')
    print(f'{"="*65}\n')

    contact_path = os.path.join(DATA_DIR, 'contact.bm')
    scenarios = load_benchmark_file(contact_path)
    n_total = scenarios.size()
    print(f'  contact.bm: {n_total} scenarios')

    results = {}

    # 1-ply (for reference)
    t0 = time.time()
    r = bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, HIDDEN_SIZES)
    t1 = time.time() - t0
    results['1-ply'] = r.score()
    print(f'  1-ply: {r.score():.2f}  ({t1:.1f}s)')

    # 2-ply, 3-ply, 4-ply
    for ply in [2, 3, 4]:
        # For 4-ply, subsample to avoid excessive time
        if ply == 4:
            step = max(1, n_total // 5000)
            sub_scenarios = load_benchmark_file(contact_path, step=step)
            print(f'  {ply}-ply (subsample {sub_scenarios.size()}/{n_total})...')
            bm = sub_scenarios
        else:
            bm = scenarios
            print(f'  {ply}-ply ({bm.size()} scenarios)...')

        t0 = time.time()
        multipy = bgbot_cpp.create_multipy_pair(
            weight_paths, HIDDEN_SIZES, n_plies=ply,
            parallel_evaluate=True, parallel_threads=16)
        r = bgbot_cpp.score_benchmarks_multipy(bm, multipy, 1)
        t_score = time.time() - t0
        results[f'{ply}-ply'] = r.score()
        print(f'  {ply}-ply: {r.score():.2f}  ({t_score:.1f}s)')

        del multipy
        import gc; gc.collect()

    return results


# ---------------------------------------------------------------------------
# Top-100 benchmark
# ---------------------------------------------------------------------------
def run_top100_benchmark(tracker):
    """Run top-100 worst 1-ply positions scored at 1-4 ply + rollout levels."""
    tracker.update_detail('Top-100 worst 1-ply benchmark')

    weight_paths = build_s8_weight_paths()

    print(f'\n{"="*65}')
    print(f'  TOP-100 WORST 1-PLY BENCHMARK (Stage 8)')
    print(f'{"="*65}\n')

    contact_file = os.path.join(DATA_DIR, 'contact.bm')
    crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

    scenarios_contact = load_benchmark_file(contact_file)
    scenarios_crashed = load_benchmark_file(crashed_file)
    n_contact = scenarios_contact.size()
    n_crashed = scenarios_crashed.size()
    total = n_contact + n_crashed
    print(f'  contact: {n_contact}, crashed: {n_crashed}, total: {total}')

    # Score all at 1-ply to find top-100 worst
    print('  Scoring all scenarios at 1-ply...')
    t0 = time.time()

    # Try pair per-scenario scoring; fall back to one-at-a-time
    try:
        errors_contact = bgbot_cpp.score_benchmarks_per_scenario_pair(
            scenarios_contact, weight_paths, HIDDEN_SIZES)
        errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_pair(
            scenarios_crashed, weight_paths, HIDDEN_SIZES)
    except AttributeError:
        # Fallback: score one scenario at a time
        print('    (per_scenario_pair not available, scoring individually...)')
        errors_contact = []
        for i in range(n_contact):
            ss = load_benchmark_scenarios_by_indices(contact_file, [i])
            r = bgbot_cpp.score_benchmarks_pair(ss, weight_paths, HIDDEN_SIZES)
            errors_contact.append(r.total_error / max(r.count, 1))
        errors_crashed = []
        for i in range(n_crashed):
            ss = load_benchmark_scenarios_by_indices(crashed_file, [i])
            r = bgbot_cpp.score_benchmarks_pair(ss, weight_paths, HIDDEN_SIZES)
            errors_crashed.append(r.total_error / max(r.count, 1))

    print(f'  Done in {time.time()-t0:.1f}s')

    # Combine and sort
    all_errors = []
    for i, err in enumerate(errors_contact):
        all_errors.append((err, 'contact', i))
    for i, err in enumerate(errors_crashed):
        all_errors.append((err, 'crashed', i))
    all_errors.sort(key=lambda x: -x[0])

    overall_er = sum(e[0] for e in all_errors) / total * 1000
    print(f'  Overall 1-ply ER: {overall_er:.2f}')

    top_n = 100
    top_errors = all_errors[:top_n]
    contact_indices = sorted([e[2] for e in top_errors if e[1] == 'contact'])
    crashed_indices = sorted([e[2] for e in top_errors if e[1] == 'crashed'])
    print(f'  Top {top_n} worst: contact={len(contact_indices)}, crashed={len(crashed_indices)}')

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

    print()
    print(f"  {'Strategy':<50} {'ER':>8}  {'Time':>9}")
    print(f"  {'-'*50} {'-'*8}  {'-'*9}")

    # 1-ply
    er, t = score_subset('1-ply',
        lambda ss: bgbot_cpp.score_benchmarks_pair(ss, weight_paths, HIDDEN_SIZES))
    print(f"  {'1-ply':<50} {er:>8.2f}  {t:>9.1f}s")
    results.append(('1-ply', er, t))

    # N-ply: 4 threads to avoid pair strategy segfault at 16 threads
    for ply in [2, 3, 4]:
        multipy = bgbot_cpp.create_multipy_pair(
            weight_paths, HIDDEN_SIZES, n_plies=ply,
            parallel_evaluate=True, parallel_threads=4)
        er, t = score_subset(f'{ply}-ply',
            lambda ss: (multipy.clear_cache(), bgbot_cpp.score_benchmarks_multipy(ss, multipy, 1))[1])
        print(f"  {f'{ply}-ply':<50} {er:>8.2f}  {t:>9.1f}s")
        results.append((f'{ply}-ply', er, t))
        del multipy
        import gc; gc.collect()

    # Rollout levels in subprocesses
    rollout_configs = [
        ('XG Roller (42t, trunc=5, dp=1)',
         dict(n_trials=42, truncation_depth=5, decision_ply=1, n_threads=4)),
        ('XG Roller+ (360t, trunc=7, dp=2, late=1@2)',
         dict(n_trials=360, truncation_depth=7, decision_ply=2, n_threads=4,
              late_ply=1, late_threshold=2)),
        ('XG Roller++ (360t, trunc=5, dp=3, late=2@2)',
         dict(n_trials=360, truncation_depth=5, decision_ply=3, n_threads=4,
              late_ply=2, late_threshold=2)),
    ]

    for name, kwargs in rollout_configs:
        er, t = _run_rollout_subprocess_pair(name, kwargs, weight_paths,
                                              contact_indices, crashed_indices)
        if er is not None:
            print(f"  {name:<50} {er:>8.2f}  {t:>9.1f}s")
            results.append((name, er, t))
        else:
            print(f"  {name:<50} CRASHED")
            results.append((name, float('nan'), 0))

    return results


def _run_rollout_subprocess_pair(level_name, create_kwargs, weight_paths,
                                  contact_indices, crashed_indices):
    """Run a single rollout level in a subprocess (pair strategy)."""
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
sys.path.insert(0, {repr(os.path.join(project_dir, 'bgsage', 'python'))})

import bgbot_cpp
from bgsage.data import load_benchmark_scenarios_by_indices
bgbot_cpp.init_escape_tables()

weight_paths = {repr(weight_paths)}
hidden_sizes = {repr(HIDDEN_SIZES)}

contact_file = {repr(os.path.join(DATA_DIR, 'contact.bm'))}
crashed_file = {repr(os.path.join(DATA_DIR, 'crashed.bm'))}
ci = {repr(contact_indices)}
ki = {repr(crashed_indices)}

tc = load_benchmark_scenarios_by_indices(contact_file, ci) if ci else bgbot_cpp.ScenarioSet()
tk = load_benchmark_scenarios_by_indices(crashed_file, ki) if ki else bgbot_cpp.ScenarioSet()

strat = bgbot_cpp.create_rollout_pair(weight_paths, hidden_sizes, **{repr(create_kwargs)})

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
            capture_output=True, text=True, timeout=1800,
            cwd=project_dir
        )
        if result.returncode != 0:
            print(f"  {level_name}: CRASHED (exit {result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[:300]}")
            return None, None
        data = json.loads(result.stdout.strip())
        return data['er'], data['elapsed']
    except Exception as e:
        print(f"  {level_name}: ERROR {e}")
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage 8 Training (17-NN Pair, 400h)')
    parser.add_argument('--sl-only', action='store_true',
                        help='Skip TD training, use existing TD weights')
    parser.add_argument('--phase34-only', action='store_true',
                        help='Skip TD+scan, run phases 3-4 only (needs s8_gpw_scan/)')
    parser.add_argument('--score-only', action='store_true',
                        help='Skip training, score existing weights')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Run multi-ply + top-100 benchmarks only')
    parser.add_argument('--nn', type=str, nargs='+', default=None,
                        help='Train specific NNs (e.g., --nn race_race att_att)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Initialize progress tracker
    tracker = ProgressTracker(interval=600)
    tracker.start()

    try:
        _run_pipeline(args, tracker)
    finally:
        tracker.stop()

    print(f'\n{"="*65}')
    total_elapsed = time.time() - tracker.start_time
    print(f'  STAGE 8 TRAINING COMPLETE ({total_elapsed/3600:.1f}h total)')
    print(f'{"="*65}\n')


def _run_pipeline(args, tracker):
    """Execute the S8 training pipeline."""

    # ---------------------------------------------------------------
    # Score-only / benchmark-only modes
    # ---------------------------------------------------------------
    if args.score_only:
        bgbot_cpp.init_escape_tables()
        print('Building pair-filtered benchmarks...')
        bm_data = load_bm_data()
        score_pair_filtered(bm_data, tracker)
        score_standard_benchmarks()
        return

    if args.benchmark_only:
        bgbot_cpp.init_escape_tables()
        multipy_results = run_multiply_benchmarks(tracker)
        top100_results = run_top100_benchmark(tracker)
        return

    # ---------------------------------------------------------------
    # Step 0: Copy PureRace weights from S7
    # ---------------------------------------------------------------
    s7_pr = os.path.join(MODELS_DIR, 'sl_s7_purerace.weights.best')
    s8_pr = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_purerace.weights.best')
    if os.path.exists(s7_pr):
        shutil.copy2(s7_pr, s8_pr)
        print(f'Copied PureRace weights: S7 -> S8')
    else:
        print(f'WARNING: S7 PureRace weights not found at {s7_pr}')

    # ---------------------------------------------------------------
    # Step 1: TD Training
    # ---------------------------------------------------------------
    td_model_name = f'{TD_MODEL_NAME}_1800k'

    if not args.sl_only and not args.phase34_only:
        td_model_name = run_td_training(tracker)
    else:
        print(f'Skipping TD training, using existing weights: {td_model_name}')
        # Verify TD weights exist
        for i, name in enumerate(PAIR_NAMES):
            if CANONICAL_MAP[i] != i:
                continue
            td_path = os.path.join(MODELS_DIR, f'{td_model_name}_{name}.weights')
            if not os.path.exists(td_path):
                print(f'  ERROR: TD weights not found: {td_path}')
                sys.exit(1)
        print('  All TD weights found.')

    # ---------------------------------------------------------------
    # Step 2: Load data + classify
    # ---------------------------------------------------------------
    if not bgbot_cpp.cuda_available():
        print('ERROR: CUDA not available for SL training')
        sys.exit(1)
    print('CUDA GPU detected')

    boards, targets = load_training_data()
    pair_ids = classify_pairs(boards)

    print('\nBuilding pair-filtered benchmarks...')
    bm_data = load_bm_data()

    # ---------------------------------------------------------------
    # Step 3: GPW Scan (Phases 1-2)
    # ---------------------------------------------------------------
    scan_dir = os.path.join(MODELS_DIR, 's8_gpw_scan')
    gpw_results_path = os.path.join(scan_dir, 'optimal_gpw.json')

    if args.phase34_only:
        # Load saved GPW results
        if os.path.exists(gpw_results_path):
            with open(gpw_results_path) as f:
                optimal_gpw = json.load(f)
            print(f'\nLoaded optimal GPW from {gpw_results_path}:')
            for name, gpw in optimal_gpw.items():
                print(f'  {name:20s}: {gpw}')
        else:
            print(f'ERROR: No saved GPW at {gpw_results_path}')
            sys.exit(1)
    else:
        optimal_gpw = run_gpw_scan(boards, targets, pair_ids, bm_data,
                                    td_model_name, tracker, nn_filter=args.nn)

    # ---------------------------------------------------------------
    # Step 4: SL Phases 3-4
    # ---------------------------------------------------------------
    sl_results = run_sl_phase34(boards, targets, pair_ids, bm_data,
                                 optimal_gpw, tracker, nn_filter=args.nn)

    # ---------------------------------------------------------------
    # Step 5: S5 Fallback — replace any NN worse than S5
    # ---------------------------------------------------------------
    s5_replacements = apply_s5_fallback(bm_data, tracker)

    # ---------------------------------------------------------------
    # Step 6: Score
    # ---------------------------------------------------------------
    tracker.set_phase('scoring', 'Final scoring')

    # Pair-filtered comparison with S5
    pair_results = score_pair_filtered(bm_data, tracker)

    # Standard benchmarks (1-ply)
    std_scores = score_standard_benchmarks()

    # ---------------------------------------------------------------
    # Step 6: Multi-ply + Top-100 benchmarks
    # ---------------------------------------------------------------
    tracker.set_phase('benchmarks', 'Multi-ply + top-100')

    multipy_results = run_multiply_benchmarks(tracker)
    top100_results = run_top100_benchmark(tracker)

    # ---------------------------------------------------------------
    # Save all results
    # ---------------------------------------------------------------
    results_dir = os.path.join(project_dir, 'experiments', 'stage8')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_results = {
        'optimal_gpw': optimal_gpw,
        's5_replacements': s5_replacements,
        'pair_filtered': pair_results,
        'standard_benchmarks': std_scores,
        'multipy_contact': multipy_results,
        'top100': [(n, e, t) for n, e, t in (top100_results or [])],
    }
    results_path = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    main()
