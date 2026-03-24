"""
Stage 7 Training: 17-NN ordered game plan pair model.

1 PureRace (100h, 196 inputs) + 16 ordered (player, opponent) contact pairs
(300h each, 244 inputs). Each contact NN specializes on positions where the
player has a specific game plan AND the opponent has a specific game plan.

Training pipeline:
  TD: 300k@0.1 + 1.5M@0.02 (from scratch)
  GPW scan: For each contact NN, try gpw in (1.5, 3, 5, 7, 10) with first 2
            SL phases (300 epochs). Pick gpw with lowest ER.
  SL: Full 4-phase training with optimal gpw per NN.
      Contact: 100ep@a=20 -> 200ep@a=10 -> 200ep@a=3.1 -> 500ep@a=1.0
      PureRace: 200ep@a=20 -> 200ep@a=6.3 -> 1000ep@a=2.0

After TD training, check outcome distribution:
  Expected: singles ~66%, gammons ~30%, backgammons ~4%
  If significantly off, TD training may need more games.

Usage:
    python scripts/run_stage7_training.py                  # Full pipeline
    python scripts/run_stage7_training.py --sl-only        # Skip TD
    python scripts/run_stage7_training.py --gpw-scan-only  # Only run GPW scan
    python scripts/run_stage7_training.py --skip-gpw-scan  # Skip scan, use default gpw
    python scripts/run_stage7_training.py --score-only     # Just score
    python scripts/run_stage7_training.py --nn race_att anch_prim  # Train specific NNs
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
bgsage_dir = os.path.dirname(script_dir)

# Resolve main project root: handles both direct (bgsage/scripts/) and
# worktree (bgsage/.claude/worktrees/<name>/scripts/) layouts
_parts = os.path.normpath(bgsage_dir).replace('\\', '/').split('/')
if '.claude' in _parts:
    _idx = _parts.index('.claude')
    bgsage_dir = '/'.join(_parts[:_idx])
project_dir = os.path.dirname(bgsage_dir)

build_dirs = [
    os.path.join(project_dir, 'build_msvc_s7'),  # S7 worktree build (highest priority)
    os.path.join(project_dir, 'build'),
    os.path.join(project_dir, 'build_msvc'),
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
from bgsage.data import load_benchmark_file, load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')

# Stage 7 network config
N_INPUTS = 244
N_HIDDEN = 300            # 300h contact NNs
N_HIDDEN_PURERACE = 100   # 100h purerace
N_INPUTS_PURERACE = 196

MODEL_PREFIX = 'sl_s7'
TD_MODEL_NAME = 'td_s7'

# 17 pair names (index 0 = purerace, 1-16 = contact pairs)
PAIR_NAMES = [
    'purerace',
    'race_race', 'race_att', 'race_prim', 'race_anch',
    'att_race', 'att_att', 'att_prim', 'att_anch',
    'prim_race', 'prim_att', 'prim_prim', 'prim_anch',
    'anch_race', 'anch_att', 'anch_prim', 'anch_anch',
]

# Player plan ID for each contact pair (1=racing, 2=attacking, 3=priming, 4=anchoring)
PLAYER_PLAN_IDS = {}
OPP_PLAN_IDS = {}
for i, name in enumerate(PAIR_NAMES[1:], start=1):
    p, o = name.split('_')
    plan_map = {'race': 1, 'att': 2, 'prim': 3, 'anch': 4}
    PLAYER_PLAN_IDS[name] = plan_map[p]
    OPP_PLAN_IDS[name] = plan_map[o]

# Player plan name -> benchmark file name
PLAN_BM_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}

# SL phases for contact NNs
CONTACT_PHASES = [(100, 20.0), (200, 10.0), (200, 3.1), (500, 1.0)]
# SL phases for GPW scan (first 2 phases only)
SCAN_PHASES = [(100, 20.0), (200, 10.0)]
# SL phases for purerace
PURERACE_PHASES = [(200, 20.0), (200, 6.3), (1000, 2.0)]

# NN sharing: (Prim,Prim), (Anch,Prim), (Anch,Anch) share NN with (Prim,Anch).
# Canonical index = 12 (prim_anch). Indices 11, 15, 16 are aliases.
# Combined frequency: ~6.8% of contact training data.
CANONICAL_MAP = list(range(17))  # identity by default
CANONICAL_MAP[11] = 12   # prim_prim -> prim_anch
CANONICAL_MAP[15] = 12   # anch_prim -> prim_anch
CANONICAL_MAP[16] = 12   # anch_anch -> prim_anch

# The set of NNs that actually need training (canonical indices only)
CANONICAL_CONTACT_NAMES = [
    name for i, name in enumerate(PAIR_NAMES[1:], start=1)
    if CANONICAL_MAP[i] == i
]
# ['race_race','race_att','race_prim','race_anch',
#  'att_race','att_att','att_prim','att_anch',
#  'prim_race','prim_att','prim_anch',
#  'anch_race','anch_att']

# Default gpw values (used when --skip-gpw-scan)
DEFAULT_GPWS = {name: 5.0 for name in CANONICAL_CONTACT_NAMES}

HIDDEN_SIZES = [N_HIDDEN_PURERACE] + [N_HIDDEN] * 16

# GPW candidates to scan
GPW_CANDIDATES = [1.5, 3.0, 5.0, 7.0, 10.0]


def flip_boards_numpy(boards):
    """Flip a batch of boards (numpy array [N, 26])."""
    flipped = np.zeros_like(boards)
    flipped[:, 0] = boards[:, 25]
    flipped[:, 25] = boards[:, 0]
    for i in range(1, 25):
        flipped[:, i] = -boards[:, 25 - i]
    return flipped


def run_td_training():
    """Run TD self-play training: 300k@0.1 + 1.5M@0.02."""

    print(f'\n{"="*60}')
    print(f'  TD SELF-PLAY TRAINING (Stage 7 — 17-NN Pair)')
    print(f'  PureRace: {N_HIDDEN_PURERACE}h, {N_INPUTS_PURERACE} inputs')
    print(f'  Contact NNs: {N_HIDDEN}h, {N_INPUTS} inputs (16 pairs)')
    print(f'{"="*60}\n')

    # Load benchmark scenarios
    benchmark_sets = {}
    for bm_name in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if os.path.exists(bm_path):
            print(f'Loading {bm_name} benchmark (step=10)...')
            t0 = time.time()
            ss = load_benchmark_file(bm_path, step=10)
            print(f'  Loaded {len(ss)} scenarios in {time.time()-t0:.1f}s')
            benchmark_sets[bm_name] = ss

    # Build benchmark list: only purerace + one per player plan (at canonical indices).
    # Using too many benchmarks causes segfaults in multithreaded scoring — limit to
    # one per player plan until the root cause is fixed.
    benchmarks = [None] * 17
    benchmarks[0] = benchmark_sets.get('purerace')
    # One benchmark per player plan at the first canonical contact index.
    # NOTE: priming benchmark excluded due to segfault in pair strategy scoring —
    # needs investigation. Other benchmarks work fine.
    for plan_bm_name, idx in [('racing', 1), ('attacking', 5), ('anchoring', 13)]:
        if CANONICAL_MAP[idx] == idx:  # only canonical indices
            benchmarks[idx] = benchmark_sets.get(plan_bm_name)

    print()

    # Phase 1: 300k @ alpha=0.1
    print(f'=== TD Phase 1: 300k @ alpha=0.1 ===')
    print(flush=True)

    # NOTE: TD training uses identity canonical_map (no sharing) because
    # shared NNs cause forward_with_gradients/forward conflicts during self-play.
    # After TD training, shared pairs will be combined by copying weights.
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
    print(f'Phase 1 done: {result1.games_played} games in {result1.total_seconds:.1f}s')

    # Phase 2: 1.5M @ alpha=0.02 (resume from Phase 1)
    td_phase2_name = f'{TD_MODEL_NAME}_1800k'
    print(f'\n=== TD Phase 2: 1.5M @ alpha=0.02 ===')
    print(flush=True)

    # Resume from Phase 1 — all 17 NNs (no sharing in TD)
    resume_paths = [
        os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_{name}.weights')
        for name in PAIR_NAMES
    ]

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
    print(f'Phase 2 done: {result2.games_played} games in {result2.total_seconds:.1f}s')
    total_td_time = result1.total_seconds + result2.total_seconds
    print(f'Total TD time: {total_td_time:.1f}s ({total_td_time/3600:.1f}h)')

    # After TD: for shared pairs, copy prim_anch weights to aliases.
    # TD trained all 17 independently; now we unify the shared group by
    # using prim_anch (the most frequent of the group) as canonical.
    import shutil
    canonical_name = 'prim_anch'
    canonical_src = os.path.join(MODELS_DIR, f'{td_phase2_name}_{canonical_name}.weights')
    for alias_name in ['prim_prim', 'anch_prim', 'anch_anch']:
        alias_dst = os.path.join(MODELS_DIR, f'{td_phase2_name}_{alias_name}.weights')
        if os.path.exists(canonical_src):
            shutil.copy2(canonical_src, alias_dst)
            print(f'  Copied TD weights: {canonical_name} -> {alias_name}')

    return td_phase2_name


def load_training_data():
    """Load contact+crashed training data."""
    print('Loading training data...')
    t0 = time.time()
    boards_c, targets_c = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    print(f'  contact-train-data: {len(boards_c)} positions ({time.time()-t0:.1f}s)')
    t0 = time.time()
    boards_r, targets_r = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    print(f'  crashed-train-data: {len(boards_r)} positions ({time.time()-t0:.1f}s)')
    boards = np.concatenate([boards_c, boards_r], axis=0)
    targets = np.concatenate([targets_c, targets_r], axis=0)
    print(f'  Total: {len(boards)} positions')
    return boards, targets


def load_purerace_training_data():
    """Load purerace training data."""
    print('Loading purerace training data...')
    t0 = time.time()
    boards, targets = load_gnubg_training_data(os.path.join(DATA_DIR, 'purerace-train-data'))
    print(f'  purerace-train-data: {len(boards)} positions ({time.time()-t0:.1f}s)')
    return boards, targets


def load_benchmarks():
    """Load per-plan benchmarks (step=10 for training progress)."""
    benchmarks = {}
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if os.path.exists(bm_path):
            benchmarks[bm_type] = load_benchmark_file(bm_path, step=10)
    return benchmarks


def classify_pairs(boards):
    """Classify all boards by (player, opponent) pair.

    Returns pair_ids array where:
      -1 = purerace position
      0-15 = contact pair index: (player_plan-1)*4 + (opponent_plan-1)
    """
    print('Classifying game plan pairs...')
    t0 = time.time()
    player_gps = bgbot_cpp.classify_game_plans_batch(boards)
    flipped = flip_boards_numpy(boards)
    opp_gps = bgbot_cpp.classify_game_plans_batch(flipped)

    # Pair index: (player-1)*4 + (opponent-1) for contact, -1 for purerace
    # These are raw pair indices (0-15), then mapped through CANONICAL_MAP.
    pair_ids = np.full(len(boards), -1, dtype=np.int32)
    contact_mask = player_gps > 0
    # Handle the rare case where player has contact but opponent is purerace
    opp_adjusted = opp_gps.copy()
    opp_adjusted[opp_adjusted == 0] = 1  # default to racing
    raw_pair_ids = (player_gps[contact_mask] - 1) * 4 + (opp_adjusted[contact_mask] - 1)
    # Apply canonical map (subtract 1 because CANONICAL_MAP is 0-based with purerace at 0)
    canonical_contact = np.array([CANONICAL_MAP[1 + pid] - 1 for pid in range(16)], dtype=np.int32)
    pair_ids[contact_mask] = canonical_contact[raw_pair_ids]

    # Print frequency table
    n = len(boards)
    n_pr = int(np.sum(~contact_mask))
    print(f'  PureRace: {n_pr}/{n} ({100*n_pr/n:.1f}%)')
    for i, name in enumerate(PAIR_NAMES[1:]):
        cnt = int(np.sum(pair_ids == i))
        print(f'  {name:12s}: {cnt:7d}/{n} ({100*cnt/n:.1f}%)')
    print(f'  Classified in {time.time()-t0:.1f}s')
    return pair_ids


def build_sample_weights(pair_ids, target_pair_idx, gpw):
    """Build sample weights array: gpw for matching pairs, 1.0 for rest."""
    weights = np.ones(len(pair_ids), dtype=np.float32)
    weights[pair_ids == target_pair_idx] = gpw
    return weights


def train_one_nn(nn_name, gpw, boards, targets, pair_ids, benchmark_scenarios,
                 td_model_name, phases, n_hidden, n_inputs):
    """Train a single NN through all SL phases."""

    td_weights = os.path.join(MODELS_DIR, f'{td_model_name}_{nn_name}.weights')
    save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{nn_name}.weights')

    if not os.path.exists(td_weights):
        print(f'  ERROR: TD weights not found: {td_weights}')
        return None

    # Build sample weights
    # For shared NNs (e.g., prim_anch), the pair_ids already map all aliases
    # to the canonical index, so a single match on that index covers all shared pairs.
    sample_weights = None
    n_match = len(boards)
    if nn_name != 'purerace' and gpw != 1.0:
        canonical_nn_idx = PAIR_NAMES.index(nn_name)
        pair_idx = canonical_nn_idx - 1  # 0-based contact pair index
        sample_weights = build_sample_weights(pair_ids, pair_idx, gpw)
        n_match = int(np.sum(pair_ids == pair_idx))

    total_epochs = sum(ep for ep, _ in phases)
    phase_desc = ' -> '.join(f'{ep}ep@a={a}' for ep, a in phases)
    print(f'\n{"="*60}')
    print(f'  {nn_name.upper()} ({n_hidden}h, {n_inputs} inputs, gpw={gpw}, {total_epochs} total epochs)')
    print(f'  Schedule: {phase_desc}')
    print(f'  TD weights: {td_weights}')
    print(f'  Save path:  {save_path}')
    if nn_name != 'purerace':
        print(f'  Matching positions: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
    else:
        print(f'  Training positions: {len(boards)}')
    print(f'{"="*60}')

    current_weights = td_weights
    best_score = float('inf')
    best_epoch_total = 0
    total_time = 0.0
    epoch_offset = 0
    phase_results = []

    for phase_idx, (epochs, alpha) in enumerate(phases):
        phase_label = f'Phase {phase_idx+1}/{len(phases)}'
        print(f'\n  {phase_label}: {epochs} epochs @ alpha={alpha}')

        print_interval = min(10, max(1, epochs // 20))

        result = bgbot_cpp.cuda_supervised_train(
            boards=boards,
            targets=targets,
            weights_path=current_weights,
            n_hidden=n_hidden,
            n_inputs=n_inputs,
            alpha=alpha,
            epochs=epochs,
            batch_size=128,
            seed=42,
            print_interval=print_interval,
            save_path=save_path,
            benchmark_scenarios=benchmark_scenarios,
            sample_weights=sample_weights,
        )

        phase_best = result['best_score']
        phase_best_epoch = result['best_epoch']
        phase_time = result['total_seconds']
        total_time += phase_time

        if phase_best < best_score:
            best_score = phase_best
            best_epoch_total = epoch_offset + phase_best_epoch

        phase_results.append({
            'phase': phase_idx + 1,
            'epochs': epochs,
            'alpha': alpha,
            'best_score': phase_best,
            'best_epoch': phase_best_epoch,
            'time': phase_time,
        })
        print(f'  {phase_label} done: best={phase_best:.2f} (epoch {phase_best_epoch}), time={phase_time:.1f}s')

        current_weights = save_path + '.best'
        epoch_offset += epochs

    print(f'\n  FINAL: {nn_name} {n_hidden}h gpw={gpw} -> best={best_score:.2f} (epoch {best_epoch_total}), total time={total_time:.1f}s')
    return {
        'best_score': best_score,
        'best_epoch': best_epoch_total,
        'total_time': total_time,
        'phases': phase_results,
    }


def gpw_scan(nn_name, boards, targets, pair_ids, benchmark_scenarios,
             td_model_name):
    """Scan GPW values for a single contact NN.

    Runs the first 2 SL phases (300 epochs) for each candidate GPW,
    returns the gpw with the lowest benchmark score.
    """
    print(f'\n{"="*60}')
    print(f'  GPW SCAN: {nn_name.upper()}')
    print(f'{"="*60}')

    results = {}
    for gpw in GPW_CANDIDATES:
        # Use a temporary save path to avoid overwriting real weights
        pair_idx = PAIR_NAMES.index(nn_name) - 1
        sample_weights = build_sample_weights(pair_ids, pair_idx, gpw)
        n_match = int(np.sum(pair_ids == pair_idx))

        td_weights = os.path.join(MODELS_DIR, f'{td_model_name}_{nn_name}.weights')
        save_path = os.path.join(MODELS_DIR, f'gpw_scan_{nn_name}_gpw{gpw:.1f}.weights')

        current_weights = td_weights
        best_score = float('inf')

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
                benchmark_scenarios=benchmark_scenarios,
                sample_weights=sample_weights,
                label=f'{nn_name} gpw={gpw}',
            )
            if result['best_score'] < best_score:
                best_score = result['best_score']
            current_weights = save_path + '.best'

        results[gpw] = best_score
        print(f'  gpw={gpw:.1f}: best ER={best_score:.2f} (match={n_match}, {100*n_match/len(boards):.1f}%)')

        # Clean up scan files
        for suffix in ['', '.best']:
            p = save_path + suffix
            if os.path.exists(p):
                os.remove(p)

    # Find best gpw
    best_gpw = min(results, key=results.get)
    print(f'\n  BEST GPW for {nn_name}: {best_gpw:.1f} (ER={results[best_gpw]:.2f})')
    print(f'  All results: {" | ".join(f"gpw={g:.1f}:{s:.2f}" for g, s in sorted(results.items()))}')
    return best_gpw, results


def score_stage7():
    """Score the Stage 7 trained models."""
    # Build weight paths — aliased indices use the canonical NN's path
    weight_paths = []
    for i, name in enumerate(PAIR_NAMES):
        canonical_idx = CANONICAL_MAP[i]
        canonical_name = PAIR_NAMES[canonical_idx]
        weight_paths.append(os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{canonical_name}.weights.best'))

    # Check only canonical paths
    for i, path in enumerate(weight_paths):
        if CANONICAL_MAP[i] == i and not os.path.exists(path):
            print(f'  Missing: {path}')
            return None

    print(f'\n{"="*60}')
    print(f'  STAGE 7 BENCHMARKS')
    print(f'{"="*60}\n')
    for i, name in enumerate(PAIR_NAMES):
        h = N_HIDDEN_PURERACE if i == 0 else N_HIDDEN
        print(f'  {name:12s}: {weight_paths[i]} ({h}h)')
    print()

    scores = {}

    # Per-plan benchmarks
    print('--- Game Plan benchmarks (full) ---')
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_pair(
            scenarios, weight_paths, HIDDEN_SIZES)
        t_score = time.time() - t0
        scores[bm_type] = result.score()
        print(f'  {bm_type:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # Old-style benchmarks
    print()
    print('--- Old-style benchmarks (for comparison) ---')
    for bm_name in ['contact', 'crashed', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        result = bgbot_cpp.score_benchmarks_pair(
            scenarios, weight_paths, HIDDEN_SIZES)
        t_score = time.time() - t0
        scores[bm_name] = result.score()
        print(f'  {bm_name:10s}: {result.score():8.2f}  ({result.count} scenarios, {t_score:.1f}s)')

    # vs PubEval
    print()
    print('=== vs PubEval (10k games) ===')
    t0 = time.time()
    stats = bgbot_cpp.play_games_pair_vs_pubeval(
        weight_paths, HIDDEN_SIZES, n_games=10000, seed=42)
    t_games = time.time() - t0
    scores['vs_pubeval'] = stats.avg_ppg()
    print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
    print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
    print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')

    # Self-play
    print()
    print('=== Self-play outcome distribution (10k games) ===')
    t0 = time.time()
    ss = bgbot_cpp.play_games_pair_vs_self(
        weight_paths, HIDDEN_SIZES, n_games=10000, seed=42)
    t_sp = time.time() - t0
    total = ss.n_games
    singles = ss.p1_wins + ss.p2_wins
    gammons = ss.p1_gammons + ss.p2_gammons
    backgammons = ss.p1_backgammons + ss.p2_backgammons
    print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
          f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
          f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
          f'({total} games in {t_sp:.1f}s)')

    # Sanity check outcome distribution
    s_pct = 100 * singles / total
    g_pct = 100 * gammons / total
    if s_pct < 55 or s_pct > 75 or g_pct < 20 or g_pct > 40:
        print(f'\n  WARNING: Outcome distribution looks unusual (singles={s_pct:.1f}%, gammons={g_pct:.1f}%)')
        print(f'  Expected: singles ~66%, gammons ~30%, backgammons ~4%')

    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage 7 Training (17-NN Pair)')
    parser.add_argument('--score-only', action='store_true',
                        help='Skip training, just score existing weight files')
    parser.add_argument('--sl-only', action='store_true',
                        help='Skip TD training, use existing TD weights')
    parser.add_argument('--gpw-scan-only', action='store_true',
                        help='Only run GPW scan (requires TD weights)')
    parser.add_argument('--skip-gpw-scan', action='store_true',
                        help='Skip GPW scan, use default gpw values')
    parser.add_argument('--nn', type=str, nargs='+', default=None,
                        help='Train only specific NNs (e.g., --nn race_att anch_prim purerace)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.score_only:
        scores = score_stage7()
        if scores:
            results_dir = os.path.join(project_dir, 'experiments', 'stage7')
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(results_dir, f'results_{timestamp}.json')
            with open(results_path, 'w') as f:
                json.dump(scores, f, indent=2)
            print(f'\nResults saved to: {results_path}')
        return

    # CUDA is only needed for SL training, not TD training
    needs_cuda = not (args.sl_only is False and args.gpw_scan_only is False)
    # Actually: check if we'll do any SL work
    needs_sl = not args.score_only  # we might do SL after TD
    if needs_sl and bgbot_cpp.cuda_available():
        print('CUDA GPU detected')
    elif needs_sl:
        print('NOTE: CUDA not available — TD training will proceed, but SL training will require CUDA.')
    print()

    # TD training
    td_model_name = f'{TD_MODEL_NAME}_1800k'
    if not args.sl_only and not args.gpw_scan_only:
        td_model_name = run_td_training()
    else:
        print(f'Skipping TD training, using existing weights: {td_model_name}')
        for i, name in enumerate(PAIR_NAMES):
            if CANONICAL_MAP[i] != i:
                continue  # skip aliases
            td_path = os.path.join(MODELS_DIR, f'{td_model_name}_{name}.weights')
            if not os.path.exists(td_path):
                print(f'  ERROR: TD weights not found: {td_path}')
                sys.exit(1)
        print('  All TD weights found.')
        print()

    # Determine which NNs to train (only canonical NNs by default)
    nn_types = args.nn if args.nn else (['purerace'] + CANONICAL_CONTACT_NAMES)

    # Load data
    benchmarks = load_benchmarks()
    all_results = {}

    # SL training requires CUDA
    if not bgbot_cpp.cuda_available():
        print('\nCUDA not available — skipping SL training.')
        print('Run with --sl-only once CUDA build is available.')
        scores = score_stage7()
        return

    # Train purerace separately (different data)
    if 'purerace' in nn_types and not args.gpw_scan_only:
        pr_boards, pr_targets = load_purerace_training_data()
        pr_benchmark = benchmarks.get('purerace')
        result = train_one_nn('purerace', 1.0,
                              pr_boards, pr_targets, None, pr_benchmark,
                              td_model_name, PURERACE_PHASES,
                              N_HIDDEN_PURERACE, N_INPUTS_PURERACE)
        if result:
            all_results['purerace'] = result

    # Train contact NNs
    contact_types = [t for t in nn_types if t != 'purerace']
    if contact_types:
        boards, targets = load_training_data()
        pair_ids = classify_pairs(boards)

        # GPW scan
        optimal_gpws = dict(DEFAULT_GPWS)
        if not args.skip_gpw_scan:
            gpw_scan_results = {}
            for nn_name in contact_types:
                plan_id = PLAYER_PLAN_IDS[nn_name]
                plan_bm_name = PLAN_BM_NAMES[plan_id]
                bm_scenarios = benchmarks.get(plan_bm_name)
                best_gpw, scan_results = gpw_scan(
                    nn_name, boards, targets, pair_ids, bm_scenarios,
                    td_model_name)
                optimal_gpws[nn_name] = best_gpw
                gpw_scan_results[nn_name] = scan_results

            # Print GPW scan summary
            print(f'\n{"="*60}')
            print(f'  GPW SCAN SUMMARY')
            print(f'{"="*60}')
            for nn_name in contact_types:
                print(f'  {nn_name:12s}: gpw={optimal_gpws[nn_name]:.1f}')
            print()

            # Save scan results
            results_dir = os.path.join(project_dir, 'experiments', 'stage7')
            os.makedirs(results_dir, exist_ok=True)
            scan_path = os.path.join(results_dir, 'gpw_scan_results.json')
            with open(scan_path, 'w') as f:
                json.dump({k: {str(g): s for g, s in v.items()}
                          for k, v in gpw_scan_results.items()}, f, indent=2)
            print(f'Scan results saved to: {scan_path}')

        if args.gpw_scan_only:
            return

        # Full SL training
        for nn_name in contact_types:
            gpw = optimal_gpws[nn_name]
            plan_id = PLAYER_PLAN_IDS[nn_name]
            plan_bm_name = PLAN_BM_NAMES[plan_id]
            bm_scenarios = benchmarks.get(plan_bm_name)
            result = train_one_nn(nn_name, gpw, boards, targets, pair_ids,
                                  bm_scenarios, td_model_name,
                                  CONTACT_PHASES, N_HIDDEN, N_INPUTS)
            if result:
                all_results[nn_name] = result

    # Print training summary
    if all_results:
        print(f'\n{"="*60}')
        print(f'  TRAINING SUMMARY')
        print(f'{"="*60}\n')
        for nn_name, result in all_results.items():
            n_h = N_HIDDEN_PURERACE if nn_name == 'purerace' else N_HIDDEN
            gpw = 1.0 if nn_name == 'purerace' else optimal_gpws.get(nn_name, 5.0)
            print(f'  {nn_name:12s}: best={result["best_score"]:.2f} (epoch {result["best_epoch"]}), '
                  f'{n_h}h, gpw={gpw}, time={result["total_time"]:.0f}s')
        total_time = sum(r['total_time'] for r in all_results.values())
        print(f'\n  Total SL training time: {total_time:.0f}s ({total_time/3600:.1f}h)')

    # Score
    scores = score_stage7()
    if scores:
        results_dir = os.path.join(project_dir, 'experiments', 'stage7')
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join(results_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    main()
