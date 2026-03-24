"""
Stage 7 SL Training: 17-NN pair strategy (100h purerace, 300h contact).

Phase 1: GPW Scan — for each of 14 canonical NNs, try gpw=(1.5,3,5,7,10)
         with a short SL schedule (200ep @ alpha=10) and pick lowest ER.
Phase 2: Full SL — train each NN with optimal gpw through full schedule.

The 4 pairs (prim_prim, prim_anch, anch_prim, anch_anch) share one NN.

Usage:
    python scripts/run_s7_sl_training.py                # Full: GPW scan + SL
    python scripts/run_s7_sl_training.py --skip-scan     # Skip scan, use default gpw
    python scripts/run_s7_sl_training.py --sl-only       # Skip scan, use saved optimal gpw
    python scripts/run_s7_sl_training.py --nn race_race att_att  # Train specific NNs only
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

# Walk up from script dir to find project root (parent of bgsage/)
project_dir = script_dir
while project_dir != os.path.dirname(project_dir):
    if os.path.isdir(os.path.join(project_dir, 'backend')):
        break
    project_dir = os.path.dirname(project_dir)

build_dir = os.path.join(project_dir, 'build_msvc')
if not os.path.isdir(build_dir):
    build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')

# Stage 7 config
N_INPUTS = 244
N_HIDDEN = 300
N_HIDDEN_PURERACE = 100
N_INPUTS_PURERACE = 196

MODEL_PREFIX = 'sl_s7'
TD_MODEL_NAME = 'td_s7_1800k'

GP_IDS = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}
GP_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}

# All 16 ordered contact pairs
ALL_CONTACT_PAIRS = [
    (p, o) for p in ['racing', 'attacking', 'priming', 'anchoring']
    for o in ['racing', 'attacking', 'priming', 'anchoring']
]

# Shared group: these 4 pairs all use the same NN (canonical = prim_anch)
SHARED_PAIRS = {
    ('priming', 'priming'), ('priming', 'anchoring'),
    ('anchoring', 'priming'), ('anchoring', 'anchoring'),
}
SHARED_CANONICAL = ('priming', 'anchoring')  # prim_anch is the canonical name

# 13 canonical contact NNs (16 pairs - 3 aliases)
CANONICAL_CONTACT_PAIRS = []
_seen_shared = False
for pair in ALL_CONTACT_PAIRS:
    if pair in SHARED_PAIRS:
        if not _seen_shared:
            CANONICAL_CONTACT_PAIRS.append(SHARED_CANONICAL)
            _seen_shared = True
    else:
        CANONICAL_CONTACT_PAIRS.append(pair)

# SL schedule (same as S5)
CONTACT_PHASES = [(100, 20.0), (200, 10.0), (200, 3.1), (500, 1.0)]
PURERACE_PHASES = [(200, 20.0), (500, 6.3), (500, 2.0)]

# GPW scan candidates
GPW_CANDIDATES = [1.5, 3.0, 5.0, 7.0, 10.0]

# GPW scan schedule (short — just Phase 2 of the SL schedule)
GPW_SCAN_PHASES = [(200, 10.0)]


def pair_name(p, o):
    """e.g. ('racing', 'attacking') -> 'race_att'"""
    abbrev = {'racing': 'race', 'attacking': 'att', 'priming': 'prim', 'anchoring': 'anch'}
    return f"{abbrev[p]}_{abbrev[o]}"


def pair_display(p, o):
    """e.g. ('racing', 'attacking') -> 'Racing/Attacking'"""
    return f"{p.capitalize()}/{o.capitalize()}"


def is_shared_pair(p, o):
    return (p, o) in SHARED_PAIRS


def get_benchmark_name(p, o):
    """Which .bm file to use for this pair. Use player's plan benchmark."""
    if (p, o) == SHARED_CANONICAL:
        return 'priming'  # shared group benchmarks on priming
    return p


def flip_boards_numpy(boards):
    """Flip all boards in numpy array to opponent perspective."""
    flipped = np.zeros_like(boards)
    flipped[:, 0] = boards[:, 25]
    flipped[:, 25] = boards[:, 0]
    flipped[:, 1:25] = -boards[:, 24:0:-1]
    return flipped


def compute_pair_mask(player_gps, opponent_gps, p, o):
    """Compute boolean mask for positions matching pair (p, o).
    For the shared canonical pair, matches all 4 shared pairs."""
    if (p, o) == SHARED_CANONICAL:
        mask = np.zeros(len(player_gps), dtype=bool)
        for sp, so in SHARED_PAIRS:
            mask |= (player_gps == GP_IDS[sp]) & (opponent_gps == GP_IDS[so])
        return mask
    else:
        return (player_gps == GP_IDS[p]) & (opponent_gps == GP_IDS[o])


def compute_sample_weights(player_gps, opponent_gps, p, o, gpw):
    """Compute per-sample weights: gpw for matching pair, 1.0 for rest."""
    if gpw == 1.0:
        return None
    mask = compute_pair_mask(player_gps, opponent_gps, p, o)
    weights = np.ones(len(player_gps), dtype=np.float32)
    weights[mask] = gpw
    return weights


def load_contact_data():
    """Load contact + crashed training data."""
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


def load_purerace_data():
    """Load purerace training data."""
    print('Loading purerace training data...')
    t0 = time.time()
    boards, targets = load_gnubg_training_data(os.path.join(DATA_DIR, 'purerace-train-data'))
    print(f'  purerace: {len(boards)} positions ({time.time()-t0:.1f}s)')
    return boards, targets


def classify_pairs(boards):
    """Classify all boards by (player, opponent) game plan pair."""
    print('Classifying game plan pairs...')
    t0 = time.time()
    player_gps = bgbot_cpp.classify_game_plans_batch(boards)
    flipped = flip_boards_numpy(boards)
    opponent_gps = bgbot_cpp.classify_game_plans_batch(flipped)
    print(f'  Classified in {time.time()-t0:.1f}s')

    # Print distribution
    for p, o in CANONICAL_CONTACT_PAIRS:
        mask = compute_pair_mask(player_gps, opponent_gps, p, o)
        n = int(np.sum(mask))
        pct = 100 * n / len(boards)
        label = pair_display(p, o)
        if (p, o) == SHARED_CANONICAL:
            label += " (shared group)"
        print(f'    {label:35s}: {n:7d} ({pct:5.1f}%)')

    return player_gps, opponent_gps


def load_benchmarks():
    """Load per-plan benchmarks (step=10 for training progress)."""
    benchmarks = {}
    for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if os.path.exists(bm_path):
            benchmarks[bm_type] = load_benchmark_file(bm_path, step=10)
    return benchmarks


def load_pair_filtered_benchmarks(step=10):
    """Load benchmarks filtered by (player, opponent) game plan pair.

    For each pair NN, returns a ScenarioSet containing only benchmark positions
    where both the player's and opponent's game plans match the pair.
    This gives per-pair ER during SL training, not just per-player-plan ER.

    For the shared group (prim_prim, prim_anch, anch_prim, anch_anch),
    combines matching positions from both priming.bm and anchoring.bm.
    """
    from bgsage.data import load_benchmark_scenarios_by_indices, board_from_gnubg_position_string

    pair_benchmarks = {}

    # For each plan benchmark file, parse boards and classify opponent plans
    bm_opponent_indices = {}  # {bm_name: {opponent_plan: [indices]}}
    for bm_name in ['racing', 'attacking', 'priming', 'anchoring']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            continue

        with open(bm_path) as f:
            lines = [l for l in f if l.startswith('m ')]

        # Subsample
        lines = lines[::step]

        # Extract boards and classify opponent game plans
        boards = []
        for line in lines:
            b = board_from_gnubg_position_string(line.split()[1])
            boards.append(b)
        boards_np = np.array(boards, dtype=np.int32)
        flipped = flip_boards_numpy(boards_np)
        opp_gps = bgbot_cpp.classify_game_plans_batch(flipped)

        # Group indices by opponent plan
        opp_indices = {}
        for gp_id in [1, 2, 3, 4]:
            indices = list(np.where(opp_gps == gp_id)[0])
            opp_indices[GP_NAMES[gp_id]] = indices
        bm_opponent_indices[bm_name] = opp_indices

        print(f'  {bm_name}.bm: {len(lines)} scenarios (step={step})')
        for opp_name, idxs in opp_indices.items():
            print(f'    opp={opp_name:10s}: {len(idxs):5d} ({100*len(idxs)/len(lines):5.1f}%)')

    # Build pair-filtered ScenarioSets
    for p, o in CANONICAL_CONTACT_PAIRS:
        name = pair_name(p, o)

        if (p, o) == SHARED_CANONICAL:
            # Shared group: combine from multiple .bm files
            # Need positions from priming.bm and anchoring.bm matching any shared pair
            ss = bgbot_cpp.ScenarioSet()
            total = 0
            for sp, so in SHARED_PAIRS:
                bm_name = sp  # player's plan = benchmark file
                if bm_name in bm_opponent_indices and so in bm_opponent_indices[bm_name]:
                    indices = bm_opponent_indices[bm_name][so]
                    if indices:
                        # Convert subsampled indices back to full-file indices
                        full_indices = [i * step for i in indices]
                        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
                        part = load_benchmark_scenarios_by_indices(bm_path, full_indices)
                        # Merge into ss
                        # ScenarioSet doesn't support merge, so reload combined
                        total += len(indices)
            # Reload all at once using combined indices
            combined_indices = []
            combined_sources = []
            for sp, so in SHARED_PAIRS:
                bm_name = sp
                if bm_name in bm_opponent_indices and so in bm_opponent_indices[bm_name]:
                    indices = bm_opponent_indices[bm_name][so]
                    full_indices = [i * step for i in indices]
                    combined_indices.append((bm_name, full_indices))
            # Load each source separately and add to one ScenarioSet
            ss = bgbot_cpp.ScenarioSet()
            for bm_name, full_indices in combined_indices:
                bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
                with open(bm_path) as f:
                    all_lines = [l for l in f if l.startswith('m ')]
                for idx in full_indices:
                    if idx < len(all_lines):
                        bits = all_lines[idx].split()
                        start_board = board_from_gnubg_position_string(bits[1])
                        die1, die2 = int(bits[2]), int(bits[3])
                        ranked_boards, ranked_errors = [], []
                        i = 4
                        while i < len(bits):
                            ranked_boards.append(board_from_gnubg_position_string(bits[i]))
                            ranked_errors.append(float(bits[i+1]) if i+1 < len(bits) else 0.0)
                            i += 2
                        ss.add(start_board, die1, die2, ranked_boards, ranked_errors)
            pair_benchmarks[name] = ss
            print(f'  {name} (shared): {ss.size()} scenarios')
        else:
            # Standard pair: filter player's plan .bm by opponent plan
            bm_name = p
            if bm_name in bm_opponent_indices and o in bm_opponent_indices[bm_name]:
                indices = bm_opponent_indices[bm_name][o]
                full_indices = [i * step for i in indices]
                bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
                ss = load_benchmark_scenarios_by_indices(bm_path, full_indices)
                pair_benchmarks[name] = ss
                print(f'  {name}: {ss.size()} scenarios')
            else:
                print(f'  {name}: NO SCENARIOS FOUND')

    return pair_benchmarks


def train_single_nn(nn_name, td_weights, n_hidden, n_inputs, boards, targets,
                    sample_weights, benchmark_scenarios, save_path, phases,
                    label=None):
    """Train one NN through all SL phases. Returns result dict."""
    if not os.path.exists(td_weights):
        print(f'  ERROR: TD weights not found: {td_weights}')
        return None

    current_weights = td_weights
    best_score = float('inf')
    best_epoch_total = 0
    total_time = 0.0
    epoch_offset = 0

    for phase_idx, (epochs, alpha) in enumerate(phases):
        phase_label = f'Phase {phase_idx+1}/{len(phases)}'
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
            label=label or nn_name,
        )

        phase_best = result['best_score']
        phase_best_epoch = result['best_epoch']
        phase_time = result['total_seconds']
        total_time += phase_time

        if phase_best < best_score:
            best_score = phase_best
            best_epoch_total = epoch_offset + phase_best_epoch

        print(f'  {phase_label}: best={phase_best:.2f} (epoch {phase_best_epoch}), '
              f'time={phase_time:.1f}s')

        current_weights = save_path + '.best'
        epoch_offset += epochs

    return {
        'best_score': best_score,
        'best_epoch': best_epoch_total,
        'total_time': total_time,
    }


def run_gpw_scan(boards, targets, player_gps, opponent_gps, benchmarks):
    """Run GPW scan for all canonical contact NNs.
    Returns dict mapping pair_name -> optimal_gpw."""

    print(f'\n{"="*70}')
    print(f'  GPW SCAN: Testing {GPW_CANDIDATES} for {len(CANONICAL_CONTACT_PAIRS)} contact NNs')
    print(f'  Schedule: {" -> ".join(f"{e}ep@a={a}" for e, a in GPW_SCAN_PHASES)}')
    print(f'{"="*70}\n')

    optimal_gpw = {}
    scan_dir = os.path.join(MODELS_DIR, 's7_gpw_scan')
    os.makedirs(scan_dir, exist_ok=True)

    t0_total = time.time()

    for pair_idx, (p, o) in enumerate(CANONICAL_CONTACT_PAIRS):
        name = pair_name(p, o)
        bm_name = get_benchmark_name(p, o)
        bm_scenarios = benchmarks.get(bm_name)

        mask = compute_pair_mask(player_gps, opponent_gps, p, o)
        n_match = int(np.sum(mask))
        pct = 100 * n_match / len(boards)

        td_weights = os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_{name}.weights')

        print(f'\n--- [{pair_idx+1}/{len(CANONICAL_CONTACT_PAIRS)}] {pair_display(p, o)} '
              f'({n_match} positions, {pct:.1f}%, bm={bm_name}) ---')

        best_gpw = 1.0
        best_score = float('inf')
        results = []

        for gpw in GPW_CANDIDATES:
            sample_weights = compute_sample_weights(player_gps, opponent_gps, p, o, gpw)
            save_path = os.path.join(scan_dir, f'{name}_gpw{gpw:.1f}.weights')

            result = train_single_nn(
                nn_name=name,
                td_weights=td_weights,
                n_hidden=N_HIDDEN,
                n_inputs=N_INPUTS,
                boards=boards,
                targets=targets,
                sample_weights=sample_weights,
                benchmark_scenarios=bm_scenarios,
                save_path=save_path,
                phases=GPW_SCAN_PHASES,
                label=f'{name}_gpw{gpw}',
            )

            if result:
                score = result['best_score']
                results.append((gpw, score))
                if score < best_score:
                    best_score = score
                    best_gpw = gpw

        optimal_gpw[name] = best_gpw
        print(f'\n  {pair_display(p, o)} GPW scan results:')
        for gpw, score in results:
            marker = ' <-- BEST' if gpw == best_gpw else ''
            print(f'    gpw={gpw:4.1f}: ER={score:.2f}{marker}')
        print(f'  Optimal gpw: {best_gpw}')

    elapsed = time.time() - t0_total
    print(f'\n{"="*70}')
    print(f'  GPW SCAN COMPLETE ({elapsed:.0f}s = {elapsed/60:.1f}m)')
    print(f'{"="*70}')
    print(f'\n  Optimal GPW values:')
    for name, gpw in optimal_gpw.items():
        print(f'    {name:20s}: {gpw}')

    # Save results
    results_path = os.path.join(scan_dir, 'optimal_gpw.json')
    with open(results_path, 'w') as f:
        json.dump(optimal_gpw, f, indent=2)
    print(f'\n  Saved to: {results_path}')

    return optimal_gpw


def run_full_sl(boards, targets, player_gps, opponent_gps, benchmarks,
                optimal_gpw, nn_filter=None):
    """Run full SL training for all canonical NNs with optimal GPW."""

    print(f'\n{"="*70}')
    print(f'  FULL SL TRAINING (Stage 7)')
    print(f'  Schedule: {" -> ".join(f"{e}ep@a={a}" for e, a in CONTACT_PHASES)}')
    print(f'{"="*70}\n')

    all_results = {}
    t0_total = time.time()

    # Train contact NNs
    contact_pairs = CANONICAL_CONTACT_PAIRS
    if nn_filter:
        contact_pairs = [(p, o) for p, o in contact_pairs if pair_name(p, o) in nn_filter]

    for pair_idx, (p, o) in enumerate(contact_pairs):
        name = pair_name(p, o)
        bm_name = get_benchmark_name(p, o)
        bm_scenarios = benchmarks.get(bm_name)
        gpw = optimal_gpw.get(name, 3.0)

        mask = compute_pair_mask(player_gps, opponent_gps, p, o)
        n_match = int(np.sum(mask))

        td_weights = os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_{name}.weights')
        save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{name}.weights')

        total_epochs = sum(e for e, _ in CONTACT_PHASES)
        phase_desc = ' -> '.join(f'{e}ep@a={a}' for e, a in CONTACT_PHASES)

        print(f'\n{"="*60}')
        print(f'  [{pair_idx+1}/{len(contact_pairs)}] {pair_display(p, o).upper()} '
              f'({N_HIDDEN}h, gpw={gpw}, {total_epochs} epochs)')
        print(f'  Schedule: {phase_desc}')
        print(f'  TD weights: {td_weights}')
        print(f'  Matching: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
        print(f'{"="*60}')

        sample_weights = compute_sample_weights(player_gps, opponent_gps, p, o, gpw)

        result = train_single_nn(
            nn_name=name,
            td_weights=td_weights,
            n_hidden=N_HIDDEN,
            n_inputs=N_INPUTS,
            boards=boards,
            targets=targets,
            sample_weights=sample_weights,
            benchmark_scenarios=bm_scenarios,
            save_path=save_path,
            phases=CONTACT_PHASES,
            label=name,
        )

        if result:
            all_results[name] = {**result, 'gpw': gpw}

    # Copy shared weights to alias names
    shared_canonical = pair_name(*SHARED_CANONICAL)
    shared_best = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{shared_canonical}.weights.best')
    if os.path.exists(shared_best):
        for sp, so in SHARED_PAIRS:
            alias_name = pair_name(sp, so)
            if alias_name != shared_canonical:
                alias_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{alias_name}.weights.best')
                import shutil
                shutil.copy2(shared_best, alias_path)
                print(f'  Copied {shared_canonical} -> {alias_name}')

    elapsed = time.time() - t0_total

    # Summary
    print(f'\n{"="*60}')
    print(f'  SL TRAINING SUMMARY ({elapsed:.0f}s = {elapsed/60:.1f}m)')
    print(f'{"="*60}\n')
    for name, r in all_results.items():
        print(f'  {name:20s}: best={r["best_score"]:.2f} (epoch {r["best_epoch"]}), '
              f'gpw={r["gpw"]}, time={r["total_time"]:.0f}s')

    return all_results


def run_purerace_sl(benchmarks, nn_filter=None):
    """Train purerace NN (no GPW needed)."""
    if nn_filter and 'purerace' not in nn_filter:
        return None

    pr_boards, pr_targets = load_purerace_data()
    bm_scenarios = benchmarks.get('purerace')

    td_weights = os.path.join(MODELS_DIR, f'{TD_MODEL_NAME}_purerace.weights')
    save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_purerace.weights')

    total_epochs = sum(e for e, _ in PURERACE_PHASES)
    phase_desc = ' -> '.join(f'{e}ep@a={a}' for e, a in PURERACE_PHASES)

    print(f'\n{"="*60}')
    print(f'  PURERACE ({N_HIDDEN_PURERACE}h, {total_epochs} epochs)')
    print(f'  Schedule: {phase_desc}')
    print(f'  TD weights: {td_weights}')
    print(f'  Training positions: {len(pr_boards)}')
    print(f'{"="*60}')

    result = train_single_nn(
        nn_name='purerace',
        td_weights=td_weights,
        n_hidden=N_HIDDEN_PURERACE,
        n_inputs=N_INPUTS_PURERACE,
        boards=pr_boards,
        targets=pr_targets,
        sample_weights=None,
        benchmark_scenarios=bm_scenarios,
        save_path=save_path,
        phases=PURERACE_PHASES,
    )

    if result:
        print(f'\n  PURERACE: best={result["best_score"]:.2f} '
              f'(epoch {result["best_epoch"]}), time={result["total_time"]:.0f}s')

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage 7 SL Training (17-NN pair strategy)')
    parser.add_argument('--skip-scan', action='store_true',
                        help='Skip GPW scan, use default gpw=3.0 for all')
    parser.add_argument('--sl-only', action='store_true',
                        help='Skip GPW scan, load saved optimal_gpw.json')
    parser.add_argument('--scan-only', action='store_true',
                        help='Only run GPW scan, skip full SL training')
    parser.add_argument('--nn', type=str, nargs='+', default=None,
                        help='Train only specific NNs (e.g., --nn race_race purerace)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not bgbot_cpp.cuda_available():
        print('ERROR: CUDA not available')
        sys.exit(1)
    print('CUDA GPU detected')
    print()

    # Load data and classify pairs
    benchmarks = load_benchmarks()

    boards, targets = load_contact_data()
    player_gps, opponent_gps = classify_pairs(boards)
    print()

    # GPW Scan
    gpw_results_path = os.path.join(MODELS_DIR, 's7_gpw_scan', 'optimal_gpw.json')
    if args.skip_scan:
        optimal_gpw = {pair_name(p, o): 3.0 for p, o in CANONICAL_CONTACT_PAIRS}
        print('Skipping GPW scan, using default gpw=3.0 for all')
    elif args.sl_only:
        if os.path.exists(gpw_results_path):
            with open(gpw_results_path) as f:
                optimal_gpw = json.load(f)
            print(f'Loaded optimal GPW from {gpw_results_path}:')
            for name, gpw in optimal_gpw.items():
                print(f'  {name:20s}: {gpw}')
        else:
            print(f'No saved GPW results at {gpw_results_path}, using default gpw=3.0')
            optimal_gpw = {pair_name(p, o): 3.0 for p, o in CANONICAL_CONTACT_PAIRS}
    else:
        optimal_gpw = run_gpw_scan(boards, targets, player_gps, opponent_gps, benchmarks)
        if args.scan_only:
            print('\nScan complete. Run with --sl-only for full training.')
            return

    # Purerace SL
    run_purerace_sl(benchmarks, nn_filter=args.nn)

    # Full SL training
    run_full_sl(boards, targets, player_gps, opponent_gps, benchmarks,
                optimal_gpw, nn_filter=args.nn)

    print('\nDone!')


if __name__ == '__main__':
    main()
