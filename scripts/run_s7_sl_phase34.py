"""
Stage 7 SL Training — Phases 3+4 only (resuming from GPW scan Phase 2 weights).

Each NN starts from its optimal-gpw .best weights from the GPW scan (which did
Phase 2: 200ep @ alpha=10), and continues with:
  Phase 3: 200ep @ alpha=3.1
  Phase 4: 500ep @ alpha=1.0

Benchmarks are pair-filtered: each NN is scored only on benchmark positions
matching its (player, opponent) game plan pair.

PureRace is skipped (reuses Stage 6 weights).

Usage:
    python scripts/run_s7_sl_phase34.py                    # Train all 13 contact NNs
    python scripts/run_s7_sl_phase34.py --nn race_race     # Train specific NN(s)
"""

import os
import sys
import json
import time
import shutil
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
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
from bgsage.data import load_gnubg_training_data, board_from_gnubg_position_string

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')
SCAN_DIR = os.path.join(MODELS_DIR, 's7_gpw_scan')

N_INPUTS = 244
N_HIDDEN = 300
MODEL_PREFIX = 'sl_s7'

GP_IDS = {'racing': 1, 'attacking': 2, 'priming': 3, 'anchoring': 4}
GP_NAMES = {1: 'racing', 2: 'attacking', 3: 'priming', 4: 'anchoring'}

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

# Phases 3 and 4 only (Phase 1: 100ep@20, Phase 2: 200ep@10 were done in GPW scan)
REMAINING_PHASES = [(200, 3.1), (500, 1.0)]


def abbrev(name):
    return {'racing': 'race', 'attacking': 'att', 'priming': 'prim', 'anchoring': 'anch'}[name]


def pair_name(p, o):
    return f'{abbrev(p)}_{abbrev(o)}'


def flip_boards_numpy(boards):
    flipped = np.zeros_like(boards)
    flipped[:, 0] = boards[:, 25]
    flipped[:, 25] = boards[:, 0]
    flipped[:, 1:25] = -boards[:, 24:0:-1]
    return flipped


def compute_pair_mask(player_gps, opponent_gps, p, o):
    if (p, o) == SHARED_CANONICAL:
        mask = np.zeros(len(player_gps), dtype=bool)
        for sp, so in SHARED_PAIRS:
            mask |= (player_gps == GP_IDS[sp]) & (opponent_gps == GP_IDS[so])
        return mask
    return (player_gps == GP_IDS[p]) & (opponent_gps == GP_IDS[o])


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='S7 SL Phases 3+4')
    parser.add_argument('--nn', type=str, nargs='+', default=None,
                        help='Train specific NNs (e.g., --nn race_race att_att)')
    args = parser.parse_args()

    if not bgbot_cpp.cuda_available():
        print('ERROR: CUDA not available')
        sys.exit(1)
    print('CUDA GPU detected\n')

    # Load optimal GPW
    gpw_path = os.path.join(SCAN_DIR, 'optimal_gpw.json')
    with open(gpw_path) as f:
        optimal_gpw = json.load(f)
    print('Optimal GPW values:')
    for name, gpw in optimal_gpw.items():
        print(f'  {name:20s}: {gpw}')
    print()

    # Load training data
    print('Loading training data...')
    t0 = time.time()
    boards_c, targets_c = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    print(f'  contact: {len(boards_c)} ({time.time()-t0:.1f}s)')
    t0 = time.time()
    boards_r, targets_r = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    print(f'  crashed: {len(boards_r)} ({time.time()-t0:.1f}s)')
    boards = np.concatenate([boards_c, boards_r], axis=0)
    targets = np.concatenate([targets_c, targets_r], axis=0)
    print(f'  Total: {len(boards)}')

    # Classify pairs for sample weights
    print('Classifying game plan pairs...')
    t0 = time.time()
    player_gps = bgbot_cpp.classify_game_plans_batch(boards)
    flipped = flip_boards_numpy(boards)
    opponent_gps = bgbot_cpp.classify_game_plans_batch(flipped)
    print(f'  Classified in {time.time()-t0:.1f}s')
    print()

    # Build pair-filtered benchmarks
    print('Building pair-filtered benchmarks...')
    bm_data = load_bm_data()
    print()

    # Determine which NNs to train
    pairs_to_train = CANONICAL_CONTACT_PAIRS
    if args.nn:
        name_set = set(args.nn)
        pairs_to_train = [(p, o) for p, o in CANONICAL_CONTACT_PAIRS
                          if pair_name(p, o) in name_set]

    all_results = {}
    t0_total = time.time()

    for pair_idx, (p, o) in enumerate(pairs_to_train):
        name = pair_name(p, o)
        gpw = optimal_gpw.get(name, 5.0)

        # Starting weights: optimal gpw .best from scan
        start_weights = os.path.join(SCAN_DIR, f'{name}_gpw{gpw:.1f}.weights.best')
        save_path = os.path.join(MODELS_DIR, f'{MODEL_PREFIX}_{name}.weights')

        # Build pair-filtered benchmark
        pair_bm = build_pair_benchmark(p, o, bm_data)

        # Compute sample weights
        mask = compute_pair_mask(player_gps, opponent_gps, p, o)
        n_match = int(np.sum(mask))
        sample_weights = None
        if gpw != 1.0:
            sample_weights = np.ones(len(boards), dtype=np.float32)
            sample_weights[mask] = gpw

        phase_desc = ' -> '.join(f'{e}ep@a={a}' for e, a in REMAINING_PHASES)
        total_epochs = sum(e for e, _ in REMAINING_PHASES)

        print(f'{"="*65}')
        print(f'  [{pair_idx+1}/{len(pairs_to_train)}] {p.upper()}/{o.upper()} '
              f'({N_HIDDEN}h, gpw={gpw}, {total_epochs} epochs)')
        print(f'  Phases 3-4: {phase_desc}')
        print(f'  Resume from: {start_weights}')
        print(f'  Pair benchmark: {pair_bm.size()} scenarios')
        print(f'  Matching training positions: {n_match}/{len(boards)} ({100*n_match/len(boards):.1f}%)')
        print(f'{"="*65}')

        if not os.path.exists(start_weights):
            print(f'  ERROR: Starting weights not found: {start_weights}')
            continue

        current_weights = start_weights
        best_score = float('inf')
        best_epoch_total = 0
        total_time = 0.0
        # Epoch offset: phases 1+2 = 100+200 = 300 epochs already done
        epoch_offset = 300

        for phase_idx, (epochs, alpha) in enumerate(REMAINING_PHASES):
            phase_num = phase_idx + 3  # phases 3 and 4
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
                label=name,
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

            current_weights = save_path + '.best'
            epoch_offset += epochs

        all_results[name] = {
            'best_score': best_score,
            'best_epoch': best_epoch_total,
            'total_time': total_time,
            'gpw': gpw,
        }
        print(f'\n  FINAL: {name} -> best pair-filtered ER={best_score:.2f} '
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

    print(f'\n{"="*65}')
    print(f'  SL TRAINING SUMMARY ({elapsed:.0f}s = {elapsed/60:.1f}m)')
    print(f'{"="*65}\n')
    for name, r in all_results.items():
        print(f'  {name:20s}: pair ER={r["best_score"]:.2f} (epoch {r["best_epoch"]}), '
              f'gpw={r["gpw"]}, time={r["total_time"]:.0f}s')

    print('\nDone!')


if __name__ == '__main__':
    main()
