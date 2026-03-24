"""Measure (player, opponent) game plan pair frequencies in training data.

Prints a 4x4 table of ordered pair frequencies for contact positions,
plus purerace frequency. Used to inform gpw tuning for Stage 7.
"""

import os
import sys
import time
import numpy as np

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
    os.path.join(project_dir, 'build_msvc'),
    os.path.join(project_dir, 'build'),
    os.path.join(project_dir, 'build_msvc_s7'),
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
sys.path.insert(0, os.path.join(bgsage_dir, 'python'))

import bgbot_cpp
from bgsage.data import load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')

PLAN_NAMES = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
CONTACT_NAMES = ['racing', 'attacking', 'priming', 'anchoring']


def flip_boards_numpy(boards):
    """Flip a batch of boards (numpy array [N, 26])."""
    flipped = np.zeros_like(boards)
    flipped[:, 0] = boards[:, 25]       # P2 bar
    flipped[:, 25] = boards[:, 0]       # P1 bar
    for i in range(1, 25):
        flipped[:, i] = -boards[:, 25 - i]  # negate and reverse points
    return flipped


def main():
    # Load contact + crashed training data
    print('Loading training data...')
    t0 = time.time()
    boards_c, _ = load_gnubg_training_data(os.path.join(DATA_DIR, 'contact-train-data'))
    print(f'  contact-train-data: {len(boards_c)} positions')
    boards_r, _ = load_gnubg_training_data(os.path.join(DATA_DIR, 'crashed-train-data'))
    print(f'  crashed-train-data: {len(boards_r)} positions')
    boards = np.concatenate([boards_c, boards_r], axis=0)
    print(f'  Total: {len(boards)} positions ({time.time()-t0:.1f}s)')

    # Classify player plans
    print('\nClassifying player game plans...')
    t0 = time.time()
    player_gps = bgbot_cpp.classify_game_plans_batch(boards)
    print(f'  Done in {time.time()-t0:.1f}s')

    # Classify opponent plans (flip boards first)
    print('Flipping boards and classifying opponent game plans...')
    t0 = time.time()
    flipped = flip_boards_numpy(boards)
    opp_gps = bgbot_cpp.classify_game_plans_batch(flipped)
    print(f'  Done in {time.time()-t0:.1f}s')

    n = len(boards)

    # Count purerace (player_gp == 0)
    n_purerace = int(np.sum(player_gps == 0))
    print(f'\nPureRace positions: {n_purerace}/{n} ({100*n_purerace/n:.1f}%)')

    # Count per-plan (player only, for comparison with S5/S6)
    print('\n--- Player plan distribution (for comparison) ---')
    for gp_id, name in enumerate(PLAN_NAMES):
        cnt = int(np.sum(player_gps == gp_id))
        print(f'  {name:12s}: {cnt:7d} ({100*cnt/n:.1f}%)')

    # Count ordered (player, opponent) pairs for contact positions
    contact_mask = player_gps > 0  # exclude purerace
    n_contact = int(np.sum(contact_mask))
    p_gps = player_gps[contact_mask]
    o_gps = opp_gps[contact_mask]

    print(f'\n--- Ordered (player, opponent) pair frequencies ({n_contact} contact positions) ---')
    print(f'{"":12s}  {"Racing":>8s}  {"Attacking":>10s}  {"Priming":>8s}  {"Anchoring":>10s}  {"Total":>7s}')

    pair_counts = {}
    for p_id, p_name in enumerate(CONTACT_NAMES, start=1):
        row_counts = []
        for o_id, o_name in enumerate(CONTACT_NAMES, start=1):
            cnt = int(np.sum((p_gps == p_id) & (o_gps == o_id)))
            pair_name = f'{p_name[:4]}_{o_name[:4]}'
            pair_counts[pair_name] = cnt
            row_counts.append(cnt)
        row_total = sum(row_counts)
        pcts = [f'{100*c/n_contact:.1f}%' for c in row_counts]
        print(f'  {p_name:10s}  {row_counts[0]:5d} ({pcts[0]:>5s})  '
              f'{row_counts[1]:5d} ({pcts[1]:>5s})  '
              f'{row_counts[2]:5d} ({pcts[2]:>5s})  '
              f'{row_counts[3]:5d} ({pcts[3]:>5s})  '
              f'{row_total:5d} ({100*row_total/n_contact:.1f}%)')

    # Also check: what fraction of "contact" positions have a purerace opponent?
    n_contact_opp_pr = int(np.sum((player_gps > 0) & (opp_gps == 0)))
    print(f'\nContact player with PureRace opponent: {n_contact_opp_pr}/{n_contact} ({100*n_contact_opp_pr/n_contact:.1f}%)')

    # Print suggested gpw values
    print('\n--- Suggested gpw ranges (target ~40-50% effective gradient) ---')
    for pair_name, cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        freq = cnt / n_contact
        # gpw for 50% target: g = (1-f)/f
        gpw_50 = (1 - freq) / freq if freq > 0 else 999
        print(f'  {pair_name:12s}: {cnt:5d} ({100*freq:.1f}%), gpw for 50% = {gpw_50:.1f}')


if __name__ == '__main__':
    main()
