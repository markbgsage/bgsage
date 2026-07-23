#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Generate Paskogammon back-game training and benchmark position data.

Plays cubeless self-play money games from the Paskogammon starting position
(doubles ALLOWED on the opening roll; positive player always on roll) with the
production model at 1-ply, collecting positions that qualify as player or
opponent back games — the SAME category rules used for the Stage 9 back-game
NNs (generate_backgame_data.py):

  Player back game:   (anchoring, racing) game-plan pair, player behind in the
                      race (higher pip count), player holds >= 2 anchors in the
                      opponent's home board (points 19-24).
  Opponent back game: the mirror — (racing, anchoring), opponent behind,
                      opponent >= 2 anchors in the player's home board (1-6).

Every detected position is also flipped and recorded from the other
perspective, so the player and opponent files always hold mirror sets of
positions (rolled out separately: the rollout convention is post-move from the
positive player's perspective, so the two orientations are different targets).

The self-play game loop, RNG seeding, and seed namespaces are IDENTICAL to the
original single-filter version of this script — the same seeds replay the same
games, so positions that were already rolled out for the earlier
pasko-train/benchmark datasets can be reused as cached rollout results.

Training and benchmark sets are generated in SEPARATE self-play runs with
disjoint RNG seed namespaces; pass the benchmark files as --exclude-file when
generating the training sets to keep them disjoint (exclusion is applied to
both orientations of every excluded board).

Positions are stored exactly as encountered (on-roll perspective for the
detected orientation, its flip for the mirror), one per line as 26
space-separated ints — the same convention as generate_backgame_data.py.

Output, in bgsage/data/:
  pasko-player-backgame-train-data        pasko-player-backgame-benchmark-data
  pasko-opponent-backgame-train-data      pasko-opponent-backgame-benchmark-data

Roll these out next (from the PARENT repo, via Parallelizor):
  python scripts/rollout_pasko_positions.py pasko-player-backgame-benchmark-data --workers 750 --timeout 1800

Usage:
  # Benchmark sets only (20k per category):
  python bgsage/scripts/generate_pasko_data.py --train-target 0 --benchmark-target 20000

  # Training sets only (100k per category), disjoint from the benchmark sets:
  python bgsage/scripts/generate_pasko_data.py --train-target 100000 --benchmark-target 0 \
      --exclude-file pasko-player-backgame-benchmark-data \
      --exclude-file pasko-opponent-backgame-benchmark-data
"""

import os
import sys
import time
import random
import argparse
import multiprocessing as mp

# --- Path setup (runs in the main process AND every spawned worker) -------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../bgbot/bgsage
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)               # .../bgbot

if sys.platform == 'win32':
    _cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)

# Prefer bgsage/build (kept fresh by the MSVC copy step); the parent build/ may
# be held open by a running backend, so it is only a fallback.
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'python'))

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')

# Paskogammon starting position (positive player on roll). 15 checkers each side.
PASKO_START = [0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,
               5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2, 0]

# Disjoint seed namespaces for the two splits (Python ints are unbounded).
# UNCHANGED from the original single-filter script so games replay identically.
_SPLIT_SEED = {'train': 1 * 10**15, 'benchmark': 2 * 10**15}

# --- Worker-local strategy (set by the pool initializer) ------------------
_strategy = None


def pip_counts(board):
    """Return (player_pips, opponent_pips) for the board."""
    p1 = sum(i * board[i] for i in range(1, 25) if board[i] > 0) + 25 * board[25]
    p2 = sum((25 - i) * (-board[i]) for i in range(1, 25) if board[i] < 0) + 25 * board[0]
    return p1, p2


# --- S10 gate (matches backgame_gate in cpp/src/neural_net.cpp) -------------
# The Stage 10 Paskogammon backgame nets carry a non-zero blend weight ONLY
# inside this gate. With --gate, the new filter is the S9 backgame rules AND
# this gate — i.e. exactly the positions where the pasko weight > 0.
_GATE_MIN_ANCHORS = 3
_GATE_ARM2_BC = 7
_GATE_ARM2_PIPS = 200


def backgame_gate(anchors, back_checkers, pips):
    """True iff the backgame side is inside the S10 gate (pasko weight > 0)."""
    return (anchors >= _GATE_MIN_ANCHORS or
            (anchors == 2 and back_checkers >= _GATE_ARM2_BC and pips >= _GATE_ARM2_PIPS))


# --- Worker-local state (set by the pool initializer) ---------------------
_gate = False


def _init_worker(weight_paths, hidden_sizes, n_plies, gate):
    """Pool initializer — one production (stage9) strategy per worker."""
    global _strategy, _gate
    import bgbot_cpp
    _gate = gate
    _strategy = bgbot_cpp.create_multipy_stage9(
        weight_paths, hidden_sizes, n_plies=n_plies)


def play_games_worker(args):
    """Play n_games of cubeless self-play; return (player_bg, opponent_bg) positions.

    Classification matches generate_backgame_data.py exactly (the Stage 9
    back-game NN rules). Every detection records the on-roll board in its
    category AND the flipped board in the mirror category.

    The game loop (RNG consumption, move selection, flip cadence) is identical
    to the original single-filter version of this script.
    """
    seed, n_games = args
    import bgbot_cpp

    strategy = _strategy
    rng = random.Random(seed)
    player_bg = set()
    opponent_bg = set()

    for _ in range(n_games):
        board = list(PASKO_START)
        # Paskogammon opening: doubles ALLOWED (no re-roll), P1 always on roll.
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

        for _ in range(500):  # safety limit against pathological non-terminating play
            # --- Classify game plans for both sides ---
            p_gp = bgbot_cpp.classify_game_plan(board)
            p_plan = p_gp.name.lower() if hasattr(p_gp, 'name') else str(p_gp).split('.')[-1].lower()

            flipped = list(bgbot_cpp.flip_board(board))
            o_gp = bgbot_cpp.classify_game_plan(flipped)
            o_plan = o_gp.name.lower() if hasattr(o_gp, 'name') else str(o_gp).split('.')[-1].lower()

            # --- Player back game: (anchoring, racing), player behind, >=2 anchors in opp home ---
            if p_plan == 'anchoring' and o_plan == 'racing':
                pip1, pip2 = pip_counts(board)
                anchors = sum(1 for i in range(19, 25) if board[i] > 1)
                if pip1 > pip2 and anchors > 1:
                    # back checkers = player's bar + all player checkers on 19-24
                    back = board[25] + sum(board[i] for i in range(19, 25) if board[i] > 0)
                    if not _gate or backgame_gate(anchors, back, pip1):
                        player_bg.add(tuple(board))
                        opponent_bg.add(tuple(flipped))

            # --- Opponent back game: (racing, anchoring), opp behind, >=2 anchors in player home ---
            elif p_plan == 'racing' and o_plan == 'anchoring':
                pip1, pip2 = pip_counts(board)
                anchors = sum(1 for i in range(1, 7) if board[i] < -1)
                if pip2 > pip1 and anchors > 1:
                    # back checkers = opponent's bar + all opponent checkers on 1-6
                    back = board[0] + sum(-board[i] for i in range(1, 7) if board[i] < 0)
                    if not _gate or backgame_gate(anchors, back, pip2):
                        opponent_bg.add(tuple(board))
                        player_bg.add(tuple(flipped))

            candidates = bgbot_cpp.possible_moves(board, d1, d2)
            if candidates:
                best_idx = strategy.best_move_index(candidates, board)
                board = list(candidates[best_idx])

            if bgbot_cpp.check_game_over(board) != 0:
                break

            board = list(bgbot_cpp.flip_board(board))
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

    return [list(p) for p in player_bg], [list(p) for p in opponent_bg]


def collect(pool, split, target, n_workers, games_per_cycle, exclude):
    """Collect `target` unique positions PER CATEGORY for a split.

    Returns (player_bg, opponent_bg) sets. Positions present in `exclude` (a
    set of tuples; callers include both orientations) are never added.

    Because every detection records both orientations, the two categories grow
    in lockstep and stay the same size.
    """
    seed_base = _SPLIT_SEED[split]
    player_bg = set()
    opponent_bg = set()
    cycle = 0
    t0 = time.time()
    while min(len(player_bg), len(opponent_bg)) < target:
        # Prime-spread seeds within a large per-split namespace so train and
        # benchmark streams are disjoint and adjacent seeds are uncorrelated.
        seeds = [
            (seed_base + (cycle * n_workers + i) * 1000003, games_per_cycle)
            for i in range(n_workers)
        ]
        for p_positions, o_positions in pool.map(play_games_worker, seeds):
            for p in p_positions:
                t = tuple(p)
                if t not in exclude:
                    player_bg.add(t)
            for p in o_positions:
                t = tuple(p)
                if t not in exclude:
                    opponent_bg.add(t)
        cycle += 1
        total_games = cycle * n_workers * games_per_cycle
        elapsed = time.time() - t0
        rate = total_games / elapsed if elapsed > 0 else 0
        print(f'  [{split}] cycle {cycle:3d}: {total_games:>9,} games | '
              f'player_bg={len(player_bg):>7,}/{target:,} | '
              f'opponent_bg={len(opponent_bg):>7,}/{target:,} | '
              f'{elapsed:6.1f}s ({rate:.0f} games/s)', flush=True)
    return player_bg, opponent_bg


def write_positions(filepath, positions):
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(' '.join(str(x) for x in pos) + '\n')
    print(f'  {os.path.basename(filepath)}: {len(positions):,} positions')


def read_position_set(path):
    """Load a positions file (26 space-separated ints per line) into a set of tuples."""
    positions = set()
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 26:
                positions.add(tuple(int(x) for x in parts[:26]))
    return positions


def _shuffle_cap(positions, target):
    """Deterministic shuffle (order uncorrelated with game order), cap to target."""
    rng = random.Random(12345)
    lst = [list(p) for p in positions]
    rng.shuffle(lst)
    return lst[:target]


def main():
    parser = argparse.ArgumentParser(
        description='Generate Paskogammon back-game train/benchmark positions via cubeless self-play')
    parser.add_argument('--train-target', type=int, default=0,
                        help='Unique training positions PER CATEGORY (default: 0 = skip)')
    parser.add_argument('--benchmark-target', type=int, default=0,
                        help='Unique benchmark positions PER CATEGORY (default: 0 = skip)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Parallel self-play worker processes (default: 32)')
    parser.add_argument('--games-per-cycle', type=int, default=100,
                        help='Games per worker per cycle (default: 100)')
    parser.add_argument('--plies', type=int, default=1,
                        help='Self-play move-selection ply (default: 1)')
    parser.add_argument('--gate', action='store_true',
                        help='Filter by the S10 gate on top of the S9 backgame rules: '
                             'keep only positions where the Paskogammon blend weight > 0 '
                             '(see backgame_gate). Default off = original S9-only filter.')
    parser.add_argument('--out-prefix', default='pasko',
                        help="Output filename prefix (default: 'pasko'). Use e.g. "
                             "'pasko-gated' to write a new dataset without clobbering "
                             "the existing pasko-* files.")
    parser.add_argument('--exclude-file', action='append', default=[],
                        help='Positions file (name in bgsage/data/, or an absolute path) '
                             'whose positions — in BOTH orientations — are excluded from '
                             'all generated sets. Repeatable. Pass the benchmark files '
                             'when generating training sets to keep them disjoint.')
    args = parser.parse_args()

    import bgbot_cpp
    from bgsage.weights import WeightConfigPair
    w = WeightConfigPair.from_model('stage9')  # production model
    w.validate()
    weight_paths, hidden_sizes = w.weight_args

    os.makedirs(DATA_DIR, exist_ok=True)

    # Exclusion set: every excluded board plus its flip, so no generated set can
    # share a physical position with an excluded file in either orientation.
    base_exclude = set()
    for xfile in args.exclude_file:
        xpath = xfile if os.path.isabs(xfile) else os.path.join(DATA_DIR, xfile)
        loaded = read_position_set(xpath)
        for b in loaded:
            base_exclude.add(b)
            base_exclude.add(tuple(bgbot_cpp.flip_board(list(b))))
        print(f'Excluding {len(loaded):,} positions (+flips) from {os.path.basename(xpath)}')
    base_exclude = frozenset(base_exclude)

    print(f'Model: production stage9 ({len(weight_paths)} NNs), {args.plies}-ply self-play')
    print(f'Filter: S9 backgame rules{" + S10 gate (pasko weight > 0)" if args.gate else ""}')
    print(f'Output prefix: {args.out_prefix}')
    print(f'Targets per category: train={args.train_target:,}  benchmark={args.benchmark_target:,}')
    print(f'Workers: {args.workers}, games/worker/cycle: {args.games_per_cycle}')
    print(flush=True)

    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(weight_paths, hidden_sizes, args.plies, args.gate)) as pool:
        train_p, train_o = set(), set()
        if args.train_target > 0:
            print('Collecting TRAIN positions...')
            train_p, train_o = collect(pool, 'train', args.train_target, args.workers,
                                       args.games_per_cycle, exclude=base_exclude)
        bench_p, bench_o = set(), set()
        if args.benchmark_target > 0:
            excl = base_exclude | train_p | train_o  # disjoint from train + exclude-files
            print(f'Collecting BENCHMARK positions (disjoint seeds, '
                  f'excluding {len(excl):,} train/exclude-file positions)...')
            bench_p, bench_o = collect(pool, 'benchmark', args.benchmark_target, args.workers,
                                       args.games_per_cycle, exclude=excl)

    pfx = args.out_prefix
    print(f'\nWriting to {DATA_DIR}:')
    if args.train_target > 0:
        write_positions(os.path.join(DATA_DIR, f'{pfx}-player-backgame-train-data'),
                        _shuffle_cap(train_p, args.train_target))
        write_positions(os.path.join(DATA_DIR, f'{pfx}-opponent-backgame-train-data'),
                        _shuffle_cap(train_o, args.train_target))
    if args.benchmark_target > 0:
        write_positions(os.path.join(DATA_DIR, f'{pfx}-player-backgame-benchmark-data'),
                        _shuffle_cap(bench_p, args.benchmark_target))
        write_positions(os.path.join(DATA_DIR, f'{pfx}-opponent-backgame-benchmark-data'),
                        _shuffle_cap(bench_o, args.benchmark_target))
    print(f'\nTotal time: {time.time() - t0:.1f}s')
    print('\nNext (roll out via Parallelizor, from the PARENT repo):')
    print(f'  python scripts/rollout_pasko_positions.py {pfx}-player-backgame-benchmark-data --workers 750 --timeout 1800')
    print(f'  python scripts/rollout_pasko_positions.py {pfx}-opponent-backgame-benchmark-data --workers 750 --timeout 1800')


if __name__ == '__main__':
    mp.freeze_support()
    main()
