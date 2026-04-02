#!/usr/bin/env python3
"""Generate back game position data for training and benchmarking.

Plays cubeless self-play games using Stage 8 pair model at 1-ply with 32
parallel processes, collecting positions that qualify as player or opponent
back games.

Player back game: (anchoring, racing) pair, player behind in race (higher
pip count), player holds >= 2 anchors in opponent's home board (points 19-24).

Opponent back game: (racing, anchoring) pair, opponent behind in race,
opponent holds >= 2 anchors in player's home board (points 1-6).

Every detected position is also flipped and recorded from the other
perspective, so player_bg and opponent_bg counts are always equal.

Output: Four files in bgsage/data/:
  player-backgame-train-data       (~90% of player BG positions)
  player-backgame-benchmark-data   (~10% of player BG positions)
  opponent-backgame-train-data     (~90% of opponent BG positions)
  opponent-backgame-benchmark-data (~10% of opponent BG positions)

Each file: one position per line, 26 space-separated integers.

Usage:
  python bgsage/scripts/generate_backgame_data.py
"""

import os
import sys
import time
import random
import multiprocessing as mp

# --- Path setup (runs in both main and spawned worker processes) ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

# --- Constants ------------------------------------------------------------
TARGET_POSITIONS = 100_000
N_WORKERS = 32
GAMES_PER_CYCLE = 100

STARTING_BOARD = [
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')

# --- Worker-local state (set by pool initializer) -------------------------
_strategy = None


def pip_counts(board):
    """Return (player_pips, opponent_pips) for the board."""
    p1 = sum(i * board[i] for i in range(1, 25) if board[i] > 0) + 25 * board[25]
    p2 = sum((25 - i) * (-board[i]) for i in range(1, 25) if board[i] < 0) + 25 * board[0]
    return p1, p2


def _init_worker(weight_paths, hidden_sizes):
    """Pool initializer — create one 1-ply pair strategy per worker."""
    global _strategy
    import bgbot_cpp
    _strategy = bgbot_cpp.create_multipy_pair(
        weight_paths, hidden_sizes, n_plies=1)


def play_games_worker(args):
    """Play n_games of cubeless self-play, return back game positions."""
    seed, n_games = args
    import bgbot_cpp

    strategy = _strategy
    rng = random.Random(seed)
    player_bg = set()
    opponent_bg = set()

    for _ in range(n_games):
        board = list(STARTING_BOARD)

        # Roll for first turn (re-roll ties)
        while True:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if d1 != d2:
                break

        for _ in range(500):  # safety limit
            # --- Classify game plans for both sides ---
            p_gp = bgbot_cpp.classify_game_plan(board)
            p_plan = p_gp.name.lower() if hasattr(p_gp, 'name') else str(p_gp).split('.')[-1].lower()

            flipped = list(bgbot_cpp.flip_board(board))
            o_gp = bgbot_cpp.classify_game_plan(flipped)
            o_plan = o_gp.name.lower() if hasattr(o_gp, 'name') else str(o_gp).split('.')[-1].lower()

            # --- Player back game: (anchoring, racing), player behind, >=2 anchors in opp home ---
            if p_plan == 'anchoring' and o_plan == 'racing':
                pip1, pip2 = pip_counts(board)
                if pip1 > pip2 and sum(1 for i in range(19, 25) if board[i] > 1) > 1:
                    player_bg.add(tuple(board))
                    opponent_bg.add(tuple(flipped))

            # --- Opponent back game: (racing, anchoring), opp behind, >=2 anchors in player home ---
            elif p_plan == 'racing' and o_plan == 'anchoring':
                pip1, pip2 = pip_counts(board)
                if pip2 > pip1 and sum(1 for i in range(1, 7) if board[i] < -1) > 1:
                    opponent_bg.add(tuple(board))
                    player_bg.add(tuple(flipped))

            # --- Make best move ---
            candidates = bgbot_cpp.possible_moves(board, d1, d2)
            if candidates:
                best_idx = strategy.best_move_index(candidates, board)
                board = list(candidates[best_idx])

            # --- Check game over ---
            if bgbot_cpp.check_game_over(board) != 0:
                break

            # --- Flip for opponent's turn ---
            board = list(bgbot_cpp.flip_board(board))
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)

    return [list(p) for p in player_bg], [list(p) for p in opponent_bg]


def write_positions(filepath, positions):
    """Write positions to file: one per line, 26 space-separated ints."""
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(' '.join(str(x) for x in pos) + '\n')
    print(f'  {os.path.basename(filepath)}: {len(positions):,} positions')


def main():
    from bgsage.weights import WeightConfigPair

    w = WeightConfigPair.from_model('stage8')
    w.validate()
    weight_paths, hidden_sizes = w.weight_args
    print(f'Model: stage8 ({len(weight_paths)} NNs)')
    print(f'Target: {TARGET_POSITIONS:,} unique positions per type')
    print(f'Workers: {N_WORKERS}, games/worker/cycle: {GAMES_PER_CYCLE}')
    print()

    all_player_bg = set()
    all_opponent_bg = set()
    cycle = 0
    start_time = time.time()

    with mp.Pool(N_WORKERS, initializer=_init_worker,
                 initargs=(weight_paths, hidden_sizes)) as pool:
        while min(len(all_player_bg), len(all_opponent_bg)) < TARGET_POSITIONS:
            seeds = [
                (42 + cycle * N_WORKERS * 1000003 + i * 1000003, GAMES_PER_CYCLE)
                for i in range(N_WORKERS)
            ]
            results = pool.map(play_games_worker, seeds)

            for player_positions, opponent_positions in results:
                all_player_bg.update(tuple(p) for p in player_positions)
                all_opponent_bg.update(tuple(p) for p in opponent_positions)

            cycle += 1
            total_games = cycle * N_WORKERS * GAMES_PER_CYCLE
            elapsed = time.time() - start_time
            rate = total_games / elapsed if elapsed > 0 else 0
            print(f'Cycle {cycle:3d}: {total_games:>8,} games | '
                  f'player_bg={len(all_player_bg):>7,} | '
                  f'opponent_bg={len(all_opponent_bg):>7,} | '
                  f'{elapsed:6.1f}s ({rate:.0f} games/s)')

    elapsed = time.time() - start_time
    total_games = cycle * N_WORKERS * GAMES_PER_CYCLE
    print(f'\nCollection done: {total_games:,} games in {elapsed:.1f}s')
    print(f'Unique: player_bg={len(all_player_bg):,}, '
          f'opponent_bg={len(all_opponent_bg):,}')

    # Shuffle deterministically
    rng = random.Random(12345)
    player_list = [list(p) for p in all_player_bg]
    opponent_list = [list(p) for p in all_opponent_bg]
    rng.shuffle(player_list)
    rng.shuffle(opponent_list)

    # 90/10 train/benchmark split
    sp = int(len(player_list) * 0.9)
    so = int(len(opponent_list) * 0.9)

    print(f'\nWriting to {DATA_DIR}:')
    write_positions(os.path.join(DATA_DIR, 'player-backgame-train-data'),
                    player_list[:sp])
    write_positions(os.path.join(DATA_DIR, 'player-backgame-benchmark-data'),
                    player_list[sp:])
    write_positions(os.path.join(DATA_DIR, 'opponent-backgame-train-data'),
                    opponent_list[:so])
    write_positions(os.path.join(DATA_DIR, 'opponent-backgame-benchmark-data'),
                    opponent_list[so:])

    print(f'\nTotal time: {time.time() - start_time:.1f}s')


if __name__ == '__main__':
    mp.freeze_support()
    main()
