#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest snake positions for rollout (training targets for the snake NN).

The snake region — a far-side prime trapping a straggler against a crunched
board (scripts/snake_rule.py) — has essentially no rows anywhere: ~100 in
the 1.8M-row GNUbg corpus, ~36 in the Stage 11 piles, none in the money or
pasko benchmarks. Everything the net learns from has to be generated, and
plain self-play from the benchmark seeds cannot do it: measured 2026-09-03,
stage11p (snake PR ~54) breaks the prime or runs within a move or two, so
900 games yield ~2 snake decisions each.

So the region is sampled directly. Random synthetic snakes — prime length
and location, spares spread from the far side to home, a crunched opponent
with 0-5 off and 1-3 stragglers — seed short 2-ply self-play games in which
the HOLDER prefers structure-keeping moves whenever it has one (the crunched
side plays normally), so trajectories stay in the region for a dozen
half-moves instead of one. For every recorded decision whose pre-move board
is a snake we keep

* the pre-move board FLIPPED (the post-move convention the rollout runner
  and the SL trainer use: the side that just moved is positive), and
* for a checker decision, the top ``--top`` candidates' post-move boards
  plus up to ``--exits`` of the best-ranked candidates that LEAVE the
  region (the prime breaks or the straggler is released). Routing is by the
  pre-move board, so the snake NN is the one asked to value exactly those
  release moves; the rollouts of both branches are what teach it when to
  hold and when to let go.

Every board that appears among the benchmark's decisions or their candidate
moves — both orientations — is excluded.

Output: data/s11-bg-snake-data (26 ints per line), for
    python scripts/rollout_backgame_positions.py s11-bg-snake-data ...
(parent repo), which appends the rolled-out probabilities as
data/s11-bg-snake-rollout.

Usage:
    py -3.14 scripts/harvest_snake_positions.py --seeds 5000 --workers 6 --limit 44000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    os.add_dll_directory(str(_PROJECT_ROOT / "build"))

_DATA = _PROJECT_ROOT / "data"
_BENCH = _PROJECT_ROOT / "backgame_ref_positions" / "benchmark"
MASTER_SEED = 11         # the benchmark folder used seed 1; containment used 7
LEVEL = "2ply"
MODEL = "stage11p"
MAX_HALF_MOVES = 24      # per seed game; the region usually ends sooner


def _flip(b):
    return [b[25]] + [-b[25 - i] for i in range(1, 25)] + [b[0]]


def excluded_boards() -> set:
    """Benchmark decision + candidate boards, both orientations."""
    ex = set()
    for path in _BENCH.glob("* rollout.jsonl"):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            for b in [r["board"], *[m["board"] for m in r.get("moves", [])]]:
                ex.add(tuple(b))
                ex.add(tuple(_flip(b)))
    for name in ("money_benchmark", "pasko_money_benchmark"):
        for d in json.load((_DATA / name / "benchmark.json").open(encoding="utf-8"))["decisions"]:
            ex.add(tuple(d["board"]))
            ex.add(tuple(_flip(d["board"])))
    return ex


def synthetic_snake(rng: random.Random, snake) -> list[int] | None:
    """One random snake in the player-1 frame (P1 holds the prime), or None."""
    board = [0] * 26
    length = rng.choice((4, 4, 5, 5, 6))
    start = rng.randint(13, 24 - length + 1)          # prime entirely on 13..24
    prime = list(range(start, start + length))
    for pt in prime:
        board[pt] = 2
    # P1's other checkers: mostly extra on the prime and on the far side,
    # some already brought round to the outer and home boards.
    spares = 15 - 2 * length
    far = [p for p in range(13, 25) if p not in prime]
    outer, home = list(range(7, 13)), list(range(1, 7))
    for _ in range(spares):
        r = rng.random()
        if r < 0.40:
            pt = rng.choice(prime)
        elif r < 0.70:
            pt = rng.choice(far)
        elif r < 0.85:
            pt = rng.choice(outer)
        else:
            pt = rng.choice(home)
        board[pt] += 1
    # P2: crunched at home, 0-5 borne off, 1-3 stragglers on the bar or in
    # P1's home board, occasionally one more checker stuck on its way round.
    off = rng.choice((0, 0, 0, 1, 2, 3, 4, 5))
    stragglers = rng.choice((1, 1, 1, 2, 2, 3))
    extra = 1 if rng.random() < 0.15 else 0
    n_home = 15 - off - stragglers - extra
    if n_home < 10:
        return None
    home_pts = [p for p in range(19, 25) if board[p] == 0]
    if not home_pts:
        return None
    # Crunch: weight the deep points heavily.
    weights = {24: 6, 23: 5, 22: 3, 21: 2, 20: 1, 19: 1}
    for _ in range(n_home):
        pt = rng.choices(home_pts, weights=[weights[p] for p in home_pts])[0]
        board[pt] -= 1
    for _ in range(stragglers):
        r = rng.random()
        if r < 0.35:
            board[0] += 1                                # on the bar
        else:
            free = [p for p in range(1, 7) if board[p] <= 0]
            if not free:
                return None
            board[rng.choice(free)] -= 1
    if extra:
        free = [p for p in range(7, 19) if board[p] <= 0]
        if free:
            board[rng.choice(free)] -= 1
    if sum(b for b in board if b > 0) != 15 or -sum(b for b in board if b < 0) + off != 15:
        return None
    return board if snake(board) else None


def play_worker(args):
    """Seeds [start, start+count): synthesise, play, harvest. Returns (boards, n_decisions)."""
    start, count, top, exits = args
    from bgsage import BgBotAnalyzer, check_game_over, flip_board, is_race, possible_moves
    from bgsage.weights import WeightConfig
    import snake_rule as sr

    analyzer = BgBotAnalyzer(weights=WeightConfig.from_model(MODEL),
                             eval_level=LEVEL, cubeful=True, parallel_threads=4)
    found: list[tuple[int, ...]] = []
    n_dec = 0

    for g in range(start, start + count):
        rng = random.Random(f"{MASTER_SEED}:snake:seed{g}")
        board = None
        for _ in range(50):
            board = synthetic_snake(rng, sr.snake)
            if board is not None:
                break
        if board is None:
            continue
        holder_positive = True                    # P1 holds the prime at the start
        if rng.random() < 0.5:                    # the crunched side moves first
            board = flip_board(board)
            holder_positive = False
        for _ in range(MAX_HALF_MOVES):
            if is_race(board) or not sr.snake(board):
                break
            die1, die2 = rng.randint(1, 6), rng.randint(1, 6)
            moves = []
            if possible_moves(board, die1, die2):
                analyzer.set_seed(rng.getrandbits(31))
                moves = analyzer.checker_play(board, die1, die2, cube_value=1,
                                              cube_owner="centered", jacoby=True,
                                              beaver=True).moves
            if moves:
                n_dec += 1
                found.append(tuple(_flip(board)))
                stays = [sr.snake(list(m.board)) for m in moves]
                n_exit = 0
                for i, (m, s) in enumerate(zip(moves, stays)):
                    if i < top or (not s and n_exit < exits):
                        found.append(tuple(m.board))
                        if not s and i >= top:
                            n_exit += 1
                # The holder keeps the structure when it can; the crunched
                # side (and a holder with no keeping move) plays the 2-ply best.
                pick = moves[0]
                if holder_positive:
                    for m, s in zip(moves, stays):
                        if s:
                            pick = m
                            break
                board = list(pick.board)
            if check_game_over(board) != 0:
                break
            board = flip_board(board)
            holder_positive = not holder_positive
    return found, n_dec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=5, help="Top 2-ply candidates kept per decision")
    parser.add_argument("--exits", type=int, default=3,
                        help="Best-ranked region-leaving candidates kept beyond the top ones")
    parser.add_argument("--limit", type=int, default=44_000)
    args = parser.parse_args()

    ex = excluded_boards()
    print(f"exclusion set: {len(ex)} boards", flush=True)

    per = args.seeds // args.workers
    jobs = [(i * per, per, args.top, args.exits) for i in range(args.workers)]
    fresh: list[tuple[int, ...]] = []
    n_dec = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for boards, dec in pool.map(play_worker, jobs):
            fresh.extend(boards)
            n_dec += dec
    # The folder's own seed positions are known-good snakes: they always go in,
    # both orientations, ahead of the harvest.
    from backgame_benchmark import read_start_positions
    import snake_rule as sr
    seeds_in = []
    for st in read_start_positions("snake"):
        for b in (tuple(_flip(list(st.board))), tuple(st.board)):
            if sr.snake(list(b)):
                seeds_in.append(b)
    n_raw = len(fresh)
    fresh = [b for b in dict.fromkeys(seeds_in + fresh) if b not in ex]
    print(f"self-play from {args.seeds} synthetic seeds: {n_dec} snake decisions, {n_raw} boards "
          f"-> {len(fresh)} distinct boards outside the benchmark", flush=True)

    rng = random.Random(MASTER_SEED)
    rng.shuffle(fresh)
    out = fresh[:args.limit]
    path = _DATA / "s11-bg-snake-data"
    with path.open("w", encoding="utf-8") as f:
        for b in out:
            f.write(" ".join(str(x) for x in b) + "\n")
    print(f"wrote {len(out)} positions -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
