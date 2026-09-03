#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest snake positions for rollout (training targets for the snake NN).

The snake region — a far-side prime trapping a straggler against a crunched
board (scripts/snake_rule.py) — has essentially no rows anywhere: ~100 in
the 1.8M-row GNUbg corpus, ~36 in the Stage 11 piles, none in the money or
pasko benchmarks. Everything the net learns from has to be generated.

Positions come from self-play games seeded from the snake seed file (the
same seeds the benchmark used, but a different master seed, and every board
that appears among the benchmark's decisions or their candidate moves — both
orientations — is excluded). Games are played 2-ply by stage11p in N worker
processes. For every recorded decision whose pre-move board is a snake:

* the pre-move board FLIPPED (the post-move convention the rollout runner
  and the SL trainer use: the side that just moved is positive), and
* for a checker decision, the top ``--top`` candidates' post-move boards
  plus up to ``--exits`` of the best-ranked candidates that LEAVE the
  region (the prime breaks or the straggler is released). Routing is by the
  pre-move board, so the snake NN is the one asked to value exactly those
  release moves; a net that never saw them extrapolates (the 2026-08
  "flee/wedge" lesson).

Output: data/s11-bg-snake-data (26 ints per line), for
    python scripts/rollout_backgame_positions.py s11-bg-snake-data ...
(parent repo), which appends the rolled-out probabilities as
data/s11-bg-snake-rollout.

Usage:
    py -3.14 scripts/harvest_snake_positions.py --games 900 --workers 6 --limit 44000
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
FOLDER = "snake"


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


def play_worker(args):
    """Play games [start, start+count); return (boards, n_decisions)."""
    start, count, top, exits = args
    from backgame_benchmark import _play_game, _start_for_game, read_start_positions
    from bgsage import BgBotAnalyzer
    from bgsage.weights import WeightConfig
    import snake_rule as sr

    starts = read_start_positions(FOLDER)
    analyzer = BgBotAnalyzer(weights=WeightConfig.from_model(MODEL),
                             eval_level=LEVEL, cubeful=True, parallel_threads=4)
    found: list[tuple[int, ...]] = []
    n_dec = 0

    def record(decision) -> bool:
        nonlocal n_dec
        board = list(decision.board)
        if not sr.snake(board):
            return True
        n_dec += 1
        found.append(tuple(_flip(board)))
        if decision.kind == "checker":
            d1, d2 = decision.dice
            res = analyzer.checker_play(board, d1, d2, cube_value=decision.cube_value,
                                        cube_owner=decision.cube_owner, jacoby=True,
                                        beaver=True)
            n_exit = 0
            for i, m in enumerate(res.moves):
                stays = sr.snake(list(m.board))
                if i < top or (not stays and n_exit < exits):
                    found.append(tuple(m.board))
                    n_exit += (not stays) and i >= top
        return True

    for g in range(start, start + count):
        st = _start_for_game(starts, g, MASTER_SEED, FOLDER)
        rng = random.Random(f"{MASTER_SEED}:{FOLDER}:game{g}")
        _play_game(st, rng, analyzer, sr.snake, record)
    return found, n_dec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=900)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=5, help="Top 2-ply candidates kept per decision")
    parser.add_argument("--exits", type=int, default=3,
                        help="Best-ranked region-leaving candidates kept beyond the top ones")
    parser.add_argument("--limit", type=int, default=44_000)
    args = parser.parse_args()

    ex = excluded_boards()
    print(f"exclusion set: {len(ex)} boards", flush=True)

    per = args.games // args.workers
    jobs = [(i * per, per, args.top, args.exits) for i in range(args.workers)]
    fresh: list[tuple[int, ...]] = []
    n_dec = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for boards, dec in pool.map(play_worker, jobs):
            fresh.extend(boards)
            n_dec += dec
    n_raw = len(fresh)
    fresh = [b for b in dict.fromkeys(fresh) if b not in ex]
    print(f"self-play: {n_dec} snake decisions, {n_raw} boards -> {len(fresh)} distinct "
          f"boards outside the benchmark", flush=True)

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
