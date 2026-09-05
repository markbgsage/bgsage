#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest the positions a 2-ply search reaches BELOW a snake release move.

Why (measured 2026-09-04, see CLAUDE.md "Stage 11s"): the snake NN is trained
on every candidate of a snake decision — the boards that keep the prime and
the "exit" boards that release it — so at 1-ply it values them all well. A
2-ply search, though, values a candidate through the opponent's best reply,
and the replies to an exit board are one half-move beyond everything the
harvest rolled out. No net is any good there (the standard nets score RMSE
0.32 on the exit rows themselves), so release candidates get noisy 2-ply
values while holding candidates get accurate ones, and the max over the
candidates picks the noise: snake PR 15.6 at 1-ply, 35.4 at 2-ply.

This script generates those "exit+1" positions so they can be rolled out and
added to the snake NN's training data. For every exit board in the harvest
(`data/s11-bg-snake-data`, the side that just released positive) it rolls one
random pair of dice for the opponent and keeps the opponent's best reply
under the routing the retrained net will be searched with — ROOT routing, the
snake net valuing the whole tree — and, when it differs, the best reply under
the engine's current per-node routing too, so the rows cover the leaves either
search actually visits. Replies are post-move boards with the replier
positive, exactly what the rollout runner and the SL trainer expect.

Excluded: races (the pure-race net handles those whatever the routing), game-
over boards, boards already in the harvest, and every board of the snake
benchmark's decisions and candidates in both orientations.

Output: data/s11-bg-snake-exit1-data (26 ints per line) and a
.meta.jsonl sidecar (parent index, dice, which routing chose the reply), for
    python scripts/rollout_backgame_positions.py s11-bg-snake-exit1-data \\
        --backend batch --workers 64 --n-trials 648 --checker-ply 3 --timeout 1800
(parent repo), which appends the rolled-out probabilities as
data/s11-bg-snake-exit1-rollout.

Usage:
    py -3.14 scripts/harvest_snake_exit_replies.py [--model stage11] [--engine-cap 4000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    if (_PROJECT_ROOT / "build").is_dir():
        os.add_dll_directory(str(_PROJECT_ROOT / "build"))

import bgbot_cpp  # noqa: E402
from bgsage.weights import WeightConfigPair  # noqa: E402
import snake_rule as sr  # noqa: E402

DATA = _PROJECT_ROOT / "data"
BENCH = _PROJECT_ROOT / "backgame_ref_positions" / "benchmark"
HARVEST = DATA / "s11-bg-snake-data"
OUT = DATA / "s11-bg-snake-exit1-data"
MASTER_SEED = 11


def _flip(b):
    return [b[25]] + [-b[25 - i] for i in range(1, 25)] + [b[0]]


def load_boards(path: Path) -> list[list[int]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 26:
            out.append([int(x) for x in parts[:26]])
    return out


def benchmark_boards() -> set[tuple[int, ...]]:
    ex: set[tuple[int, ...]] = set()
    for name in ("snake rollout.jsonl", "snake s11play rollout.jsonl", "snake candidates rollout.jsonl"):
        p = BENCH / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for b in [r["board"], *[m["board"] for m in r.get("moves", [])]]:
                ex.add(tuple(b))
                ex.add(tuple(_flip(b)))
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stage11")
    ap.add_argument("--engine-cap", type=int, default=4000,
                    help="At most this many parents also contribute the engine-routed best reply when it differs")
    args = ap.parse_args()

    harvest = load_boards(HARVEST)
    seen = {tuple(b) for b in harvest}
    exclude = benchmark_boards()
    parents = [b for b in harvest if not sr.snake(b) and not bgbot_cpp.is_race(b)
               and bgbot_cpp.check_game_over(b) == 0]
    print(f"{len(harvest)} harvest boards, {len(parents)} exit parents, {len(exclude)} benchmark boards excluded")

    w = WeightConfigPair.from_model(args.model)
    strat = bgbot_cpp.create_strategy(w.strategy_type, w.weight_paths_list, w.hidden_sizes_list)
    # A fixed snake board stands in for "the root" in ROOT routing: any snake
    # board selects the snake net, whichever side holds the prime.
    anchor = next(b for b in harvest if sr.snake(b))
    assert strat.select_nn_idx(anchor) == 22, strat.select_nn_idx(anchor)

    rng = random.Random(f"{MASTER_SEED}:snake-exit1")
    rows: list[tuple[list[int], dict]] = []
    n_engine = 0
    stats = {"no_reply": 0, "dup": 0, "excluded": 0, "race": 0, "over": 0}
    for i, parent in enumerate(parents):
        opp_pre = _flip(parent)                        # opponent on roll, opponent positive
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        replies = bgbot_cpp.possible_moves(opp_pre, d1, d2)
        if not replies:
            stats["no_reply"] += 1
            continue
        def best(route):
            return max(replies, key=lambda r: strat.evaluate_board(r, route)["equity"])
        picks = [(best(anchor), "root")]
        if n_engine < args.engine_cap:
            eng = best(opp_pre)                        # the engine routes replies by the opponent's pre-move board
            if eng != picks[0][0]:
                picks.append((eng, "engine"))
                n_engine += 1
        for board, how in picks:
            t = tuple(board)
            if bgbot_cpp.check_game_over(board) != 0:
                stats["over"] += 1; continue
            if bgbot_cpp.is_race(board):
                stats["race"] += 1; continue
            if t in exclude:
                stats["excluded"] += 1; continue
            if t in seen:
                stats["dup"] += 1; continue
            seen.add(t)
            rows.append((board, {"parent": i, "dice": [d1, d2], "routing": how,
                                 "snake": bool(sr.snake(board))}))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(parents)} parents -> {len(rows)} rows", flush=True)

    with OUT.open("w", encoding="utf-8") as f, OUT.with_suffix(".meta.jsonl").open("w", encoding="utf-8") as m:
        for board, meta in rows:
            f.write(" ".join(str(x) for x in board) + "\n")
            m.write(json.dumps(meta) + "\n")
    still_snake = sum(1 for _, meta in rows if meta["snake"])
    print(f"wrote {len(rows)} boards to {OUT.name} ({n_engine} engine-routed extras; "
          f"{still_snake} are still snakes after the reply); skipped {stats}")


if __name__ == "__main__":
    main()
