#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest exit-descendant positions for the Stage 11 backgame NNs.

The 20-NN strategy routes candidate evaluation by the PRE-move board
(`select_nn_idx(pre_move_board)` in every candidate path), so when the
pre-move position is in a backgame category, the category NN values EVERY
candidate — including moves that leave the region. The category training
piles contain only in-region positions, so the net extrapolates on those
exit candidates; measured on the deep folders (2026-08-31), that
extrapolation is ~+0.30 optimistic and produces the dominant error mode
(57% of stage11's checker-error mass = "best stays in, model exits").

This script builds the missing training positions: one-step exit
descendants of in-region positions, in exactly the orientation the router
evaluates them (post-move boards, mover-positive).

For each row of ``data/s11-bg-<cat>-train-rollout`` (a post-move in-region
board, mover-positive), the opponent is on roll at ``flip(board)``; sample a
few dice rolls, enumerate the legal moves, and keep the candidates that
classify as NO backgame category. Rows are drawn half from the standard-game
sources and half from the pasko sources (the play benchmarks are
standard-game distributed; the pile is 84% pasko).

Excluded, so the play benchmarks stay honest: any board (either orientation)
appearing among the candidate moves of the ``backgame_ref_positions/
benchmark/* rollout.jsonl`` references — training the value net on the exact
candidates the benchmark scores would optimize on the test set.

Output: ``data/s11-bg-<cat>-exit-data`` (26 space-separated ints per line),
ready for the parent repo's distributed rollout runner, which appends the
rolled-out probabilities in the pile's own grade and format:

    python scripts/rollout_backgame_positions.py s11-bg-deep-exit-data \\
        --workers 72 --backend batch --timeout 900

Deterministic: the same seed regenerates the same file.

Usage:
    py -3.14 scripts/generate_s11_exit_positions.py --category deep
    py -3.14 scripts/generate_s11_exit_positions.py --category deep --limit 40000
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
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    os.add_dll_directory(str(_PROJECT_ROOT / "build"))

import bgbot_cpp  # noqa: E402

_DATA = _PROJECT_ROOT / "data"
_BENCH_DIR = _PROJECT_ROOT / "backgame_ref_positions" / "benchmark"

#: Boards in these files are the standard-game (S9-era) slice; everything
#: else in the category pile came from the pasko sources.
_STANDARD_SOURCES = (
    "player-backgame-train-rollout",
    "player-backgame-benchmark-rollout",
    "opponent-backgame-train-rollout",
    "opponent-backgame-benchmark-rollout",
)

#: The 21 distinct dice rolls (d1 <= d2).
_DICE = [(d1, d2) for d1 in range(1, 7) for d2 in range(d1, 7)]


def _boards_of(path: Path) -> list[tuple[int, ...]]:
    boards = []
    for line in path.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) >= 26:
            boards.append(tuple(int(x) for x in parts[:26]))
    return boards


def load_benchmark_candidate_boards() -> set[tuple[int, ...]]:
    """Every candidate board in every folder reference, both orientations."""
    excluded: set[tuple[int, ...]] = set()
    for path in sorted(_BENCH_DIR.glob("* rollout.jsonl")):
        for line in path.open(encoding="utf-8"):
            entry = json.loads(line)
            for m in entry.get("moves", []):
                b = tuple(m["board"])
                excluded.add(b)
                excluded.add(tuple(bgbot_cpp.flip_board(list(b))))
    return excluded


def harvest(rows: list[tuple[int, ...]], budget: int, rng: random.Random,
            per_row_rolls: int, seen: set[tuple[int, ...]],
            excluded: set[tuple[int, ...]], stats: dict) -> list[tuple[int, ...]]:
    """Exit candidates from ``rows`` until ``budget`` distinct ones are found."""
    out: list[tuple[int, ...]] = []
    order = list(range(len(rows)))
    rng.shuffle(order)
    for idx in order:
        if len(out) >= budget:
            break
        pre = bgbot_cpp.flip_board(list(rows[idx]))
        if bgbot_cpp.backgame_category(pre) == "none":
            stats["not_in_region"] += 1
            continue
        for die1, die2 in rng.sample(_DICE, per_row_rolls):
            for cand in bgbot_cpp.possible_moves(pre, die1, die2):
                cand_t = tuple(cand)
                if cand_t in seen:
                    stats["dup"] += 1
                    continue
                seen.add(cand_t)
                if bgbot_cpp.backgame_category(list(cand)) != "none":
                    stats["in_region_cand"] += 1
                    continue
                if bgbot_cpp.check_game_over(list(cand)) != 0:
                    stats["game_over"] += 1
                    continue
                if cand_t in excluded:
                    stats["benchmark_hit"] += 1
                    continue
                out.append(cand_t)
                if len(out) >= budget:
                    return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True,
                        choices=("deep", "middle", "double"))
    parser.add_argument("--limit", type=int, default=40_000,
                        help="Total exit positions to emit (default 40,000)")
    parser.add_argument("--standard-share", type=float, default=0.5,
                        help="Fraction drawn from standard-game source rows "
                             "(default 0.5; the shortfall of either slice is "
                             "filled from the other)")
    parser.add_argument("--per-row-rolls", type=int, default=3,
                        help="Dice rolls sampled per source row (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pile = _DATA / f"s11-bg-{args.category}-train-rollout"
    if not pile.exists():
        raise SystemExit(f"{pile} missing - run segregate_s11_backgame_data.py first")

    standard_boards: set[tuple[int, ...]] = set()
    for name in _STANDARD_SOURCES:
        p = _DATA / name
        if p.exists():
            standard_boards.update(_boards_of(p))

    rows = _boards_of(pile)
    std_rows = [b for b in rows if b in standard_boards]
    pas_rows = [b for b in rows if b not in standard_boards]
    print(f"{pile.name}: {len(rows)} rows = {len(std_rows)} standard "
          f"+ {len(pas_rows)} pasko")

    excluded = load_benchmark_candidate_boards()
    print(f"benchmark candidate exclusion set: {len(excluded)} boards "
          f"(both orientations)")

    rng = random.Random(args.seed)
    seen: set[tuple[int, ...]] = set()
    stats = {"dup": 0, "in_region_cand": 0, "game_over": 0,
             "benchmark_hit": 0, "not_in_region": 0}

    std_budget = int(args.limit * args.standard_share)
    std_exits = harvest(std_rows, std_budget, rng, args.per_row_rolls,
                        seen, excluded, stats)
    pas_exits = harvest(pas_rows, args.limit - len(std_exits), rng,
                        args.per_row_rolls, seen, excluded, stats)
    if len(std_exits) + len(pas_exits) < args.limit and len(std_exits) == std_budget:
        std_exits += harvest(std_rows, args.limit - len(std_exits) - len(pas_exits),
                             rng, args.per_row_rolls, seen, excluded, stats)

    exits = std_exits + pas_exits
    rng.shuffle(exits)
    out_path = _DATA / f"s11-bg-{args.category}-exit-data"
    with out_path.open("w", encoding="utf-8") as f:
        for b in exits:
            f.write(" ".join(str(x) for x in b) + "\n")

    print(f"\nwrote {len(exits)} exit positions -> {out_path.name}")
    print(f"  {len(std_exits)} from standard-game rows, "
          f"{len(pas_exits)} from pasko rows")
    print(f"  skipped: {stats['dup']} duplicate candidates, "
          f"{stats['in_region_cand']} still in-region, "
          f"{stats['game_over']} game-over, "
          f"{stats['benchmark_hit']} benchmark candidates, "
          f"{stats['not_in_region']} source rows not in-region")


if __name__ == "__main__":
    main()
