#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Seed positions for the back-game FAMILY benchmark folders.

Writes ``backgame_ref_positions/benchmark/<folder> manual.txt`` for the three
folders whose filters live in ``backgame_benchmark.FAMILY_FILTERS``:

* ``containment``      — late containment games (escaper >= 6 off, 1-3 trapped)
* ``massive backgame`` — 3+ anchors, or 2 anchors with 7+ checkers back
* ``snake``            — a far-side prime containing a straggler vs a crunched side

Seeds come from positions the engine has already met: the money and pasko
benchmark decisions (with their cube), rows of the S11 category / exit piles
and the pasko pile (cube centred), and — for the snake family, which the
piles barely hold — synthetic variants of the "Snake fail" shape (prime
length, prime placement, straggler location, crunch split). Every seed is
checked against its folder's filter, deduplicated, and sampled with a fixed
seed so the file regenerates identically.

Then:
    py -3.14 scripts/backgame_benchmark.py import --folder containment
    py -3.14 scripts/backgame_benchmark.py generate --folder containment --count 3000 --model stage11

Usage:
    py -3.14 scripts/build_family_seeds.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backgame_benchmark import FAMILY_FILTERS, _flip, manual_file  # noqa: E402

_DATA = _PROJECT_ROOT / "data"
_DECISION_SETS = ("money_benchmark", "pasko_money_benchmark")
_PILES = ("s11-bg-deep-train-rollout", "s11-bg-middle-train-rollout",
          "s11-bg-double-train-rollout", "s11-bg-deep-exit-rollout",
          "s11-bg-middle-exit-rollout", "s11-bg-double-exit-rollout",
          "pasko-train-rollout", "player-backgame-train-rollout",
          "opponent-backgame-train-rollout")
#: (decision seeds cap, total cap) per folder.
CAPS = {"containment": (140, 200), "massive backgame": (60, 200), "snake": (60, 100)}
SEED = 1


def decision_seeds(matches) -> list[tuple[int, str, tuple[int, ...]]]:
    out, seen = [], set()
    for name in _DECISION_SETS:
        data = json.loads((_DATA / name / "benchmark.json").read_text(encoding="utf-8"))
        for d in data["decisions"]:
            board = tuple(d["board"])
            key = (d["cube_value"], d["cube_owner"], board)
            if key in seen or not matches(board):
                continue
            seen.add(key)
            out.append((min(d["cube_value"], 2) if d["cube_owner"] != "centered" else 1,
                        d["cube_owner"], board))
    return out


def pile_seeds(matches, rng: random.Random, limit: int,
               exclude: set) -> list[tuple[int, str, tuple[int, ...]]]:
    found: list[tuple[int, ...]] = []
    seen: set = set(exclude)
    for name in _PILES:
        path = _DATA / name
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            parts = line.split()
            if len(parts) < 26:
                continue
            board = tuple(int(x) for x in parts[:26])
            if board in seen or not matches(board):
                continue
            seen.add(board)
            found.append(board)
    rng.shuffle(found)
    return [(1, "centered", b) for b in found[:limit]]


def synthetic_snakes(matches, rng: random.Random, limit: int) -> list:
    """Variants of the Snake shape: P1's whole army on the far side as a prime
    with spares, P2 crunched at home with one straggler behind the prime."""
    out, seen = [], set()
    for start in (15, 16, 17, 18):
        for length in (4, 5, 6):
            prime = list(range(start, start + length))
            if prime[-1] > 22:
                continue
            for straggler in (0, 1, 2, 3, 4, 5, 6):          # 0 = on the bar
                for split in ((7, 7), (5, 9), (9, 5), (4, 5, 5), (3, 5, 6)):
                    board = [0] * 26
                    for pt in prime:
                        board[pt] = 2
                    spare_pts = [p for p in (13, 14, 15, 16, 22) if p not in prime]
                    spares = 15 - 2 * length
                    # extra checkers first onto the prime (a third checker),
                    # then onto the free spare points, alternating.
                    slots = []
                    for i in range(spares):
                        if i % 2 == 0 and prime:
                            slots.append(prime[i // 2 % len(prime)])
                        else:
                            slots.append(spare_pts[(i // 2) % len(spare_pts)])
                    for pt in slots:
                        board[pt] += 1
                    # P2: crunched checkers on its 1/2(/3) points, one straggler.
                    home_pts = (24, 23, 22)[:len(split)]
                    if any(board[pt] > 0 for pt in home_pts):
                        continue
                    for pt, n in zip(home_pts, split):
                        board[pt] = -n
                    if straggler == 0:
                        board[0] = 1
                    else:
                        board[straggler] = -1
                    b = tuple(board)
                    if b in seen or not matches(b):
                        continue
                    seen.add(b)
                    out.append((1, "centered", b))
    rng.shuffle(out)
    return out[:limit]


def anchor_hist(seeds) -> Counter:
    def anchors(board) -> int:
        a1 = sum(1 for i in range(19, 25) if board[i] >= 2)
        a2 = sum(1 for i in range(1, 7) if board[i] <= -2)
        return max(a1, a2)
    return Counter(anchors(b) for _, _, b in seeds)


def main() -> None:
    for folder, (dec_cap, total_cap) in CAPS.items():
        matches = FAMILY_FILTERS[folder]
        rng = random.Random(f"{SEED}:{folder}")
        dec = decision_seeds(matches)
        rng.shuffle(dec)
        dec = dec[:dec_cap]
        seeds = list(dec)
        if folder == "snake":
            seeds += synthetic_snakes(matches, rng, 40)
        seeds += pile_seeds(matches, rng, total_cap - len(seeds),
                            exclude={b for _, _, b in seeds})
        lines = [f"# {folder} - seed positions (build_family_seeds.py, seed {SEED}).",
                 "# One position per line: cube_value cube_owner board(26 ints).",
                 f"# {len(dec)} from money/pasko benchmark decisions (with cube), "
                 f"{len(seeds) - len(dec)} from piles / synthetic (cube centred)."]
        for cube_value, cube_owner, board in seeds:
            assert matches(board) or matches(_flip(board))
            lines.append(f"{cube_value} {cube_owner} " + " ".join(str(x) for x in board))
        path = manual_file(folder)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{folder:18s}: {len(seeds):4d} seeds -> {path.name}  "
              f"(decisions {len(dec)}, anchors {dict(sorted(anchor_hist(seeds).items()))})")


if __name__ == "__main__":
    main()
