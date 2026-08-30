#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Segregate the Stage 9/10 backgame rollout piles into the S11 categories.

Reads every backgame-labelled rollout file from the Stage 9 and Stage 10
training eras and splits the rows by ``backgame_category(board)`` — deep
(21/31/32), middle (41/42/51/52), double (43/53/54) — writing

    data/s11-bg-<cat>-train-rollout
    data/s11-bg-<cat>-benchmark-rollout

in the same format as the sources (26 board ints + 5 cubeless probs,
post-move, positive-player perspective), so ``run_s11_backgame_sl.py`` can
train each category NN on its own region.

Sources (rollout targets only — the ``*-data`` position files and the
superseded ``*.s8`` targets are skipped, as are the general non-backgame
pasko piles):

    player/opponent-backgame-{train,benchmark}-rollout          (S9, re-rolled)
    pasko-{player,opponent}-backgame-{train,benchmark}-rollout   (S10)
    pasko-gated-{player,opponent}-backgame-{train,benchmark}-rollout
    pasko-gated-remainder-backgame-{train,benchmark}-rollout

Rows are deduplicated by exact board across all sources (first occurrence
wins, in the order above — the standard-game S9 targets first). A board that
already landed in a category's TRAIN file is dropped from its BENCHMARK file,
so best-weight selection is held out. Rows whose board classifies as no
backgame under S11 detection are dropped and counted.

Usage:
    py -3.14 scripts/segregate_s11_backgame_data.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
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

#: Source rollout files in trust order (standard-game S9 targets first).
SOURCES = [
    "player-backgame-{split}-rollout",
    "opponent-backgame-{split}-rollout",
    "pasko-gated-player-backgame-{split}-rollout",
    "pasko-gated-opponent-backgame-{split}-rollout",
    "pasko-gated-remainder-backgame-{split}-rollout",
    "pasko-player-backgame-{split}-rollout",
    "pasko-opponent-backgame-{split}-rollout",
]

CATEGORIES = ("deep", "middle", "double")


def out_path(cat: str, split: str) -> Path:
    return _DATA / f"s11-bg-{cat}-{split}-rollout"


def main() -> None:
    train_boards: set[tuple[int, ...]] = set()
    stats: Counter = Counter()
    per_source: dict[str, Counter] = {}

    for split in ("train", "benchmark"):
        outs = {cat: open(out_path(cat, split), "w", encoding="utf-8")
                for cat in CATEGORIES}
        seen: set[tuple[int, ...]] = set()
        try:
            for pattern in SOURCES:
                name = pattern.format(split=split)
                path = _DATA / name
                if not path.exists():
                    print(f"  (missing, skipped: {name})")
                    continue
                src_stats = per_source.setdefault(name, Counter())
                for line in path.open(encoding="utf-8"):
                    parts = line.split()
                    if len(parts) < 31:
                        continue
                    board = tuple(int(x) for x in parts[:26])
                    if board in seen:
                        src_stats["dup"] += 1
                        continue
                    if split == "benchmark" and board in train_boards:
                        src_stats["in_train"] += 1
                        continue
                    seen.add(board)
                    cat = bgbot_cpp.backgame_category(list(board))
                    if cat not in outs:
                        src_stats["none"] += 1
                        stats[f"{split}:none"] += 1
                        continue
                    outs[cat].write(line if line.endswith("\n") else line + "\n")
                    src_stats[cat] += 1
                    stats[f"{split}:{cat}"] += 1
        finally:
            for f in outs.values():
                f.close()
        if split == "train":
            train_boards = seen

    print("\nper-source breakdown:")
    print(f"  {'file':56s} {'deep':>7s} {'middle':>7s} {'double':>7s} "
          f"{'none':>7s} {'dup':>6s} {'leak':>5s}")
    for name, c in per_source.items():
        print(f"  {name:56s} {c['deep']:7d} {c['middle']:7d} {c['double']:7d} "
              f"{c['none']:7d} {c['dup']:6d} {c['in_train']:5d}")

    print("\noutput files:")
    for split in ("train", "benchmark"):
        for cat in CATEGORIES:
            n = stats[f"{split}:{cat}"]
            print(f"  {out_path(cat, split).name:36s} {n:8d} rows")
        print(f"  ({split}: {stats[f'{split}:none']} rows classified as no "
              f"backgame under S11 detection — dropped)")


if __name__ == "__main__":
    main()
