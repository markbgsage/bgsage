#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Build the massive-backgame NN's training files from the Stage 11 category
piles (which ``segregate_s11_backgame_data.py`` derives from the S9/S10-era
backgame rollouts) and the three exit piles::

    data/s11-bg-massive-rollout        every row the benchmark's massive
                                       family claims (the FAMILY file)
    data/s11-bg-massive-nbhd-rollout   every other row of the same piles —
                                       the neighbourhood a search below a
                                       massive root visits

Rows are deduplicated by board across the six inputs. The family is defined
by ``backgame_benchmark._massive`` (>= 3 anchors, or >= 2 anchors and >= 7
checkers back, behind in the race; never a containment game or a snake),
the same rule ``massive_category()`` implements in C++.

    py -3.14 scripts/build_massive_data.py

Train with the containment recipe::

    py -3.14 scripts/run_s11_containment_sl.py --family-data s11-bg-massive-rollout \\
        --extra-data s11-bg-massive-nbhd-rollout --family-weight 4 \\
        --out-prefix sl_massive --init-from models/sl_s11_bg_deep.weights.best --tag deep
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA = _PROJECT_ROOT / "data"
sys.path.insert(0, str(_SCRIPT_DIR))
import backgame_benchmark as bb  # noqa: E402

SOURCES = ["s11-bg-deep-train-rollout", "s11-bg-middle-train-rollout", "s11-bg-double-train-rollout",
           "s11-bg-deep-exit-rollout", "s11-bg-middle-exit-rollout", "s11-bg-double-exit-rollout"]


def main() -> None:
    t0 = time.time()
    seen: set[tuple[int, ...]] = set()
    n_fam = n_nb = n_dup = 0
    with (_DATA / "s11-bg-massive-rollout").open("w", encoding="utf-8") as fam, \
            (_DATA / "s11-bg-massive-nbhd-rollout").open("w", encoding="utf-8") as nb:
        for name in SOURCES:
            path = _DATA / name
            if not path.exists():
                print(f"  (missing: {name})", flush=True)
                continue
            for line in path.open(encoding="utf-8"):
                parts = line.split()
                if len(parts) < 31:
                    continue
                board = tuple(int(x) for x in parts[:26])
                if board in seen:
                    n_dup += 1
                    continue
                seen.add(board)
                if bb._massive(board):
                    fam.write(" ".join(parts[:31]) + "\n")
                    n_fam += 1
                else:
                    nb.write(" ".join(parts[:31]) + "\n")
                    n_nb += 1
    print(f"family (massive) rows: {n_fam}; neighbourhood rows: {n_nb}; "
          f"duplicates dropped: {n_dup} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
