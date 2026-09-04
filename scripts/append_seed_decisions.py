#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Add every folder's seed positions to its benchmark decision list.

``backgame_benchmark.py generate`` now records each seed as a cube decision
(a seed is a known-good position of its family), but the benchmarks built
before 2026-09-04 hold only the decisions that arose in play from the seeds.
This rewrites each ``<folder> benchmark.txt`` with the seeds' cube decisions
added (deduplicated, the header's counts refreshed, the run's seed / model /
level taken from the state sidecar), so the decision list matches the
reference once the seeds' rollouts are in. Idempotent.

Usage:
    py -3.14 scripts/append_seed_decisions.py [--folder "21 backgame" ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FOLDERS = ["21 backgame", "31 backgame", "32 backgame", "41 backgame", "42 backgame",
           "51 backgame", "52 backgame", "43 backgame", "53 backgame", "54 backgame",
           "containment", "massive backgame", "snake", "Positions XG gets wrong"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", nargs="*", default=FOLDERS)
    args = parser.parse_args()
    from backgame_benchmark import (Decision, _load_state, read_decisions, read_start_positions,
                                    write_decisions)

    for folder in args.folder:
        existing = read_decisions(folder)
        if not existing:
            print(f"{folder}: no decision list, skipped")
            continue
        found = {d: None for d in existing}
        before = len(found)
        for st in read_start_positions(folder):
            found.setdefault(Decision("cube", st.cube_value, st.cube_owner, None, tuple(st.board)), None)
        state = _load_state(folder)
        write_decisions(folder, list(found), int(state.get("seed", 1)),
                        str(state.get("model", "stage9")), str(state.get("level", "3ply")))
        print(f"{folder}: {before} -> {len(found)} decisions (+{len(found) - before} seeds)")


if __name__ == "__main__":
    main()
