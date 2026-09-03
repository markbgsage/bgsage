#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Split the harvested containment positions into "already rolled out" and
"needs rolling out".

harvest_containment_positions.py writes pile rows and fresh self-play boards
into one positions file. The pile rows already carry rollout targets in the
piles they came from, so re-rolling them would waste ~$100; this script
collects those targets into

    data/s11-bg-containment-pile-rollout      (26 ints + 5 probs, ready to train on)

and rewrites

    data/s11-bg-containment-data              (fresh boards only)

for the parent repo's rollout runner (-> data/s11-bg-containment-rollout).

Usage:
    py -3.14 scripts/split_containment_targets.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA = _PROJECT_ROOT / "data"

positions_path = _DATA / "s11-bg-containment-data"
boards = [tuple(int(x) for x in line.split()[:26])
          for line in positions_path.open(encoding="utf-8") if line.strip()]
wanted = set(boards)

known: dict[tuple, str] = {}
for path in sorted(glob.glob(str(_DATA / "*-rollout"))):
    if path.endswith(("s11-bg-containment-pile-rollout", "s11-bg-containment-rollout")):
        continue
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 31:
            continue
        b = tuple(int(x) for x in p[:26])
        if b in wanted and b not in known:
            known[b] = " ".join(p[:31])

pile_out = _DATA / "s11-bg-containment-pile-rollout"
with pile_out.open("w", encoding="utf-8") as f:
    for b in boards:
        if b in known:
            f.write(known[b] + "\n")
fresh = [b for b in boards if b not in known]
with positions_path.open("w", encoding="utf-8") as f:
    for b in fresh:
        f.write(" ".join(str(x) for x in b) + "\n")
print(f"{len(known)} positions already have rollout targets -> {pile_out.name}")
print(f"{len(fresh)} fresh positions remain in {positions_path.name} for the rollout runner")
