#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Pull the containment-rule rows out of the main GNUbg training corpus.

The Stage 11 containment NN was first trained only on family-style
containment fights (the containment seeds' self-play, ``data/s11-bg-
containment-*-rollout``). On the money benchmark's 237 containment-routed
decisions that net scored PR 4.62 against 2.91 for Stage 9's ordinary
nets: a narrow-subset regression on ORDINARY-game containment positions
(a late hit during a bear-off, say), which look nothing like the family
fights. This script gives the net the rest of its own region: every row
of ``contact-train-data`` / ``crashed-train-data`` whose post-move board
satisfies the C++ ``containment_category`` rule, written in the rollout
row format (26 ints + 5 probs) the SL script reads.

Usage:
    py -3.14 scripts/extract_general_containment_rows.py
    -> data/s11-bg-containment-general-rollout
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    os.add_dll_directory(str(_PROJECT_ROOT / "build"))

import bgbot_cpp  # noqa: E402
from bgsage.data import load_gnubg_training_data  # noqa: E402

_DATA = _PROJECT_ROOT / "data"
OUT = _DATA / (sys.argv[1] if len(sys.argv) > 1 else "s11-bg-containment-general-rollout")


def main() -> None:
    t0 = time.time()
    n_out = 0
    with OUT.open("w", encoding="utf-8") as out:
        for name in ("contact-train-data", "crashed-train-data"):
            boards, targets = load_gnubg_training_data(str(_DATA / name))
            n_file = 0
            for b, p in zip(boards.tolist(), targets.tolist()):
                if not bgbot_cpp.containment_category(b):
                    continue
                out.write(" ".join(str(int(x)) for x in b) + " "
                          + " ".join(f"{float(x):.4f}" for x in p) + "\n")
                n_file += 1
            print(f"  {name}: {n_file} of {len(boards)} rows satisfy the rule "
                  f"({100 * n_file / len(boards):.2f}%)", flush=True)
            n_out += n_file
    print(f"wrote {OUT.name}: {n_out} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
