#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""SL training for the Stage 11 containment-game NN (phased index 21).

Data: the pile rows that satisfy the containment rule
(``data/s11-bg-containment-pile-rollout``) plus the fresh self-play
positions rolled out for this net (``data/s11-bg-containment-rollout``).
A deterministic 10% of each (by board hash) is held out; ER = mean
|1-ply equity - rollout target| x 1000 on the holdout.

Two warm starts are worth comparing, since the region was never trained
on directly: Stage 9's prim_race net (the standard net that serves most
containment positions today) and the S9-initialised deep back-game net.
--init-from picks the start, --tag names the output under models/s11_diag/;
the winner is copied to models/sl_s11_bg_containment.weights.best by hand.

Schedule (small region, warm start): 20k epochs @ alpha 3.1, then 60k @ 1.0,
2,500-epoch chunks at batch 4096, best-ER checkpointing.

Usage:
    py -3.14 scripts/run_s11_containment_sl.py --init-from models/sl_s9_prim_race.weights.best --tag primrace
    py -3.14 scripts/run_s11_containment_sl.py --init-from models/s11_diag/sl_deep_s9init_best.weights --tag deep
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import shutil
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
import numpy as np  # noqa: E402

N_HIDDEN, N_INPUTS = 400, 244
PHASES = [(1, 20_000, 3.1), (2, 60_000, 1.0)]
CHUNK, BATCH = 2500, 4096
_DATA = _PROJECT_ROOT / "data"
_DIAG = _PROJECT_ROOT / "models" / "s11_diag"


def load_rollout(path: Path):
    boards, probs = [], []
    for line in path.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) < 31:
            continue
        boards.append([int(x) for x in parts[:26]])
        probs.append([float(x) for x in parts[26:31]])
    return boards, np.array(probs, dtype=np.float32)


def holdout(board) -> bool:
    key = " ".join(str(x) for x in board).encode()
    return hashlib.md5(key).digest()[0] % 10 == 0


def equity(p: np.ndarray) -> np.ndarray:
    return 2 * p[:, 0] - 1 + p[:, 1] - p[:, 3] + p[:, 2] - p[:, 4]


def er(boards, eq, weights_path: str) -> float:
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    tot = 0.0
    for b, e in zip(boards, eq):
        tot += abs(nn.evaluate_board(b, b)["equity"] - float(e))
    return tot / len(boards) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    tb, tp, hb, hp = [], [], [], []
    for name in ("s11-bg-containment-pile-rollout", "s11-bg-containment-rollout"):
        path = _DATA / name
        if not path.exists():
            print(f"  (missing: {name})", flush=True)
            continue
        boards, probs = load_rollout(path)
        for b, p in zip(boards, probs):
            (hb if holdout(b) else tb).append(b)
            (hp if holdout(b) else tp).append(p)
        print(f"  {name}: {len(boards)} rows", flush=True)
    tp, hp = np.array(tp, dtype=np.float32), np.array(hp, dtype=np.float32)
    heq = equity(hp)
    print(f"=== containment SL [{args.tag}]: {len(tb)} train / {len(hb)} holdout rows, "
          f"init {os.path.basename(args.init_from)} ===", flush=True)

    _DIAG.mkdir(exist_ok=True)
    wpath = str(_DIAG / f"sl_containment_{args.tag}.weights")
    best = wpath + ".best"
    shutil.copy2(args.init_from, wpath)
    shutil.copy2(args.init_from, best)
    inputs = bgbot_cpp.encode_boards_batch(np.array(tb, dtype=np.int32), N_INPUTS)
    best_er = er(hb, heq, best)
    print(f"  initial ER {best_er:.2f}", flush=True)
    t0, total = time.time(), 0
    for phase, n_epochs, alpha in PHASES:
        shutil.copy2(best, wpath)
        print(f"--- phase {phase}: {n_epochs}ep @ alpha={alpha} (from best {best_er:.2f}) ---",
              flush=True)
        done = 0
        while done < n_epochs:
            chunk = min(CHUNK, n_epochs - done)
            done += chunk
            total += chunk
            bgbot_cpp.cuda_supervised_train_preencoded(
                inputs=inputs, targets=tp, weights_path=wpath, n_hidden=N_HIDDEN,
                n_inputs=N_INPUTS, alpha=alpha, epochs=chunk, batch_size=BATCH,
                seed=4242 + total, print_interval=chunk + 1, save_path=wpath)
            e = er(hb, heq, wpath)
            gc.collect()
            tag = ""
            if e < best_er:
                best_er = e
                shutil.copy2(wpath, best)
                tag = "  *BEST*"
            print(f"  P{phase} {done:6d}/{n_epochs}  ER={e:.2f}  best={best_er:.2f}  "
                  f"{time.time() - t0:.0f}s{tag}", flush=True)
    print(f"=== [{args.tag}] done: best ER {best_er:.2f} -> {os.path.basename(best)} ===",
          flush=True)


if __name__ == "__main__":
    main()
