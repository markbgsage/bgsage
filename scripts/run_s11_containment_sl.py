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


def er(boards, eq, weights_path: str, row_weights=None) -> float:
    """Mean |1-ply equity - target| x 1000, optionally row-weighted."""
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    errs = np.array([abs(nn.evaluate_board(b, b)["equity"] - float(e))
                     for b, e in zip(boards, eq)])
    if row_weights is None:
        return float(errs.mean() * 1000.0)
    return float((errs * row_weights).sum() / row_weights.sum() * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--extra-data", action="append", default=[],
                        help="Additional rollout-format data file under data/ (repeatable), "
                             "e.g. s11-bg-containment-general-rollout: the rule's rows from "
                             "the main GNUbg corpus, which cover ordinary-game containment")
    parser.add_argument("--family-weight", type=int, default=1,
                        help="Oversample the family rows (the two containment-* files) this "
                             "many times against --extra-data, in training and in the "
                             "checkpointing ER; the general corpus is ~6x larger")
    parser.add_argument("--family-data", action="append", default=None,
                        help="Family rollout file(s) under data/ (repeatable); default the two "
                             "containment files. The snake NN trains from "
                             "s11-bg-snake-rollout with --out-prefix sl_snake")
    parser.add_argument("--out-prefix", default="sl_containment",
                        help="Output name under models/s11_diag/: <prefix>_<tag>.weights[.best]")
    parser.add_argument("--epoch-scale", type=int, default=1,
                        help="Multiply both phases' epoch counts (a schedule is passes over the "
                             "data, so a small set trains in minutes and can afford more)")
    args = parser.parse_args()
    phases = [(ph, n * args.epoch_scale, a) for ph, n, a in PHASES]

    # Family rows (the containment seeds' fights) and general rows (the rule's
    # slice of the main corpus) are held out separately so each can be weighted.
    groups = {"family": ([], [], [], []), "general": ([], [], [], [])}
    family_files = args.family_data or ["s11-bg-containment-pile-rollout",
                                        "s11-bg-containment-rollout"]
    sources = [(n, "family") for n in family_files]
    sources += [(n, "general") for n in args.extra_data]
    for name, group in sources:
        path = _DATA / name
        if not path.exists():
            print(f"  (missing: {name})", flush=True)
            continue
        boards, probs = load_rollout(path)
        tb_, tp_, hb_, hp_ = groups[group]
        for b, p in zip(boards, probs):
            (hb_ if holdout(b) else tb_).append(b)
            (hp_ if holdout(b) else tp_).append(p)
        print(f"  {name}: {len(boards)} rows ({group})", flush=True)
    w = max(1, args.family_weight)
    fam, gen = groups["family"], groups["general"]
    tb = fam[0] * w + gen[0]
    tp = np.array(fam[1] * w + gen[1], dtype=np.float32)
    hb = fam[2] + gen[2]
    hp = np.array(fam[3] + gen[3], dtype=np.float32)
    hw = np.array([float(w)] * len(fam[2]) + [1.0] * len(gen[2]), dtype=np.float64)
    heq = equity(hp)
    print(f"=== containment SL [{args.tag}]: {len(tb)} train rows "
          f"({len(fam[0])} family x{w} + {len(gen[0])} general) / "
          f"{len(hb)} holdout ({len(fam[2])} family, {len(gen[2])} general), "
          f"init {os.path.basename(args.init_from)} ===", flush=True)

    _DIAG.mkdir(exist_ok=True)
    wpath = str(_DIAG / f"{args.out_prefix}_{args.tag}.weights")
    best = wpath + ".best"
    shutil.copy2(args.init_from, wpath)
    shutil.copy2(args.init_from, best)
    inputs = bgbot_cpp.encode_boards_batch(np.array(tb, dtype=np.int32), N_INPUTS)
    best_er = er(hb, heq, best, hw)
    n_fam = len(fam[2])
    print(f"  initial ER {best_er:.2f} (family {er(hb[:n_fam], heq[:n_fam], best):.2f}"
          + (f", general {er(hb[n_fam:], heq[n_fam:], best):.2f})" if len(hb) > n_fam else ")"),
          flush=True)
    t0, total = time.time(), 0
    for phase, n_epochs, alpha in phases:
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
            e = er(hb, heq, wpath, hw)
            gc.collect()
            tag = ""
            if e < best_er:
                best_er = e
                shutil.copy2(wpath, best)
                tag = "  *BEST*"
            detail = f"  fam={er(hb[:n_fam], heq[:n_fam], wpath):.2f}"
            if len(hb) > n_fam:
                detail += f" gen={er(hb[n_fam:], heq[n_fam:], wpath):.2f}"
            print(f"  P{phase} {done:6d}/{n_epochs}  ER={e:.2f}  best={best_er:.2f}{detail}  "
                  f"{time.time() - t0:.0f}s{tag}", flush=True)
    print(f"=== [{args.tag}] done: best ER {best_er:.2f} (family "
          f"{er(hb[:n_fam], heq[:n_fam], best):.2f}) -> {os.path.basename(best)} ===",
          flush=True)


if __name__ == "__main__":
    main()
