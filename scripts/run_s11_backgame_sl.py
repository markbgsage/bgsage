#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""SL training for the Stage 11 backgame category NNs.

Trains one 400h/244-input NN per backgame category against the rollout
targets that ``segregate_s11_backgame_data.py`` produced
(``data/s11-bg-<cat>-{train,benchmark}-rollout``), following the exact recipe
that made Stage 9's backgame nets (run_backgame_sl_training.py): GPU
supervised training in 2,500-epoch chunks at batch 4096, phase 1 100k epochs
@ alpha 3.1 then phase 2 250k @ alpha 1.0, each phase resuming from the best
weights so far, best-ER checkpointing between chunks.

ER = mean |1-ply equity − rollout target equity| × 1000 over the category's
held-out benchmark rows.

Initial weights come from the truncated-TD bootstrap by default
(``models/td_s11_bg_<cat>.weights.best``); pass --init-from to override.
Output lands at ``models/sl_s11_bg_<cat>.weights(.best)`` — exactly the
filenames the ``stage11`` registry entry points at, so a finished run is
immediately loadable via ``WeightConfigPair.from_model("stage11")``.

Usage (long-running — launch detached per the CLAUDE.md pattern)::

    py -3.14 scripts/run_s11_backgame_sl.py --category deep
    py -3.14 scripts/run_s11_backgame_sl.py --category all
    py -3.14 scripts/run_s11_backgame_sl.py --category deep --smoke   # pipeline check
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
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
import numpy as np  # noqa: E402

N_HIDDEN = 400
N_INPUTS = 244
CATEGORIES = ("deep", "middle", "double")

#: The Stage 9 backgame SL schedule: (phase label, epochs, alpha).
PHASES = [(1, 100_000, 3.1), (2, 250_000, 1.0)]
CHUNK = 2500
BATCH = 4096

_DATA = _PROJECT_ROOT / "data"
_MODELS = _PROJECT_ROOT / "models"


def load_rollout(path: Path) -> tuple[list[list[int]], np.ndarray]:
    boards, probs = [], []
    for line in path.open(encoding="utf-8"):
        parts = line.split()
        if len(parts) < 31:
            continue
        boards.append([int(x) for x in parts[:26]])
        probs.append([float(x) for x in parts[26:31]])
    return boards, np.array(probs, dtype=np.float32)


def benchmark_er(bench_boards: list[list[int]], bench_eq: np.ndarray,
                 weights_path: str) -> float:
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    total = 0.0
    for board, eq in zip(bench_boards, bench_eq):
        total += abs(nn.evaluate_board(board, board)["equity"] - float(eq))
    return total / len(bench_boards) * 1000.0


def train_category(category: str, args: argparse.Namespace) -> None:
    print(f"=== S11 backgame SL: {category} ===", flush=True)
    train_path = _DATA / f"s11-bg-{category}-train-rollout"
    bench_path = _DATA / f"s11-bg-{category}-benchmark-rollout"
    for p in (train_path, bench_path):
        if not p.exists():
            raise SystemExit(f"{p} missing - run segregate_s11_backgame_data.py first")

    train_boards, train_probs = load_rollout(train_path)
    bench_boards, bench_probs = load_rollout(bench_path)
    bench_eq = (2 * bench_probs[:, 0] - 1 + bench_probs[:, 1] - bench_probs[:, 3]
                + bench_probs[:, 2] - bench_probs[:, 4])
    print(f"  {len(train_boards)} train rows, {len(bench_boards)} benchmark rows",
          flush=True)

    wpath = str(_MODELS / f"sl_s11_bg_{category}.weights")
    best_path = wpath + ".best"

    init = args.init_from or str(_MODELS / f"td_s11_bg_{category}.weights.best")
    if not os.path.exists(wpath):
        if not os.path.exists(init):
            raise SystemExit(
                f"No initial weights: {init}\n"
                f"Run the TD bootstrap first (run_s11_backgame_td.py "
                f"--category {category}) or pass --init-from.")
        print(f"  init from {os.path.basename(init)}", flush=True)
        shutil.copy2(init, wpath)
        shutil.copy2(init, best_path)
    else:
        print(f"  resuming existing {os.path.basename(wpath)}", flush=True)

    print("  pre-encoding train inputs...", flush=True)
    t0 = time.time()
    train_inputs = bgbot_cpp.encode_boards_batch(
        np.array(train_boards, dtype=np.int32), N_INPUTS)
    print(f"  encoded in {time.time() - t0:.1f}s", flush=True)

    best_er = benchmark_er(bench_boards, bench_eq, best_path
                           if os.path.exists(best_path) else wpath)
    gc.collect()
    print(f"  initial ER: {best_er:.2f}\n", flush=True)

    phases = [(0, 2 * CHUNK, 3.1)] if args.smoke else PHASES
    t_start = time.time()
    total_epochs = 0
    for phase, n_epochs, alpha in phases:
        if os.path.exists(best_path):
            shutil.copy2(best_path, wpath)
        print(f"--- phase {phase}: {n_epochs}ep @ alpha={alpha} "
              f"(from best ER={best_er:.2f}) ---", flush=True)
        done = 0
        while done < n_epochs:
            chunk = min(CHUNK, n_epochs - done)
            total_epochs += chunk
            done += chunk
            # print_interval must be > 0 (0 crashes CUDA on Python 3.14).
            bgbot_cpp.cuda_supervised_train_preencoded(
                inputs=train_inputs, targets=train_probs, weights_path=wpath,
                n_hidden=N_HIDDEN, n_inputs=N_INPUTS, alpha=alpha, epochs=chunk,
                batch_size=BATCH, seed=42 + total_epochs,
                print_interval=chunk + 1, save_path=wpath)
            er = benchmark_er(bench_boards, bench_eq, wpath)
            gc.collect()
            tag = ""
            if er < best_er:
                best_er = er
                shutil.copy2(wpath, best_path)
                tag = "  *BEST*"
            print(f"  P{phase} {done:6d}/{n_epochs}  ER={er:.2f}  "
                  f"best={best_er:.2f}  {time.time() - t_start:.0f}s{tag}",
                  flush=True)
        print(flush=True)

    print(f"=== {category} done: {total_epochs}ep in "
          f"{(time.time() - t_start) / 60:.1f}m, best ER {best_er:.2f} -> "
          f"{os.path.basename(best_path)} ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True,
                        choices=[*CATEGORIES, "all"])
    parser.add_argument("--init-from", default=None,
                        help="Initial weights (default: the category's TD "
                             "bootstrap, models/td_s11_bg_<cat>.weights.best)")
    parser.add_argument("--smoke", action="store_true",
                        help="Two chunks only - validates the pipeline")
    args = parser.parse_args()

    cats = list(CATEGORIES) if args.category == "all" else [args.category]
    for category in cats:
        train_category(category, args)


if __name__ == "__main__":
    main()
