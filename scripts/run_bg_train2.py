# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""SL training for backgame NNs (Stage 9).

Usage:
    python bgsage/scripts/run_backgame_sl_training.py --side player
    python bgsage/scripts/run_backgame_sl_training.py --side opponent
"""
import argparse
import gc
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, "build")

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.isdir(cuda_path):
    os.add_dll_directory(cuda_path)

import bgbot_cpp


def benchmark_er(bench_boards, bench_eq, weights_path):
    nn = bgbot_cpp.NNStrategy(weights_path, 400, 244)
    total_err = 0.0
    for i in range(len(bench_boards)):
        r = nn.evaluate_board(bench_boards[i].tolist(), bench_boards[i].tolist())
        total_err += abs(r["equity"] - bench_eq[i])
    return (total_err / len(bench_boards)) * 1000.0


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=["player", "opponent"])
    args = parser.parse_args()
    side = args.side

    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
    models_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))

    if side == "player":
        train_file = "player-backgame-train-rollout"
        bench_file = "player-backgame-benchmark-rollout"
        s8_nn = "anch_race"
        wname = "sl_s9_player_bg"
    else:
        train_file = "opponent-backgame-train-rollout"
        bench_file = "opponent-backgame-benchmark-rollout"
        s8_nn = "race_anch"
        wname = "sl_s9_opponent_bg"

    print(f"=== Backgame SL Training: {side} ===\n", flush=True)

    # Load training data
    boards_list, probs_list = [], []
    with open(os.path.join(data_dir, train_file)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31:
                continue
            boards_list.append([int(x) for x in parts[:26]])
            probs_list.append([float(x) for x in parts[26:31]])
    train_boards = np.array(boards_list, dtype=np.int32)
    train_probs = np.array(probs_list, dtype=np.float32)
    print(f"Loaded {len(train_boards)} training positions", flush=True)

    # Load benchmark data
    bench_boards_list, bench_probs_list = [], []
    with open(os.path.join(data_dir, bench_file)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31:
                continue
            bench_boards_list.append([int(x) for x in parts[:26]])
            bench_probs_list.append([float(x) for x in parts[26:31]])
    bench_boards = np.array(bench_boards_list, dtype=np.int32)
    bench_probs = np.array(bench_probs_list, dtype=np.float32)
    bench_eq = (2 * bench_probs[:, 0] - 1 + bench_probs[:, 1]
                - bench_probs[:, 3] + bench_probs[:, 2] - bench_probs[:, 4])
    print(f"Loaded {len(bench_boards)} benchmark positions", flush=True)

    # Weights
    wpath = os.path.join(models_dir, wname + ".weights")
    best_path = os.path.join(models_dir, wname + ".weights.best")

    if not os.path.exists(wpath):
        s8_src = os.path.join(models_dir, f"sl_s8_{s8_nn}.weights.best")
        print(f"Init from S8 {s8_nn}: {s8_src}", flush=True)
        shutil.copy2(s8_src, wpath)
        shutil.copy2(s8_src, best_path)

    best_er = benchmark_er(bench_boards, bench_eq, wpath)
    gc.collect()
    print(f"Initial ER: {best_er:.2f}\n", flush=True)

    phases = [(3, 200, 3.1), (4, 500, 1.0)]
    t_start = time.time()
    total_epochs = 0

    for phase_num, n_epochs, alpha in phases:
        print(f"--- Phase {phase_num}: {n_epochs} epochs @ alpha={alpha} ---", flush=True)

        epochs_done = 0
        while epochs_done < n_epochs:
            chunk = min(10, n_epochs - epochs_done)
            total_epochs += chunk
            epochs_done += chunk

            bgbot_cpp.cuda_supervised_train(
                boards=train_boards, targets=train_probs,
                weights_path=wpath, n_hidden=400, n_inputs=244,
                alpha=alpha, epochs=chunk, batch_size=4096,
                seed=42 + total_epochs, print_interval=0, save_path=wpath,
            )

            er = benchmark_er(bench_boards, bench_eq, wpath)
            gc.collect()

            improved = ""
            if er < best_er:
                best_er = er
                shutil.copy2(wpath, best_path)
                improved = " *BEST*"

            elapsed = time.time() - t_start
            print(f"  Phase {phase_num} ep {epochs_done:3d}/{n_epochs}  "
                  f"ER={er:.2f}  best={best_er:.2f}  "
                  f"elapsed={elapsed:.0f}s{improved}", flush=True)

        print(flush=True)

    total_time = time.time() - t_start
    print(f"=== Training complete ===")
    print(f"Total: {total_epochs} epochs in {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"Best ER: {best_er:.2f}")
    print(f"Best weights: {best_path}")


run()
