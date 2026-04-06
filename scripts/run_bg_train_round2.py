"""Round 2 SL training for back game NNs (Stage 9).

Starts from current .weights.best, saves best to .weights.best.

Usage (from project root):
    python bgsage/scripts/run_bg_train_round2.py --side player
    python bgsage/scripts/run_bg_train_round2.py --side opponent
    python bgsage/scripts/run_bg_train_round2.py --side player --alpha 1.0 --epochs 100000 --bench-interval 10000
"""
import argparse
import gc
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, "build")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
import bgbot_cpp

N_HIDDEN = 400
N_INPUTS = 244
BATCH_SIZE = 4096


def load_rollout(filepath):
    boards_list, probs_list = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31:
                continue
            boards_list.append([int(x) for x in parts[:26]])
            probs_list.append([float(x) for x in parts[26:31]])
    return np.array(boards_list, dtype=np.int32), np.array(probs_list, dtype=np.float32)


def benchmark_er(bench_boards, bench_eq, weights_path):
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    total_err = 0.0
    for i in range(len(bench_boards)):
        r = nn.evaluate_board(bench_boards[i].tolist(), bench_boards[i].tolist())
        total_err += abs(r["equity"] - bench_eq[i])
    er = (total_err / len(bench_boards)) * 1000.0
    del nn
    gc.collect()
    return er


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=["player", "opponent"])
    parser.add_argument("--alpha", type=float, default=3.1)
    parser.add_argument("--epochs", type=int, default=100_000)
    parser.add_argument("--bench-interval", type=int, default=2_500)
    args = parser.parse_args()

    side = args.side
    alpha = args.alpha
    total_epochs = args.epochs
    bench_interval = args.bench_interval

    wname = f"sl_s9_{side}_bg"
    wpath = f"bgsage/models/{wname}.weights"
    best_path = f"bgsage/models/{wname}.weights.best"
    train_file = f"bgsage/data/{side}-backgame-train-rollout"
    bench_file = f"bgsage/data/{side}-backgame-benchmark-rollout"

    print(f"=== {side.title()} BG Round 2: {total_epochs} epochs @ alpha={alpha} ===\n", flush=True)

    train_boards, train_probs = load_rollout(train_file)
    print(f"Training: {len(train_boards)} positions", flush=True)

    bench_boards, bench_probs = load_rollout(bench_file)
    bench_eq = 2 * bench_probs[:, 0] - 1 + bench_probs[:, 1] - bench_probs[:, 3] + bench_probs[:, 2] - bench_probs[:, 4]
    print(f"Benchmark: {len(bench_boards)} positions", flush=True)

    shutil.copy2(best_path, wpath)
    best_er = benchmark_er(bench_boards, bench_eq, wpath)
    print(f"Initial ER: {best_er:.2f}\n", flush=True)

    t_start = time.time()
    epochs_done = 0

    while epochs_done < total_epochs:
        chunk = min(bench_interval, total_epochs - epochs_done)
        epochs_done += chunk

        bgbot_cpp.cuda_supervised_train(
            boards=train_boards, targets=train_probs,
            weights_path=wpath, n_hidden=N_HIDDEN, n_inputs=N_INPUTS,
            alpha=alpha, epochs=chunk, batch_size=BATCH_SIZE,
            seed=42 + epochs_done, print_interval=500, save_path=wpath,
        )

        er = benchmark_er(bench_boards, bench_eq, wpath)

        improved = ""
        if er < best_er:
            best_er = er
            shutil.copy2(wpath, best_path)
            improved = " *BEST*"

        elapsed = time.time() - t_start
        print(f"  ep {epochs_done:6d}/{total_epochs}  ER={er:.2f}  best={best_er:.2f}  "
              f"elapsed={elapsed:.0f}s ({elapsed/60:.1f}m){improved}", flush=True)

    total_time = time.time() - t_start
    print(f"\n=== Training complete ===")
    print(f"Total: {epochs_done} epochs in {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"Best ER: {best_er:.2f}")
    print(f"Best weights: {best_path}")


if __name__ == "__main__":
    main()
