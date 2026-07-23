# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Experiment: higher learning rate SL for player backgame NN.

Tests whether the current S9 player_bg NN is stuck in a local minimum by
training with alpha=31 (25k epochs) then alpha=10 (100k epochs), benchmarking
every 2.5k epochs. Starts from a COPY of the current S9 player_bg weights
so the production model is not affected.

Usage:
    python bgsage/scripts/run_backgame_high_lr_experiment.py
"""
import gc, os, shutil, sys, time
sys.path.insert(0, 'build')
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
if os.path.isdir(cuda_path):
    os.add_dll_directory(cuda_path)
import bgbot_cpp
import numpy as np

N_HIDDEN = 400
N_INPUTS = 244
EXPERIMENT = 'bg_high_lr'
WNAME = f'exp_{EXPERIMENT}_player_bg'

def benchmark_er(bench_boards_raw, bench_eq, weights_path):
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    total_err = 0.0
    n = len(bench_boards_raw)
    for i in range(n):
        board = bench_boards_raw[i]
        r = nn.evaluate_board(board, board)
        total_err += abs(r['equity'] - bench_eq[i])
    return (total_err / n) * 1000.0

def run():
    print(f'=== Backgame High-LR Experiment: player_bg ===', flush=True)

    # Load training data
    boards_list, probs_list = [], []
    with open('bgsage/data/player-backgame-train-rollout') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31: continue
            boards_list.append([int(x) for x in parts[:26]])
            probs_list.append([float(x) for x in parts[26:31]])
    train_boards = np.array(boards_list, dtype=np.int32)
    train_probs = np.array(probs_list, dtype=np.float32)
    print(f'Loaded {len(train_boards)} train positions', flush=True)

    # Load benchmark data
    bench_boards_list, bench_probs_list = [], []
    with open('bgsage/data/player-backgame-benchmark-rollout') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31: continue
            bench_boards_list.append([int(x) for x in parts[:26]])
            bench_probs_list.append([float(x) for x in parts[26:31]])
    bench_boards_raw = [b for b in bench_boards_list]
    bench_probs = np.array(bench_probs_list, dtype=np.float32)
    bench_eq = 2*bench_probs[:,0]-1+bench_probs[:,1]-bench_probs[:,3]+bench_probs[:,2]-bench_probs[:,4]
    print(f'Loaded {len(bench_boards_raw)} bench positions', flush=True)

    # Pre-encode training inputs
    print('Pre-encoding training inputs...', flush=True)
    t_enc = time.time()
    train_inputs = bgbot_cpp.encode_boards_batch(train_boards, N_INPUTS)
    print(f'Pre-encoded in {time.time()-t_enc:.1f}s', flush=True)

    # Copy current S9 player_bg weights for the experiment
    src = 'bgsage/models/sl_s9_player_bg.weights.best'
    wpath = f'bgsage/models/{WNAME}.weights'
    best_path = f'bgsage/models/{WNAME}.weights.best'
    shutil.copy2(src, wpath)
    shutil.copy2(src, best_path)
    print(f'Copied S9 player_bg weights to {wpath}', flush=True)

    best_er = benchmark_er(bench_boards_raw, bench_eq, wpath)
    gc.collect()
    print(f'Initial ER: {best_er:.2f}\n', flush=True)

    t_start = time.time()
    total_epochs = 0
    schedule = [
        (1, 25000, 31.0),
        (2, 100000, 10.0),
    ]
    for phase_num, n_epochs, alpha in schedule:
        if os.path.exists(best_path):
            shutil.copy2(best_path, wpath)
            print(f'--- Phase {phase_num}: {n_epochs}ep @ alpha={alpha} (starting from best ER={best_er:.2f}) ---', flush=True)

        epochs_done = 0
        while epochs_done < n_epochs:
            chunk = min(2500, n_epochs - epochs_done)
            total_epochs += chunk
            epochs_done += chunk
            bgbot_cpp.cuda_supervised_train_preencoded(
                inputs=train_inputs, targets=train_probs, weights_path=wpath,
                n_hidden=N_HIDDEN, n_inputs=N_INPUTS, alpha=alpha, epochs=chunk,
                batch_size=4096, seed=42+total_epochs,
                print_interval=chunk+1, save_path=wpath)
            er = benchmark_er(bench_boards_raw, bench_eq, wpath)
            gc.collect()
            improved = ''
            if er < best_er:
                best_er = er
                shutil.copy2(wpath, best_path)
                improved = ' *BEST*'
            elapsed = time.time() - t_start
            print(f'  P{phase_num} ep {epochs_done:6d}/{n_epochs} ER={er:.2f} best={best_er:.2f} {elapsed:.0f}s{improved}', flush=True)
        print(flush=True)

    total_time = time.time() - t_start
    print(f'=== Done: {total_epochs}ep in {total_time:.0f}s ({total_time/60:.1f}m) ===')
    print(f'Best ER: {best_er:.2f}')
    print(f'Best weights: {best_path}')

run()
