# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
import sys, os, shutil, gc
sys.path.insert(0, 'build')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64')
import bgbot_cpp
import numpy as np

def benchmark_er(bench_boards, bench_eq, weights_path):
    nn = bgbot_cpp.NNStrategy(weights_path, 400, 244)
    total_err = 0.0
    for i in range(len(bench_boards)):
        r = nn.evaluate_board(bench_boards[i].tolist(), bench_boards[i].tolist())
        total_err += abs(r['equity'] - bench_eq[i])
    return (total_err / len(bench_boards)) * 1000.0

def run():
    boards_list, probs_list = [], []
    with open('bgsage/data/player-backgame-train-rollout') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31: continue
            boards_list.append([int(x) for x in parts[:26]])
            probs_list.append([float(x) for x in parts[26:31]])
    boards = np.array(boards_list, dtype=np.int32)
    probs = np.array(probs_list, dtype=np.float32)

    bench_boards_list, bench_probs_list = [], []
    with open('bgsage/data/player-backgame-benchmark-rollout') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31: continue
            bench_boards_list.append([int(x) for x in parts[:26]])
            bench_probs_list.append([float(x) for x in parts[26:31]])
    bench_boards = np.array(bench_boards_list, dtype=np.int32)
    bench_probs = np.array(bench_probs_list, dtype=np.float32)
    bench_eq = 2*bench_probs[:,0] - 1 + bench_probs[:,1] - bench_probs[:,3] + bench_probs[:,2] - bench_probs[:,4]

    wpath = 'bgsage/models/sl_s9_player_bg.weights'
    shutil.copy2('bgsage/models/sl_s8_anch_race.weights.best', wpath)

    er = benchmark_er(bench_boards, bench_eq, wpath)
    gc.collect()
    print(f'ER: {er:.2f}', flush=True)

    print('Training...', flush=True)
    result = bgbot_cpp.cuda_supervised_train(
        boards=boards, targets=probs, weights_path=wpath,
        n_hidden=400, n_inputs=244, alpha=3.1, epochs=10,
        batch_size=4096, seed=52, print_interval=1, save_path=wpath,
    )
    print(f'Done: {result["epochs_completed"]}', flush=True)

    er2 = benchmark_er(bench_boards, bench_eq, wpath)
    print(f'Post-train ER: {er2:.2f}', flush=True)

run()
