# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Supervised-learning training for the Paskogammon net.

Refines the TD-trained Paskogammon net (single 400h / 244-input extended-contact
NN) against the Stage 9 rollout targets in pasko-train-rollout, selecting the
best weights on the held-out pasko-benchmark-rollout. This is the TD->SL
bootstrap: TD self-play gives realistic distributions, SL sharpens them against
the 1296-trial rollouts (same recipe as the Stage 9 back-game NNs).

Starts from the TD weights (td_pasko.weights.best) and writes sl_pasko.weights /
sl_pasko.weights.best. Score the result with:
    python bgsage/scripts/score_pasko_benchmark.py --pasko-weights models/sl_pasko.weights.best

Run after the training rollout has finished (positions still missing from
pasko-train-rollout are simply not trained on):
    python bgsage/scripts/run_pasko_sl_training.py
    python bgsage/scripts/run_pasko_sl_training.py --init models/td_pasko.weights --out sl_pasko_fromfinal
"""

import argparse
import gc
import os
import shutil
import sys
import time

# --- paths / imports ------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)

if sys.platform == 'win32':
    _cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    for _d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(_d):
            os.add_dll_directory(_d)
# Prefer bgsage/build (kept fresh, CUDA-enabled); parent build/ is a fallback.
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))

import bgbot_cpp
import numpy as np

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')
MODELS_DIR = os.path.join(_BGSAGE_ROOT, 'models')
N_HIDDEN = 400
N_INPUTS = 244

# SL schedule (same as the Stage 9 back-game NNs): (phase, steps, alpha).
SCHEDULE = [(1, 100000, 3.1), (2, 250000, 1.0)]


def load_rollout(path):
    """Boards (list[26 int]) + probs (Nx5 float32) from a *-rollout file."""
    boards, probs = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 31:
                continue
            boards.append([int(x) for x in parts[:26]])
            probs.append([float(x) for x in parts[26:31]])
    return boards, np.array(probs, dtype=np.float32)


def benchmark_er(bench_boards, bench_eq, weights_path):
    """Mean |1-ply cubeless equity - rollout equity| * 1000 on the benchmark."""
    nn = bgbot_cpp.NNStrategy(weights_path, N_HIDDEN, N_INPUTS)
    total_err = 0.0
    for board, target_eq in zip(bench_boards, bench_eq):
        total_err += abs(nn.evaluate_board(board, board)['equity'] - target_eq)
    return (total_err / len(bench_boards)) * 1000.0


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(_BGSAGE_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description="SL-train the Paskogammon net against Stage 9 rollouts")
    parser.add_argument('--train', default='pasko-train-rollout',
                        help='Training rollout file in bgsage/data (default: pasko-train-rollout)')
    parser.add_argument('--bench', default='pasko-benchmark-rollout',
                        help='Benchmark rollout file in bgsage/data (default: pasko-benchmark-rollout)')
    parser.add_argument('--init', default='models/td_pasko.weights.best',
                        help='Initial weights, abs or relative to bgsage/ (default: models/td_pasko.weights.best)')
    parser.add_argument('--out', default='sl_pasko',
                        help='Output model name in bgsage/models (default: sl_pasko -> sl_pasko.weights[.best])')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Cap total SL steps (0 = full schedule; use a small value to smoke-test).')
    args = parser.parse_args()

    train_path = os.path.join(DATA_DIR, args.train)
    bench_path = os.path.join(DATA_DIR, args.bench)
    init_path = _resolve(args.init)
    wpath = os.path.join(MODELS_DIR, f'{args.out}.weights')
    best_path = wpath + '.best'

    print('=== Paskogammon SL Training ===', flush=True)
    print(f'  init:  {init_path}', flush=True)
    print(f'  out:   {wpath}', flush=True)

    # Load data.
    train_boards_list, train_probs = load_rollout(train_path)
    train_boards = np.array(train_boards_list, dtype=np.int32)
    print(f'Loaded {len(train_boards):,} train positions from {args.train}', flush=True)

    bench_boards, bench_probs = load_rollout(bench_path)
    bench_eq = (2 * bench_probs[:, 0] - 1 + bench_probs[:, 1] - bench_probs[:, 3]
                + bench_probs[:, 2] - bench_probs[:, 4])
    print(f'Loaded {len(bench_boards):,} benchmark positions from {args.bench}', flush=True)

    # Pre-encode training inputs once (244 extended-contact features).
    t_enc = time.time()
    train_inputs = bgbot_cpp.encode_boards_batch(train_boards, N_INPUTS)
    print(f'Pre-encoded {len(train_boards):,} positions in {time.time() - t_enc:.1f}s\n', flush=True)

    # Initialise from the TD weights.
    if not os.path.exists(wpath):
        if not os.path.exists(init_path):
            print(f'ERROR: init weights not found: {init_path}')
            sys.exit(1)
        shutil.copy2(init_path, wpath)
        shutil.copy2(init_path, best_path)
        print(f'Initialised {args.out} from {os.path.basename(init_path)}', flush=True)

    best_er = benchmark_er(bench_boards, bench_eq, wpath)
    gc.collect()
    print(f'Initial benchmark ER: {best_er:.2f}  (PR {best_er/2:.2f})\n', flush=True)

    t_start = time.time()
    total_steps = 0
    for phase, n_steps, alpha in SCHEDULE:
        if args.max_steps:  # smoke-test / early-stop cap
            if total_steps >= args.max_steps:
                break
            n_steps = min(n_steps, args.max_steps - total_steps)
        # Resume each phase from the best weights so far.
        if os.path.exists(best_path):
            shutil.copy2(best_path, wpath)
        print(f'--- SL phase {phase}: {n_steps:,} steps @ alpha={alpha} (from best ER={best_er:.2f}) ---', flush=True)

        done = 0
        while done < n_steps:
            chunk = min(2500, n_steps - done)
            total_steps += chunk
            done += chunk
            # print_interval must be > 0 (0 crashes CUDA on Python 3.14).
            bgbot_cpp.cuda_supervised_train_preencoded(
                inputs=train_inputs, targets=train_probs, weights_path=wpath,
                n_hidden=N_HIDDEN, n_inputs=N_INPUTS, alpha=alpha, epochs=chunk,
                batch_size=args.batch_size, seed=42 + total_steps,
                print_interval=chunk + 1, save_path=wpath)
            er = benchmark_er(bench_boards, bench_eq, wpath)
            gc.collect()
            improved = ''
            if er < best_er:
                best_er = er
                shutil.copy2(wpath, best_path)
                improved = ' *BEST*'
            elapsed = time.time() - t_start
            print(f'  P{phase} {done:6d}/{n_steps} ER={er:.2f} best={best_er:.2f} '
                  f'{elapsed:.0f}s{improved}', flush=True)
        print(flush=True)

    total_time = time.time() - t_start
    print(f'=== Done: {total_steps:,} steps in {total_time/60:.1f} min ===')
    print(f'Best benchmark ER: {best_er:.2f}  (PR {best_er/2:.2f})')
    print(f'Best weights: {best_path}')
    print(f'\nScore it: python bgsage/scripts/score_pasko_benchmark.py --pasko-weights models/{args.out}.weights.best')


if __name__ == '__main__':
    main()
