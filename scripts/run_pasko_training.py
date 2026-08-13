# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""
TD(0) self-play training for "Paskogammon" — a single 244-input extended-contact
neural net (same hidden size / inputs as a Stage 9 contact NN), trained from
scratch (random weights) on games that start from the Paskogammon opening.

Paskogammon differs from standard backgammon in two ways that matter for
self-play:
  * every game starts from a fixed non-standard position (PASKO_START below),
  * the positive player is ALWAYS on roll, and the opening roll MAY be doubles
    (no re-roll, no perspective flip).
The doubling cube is not part of cubeless TD self-play — it is applied at
analysis/play time via the usual Janowski machinery, so nothing special is
needed here for it.

Progress is reported with the player back-game equity benchmark (mean
|equity - rollout_target| * 1000) — the same benchmark built for the custom
back-game training — instead of the contact/race GNUbg benchmarks, because the
Paskogammon opening produces lots of back games.

Default schedule matches the regular TD training LR schedule:
    Phase 1: 200k games @ alpha=0.1
    Phase 2: 1M   games @ alpha=0.02   (resumes from Phase 1)

Usage:
    python bgsage/scripts/run_pasko_training.py
    python bgsage/scripts/run_pasko_training.py --games 200000 --games2 1000000
    python bgsage/scripts/run_pasko_training.py --games 5000 --games2 0   # quick smoke test
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime

# ---- Import paths -----------------------------------------------------------
# The compiled bgbot_cpp lives in the parent project's build/ (same convention
# as run_td_training.py); back-game data + models live inside the bgsage repo.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BGSAGE_ROOT = os.path.dirname(_SCRIPT_DIR)                 # .../bgbot/bgsage
_PARENT_ROOT = os.path.dirname(_BGSAGE_ROOT)               # .../bgbot

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    for d in (os.path.join(_PARENT_ROOT, 'build'), os.path.join(_BGSAGE_ROOT, 'build')):
        if os.path.isdir(d):
            os.add_dll_directory(d)

# Prefer bgsage/build (kept in sync by the MSVC copy step); the parent build/
# may be held open by a running backend, so it is only a fallback.
sys.path.insert(0, os.path.join(_PARENT_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'build'))
sys.path.insert(0, os.path.join(_BGSAGE_ROOT, 'python'))

import bgbot_cpp

DATA_DIR = os.path.join(_BGSAGE_ROOT, 'data')
MODELS_DIR = os.path.join(_BGSAGE_ROOT, 'models')

# Paskogammon starting position (positive player on roll). 15 checkers each side.
PASKO_START = [0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,
               5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2, 0]


def load_equity_benchmark(filepath, step=1):
    """Load (boards, target_equities) from a *-backgame-*-rollout file.

    Each line: 26 ints (board) + 5 floats (W, Gw, Bw, Gl, Bl). Target equity is
    the cubeless post-move equity from the positive player's perspective.
    """
    boards, targets = [], []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if step > 1 and (i % step != 0):
                continue
            parts = line.split()
            if len(parts) < 31:
                continue
            board = [int(x) for x in parts[:26]]
            p = [float(x) for x in parts[26:31]]
            eq = 2 * p[0] - 1 + p[1] - p[3] + p[2] - p[4]
            boards.append(board)
            targets.append(eq)
    return boards, targets


def run_phase(phase, n_games, alpha, args, resume_from, bench_boards, bench_targets):
    print(f'=== Pasko TD Phase {phase}: {n_games // 1000}k @ alpha={alpha} ===')
    print(f'  Hidden:  {args.hidden}, 244 extended-contact inputs')
    print(f'  Model:   {os.path.join(MODELS_DIR, args.model_name)}.weights')
    if resume_from:
        print(f'  Resume:  {resume_from}')
    print(flush=True)

    t0 = time.time()
    result = bgbot_cpp.td_train_pasko(
        n_games=n_games,
        alpha=alpha,
        n_hidden=args.hidden,
        eps=args.eps,
        seed=args.seed,
        benchmark_interval=args.benchmark_interval,
        model_name=args.model_name,
        models_dir=MODELS_DIR,
        resume_from=resume_from,
        start_board=PASKO_START,
        bench_boards=bench_boards,
        bench_targets=bench_targets,
    )
    print(f'Phase {phase} done: {result.games_played} games in {time.time() - t0:.1f}s',
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Paskogammon TD self-play training (single 244-input NN)')
    parser.add_argument('--games', type=int, default=200000, help='Phase 1 games (default: 200000)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Phase 1 learning rate (default: 0.1)')
    parser.add_argument('--games2', type=int, default=1000000, help='Phase 2 games (default: 1000000; 0 = skip)')
    parser.add_argument('--alpha2', type=float, default=0.02, help='Phase 2 learning rate (default: 0.02)')
    parser.add_argument('--hidden', type=int, default=400, help='Hidden nodes (default: 400 = Stage 9 contact size)')
    parser.add_argument('--eps', type=float, default=0.1, help='Weight init scale (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: 42)')
    parser.add_argument('--benchmark-interval', type=int, default=10000, help='Benchmark every N games (default: 10000)')
    parser.add_argument('--benchmark-step', type=int, default=1, help='Use every Nth benchmark position (default: 1 = all)')
    parser.add_argument('--model-name', type=str, default='td_pasko', help='Model name prefix (default: td_pasko)')
    parser.add_argument('--resume-from', type=str, default='', help='Resume Phase 1 from this .weights file')
    parser.add_argument('--benchmark-side', choices=['player', 'opponent'], default='player',
                        help='Which back-game benchmark to report (default: player)')
    parser.add_argument('--log-dir', type=str, default='', help='Directory for log file (default: <bgsage>/logs)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Tee all output (incl. C++ std::cout) to console + log file ----------
    log_dir = args.log_dir if args.log_dir else os.path.join(_BGSAGE_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_k = (args.games + args.games2) // 1000
    log_path = os.path.join(log_dir, f'pasko_{total_k}k_{timestamp}.log')

    log_file = open(log_path, 'w')
    original_stdout_fd = os.dup(1)
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, 1)
    os.close(pipe_w)

    def tee_thread():
        with os.fdopen(pipe_r, 'r', errors='replace') as reader:
            for line in reader:
                os.write(original_stdout_fd, line.encode())
                log_file.write(line)
                log_file.flush()

    tee = threading.Thread(target=tee_thread, daemon=True)
    tee.start()
    sys.stdout = os.fdopen(os.dup(1), 'w')

    print(f'Log file: {log_path}')
    print(f'bgbot_cpp: {bgbot_cpp.__file__}')
    print()

    # ---- Load back-game benchmark --------------------------------------------
    bench_file = os.path.join(DATA_DIR, f'{args.benchmark_side}-backgame-benchmark-rollout')
    if not os.path.exists(bench_file):
        print(f'ERROR: benchmark file not found: {bench_file}')
        sys.exit(1)
    t0 = time.time()
    bench_boards, bench_targets = load_equity_benchmark(bench_file, step=args.benchmark_step)
    print(f'Loaded {len(bench_boards)} {args.benchmark_side} back-game benchmark positions '
          f'from {bench_file} in {time.time() - t0:.1f}s')
    print(f'Reported score "bg=" is {args.benchmark_side} back-game ER '
          f'(mean |equity - rollout_target| * 1000; lower is better)')
    print(flush=True)

    weights_path = os.path.join(MODELS_DIR, f'{args.model_name}.weights')

    # ---- Phase 1 (from scratch unless --resume-from) -------------------------
    run_phase(1, args.games, args.alpha, args, args.resume_from, bench_boards, bench_targets)

    # ---- Phase 2 (resumes from Phase 1 final weights) ------------------------
    if args.games2 > 0:
        print()
        run_phase(2, args.games2, args.alpha2, args, weights_path, bench_boards, bench_targets)

    print()
    print('Final weights:')
    print(f'  {weights_path}')
    print(f'  {weights_path}.best   (best benchmark ER)')

    sys.stdout.close()
    tee.join(timeout=5)
    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    main()
