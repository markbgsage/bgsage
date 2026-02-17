"""
Capacity experiment: single NN with 196 Tesauro inputs, varying hidden layer sizes.
Tests TD→SL pipeline for each hidden size to find the optimal capacity.

All NNs use the same encoding (196 Tesauro inputs), same training data (contact+crashed),
and same benchmark (contact.bm). Only hidden layer size varies.

TD phases run in parallel (CPU-bound, one per core). SL phases run sequentially (GPU-bound).

Usage:
    python run_capacity_experiment.py
    python run_capacity_experiment.py --hidden-sizes 40 80 120 160 200 250
    python run_capacity_experiment.py --td-games 100000 --sl-epochs 300
    python run_capacity_experiment.py --skip-td   # reuse existing TD weights, SL only
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file, load_gnubg_training_data

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')
LOGS_DIR = os.path.join(project_dir, 'logs')


def run_td_training(n_hidden, n_games, alpha, seed=42, benchmark_step=10):
    """Run TD self-play training for a single NN. Returns weight file path."""
    model_name = f'capacity_{n_hidden}h'
    weights_path = os.path.join(MODELS_DIR, f'{model_name}.weights')

    # Load benchmark for progress tracking
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    benchmark = None
    if os.path.exists(bm_path):
        benchmark = load_benchmark_file(bm_path, step=benchmark_step)

    print(f'[TD {n_hidden}h] Starting {n_games//1000}k games @ alpha={alpha}', flush=True)
    t0 = time.time()

    result = bgbot_cpp.td_train(
        n_games=n_games,
        alpha=alpha,
        n_hidden=n_hidden,
        eps=0.1,
        seed=seed,
        benchmark_interval=10000,
        model_name=model_name,
        models_dir=MODELS_DIR,
        resume_from='',
        scenarios=benchmark,
    )

    elapsed = time.time() - t0
    print(f'[TD {n_hidden}h] Done: {result.games_played} games in {elapsed:.1f}s', flush=True)
    return weights_path


def run_td_worker(args):
    """Worker function for parallel TD training (runs in subprocess)."""
    n_hidden, n_games, alpha, seed = args

    # Write a temp script to avoid f-string nesting issues
    import tempfile
    script = (
        'import os, sys, time\n'
        'if sys.platform == "win32":\n'
        '    cuda_bin = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin\\x64"\n'
        '    if os.path.isdir(cuda_bin): os.add_dll_directory(cuda_bin)\n'
        f'    bd = r"{build_dir}"\n'
        '    if os.path.isdir(bd): os.add_dll_directory(bd)\n'
        f'sys.path.insert(0, r"{build_dir}")\n'
        f'sys.path.insert(0, r"{os.path.join(project_dir, "python")}")\n'
        'import bgbot_cpp\n'
        'from bgsage.data import load_benchmark_file\n'
        f'bm = load_benchmark_file(r"{os.path.join(DATA_DIR, "contact.bm")}", step=10)\n'
        f'print("[TD {n_hidden}h] Starting {n_games // 1000}k games @ alpha={alpha}", flush=True)\n'
        't0 = time.time()\n'
        f'result = bgbot_cpp.td_train(\n'
        f'    n_games={n_games}, alpha={alpha}, n_hidden={n_hidden},\n'
        f'    eps=0.1, seed={seed}, benchmark_interval=10000,\n'
        f'    model_name="capacity_{n_hidden}h",\n'
        f'    models_dir=r"{MODELS_DIR}", resume_from="", scenarios=bm)\n'
        'print(f"[TD ' + str(n_hidden) + 'h] Done in {time.time()-t0:.1f}s", flush=True)\n'
    )

    fd, script_path = tempfile.mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(script)
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True, text=True, timeout=3600)
        return n_hidden, result.stdout, result.stderr
    finally:
        os.unlink(script_path)


def run_sl_training(n_hidden, td_weights_path, n_inputs, epochs, alpha, batch_size=128, seed=42):
    """Run GPU SL training for a single NN. Returns best score."""
    save_prefix = f'capacity_{n_hidden}h'
    save_path = os.path.join(MODELS_DIR, save_prefix)

    # Load training data (contact + crashed)
    print(f'\n[SL {n_hidden}h] Loading training data...', flush=True)
    t0 = time.time()

    boards1, targets1 = load_gnubg_training_data(
        os.path.join(DATA_DIR, 'contact-train-data'))
    boards2, targets2 = load_gnubg_training_data(
        os.path.join(DATA_DIR, 'crashed-train-data'))

    import numpy as np
    boards = np.concatenate([boards1, boards2])
    targets = np.concatenate([targets1, targets2])
    print(f'  Loaded {len(boards)} positions in {time.time()-t0:.1f}s', flush=True)

    # Load benchmark
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    benchmark = load_benchmark_file(bm_path, step=10)
    print(f'  Loaded {len(benchmark)} benchmark scenarios', flush=True)

    # Run GPU SL
    print(f'[SL {n_hidden}h] Training {epochs} epochs @ alpha={alpha}, '
          f'starting from {td_weights_path}', flush=True)

    result = bgbot_cpp.cuda_supervised_train(
        boards, targets,
        weights_path=td_weights_path,
        n_hidden=n_hidden,
        n_inputs=n_inputs,
        alpha=alpha,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        print_interval=1,
        save_path=save_path,
        benchmark_scenarios=benchmark,
    )

    print(f'[SL {n_hidden}h] Best: {result["best_score"]:.2f} @ epoch {result["best_epoch"]}',
          flush=True)
    return result


def score_final_benchmark(weights_path, n_hidden, n_inputs=196):
    """Score a single NN against the full contact benchmark."""
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    scenarios = load_benchmark_file(bm_path)
    result = bgbot_cpp.score_benchmarks_nn(scenarios, weights_path, n_hidden, n_inputs)
    return result.score()


def main():
    parser = argparse.ArgumentParser(description='Capacity Experiment: Single NN, varying hidden sizes')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[40, 80, 120, 160, 200, 250],
                        help='Hidden layer sizes to test (default: 40 80 120 160 200 250)')
    parser.add_argument('--td-games', type=int, default=50000,
                        help='TD games per hidden size (default: 50000)')
    parser.add_argument('--td-alpha', type=float, default=0.1,
                        help='TD learning rate (default: 0.1)')
    parser.add_argument('--sl-epochs', type=int, default=200,
                        help='SL epochs per hidden size (default: 200)')
    parser.add_argument('--sl-alpha', type=float, default=1.0,
                        help='SL learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base RNG seed (default: 42)')
    parser.add_argument('--skip-td', action='store_true',
                        help='Skip TD phase, reuse existing TD weights')
    parser.add_argument('--skip-sl', action='store_true',
                        help='Skip SL phase, just report TD scores')
    parser.add_argument('--n-inputs', type=int, default=196,
                        help='Number of inputs (default: 196 = Tesauro)')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sizes_str = '_'.join(str(h) for h in args.hidden_sizes)
    log_path = os.path.join(LOGS_DIR, f'capacity_{sizes_str}_{timestamp}.log')

    # Tee output
    import threading
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

    print(f'=== Capacity Experiment ===')
    print(f'Log: {log_path}')
    print(f'Hidden sizes: {args.hidden_sizes}')
    print(f'Inputs: {args.n_inputs}')
    print(f'TD: {args.td_games//1000}k games @ alpha={args.td_alpha}')
    print(f'SL: {args.sl_epochs} epochs @ alpha={args.sl_alpha}')
    print(f'Seed: {args.seed}')
    print(flush=True)

    # ========== Phase 1: TD self-play (parallel) ==========
    td_weights = {}

    if not args.skip_td:
        print('\n--- Phase 1: TD Self-Play (parallel) ---', flush=True)
        t0 = time.time()

        # Run TD training for each hidden size in parallel as subprocesses
        workers = []
        for i, n_hidden in enumerate(args.hidden_sizes):
            workers.append((n_hidden, args.td_games, args.td_alpha, args.seed + i * 100))

        with ProcessPoolExecutor(max_workers=len(args.hidden_sizes)) as executor:
            futures = {executor.submit(run_td_worker, w): w[0] for w in workers}
            for future in as_completed(futures):
                n_hidden = futures[future]
                try:
                    _, stdout, stderr = future.result()
                    if stdout:
                        print(stdout, end='', flush=True)
                    if stderr:
                        print(f'[TD {n_hidden}h STDERR] {stderr}', flush=True)
                except Exception as e:
                    print(f'[TD {n_hidden}h] ERROR: {e}', flush=True)

        td_elapsed = time.time() - t0
        print(f'\nAll TD runs complete in {td_elapsed:.1f}s', flush=True)

    # Collect TD weight paths
    for n_hidden in args.hidden_sizes:
        path = os.path.join(MODELS_DIR, f'capacity_{n_hidden}h.weights')
        if os.path.exists(path):
            td_weights[n_hidden] = path
        else:
            print(f'WARNING: TD weights not found for {n_hidden}h: {path}')

    # Score TD results
    print('\n--- TD Benchmark Scores ---', flush=True)
    td_scores = {}
    for n_hidden in args.hidden_sizes:
        if n_hidden in td_weights:
            score = score_final_benchmark(td_weights[n_hidden], n_hidden, args.n_inputs)
            td_scores[n_hidden] = score
            print(f'  {n_hidden:4d}h: {score:.2f}', flush=True)

    if args.skip_sl:
        print('\n--- Skipping SL phase ---', flush=True)
        print_summary(args.hidden_sizes, td_scores, {})
        cleanup_stdout(log_file, original_stdout_fd, tee, log_path)
        return

    # ========== Phase 2: SL training (sequential, GPU) ==========
    print('\n--- Phase 2: Supervised Learning (sequential, GPU) ---', flush=True)
    sl_scores = {}

    for n_hidden in args.hidden_sizes:
        if n_hidden not in td_weights:
            print(f'  Skipping {n_hidden}h (no TD weights)', flush=True)
            continue

        result = run_sl_training(
            n_hidden=n_hidden,
            td_weights_path=td_weights[n_hidden],
            n_inputs=args.n_inputs,
            epochs=args.sl_epochs,
            alpha=args.sl_alpha,
            seed=args.seed,
        )
        sl_scores[n_hidden] = result['best_score']

    # Final full benchmark scores on SL best weights
    print('\n--- Final SL Benchmark Scores (full contact.bm) ---', flush=True)
    final_sl_scores = {}
    for n_hidden in args.hidden_sizes:
        best_path = os.path.join(MODELS_DIR, f'capacity_{n_hidden}h.weights.best')
        if os.path.exists(best_path):
            score = score_final_benchmark(best_path, n_hidden, args.n_inputs)
            final_sl_scores[n_hidden] = score
            print(f'  {n_hidden:4d}h: {score:.2f}', flush=True)

    # ========== Summary ==========
    print_summary(args.hidden_sizes, td_scores, final_sl_scores)

    cleanup_stdout(log_file, original_stdout_fd, tee, log_path)


def print_summary(hidden_sizes, td_scores, sl_scores):
    print('\n' + '=' * 50)
    print('CAPACITY EXPERIMENT SUMMARY')
    print('=' * 50)
    print(f'{"Hidden":>8s}  {"TD Score":>10s}  {"SL Score":>10s}')
    print(f'{"------":>8s}  {"--------":>10s}  {"--------":>10s}')
    for n_hidden in hidden_sizes:
        td = f'{td_scores[n_hidden]:.2f}' if n_hidden in td_scores else '-'
        sl = f'{sl_scores[n_hidden]:.2f}' if n_hidden in sl_scores else '-'
        print(f'{n_hidden:8d}  {td:>10s}  {sl:>10s}')
    print('=' * 50)
    print(flush=True)


def cleanup_stdout(log_file, original_stdout_fd, tee, log_path):
    sys.stdout.close()
    tee.join(timeout=5)
    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    main()
