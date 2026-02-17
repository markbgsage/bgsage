"""Reference-position benchmark for MultiPly serial/parallel parity and speed.

Usage:
  python python/bench_multipy_reference.py --build-dir build_macos --repeats 25
  python python/bench_multipy_reference.py --plies 1 2 3 --threads 8
"""

import argparse
import os
import statistics
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', default='build_macos')
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--plies', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--full-depth', action='store_true')
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_dir, args.build_dir)
    std_build = os.path.join(project_dir, 'build')

    sys.path.insert(0, build_dir)
    sys.path.insert(0, std_build)
    sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

    import bgbot_cpp

    # 26-element reference position from request
    board = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 1, 0, 0, -3, 0, -5, 0, 0, 0, 0, 1, 0, 0][:26]
    if len(board) != 26:
        raise SystemExit(f'Ref board must be 26 elements, got {len(board)}')

    models = {
        'purerace': os.path.join(project_dir, 'models', 'sl_s5_purerace.weights.best'),
        'racing': os.path.join(project_dir, 'models', 'sl_s5_racing.weights.best'),
        'attacking': os.path.join(project_dir, 'models', 'sl_s5_attacking.weights.best'),
        'priming': os.path.join(project_dir, 'models', 'sl_s5_priming.weights.best'),
        'anchoring': os.path.join(project_dir, 'models', 'sl_s5_anchoring.weights.best'),
    }
    for p in models.values():
        if not os.path.exists(p):
            raise SystemExit(f'Missing model: {p}')

    # Stage-5 hidden sizes
    nh = (200, 400, 400, 400, 400)

    def run_case(ply: int, parallel: bool):
        mp = bgbot_cpp.create_multipy_5nn(
            models['purerace'], models['racing'], models['attacking'],
            models['priming'], models['anchoring'],
            nh[0], nh[1], nh[2], nh[3], nh[4],
            n_plies=ply,
            full_depth_opponent=args.full_depth,
            parallel_evaluate=parallel,
            parallel_threads=args.threads,
        )

        times_us = []
        probs = []
        eqs = []
        for _ in range(args.repeats):
            mp.clear_cache()  # cold run behavior
            t0 = time.perf_counter()
            r = mp.evaluate_board(board, board)
            times_us.append((time.perf_counter() - t0) * 1e6)
            eqs.append(round(r['equity'], 12))
            probs.append(tuple(round(v, 10) for v in r['probs']))

        return {
            'times_us': times_us,
            'median_us': statistics.median(times_us),
            'mean_us': statistics.mean(times_us),
            'min_us': min(times_us),
            'max_us': max(times_us),
            'equity': eqs[0],
            'prob_variants': len(set(probs)),
            'probs': probs[0],
        }

    print(f'Board: {board}')
    print(f'Build dir: {build_dir}')
    print(f'Repeats: {args.repeats}')
    print(f'Threads arg: {args.threads or "auto"}, full_depth: {args.full_depth}')

    for ply in args.plies:
        serial = run_case(ply, parallel=False)
        parallel = run_case(ply, parallel=True)

        eq_match = abs(serial['equity'] - parallel['equity']) < 1e-9
        speedup = serial['median_us'] / parallel['median_us'] if parallel['median_us'] > 0 else float('inf')

        print(f'\n=== {ply}-ply ===')
        print(f'Serial   med/mean (us): {serial["median_us"]:10.1f} / {serial["mean_us"]:10.1f}  | eq={serial["equity"]:.12f}  probs_variants={serial["prob_variants"]}')
        print(f'Parallel med/mean (us): {parallel["median_us"]:10.1f} / {parallel["mean_us"]:10.1f}  | eq={parallel["equity"]:.12f}  probs_variants={parallel["prob_variants"]}')
        print(f'Speedup (serial/parallel): {speedup:.2f}x')
        print(f'Range us: serial {serial["min_us"]:.1f}-{serial["max_us"]:.1f} | parallel {parallel["min_us"]:.1f}-{parallel["max_us"]:.1f}')
        print(f'Parity check: {"PASS" if eq_match else "FAIL"}')


if __name__ == '__main__':
    main()
