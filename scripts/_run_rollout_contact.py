"""Helper: run a rollout contact benchmark in a clean subprocess."""
import os
import sys
import time
import json
import argparse

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build'))
if not os.path.isdir(build_dir):
    build_dir = os.path.abspath(r'C:\Users\mghig\Dropbox\agents\bgbot\build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)

python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python'))
if not os.path.isdir(python_dir):
    python_dir = os.path.abspath(r'C:\Users\mghig\Dropbox\agents\bgbot\bgsage\python')
sys.path.insert(0, python_dir)

import bgbot_cpp
from bgsage.data import load_benchmark_file
from bgsage.weights import WeightConfig

bgbot_cpp.init_escape_tables()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, required=True)
    parser.add_argument('--truncation-depth', type=int, required=True)
    parser.add_argument('--decision-ply', type=int, required=True)
    parser.add_argument('--late-ply', type=int, default=-1)
    parser.add_argument('--late-threshold', type=int, default=20)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--step', type=int, default=1,
                        help='Subsample every Nth scenario (default: 1 = all)')
    args = parser.parse_args()

    w = WeightConfig.default()
    w.validate()

    # Always use main repo data dir (worktrees don't have data/)
    data_dir = os.path.abspath(r'C:\Users\mghig\Dropbox\agents\bgbot\bgsage\data')

    contact_file = os.path.join(data_dir, 'contact.bm')
    scenarios = load_benchmark_file(contact_file, step=args.step)
    n = scenarios.size()
    print(f'Loaded {n} scenarios (step={args.step})', file=sys.stderr, flush=True)

    kwargs = dict(
        n_trials=args.n_trials,
        truncation_depth=args.truncation_depth,
        decision_ply=args.decision_ply,
        n_threads=args.threads,
    )
    if args.late_ply >= 0:
        kwargs['late_ply'] = args.late_ply
        kwargs['late_threshold'] = args.late_threshold

    strat = bgbot_cpp.create_rollout_5nn(*w.weight_args, **kwargs)
    print('Rollout strategy created, scoring...', file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_rollout(scenarios, strat, 1)
    elapsed = time.perf_counter() - t0
    er = r.total_error / r.count * 1000 if r.count > 0 else 0
    print(json.dumps({'er': er, 'elapsed': elapsed, 'count': r.count}))


if __name__ == '__main__':
    main()
