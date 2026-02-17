"""
Score a strategy against the Benchmark PR data.

Takes pre-generated benchmark PR data (decisions + rollout equities) and
scores any strategy by having it choose moves for each decision, then
measuring the equity error against the rollout-determined best move.

Benchmark PR = mean(error) * 500

Usage:
  python bgsage/scripts/score_benchmark_pr.py                      # production model at 0-ply
  python bgsage/scripts/score_benchmark_pr.py --plies 1            # production model at 1-ply
  python bgsage/scripts/score_benchmark_pr.py --model stage3       # specific model
  python bgsage/scripts/score_benchmark_pr.py --all-models         # all registered models
"""

import os
import sys
import json
import time
import argparse

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.weights import MODELS, WeightConfig

DATA_DIR = os.path.join(project_dir, 'data')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark_pr')

GAME_PLANS = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']

FILTER_MAX_MOVES = 5
FILTER_THRESHOLD = 0.08


def load_benchmark_data():
    """Load decisions and rollout equities, returning decisions with
    rollout equities pre-attached to each candidate."""
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')

    if not os.path.exists(decisions_file):
        print(f"ERROR: {decisions_file} not found. Run generate_benchmark_pr.py first.")
        sys.exit(1)

    with open(decisions_file, 'r') as f:
        saved = json.load(f)
    decisions = saved['decisions']

    # Load rollout equities
    rollout_equities = {}
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = tuple(rec['board'])
                rollout_equities[key] = rec['equity']

    # Pre-attach rollout equities to each decision for C++ consumption.
    # Skip decisions where not all candidates have rollout data.
    prepared = []
    n_missing = 0
    for dec in decisions:
        candidates = dec['candidates']
        eqs = []
        all_have = True
        for cand in candidates:
            key = tuple(cand)
            if key not in rollout_equities:
                all_have = False
                break
            eqs.append(rollout_equities[key])
        if not all_have:
            n_missing += 1
            continue
        prepared.append({
            'board': dec['board'],
            'dice': dec['dice'],
            'candidates': candidates,
            'rollout_equities': eqs,
        })

    if n_missing:
        print(f"  ({n_missing} decisions missing rollout data, excluded)")

    return prepared, saved


def print_result(label, result):
    """Print PR scoring result from C++ dict."""
    total = result['total_decisions']
    if total == 0:
        print(f"  {label}: No decisions could be scored!")
        return None

    overall = result['overall_pr']
    n_outside = result['n_outside']
    n_skipped = result['n_skipped']
    gp_pr = result['per_plan_pr']
    gp_n = result['per_plan_n']
    gp_n_err = result['per_plan_n_with_error']

    total_with_error = sum(gp_n_err.values())
    print(f"  {label}: PR = {overall:.2f} "
          f"({total} decisions, "
          f"{total_with_error}/{total} with error, "
          f"{n_outside} outside filtered set)")
    if n_skipped:
        print(f"    ({n_skipped} decisions skipped)")

    parts = []
    for gp in GAME_PLANS:
        if gp_n[gp] > 0:
            parts.append(f"{gp}={gp_pr[gp]:.2f}({gp_n[gp]})")
    print(f"    Per plan: {', '.join(parts)}")

    return {
        'overall': overall,
        'per_plan': {gp: gp_pr[gp] if gp_n[gp] > 0 else None for gp in GAME_PLANS},
        'per_plan_n': dict(gp_n),
    }


def main():
    parser = argparse.ArgumentParser(description='Score strategies on Benchmark PR')
    WeightConfig.add_model_arg(parser)
    parser.add_argument('--plies', type=int, default=0,
                        help='Ply depth (default: 0)')
    parser.add_argument('--all-models', action='store_true',
                        help='Score all registered models at the given ply depth')
    parser.add_argument('--all-plies', action='store_true',
                        help='Score at 0-ply and 1-ply')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0 = auto)')
    args = parser.parse_args()

    bgbot_cpp.init_escape_tables()

    print("Loading benchmark data...")
    decisions, meta = load_benchmark_data()
    print(f"  {len(decisions)} scoreable decisions")
    print(f"  Data from {meta['n_games']} games, {meta['n_turns']} turns")
    print()

    # Determine which models and plies to score
    if args.all_models:
        model_names = sorted(MODELS.keys())
    else:
        model_names = [args.model]

    if args.all_plies:
        plies_list = [0, 1]
    else:
        plies_list = [args.plies]

    results = {}

    for model_name in model_names:
        try:
            w = WeightConfig.from_model(model_name)
            w.validate()
        except (KeyError, FileNotFoundError) as e:
            print(f"Model {model_name}: {e}, skipping.")
            continue

        wt = (w.purerace, w.racing, w.attacking, w.priming, w.anchoring)
        ht = w.hidden_sizes

        for plies in plies_list:
            label = f"{model_name} ({plies}-ply)"
            t0 = time.perf_counter()

            if plies == 0:
                raw = bgbot_cpp.score_benchmark_pr_0ply(
                    decisions, *wt, *ht, n_threads=args.threads)
            else:
                raw = bgbot_cpp.score_benchmark_pr_nply(
                    decisions, *wt, *ht,
                    n_plies=plies,
                    filter_max_moves=FILTER_MAX_MOVES,
                    filter_threshold=FILTER_THRESHOLD,
                    n_threads=args.threads)

            elapsed = time.perf_counter() - t0
            print(f"  [{elapsed:.1f}s] ", end='')
            result = print_result(label, raw)
            if result is not None:
                results[label] = result

    if results:
        print()
        print("=" * 60)
        print("Benchmark PR Results (lower is better)")
        print("=" * 60)

        # Header
        gp_short = {'purerace': 'PureRace', 'racing': 'Racing',
                     'attacking': 'Attacking', 'priming': 'Priming',
                     'anchoring': 'Anchoring'}
        header = f"  {'Strategy':<22} {'Overall':>7}"
        for gp in GAME_PLANS:
            header += f"  {gp_short[gp]:>9}"
        print(header)
        print("  " + "-" * 58)

        for label, res in results.items():
            row = f"  {label:<22} {res['overall']:>7.2f}"
            for gp in GAME_PLANS:
                pr = res['per_plan'].get(gp)
                if pr is not None:
                    row += f"  {pr:>9.2f}"
                else:
                    row += f"  {'---':>9}"
            print(row)

        # Print decision counts
        print()
        first_res = next(iter(results.values()))
        counts = f"  {'(N decisions)':<22} {'':>7}"
        for gp in GAME_PLANS:
            n = first_res['per_plan_n'].get(gp, 0)
            counts += f"  {n:>9}"
        print(counts)


if __name__ == '__main__':
    main()
