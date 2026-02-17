"""
Score a strategy against the Cubeful Benchmark PR data.

Scores both:
1. Checker-play decisions: strategy picks best move, error = best_rollout_eq - chosen_rollout_eq
2. Cube decisions: strategy's cube action vs rollout-derived cubeful equities

For cube decisions:
- Load rollout probs for the flipped pre-roll board
- Invert probs to get cubeless pre-roll probabilities
- Apply Janowski conversion to get ND/DT/DP cubeful equities
- Score: error based on whether strategy makes correct double/take decision
  and the equity difference from optimal play

Cubeful PR = mean(all_errors) * 500

Usage:
  python python/score_cubeful_benchmark_pr.py                # Score Stage 5 at 0-ply
  python python/score_cubeful_benchmark_pr.py --plies 1      # Score at 1-ply
  python python/score_cubeful_benchmark_pr.py --checker-only  # Checker-play only
  python python/score_cubeful_benchmark_pr.py --cube-only     # Cube only
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

MODELS_DIR = os.path.join(project_dir, 'models')
DATA_DIR = os.path.join(project_dir, 'data')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark_cubeful_pr')

STAGES = {
    5: {
        'hidden': (200, 400, 400, 400, 400),
        'weights': {
            'purerace': 'sl_s5_purerace.weights.best',
            'racing': 'sl_s5_racing.weights.best',
            'attacking': 'sl_s5_attacking.weights.best',
            'priming': 'sl_s5_priming.weights.best',
            'anchoring': 'sl_s5_anchoring.weights.best',
        },
    },
}

FILTER_MAX_MOVES = 5
FILTER_THRESHOLD = 0.08

GAME_PLANS = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']


def get_stage_weights(stage):
    cfg = STAGES[stage]
    paths = {}
    for t in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        path = os.path.join(MODELS_DIR, cfg['weights'][t])
        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            return None, None
        paths[t] = path
    return paths, cfg['hidden']


def load_benchmark_data():
    """Load cubeful benchmark decisions and rollout data."""
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')

    if not os.path.exists(decisions_file):
        print(f"ERROR: {decisions_file} not found.")
        sys.exit(1)

    with open(decisions_file, 'r') as f:
        saved = json.load(f)

    checker_decisions = saved['checker_decisions']
    cube_decisions = saved['cube_decisions']

    # Load rollout data (with probs)
    rollout_data = {}  # key -> {equity, se, probs}
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = tuple(rec['board'])
                rollout_data[key] = rec

    return checker_decisions, cube_decisions, rollout_data, saved


def prepare_checker_decisions(checker_decisions, rollout_data):
    """Prepare checker-play decisions for C++ scoring (same as existing PR)."""
    prepared = []
    n_missing = 0
    for dec in checker_decisions:
        candidates = dec['candidates']
        eqs = []
        all_have = True
        for cand in candidates:
            key = tuple(cand)
            if key not in rollout_data:
                all_have = False
                break
            eqs.append(rollout_data[key]['equity'])
        if not all_have:
            n_missing += 1
            continue
        prepared.append({
            'board': dec['board'],
            'dice': dec['dice'],
            'candidates': candidates,
            'rollout_equities': eqs,
        })
    return prepared, n_missing


def score_cube_decisions(cube_decisions, rollout_data, weights, hidden, n_plies=0):
    """Score cube decisions using rollout-derived reference equities.

    For each cube decision:
    1. Get rollout probs for the flipped pre-roll board (cubeless post-move probs)
    2. Invert to get cubeless pre-roll probs
    3. Apply Janowski to get reference ND/DT/DP equities
    4. Evaluate strategy's cube decision at the given ply depth
    5. Compute error = |reference_optimal_equity - strategy_equity|

    Returns dict with per-plan and overall statistics.
    """
    wt = (weights['purerace'], weights['racing'], weights['attacking'],
          weights['priming'], weights['anchoring'])
    ht = hidden

    total_error = 0.0
    n_scored = 0
    n_missing = 0
    n_trivial_ref = 0

    per_plan_error = {gp: 0.0 for gp in GAME_PLANS}
    per_plan_n = {gp: 0 for gp in GAME_PLANS}
    per_type_error = {'double': 0.0, 'take': 0.0}
    per_type_n = {'double': 0, 'take': 0}
    wrong_decisions = 0
    errors_list = []

    for dec in cube_decisions:
        board = dec['board']
        flipped = bgbot_cpp.flip_board(board)
        key = tuple(flipped)

        if key not in rollout_data:
            n_missing += 1
            continue

        rollout_rec = rollout_data[key]
        rollout_probs_raw = rollout_rec.get('probs')
        if rollout_probs_raw is None:
            n_missing += 1
            continue

        # Invert rollout probs to get pre-roll cubeless probs
        # rollout gives post-move probs from the "flipped" player's perspective
        # invert_probs converts to current player's pre-roll perspective
        ref_probs = bgbot_cpp.invert_probs_py(rollout_probs_raw)

        # Reference cube decision from rollout probs
        cube_value = dec['cube_value']
        is_race = bgbot_cpp.is_race(board)
        cube_x = bgbot_cpp.cube_efficiency(board, is_race)

        if dec['decision_type'] == 'double':
            # Doubler's decision: need cube owner from doubler's perspective
            cube_owner_str = dec['cube_owner']
            if cube_owner_str == 'centered':
                owner_enum = bgbot_cpp.CubeOwner.CENTERED
            elif cube_owner_str == f"player{dec['player']}":
                owner_enum = bgbot_cpp.CubeOwner.PLAYER
            else:
                owner_enum = bgbot_cpp.CubeOwner.OPPONENT

            ref_cd = bgbot_cpp.cube_decision_0ply(ref_probs, cube_value, owner_enum, cube_x)
            ref_optimal_eq = ref_cd.optimal_equity

            # Strategy's cube decision
            if n_plies == 0:
                strat_result = bgbot_cpp.evaluate_cube_decision(
                    board, cube_value, owner_enum,
                    *wt, *ht)
                strat_should_double = strat_result['should_double']
                strat_optimal_eq = strat_result['optimal_equity']
            else:
                strat_result = bgbot_cpp.cube_decision_nply(
                    board, cube_value, owner_enum,
                    n_plies=n_plies,
                    purerace_weights=wt[0], racing_weights=wt[1],
                    attacking_weights=wt[2], priming_weights=wt[3],
                    anchoring_weights=wt[4],
                    n_hidden_purerace=ht[0], n_hidden_racing=ht[1],
                    n_hidden_attacking=ht[2], n_hidden_priming=ht[3],
                    n_hidden_anchoring=ht[4])
                strat_should_double = strat_result['should_double']
                strat_optimal_eq = strat_result['optimal_equity']

            # Error: strategy picked wrong action
            # If strategy doubles when it shouldn't: error = ND - min(DT, DP)
            # If strategy doesn't double when it should: error = min(DT, DP) - ND
            # All using reference equities
            ref_should_double = ref_cd.should_double
            if strat_should_double == ref_should_double:
                error = 0.0
            else:
                # Use reference equities to measure cost
                error = abs(ref_cd.equity_nd - min(ref_cd.equity_dt, ref_cd.equity_dp))
                wrong_decisions += 1

        elif dec['decision_type'] == 'take':
            # Opponent's take/pass decision
            # From opponent's perspective, the equities are negated
            # But we stored equities from the doubler's perspective
            # Reference: opponent should take if DT <= DP (from doubler's view)
            # Which means -DT >= -DP from opponent's view

            # Get cube owner for the doubler
            cube_owner_str = dec['cube_owner']
            doubler_player = 1 - dec['player']  # the doubler
            if cube_owner_str == 'centered':
                owner_enum = bgbot_cpp.CubeOwner.CENTERED
            elif cube_owner_str == f"player{doubler_player}":
                owner_enum = bgbot_cpp.CubeOwner.PLAYER
            else:
                owner_enum = bgbot_cpp.CubeOwner.OPPONENT

            ref_cd = bgbot_cpp.cube_decision_0ply(ref_probs, cube_value, owner_enum, cube_x)

            # Strategy's take/pass decision
            if n_plies == 0:
                strat_result = bgbot_cpp.evaluate_cube_decision(
                    board, cube_value, owner_enum,
                    *wt, *ht)
                strat_should_take = strat_result['should_take']
            else:
                strat_result = bgbot_cpp.cube_decision_nply(
                    board, cube_value, owner_enum,
                    n_plies=n_plies,
                    purerace_weights=wt[0], racing_weights=wt[1],
                    attacking_weights=wt[2], priming_weights=wt[3],
                    anchoring_weights=wt[4],
                    n_hidden_purerace=ht[0], n_hidden_racing=ht[1],
                    n_hidden_attacking=ht[2], n_hidden_priming=ht[3],
                    n_hidden_anchoring=ht[4])
                strat_should_take = strat_result['should_take']

            ref_should_take = ref_cd.should_take

            if strat_should_take == ref_should_take:
                error = 0.0
            else:
                # Cost of wrong take/pass (from doubler's perspective, then abs)
                # Wrong take: should have passed, took instead → cost = |DP - DT|
                # Wrong pass: should have taken, passed instead → cost = |DT - DP|
                error = abs(ref_cd.equity_dt - ref_cd.equity_dp)
                wrong_decisions += 1

        else:
            continue

        # Classify game plan
        gp_name = bgbot_cpp.classify_game_plan(board)

        total_error += error
        n_scored += 1
        per_plan_error[gp_name] += error
        per_plan_n[gp_name] += 1
        per_type_error[dec['decision_type']] += error
        per_type_n[dec['decision_type']] += 1

        errors_list.append({
            'error': error,
            'decision_type': dec['decision_type'],
            'game_plan': gp_name,
            'turn': dec['turn'],
            'game': dec.get('game', -1),
            'board': dec['board'],
            'cube_value': dec['cube_value'],
        })

    # Compute PRs
    overall_pr = (total_error / n_scored * 500) if n_scored > 0 else 0.0
    per_plan_pr = {}
    for gp in GAME_PLANS:
        if per_plan_n[gp] > 0:
            per_plan_pr[gp] = per_plan_error[gp] / per_plan_n[gp] * 500
        else:
            per_plan_pr[gp] = 0.0

    per_type_pr = {}
    for dt in ['double', 'take']:
        if per_type_n[dt] > 0:
            per_type_pr[dt] = per_type_error[dt] / per_type_n[dt] * 500
        else:
            per_type_pr[dt] = 0.0

    return {
        'overall_pr': overall_pr,
        'n_scored': n_scored,
        'n_missing': n_missing,
        'n_trivial_ref': n_trivial_ref,
        'wrong_decisions': wrong_decisions,
        'per_plan_pr': per_plan_pr,
        'per_plan_n': per_plan_n,
        'per_type_pr': per_type_pr,
        'per_type_n': per_type_n,
        'errors': sorted(errors_list, key=lambda x: -x['error']),
    }


def print_checker_result(label, result):
    """Print checker-play PR result from C++ dict."""
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


def print_cube_result(label, result):
    """Print cube PR result."""
    n_scored = result['n_scored']
    if n_scored == 0:
        print(f"  {label}: No cube decisions could be scored!")
        return None

    overall = result['overall_pr']
    wrong = result['wrong_decisions']
    n_missing = result['n_missing']

    print(f"  {label}: PR = {overall:.2f} "
          f"({n_scored} decisions, "
          f"{wrong} wrong, "
          f"{n_missing} missing rollout data)")

    # Per-type
    for dt in ['double', 'take']:
        n = result['per_type_n'][dt]
        if n > 0:
            pr = result['per_type_pr'][dt]
            print(f"    {dt}: PR={pr:.2f} ({n} decisions)")

    # Per-plan
    parts = []
    for gp in GAME_PLANS:
        n = result['per_plan_n'][gp]
        if n > 0:
            parts.append(f"{gp}={result['per_plan_pr'][gp]:.2f}({n})")
    if parts:
        print(f"    Per plan: {', '.join(parts)}")

    # Top 10 worst errors
    errors = result['errors']
    if errors and errors[0]['error'] > 0:
        print(f"    Top 10 worst cube errors:")
        for i, e in enumerate(errors[:10]):
            if e['error'] > 0:
                print(f"      {i+1}. error={e['error']*500:.2f} "
                      f"type={e['decision_type']} gp={e['game_plan']} "
                      f"cube={e['cube_value']}")

    return {
        'overall': overall,
        'n_scored': n_scored,
        'wrong_decisions': wrong,
        'per_plan_pr': result['per_plan_pr'],
        'per_plan_n': result['per_plan_n'],
        'per_type_pr': result['per_type_pr'],
        'per_type_n': result['per_type_n'],
    }


def main():
    parser = argparse.ArgumentParser(description='Score strategies on Cubeful Benchmark PR')
    parser.add_argument('--stage', type=int, default=5,
                        choices=[5], help='Model stage (default: 5)')
    parser.add_argument('--plies', type=int, default=0,
                        help='Ply depth for cube scoring (default: 0)')
    parser.add_argument('--checker-plies', type=int, default=None,
                        help='Ply depth for checker scoring (default: same as --plies)')
    parser.add_argument('--checker-only', action='store_true',
                        help='Score checker-play only')
    parser.add_argument('--cube-only', action='store_true',
                        help='Score cube decisions only')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads for checker scoring (0 = auto)')
    args = parser.parse_args()

    if args.checker_plies is None:
        args.checker_plies = args.plies

    bgbot_cpp.init_escape_tables()

    print("Loading cubeful benchmark data...")
    checker_decisions, cube_decisions, rollout_data, meta = load_benchmark_data()
    print(f"  {len(checker_decisions)} checker-play decisions")
    print(f"  {len(cube_decisions)} cube decisions")
    print(f"  {len(rollout_data)} rollout records")
    print(f"  Data from {meta['n_games']} games, {meta['n_turns']} turns")
    print()

    weights, hidden = get_stage_weights(args.stage)
    if weights is None:
        print("Weight files not found.")
        sys.exit(1)

    wt = (weights['purerace'], weights['racing'], weights['attacking'],
          weights['priming'], weights['anchoring'])
    ht = hidden

    checker_result = None
    cube_result = None

    # Score checker-play
    if not args.cube_only:
        prepared, n_missing = prepare_checker_decisions(checker_decisions, rollout_data)
        if n_missing:
            print(f"  ({n_missing} checker decisions missing rollout data)")
        print(f"  {len(prepared)} scoreable checker-play decisions")

        if prepared:
            t0 = time.perf_counter()
            if args.checker_plies == 0:
                raw = bgbot_cpp.score_benchmark_pr_0ply(
                    prepared, *wt, *ht, n_threads=args.threads)
            else:
                raw = bgbot_cpp.score_benchmark_pr_nply(
                    prepared, *wt, *ht,
                    n_plies=args.checker_plies,
                    filter_max_moves=FILTER_MAX_MOVES,
                    filter_threshold=FILTER_THRESHOLD,
                    n_threads=args.threads)
            elapsed = time.perf_counter() - t0
            print(f"  [{elapsed:.1f}s] ", end='')
            checker_result = print_checker_result(
                f"Checker-play ({args.checker_plies}-ply)", raw)
        print()

    # Score cube decisions
    if not args.checker_only:
        print(f"Scoring cube decisions at {args.plies}-ply...")
        t0 = time.perf_counter()
        cube_raw = score_cube_decisions(
            cube_decisions, rollout_data, weights, hidden, n_plies=args.plies)
        elapsed = time.perf_counter() - t0
        print(f"  [{elapsed:.1f}s] ", end='')
        cube_result = print_cube_result(
            f"Cube ({args.plies}-ply)", cube_raw)
        print()

    # Combined PR
    if checker_result and cube_result:
        # Compute combined PR as weighted average
        checker_total = checker_result['overall']
        cube_total = cube_result['overall']
        checker_n = sum(checker_result['per_plan_n'].values())
        cube_n = cube_result['n_scored']
        combined_n = checker_n + cube_n
        if combined_n > 0:
            # Weighted by number of decisions
            combined_pr = (checker_total * checker_n + cube_total * cube_n) / combined_n
            print("=" * 60)
            print(f"Combined Cubeful PR: {combined_pr:.2f}")
            print(f"  Checker-play: PR={checker_total:.2f} ({checker_n} decisions)")
            print(f"  Cube:         PR={cube_total:.2f} ({cube_n} decisions)")
            print(f"  Total:        {combined_n} decisions")
            print("=" * 60)


if __name__ == '__main__':
    main()
