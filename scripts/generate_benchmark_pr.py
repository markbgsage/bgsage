"""
Generate Benchmark PR data: simulate games, collect decision positions,
rollout all candidates, save incrementally.

The Benchmark PR is the average equity error per decision * 500, analogous
to XG's Performance Rating. Decisions exclude forced moves, dances, and
trivial positions (best-worst equity spread < 0.001).

Only the top candidates (filtered by TINY: top 5 within 0.08 of best) are
rolled out, plus the strategy's chosen move.

Usage:
  python python/generate_benchmark_pr.py [--n-games 200] [--seed 42] [--resume]
  python python/generate_benchmark_pr.py --estimate-only  # time estimate
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import concurrent.futures

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
LOGS_DIR = os.path.join(project_dir, 'logs')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark_pr')

# Stage 5 hidden sizes
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 200, 400, 400, 400, 400

STARTING_BOARD = [
    0,
    -2, 0, 0, 0, 0, 5,
    0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0,
    -5, 0, 0, 0, 0, 2,
    0
]

# XG decision threshold: positions where best-worst equity < 0.001 are "trivial"
TRIVIAL_THRESHOLD = 0.001

# TINY filter for candidate selection
FILTER_MAX_MOVES = 5
FILTER_THRESHOLD = 0.08


def get_weights():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    paths = {}
    for t in types:
        path = os.path.join(MODELS_DIR, f'sl_s5_{t}.weights.best')
        if os.path.exists(path):
            paths[t] = path
        else:
            print(f"WARNING: {path} not found")
            return None
    return paths


def weight_tuple(w):
    return (w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'])


def is_game_over(board):
    """Check if player on roll has borne off all 15 checkers."""
    player_on_board = board[25]  # bar
    for i in range(1, 25):
        if board[i] > 0:
            player_on_board += board[i]
    return player_on_board == 0


def roll_dice(rng):
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    return (max(d1, d2), min(d1, d2))


def filter_candidates(candidates, equities, chosen_idx):
    """
    Filter to top FILTER_MAX_MOVES candidates within FILTER_THRESHOLD of best,
    always including the chosen move. Returns (filtered_candidates, filtered_equities,
    new_chosen_idx).
    """
    best_eq = max(equities)
    threshold = best_eq - FILTER_THRESHOLD

    # Build list of (index, equity) pairs that pass filter
    passing = []
    for i, eq in enumerate(equities):
        if eq >= threshold:
            passing.append((i, eq))

    # Sort by equity descending, take top N
    passing.sort(key=lambda x: -x[1])
    passing = passing[:FILTER_MAX_MOVES]

    # Ensure chosen move is included
    passing_indices = {p[0] for p in passing}
    if chosen_idx not in passing_indices:
        passing.append((chosen_idx, equities[chosen_idx]))

    # Build filtered lists
    filtered_cands = []
    filtered_eqs = []
    new_chosen_idx = 0
    for i, (orig_idx, eq) in enumerate(passing):
        filtered_cands.append(candidates[orig_idx])
        filtered_eqs.append(eq)
        if orig_idx == chosen_idx:
            new_chosen_idx = i

    return filtered_cands, filtered_eqs, new_chosen_idx


def simulate_game(strategy, rng, max_turns=300):
    """
    Simulate one game using the given strategy for move selection.
    Returns (decisions, total_turns).
    """
    board = list(STARTING_BOARD)
    decisions = []
    turn = 0
    player = 0

    while turn < max_turns:
        if is_game_over(board):
            break

        d1, d2 = roll_dice(rng)
        candidates = bgbot_cpp.possible_moves(board, d1, d2)

        if len(candidates) <= 1:
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = list(board)
            board = bgbot_cpp.flip_board(chosen)
            turn += 1
            player = 1 - player
            continue

        # Evaluate all candidates at 1-ply
        equities = []
        for cand in candidates:
            result = strategy.evaluate_board(cand, board)
            equities.append(result['equity'])

        best_eq = max(equities)
        worst_eq = min(equities)
        spread = best_eq - worst_eq

        if spread < TRIVIAL_THRESHOLD:
            best_idx = equities.index(best_eq)
            chosen = candidates[best_idx]
            board = bgbot_cpp.flip_board(chosen)
            turn += 1
            player = 1 - player
            continue

        # Real decision
        best_idx = equities.index(best_eq)

        # Filter candidates to top N
        f_cands, f_eqs, f_chosen = filter_candidates(
            candidates, equities, best_idx)

        decisions.append({
            'board': list(board),
            'dice': [d1, d2],
            'candidates': [list(c) for c in f_cands],
            'candidate_equities_1ply': f_eqs,
            'chosen_idx': f_chosen,
            'n_total_candidates': len(candidates),
            'turn': turn,
            'player': player,
        })

        chosen = candidates[best_idx]
        board = bgbot_cpp.flip_board(chosen)
        turn += 1
        player = 1 - player

    return decisions, turn


def simulate_games_parallel(weights, n_games, seed, n_workers=4):
    """Simulate games across multiple workers."""
    # Each worker gets a range of games with deterministic seeds
    master_rng = random.Random(seed)
    game_seeds = [master_rng.randint(0, 2**31) for _ in range(n_games)]

    all_decisions = []
    total_turns = 0

    def worker(game_indices_seeds):
        """Worker simulates a batch of games."""
        strategy = bgbot_cpp.create_multipy_5nn(
            *weight_tuple(weights), NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_plies=1)
        results = []
        for game_idx, game_seed in game_indices_seeds:
            rng = random.Random(game_seed)
            decisions, n_turns = simulate_game(strategy, rng)
            for d in decisions:
                d['game'] = game_idx
            results.append((decisions, n_turns))
        return results

    # Split games into batches for workers
    batch_size = max(1, n_games // n_workers)
    batches = []
    for i in range(0, n_games, batch_size):
        batch = [(i + j, game_seeds[i + j])
                 for j in range(min(batch_size, n_games - i))]
        batches.append(batch)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, batch): batch for batch in batches}
        done_games = 0
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            for decisions, n_turns in results:
                all_decisions.extend(decisions)
                total_turns += n_turns
                done_games += 1

    elapsed = time.perf_counter() - t0
    return all_decisions, total_turns, elapsed


def collect_unique_positions(all_decisions):
    """Collect all unique post-move candidate positions needing rollouts."""
    positions = {}
    for dec in all_decisions:
        keys = []
        for cand in dec['candidates']:
            key = tuple(cand)
            if key not in positions:
                positions[key] = cand
            keys.append(key)
        dec['candidate_keys'] = [list(k) for k in keys]
    return positions


def load_completed_rollouts(rollout_file):
    """Load already-completed rollouts from the incremental save file."""
    completed = {}
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = tuple(rec['board'])
                completed[key] = rec['equity']
    return completed


def run_rollouts(positions, rollout_strategy, rollout_file, log_file, completed=None):
    """Run rollouts for all positions, saving incrementally."""
    if completed is None:
        completed = {}

    remaining_keys = [k for k in positions if k not in completed]
    total = len(positions)
    done = len(completed)

    log(log_file, f"Rollout: {done}/{total} already done, {len(remaining_keys)} remaining")

    if not remaining_keys:
        return dict(completed)

    results = dict(completed)
    start_time = time.perf_counter()
    batch_start = start_time
    initial_done = done

    with open(rollout_file, 'a') as f:
        for i, key in enumerate(remaining_keys):
            board = list(key)
            t0 = time.perf_counter()
            result = rollout_strategy.rollout_position(board)
            elapsed_pos = time.perf_counter() - t0

            equity = result.equity
            results[key] = equity

            rec = {'board': board, 'equity': equity, 'se': result.std_error}
            f.write(json.dumps(rec) + '\n')
            f.flush()

            done += 1

            if done % 50 == 0 or (time.perf_counter() - batch_start) > 300:
                total_elapsed = time.perf_counter() - start_time
                n_done_this_run = done - initial_done
                rate = n_done_this_run / total_elapsed if total_elapsed > 0 else 0
                eta_s = (total - done) / rate if rate > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_s)))
                log(log_file, f"  Rollout progress: {done}/{total} "
                    f"({done/total*100:.1f}%) | "
                    f"{rate:.2f} pos/s | "
                    f"last: {elapsed_pos:.1f}s | "
                    f"ETA: {eta_str}")
                batch_start = time.perf_counter()

    total_elapsed = time.perf_counter() - start_time
    log(log_file, f"Rollout complete: {total} positions in "
        f"{datetime.timedelta(seconds=int(total_elapsed))}")
    return results


def compute_benchmark_pr(all_decisions, rollout_equities):
    """
    Compute Benchmark PR from decisions and rollout equities.

    For each decision:
    - Find the candidate with the highest rollout equity (the "correct" move)
    - The strategy chose candidate at chosen_idx
    - Error = correct_equity - chosen_equity

    Benchmark PR = mean(error) * 500
    """
    total_error = 0.0
    n_decisions = 0
    errors = []

    for dec in all_decisions:
        keys = [tuple(k) for k in dec['candidate_keys']]
        rollout_eqs = [rollout_equities[k] for k in keys]

        best_rollout_eq = max(rollout_eqs)
        chosen_eq = rollout_eqs[dec['chosen_idx']]
        error = best_rollout_eq - chosen_eq

        total_error += error
        n_decisions += 1
        errors.append({
            'error': error,
            'turn': dec['turn'],
            'player': dec['player'],
            'game': dec.get('game', -1),
            'n_candidates': len(dec['candidates']),
            'n_total_candidates': dec.get('n_total_candidates', len(dec['candidates'])),
            'dice': dec['dice'],
            'chosen_eq': chosen_eq,
            'best_eq': best_rollout_eq,
            'board': dec['board'],
        })

    if n_decisions == 0:
        return 0.0, errors

    benchmark_pr = (total_error / n_decisions) * 500
    return benchmark_pr, errors


def log(log_file, msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate Benchmark PR data')
    parser.add_argument('--n-games', type=int, default=200,
                        help='Number of games to simulate (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from saved rollout data')
    parser.add_argument('--estimate-only', action='store_true',
                        help='Only estimate time, do not run full rollouts')
    parser.add_argument('--threads', type=int, default=0,
                        help='Rollout threads (0=auto)')
    parser.add_argument('--sim-workers', type=int, default=4,
                        help='Parallel workers for game simulation')
    parser.add_argument('--n-trials', type=int, default=1296,
                        help='Rollout trials per position (default: 1296)')
    parser.add_argument('--skip-rollout', action='store_true',
                        help='Stop after saving decisions (skip local rollout)')
    args = parser.parse_args()

    weights = get_weights()
    if weights is None:
        print("Cannot find weight files. Exiting.")
        sys.exit(1)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, 'benchmark_pr_generation.log')
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')
    results_file = os.path.join(BENCHMARK_DIR, 'results.json')

    bgbot_cpp.init_escape_tables()

    log(log_file, f"=== Benchmark PR Generation ===")
    log(log_file, f"Games: {args.n_games}, Seed: {args.seed}")

    # Step 1: Simulate games and collect decisions
    if args.resume and os.path.exists(decisions_file):
        log(log_file, "Loading saved decisions...")
        with open(decisions_file, 'r') as f:
            saved = json.load(f)
        all_decisions = saved['decisions']
        n_games = saved['n_games']
        n_turns = saved['n_turns']
        log(log_file, f"  Loaded {len(all_decisions)} decisions from "
            f"{n_games} games, {n_turns} turns")
    else:
        log(log_file, f"Simulating {args.n_games} games with "
            f"{args.sim_workers} workers...")

        all_decisions, n_turns, sim_elapsed = simulate_games_parallel(
            weights, args.n_games, args.seed, args.sim_workers)
        n_games = args.n_games

        # Sort decisions by game index then turn for consistency
        all_decisions.sort(key=lambda d: (d.get('game', 0), d['turn']))

        log(log_file, f"Simulation complete: {n_games} games, {n_turns} turns, "
            f"{len(all_decisions)} decisions in {sim_elapsed:.0f}s")
        log(log_file, f"  Avg turns/game: {n_turns/n_games:.1f}")
        log(log_file, f"  Avg decisions/game: {len(all_decisions)/n_games:.1f}")
        log(log_file, f"  Avg candidates/decision: "
            f"{sum(len(d['candidates']) for d in all_decisions)/len(all_decisions):.1f}")

        # Save decisions
        with open(decisions_file, 'w') as f:
            json.dump({
                'n_games': n_games,
                'n_turns': n_turns,
                'seed': args.seed,
                'filter_max_moves': FILTER_MAX_MOVES,
                'filter_threshold': FILTER_THRESHOLD,
                'trivial_threshold': TRIVIAL_THRESHOLD,
                'decisions': all_decisions,
            }, f)
        log(log_file, f"  Saved decisions to {decisions_file}")

    # Step 2: Collect unique positions needing rollouts
    positions = collect_unique_positions(all_decisions)
    log(log_file, f"Unique positions to rollout: {len(positions)}")
    log(log_file, f"  Total candidate evaluations: "
        f"{sum(len(d['candidates']) for d in all_decisions)}")

    if args.skip_rollout:
        log(log_file, f"--skip-rollout: stopping after decisions. "
            f"Use cloud_rollout.py to roll out positions.")
        return

    # Step 3: Load any completed rollouts
    completed = load_completed_rollouts(rollout_file) if args.resume else {}
    if completed:
        log(log_file, f"  Resuming: {len(completed)} rollouts already done")

    # Step 4: Estimate time
    n_to_rollout = len(positions) - len(completed)
    if n_to_rollout > 0:
        log(log_file, "Timing estimate: rolling out 5 sample positions...")
        rollout_est = bgbot_cpp.create_rollout_5nn(
            *weight_tuple(weights), NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
            n_trials=args.n_trials, truncation_depth=0,
            decision_ply=1, vr_ply=0,
            n_threads=args.threads,
            late_ply=0, late_threshold=3)

        sample_keys = [k for k in positions if k not in completed][:5]
        times = []
        for k in sample_keys:
            t0 = time.perf_counter()
            rollout_est.rollout_position(list(k))
            times.append(time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        total_est = avg_time * n_to_rollout
        est_str = str(datetime.timedelta(seconds=int(total_est)))
        log(log_file, f"  Avg time per position: {avg_time:.1f}s")
        log(log_file, f"  Estimated total time for {n_to_rollout} positions: {est_str}")

        if args.estimate_only:
            return

    elif args.estimate_only:
        log(log_file, "All positions already rolled out!")
        return

    # Step 5: Run rollouts
    log(log_file, f"Starting rollouts (1-ply, {args.n_trials} trials, VR=0, late=0@3)...")
    rollout = bgbot_cpp.create_rollout_5nn(
        *weight_tuple(weights), NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_trials=args.n_trials, truncation_depth=0,
        decision_ply=1, vr_ply=0,
        n_threads=args.threads,
        late_ply=0, late_threshold=3)

    rollout_equities = run_rollouts(
        positions, rollout, rollout_file, log_file, completed)

    # Step 6: Compute Benchmark PR
    benchmark_pr, errors = compute_benchmark_pr(all_decisions, rollout_equities)
    errors.sort(key=lambda x: -x['error'])

    log(log_file, f"\n{'='*60}")
    log(log_file, f"Benchmark PR (1-ply strategy, Stage 5): {benchmark_pr:.2f}")
    log(log_file, f"Total decisions: {len(all_decisions)}")
    log(log_file, f"Total unique positions rolled out: {len(positions)}")
    log(log_file, f"{'='*60}")

    # Stats
    nonzero_errors = [e['error'] for e in errors if e['error'] > 0]
    log(log_file, f"Decisions with error > 0: {len(nonzero_errors)} "
        f"({len(nonzero_errors)/len(errors)*100:.1f}%)")

    # Show top 20 worst errors
    log(log_file, "\nTop 20 worst errors:")
    for i, e in enumerate(errors[:20]):
        log(log_file, f"  {i+1}. error={e['error']*500:.2f} "
            f"dice={e['dice']} cands={e['n_candidates']} "
            f"total_cands={e['n_total_candidates']} "
            f"chosen_eq={e['chosen_eq']:.4f} best_eq={e['best_eq']:.4f}")

    # Save results
    with open(results_file, 'w') as f:
        json.dump({
            'benchmark_pr': benchmark_pr,
            'n_games': n_games,
            'n_decisions': len(all_decisions),
            'n_unique_positions': len(positions),
            'n_turns': n_turns,
            'errors_summary': {
                'mean_error': sum(e['error'] for e in errors) / len(errors),
                'max_error': errors[0]['error'] if errors else 0,
                'median_error': errors[len(errors)//2]['error'] if errors else 0,
                'pct_with_error': len(nonzero_errors) / len(errors) * 100,
            },
            'top_50_errors': errors[:50],
        }, f, indent=2)
    log(log_file, f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
