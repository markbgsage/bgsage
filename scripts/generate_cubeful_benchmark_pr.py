"""
Generate Cubeful Benchmark PR data: simulate money games with cube decisions,
collect both checker-play and cube decision positions, prepare for rollout.

The Cubeful Benchmark PR measures both:
1. Checker-play decisions (same as existing PR)
2. Cube decisions (double/no-double and take/pass)

Game simulation:
- Both players use Stage 5 0-ply strategy for checker play
- Cube decisions use 0-ply Janowski (evaluate_cube_decision)
- First move: both roll one die, higher die goes first with that roll
- Every subsequent turn: player considers doubling before rolling

Cube decision filtering (XG-style):
- Include if |ND - min(DT, DP)| < 0.200 (non-trivial)
- Exclude obvious no-doubles and too-good-to-double positions

Rollout needs:
- Checker-play: post-move board rollouts (cubeless), same as existing
- Cube decisions: pre-roll board rollouts (cubeless probs), then Janowski conversion

Usage:
  python python/generate_cubeful_benchmark_pr.py [--n-games 200] [--seed 42]
  python python/generate_cubeful_benchmark_pr.py --skip-rollout
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
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark_cubeful_pr')

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

# XG decision thresholds
TRIVIAL_THRESHOLD = 0.001          # checker-play: best-worst equity < this → trivial
CUBE_TRIVIAL_THRESHOLD = 0.200     # cube: |ND - optimal_action| >= this → trivial (excluded)

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


def count_result(board):
    """Return game result from perspective of the player who just won.
    The board is from the loser's perspective (player on roll lost).
    Returns (points, is_gammon, is_backgammon).
    Loser is the player on roll. Check if loser has borne off any,
    and if they have checkers in opponent's home or on bar."""
    # Player on roll is the loser - check their checkers
    loser_borne_off = 15
    loser_on_bar = board[25]
    loser_in_opp_home = 0  # loser's checkers in winner's home (points 1-6 from winner = 19-24 for loser)

    loser_borne_off -= loser_on_bar
    for i in range(1, 25):
        if board[i] > 0:
            loser_borne_off -= board[i]
            if i >= 19:  # loser's checkers in opponent's home board
                loser_in_opp_home += board[i]

    if loser_borne_off == 0:
        # Gammon or backgammon
        if loser_on_bar > 0 or loser_in_opp_home > 0:
            return 3  # backgammon
        else:
            return 2  # gammon
    return 1  # single


def roll_dice(rng):
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    return (max(d1, d2), min(d1, d2))


def filter_candidates(candidates, equities, chosen_idx):
    """Filter to top FILTER_MAX_MOVES candidates within FILTER_THRESHOLD of best."""
    best_eq = max(equities)
    threshold = best_eq - FILTER_THRESHOLD

    passing = []
    for i, eq in enumerate(equities):
        if eq >= threshold:
            passing.append((i, eq))

    passing.sort(key=lambda x: -x[1])
    passing = passing[:FILTER_MAX_MOVES]

    passing_indices = {p[0] for p in passing}
    if chosen_idx not in passing_indices:
        passing.append((chosen_idx, equities[chosen_idx]))

    filtered_cands = []
    filtered_eqs = []
    new_chosen_idx = 0
    for i, (orig_idx, eq) in enumerate(passing):
        filtered_cands.append(candidates[orig_idx])
        filtered_eqs.append(eq)
        if orig_idx == chosen_idx:
            new_chosen_idx = i

    return filtered_cands, filtered_eqs, new_chosen_idx


def simulate_game(strategy, wt, rng, max_turns=300):
    """
    Simulate one money game with cube decisions.

    Returns (checker_decisions, cube_decisions, total_turns, game_result).
    game_result: dict with 'winner' (0 or 1), 'points' (1/2/3), 'cube_value',
                 'resigned' (bool), 'double_passed' (bool).
    """
    board = list(STARTING_BOARD)
    checker_decisions = []
    cube_decisions = []
    turn = 0
    player = -1  # determined by opening roll

    cube_value = 1
    cube_owner = 'centered'  # 'centered', 'player0', 'player1'
    game_result = None

    # Opening roll: both roll one die, higher goes first
    while True:
        d1 = rng.randint(1, 6)
        d2 = rng.randint(1, 6)
        if d1 != d2:
            break
    if d1 > d2:
        player = 0
    else:
        player = 1
        d1, d2 = d2, d1
    dice = (max(d1, d2), min(d1, d2))
    is_opening = True

    while turn < max_turns:
        if is_game_over(board):
            # Previous player won
            prev_player = 1 - player
            result_points = count_result(board)
            game_result = {
                'winner': prev_player,
                'points': result_points * cube_value,
                'cube_value': cube_value,
                'resigned': False,
                'double_passed': False,
            }
            break

        # Cube decision (not on opening move)
        if not is_opening:
            # Can this player double?
            can_dbl = (cube_owner == 'centered' or
                       (cube_owner == f'player{player}'))

            if can_dbl:
                # Get cube owner enum for C++
                if cube_owner == 'centered':
                    owner_enum = bgbot_cpp.CubeOwner.CENTERED
                elif cube_owner == f'player{player}':
                    owner_enum = bgbot_cpp.CubeOwner.PLAYER
                else:
                    owner_enum = bgbot_cpp.CubeOwner.OPPONENT

                cube_result = bgbot_cpp.evaluate_cube_decision(
                    board, cube_value, owner_enum,
                    *wt, NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)

                equity_nd = cube_result['equity_nd']
                equity_dt = cube_result['equity_dt']
                equity_dp = cube_result['equity_dp']
                should_double = cube_result['should_double']
                should_take = cube_result['should_take']

                # Optimal action equity from opponent's perspective
                optimal_action_eq = min(equity_dt, equity_dp)
                cube_error_margin = abs(equity_nd - optimal_action_eq)

                # Collect cube decision if non-trivial
                if cube_error_margin < CUBE_TRIVIAL_THRESHOLD:
                    cube_decisions.append({
                        'board': list(board),
                        'cube_value': cube_value,
                        'cube_owner': cube_owner,
                        'player': player,
                        'turn': turn,
                        'equity_nd': equity_nd,
                        'equity_dt': equity_dt,
                        'equity_dp': equity_dp,
                        'should_double': should_double,
                        'should_take': should_take,
                        'probs': list(cube_result['probs']),
                        'cubeless_equity': cube_result['cubeless_equity'],
                        'cube_x': cube_result['cube_x'],
                        'is_race': cube_result['is_race'],
                        'decision_type': 'double',  # roller's decision
                    })

                # Execute cube action
                if should_double:
                    if should_take:
                        # Record take/pass decision BEFORE updating cube state
                        take_error_margin = abs(equity_dt - equity_dp)
                        if take_error_margin < CUBE_TRIVIAL_THRESHOLD:
                            cube_decisions.append({
                                'board': list(board),
                                'cube_value': cube_value,  # pre-double value
                                'cube_owner': cube_owner,  # pre-double owner
                                'player': 1 - player,  # opponent is deciding
                                'turn': turn,
                                'equity_nd': equity_nd,
                                'equity_dt': equity_dt,
                                'equity_dp': equity_dp,
                                'should_double': should_double,
                                'should_take': should_take,
                                'probs': list(cube_result['probs']),
                                'cubeless_equity': cube_result['cubeless_equity'],
                                'cube_x': cube_result['cube_x'],
                                'is_race': cube_result['is_race'],
                                'decision_type': 'take',  # opponent's decision
                            })

                        # Double/Take: always update cube state (regardless of whether decision was recorded)
                        cube_value *= 2
                        cube_owner = f'player{1 - player}'
                    else:
                        # Double/Pass - game over
                        game_result = {
                            'winner': player,
                            'points': cube_value,  # wins current cube value
                            'cube_value': cube_value,
                            'resigned': False,
                            'double_passed': True,
                        }
                        break

            # Roll dice for this turn
            dice = roll_dice(rng)

        # Checker play
        candidates = bgbot_cpp.possible_moves(board, dice[0], dice[1])

        if len(candidates) <= 1:
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = list(board)
            board = bgbot_cpp.flip_board(chosen)
            turn += 1
            player = 1 - player
            is_opening = False
            continue

        # Evaluate all candidates
        equities = []
        for cand in candidates:
            result = strategy.evaluate_board(cand, board)
            equities.append(result['equity'])

        best_eq = max(equities)
        worst_eq = min(equities)
        spread = best_eq - worst_eq

        best_idx = equities.index(best_eq)

        if spread >= TRIVIAL_THRESHOLD:
            # Real checker-play decision
            f_cands, f_eqs, f_chosen = filter_candidates(
                candidates, equities, best_idx)

            checker_decisions.append({
                'board': list(board),
                'dice': list(dice),
                'candidates': [list(c) for c in f_cands],
                'candidate_equities_0ply': f_eqs,
                'chosen_idx': f_chosen,
                'n_total_candidates': len(candidates),
                'turn': turn,
                'player': player,
                'cube_value': cube_value,
                'cube_owner': cube_owner,
            })

        chosen = candidates[best_idx]
        board = bgbot_cpp.flip_board(chosen)
        turn += 1
        player = 1 - player
        is_opening = False

    if game_result is None:
        # Max turns reached
        game_result = {
            'winner': -1,
            'points': 0,
            'cube_value': cube_value,
            'resigned': False,
            'double_passed': False,
        }

    return checker_decisions, cube_decisions, turn, game_result


def simulate_games_parallel(weights, n_games, seed, n_workers=4):
    """Simulate games across multiple workers."""
    master_rng = random.Random(seed)
    game_seeds = [master_rng.randint(0, 2**31) for _ in range(n_games)]

    all_checker_decisions = []
    all_cube_decisions = []
    total_turns = 0
    game_results = []

    def worker(game_indices_seeds):
        wt = weight_tuple(weights)
        strategy = bgbot_cpp.GamePlanStrategy(
            *wt, NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)
        results = []
        for game_idx, game_seed in game_indices_seeds:
            rng = random.Random(game_seed)
            checker_decs, cube_decs, n_turns, game_res = simulate_game(
                strategy, wt, rng)
            for d in checker_decs:
                d['game'] = game_idx
            for d in cube_decs:
                d['game'] = game_idx
            results.append((checker_decs, cube_decs, n_turns, game_res))
        return results

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
            for checker_decs, cube_decs, n_turns, game_res in results:
                all_checker_decisions.extend(checker_decs)
                all_cube_decisions.extend(cube_decs)
                total_turns += n_turns
                game_results.append(game_res)
                done_games += 1

    elapsed = time.perf_counter() - t0
    return all_checker_decisions, all_cube_decisions, total_turns, game_results, elapsed


def collect_unique_checker_positions(all_decisions):
    """Collect unique post-move candidate positions needing rollouts."""
    positions = {}
    for dec in all_decisions:
        for cand in dec['candidates']:
            key = tuple(cand)
            if key not in positions:
                positions[key] = cand
    return positions


def collect_unique_cube_positions(all_cube_decisions):
    """Collect unique pre-roll board positions needing rollouts for cube decisions.
    These are flipped boards (opponent's post-move = our pre-roll) that need
    cubeless rollout to get pre-roll probs via inversion."""
    positions = {}
    for dec in all_cube_decisions:
        board = dec['board']
        # For cube decisions, we need to rollout the flipped board
        # (opponent's post-move position), then invert probs to get
        # pre-roll cubeless probs for Janowski conversion.
        flipped = bgbot_cpp.flip_board(board)
        key = tuple(flipped)
        if key not in positions:
            positions[key] = list(flipped)
    return positions


def log(log_file, msg):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate Cubeful Benchmark PR data')
    parser.add_argument('--n-games', type=int, default=200,
                        help='Number of games to simulate (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sim-workers', type=int, default=4,
                        help='Parallel workers for game simulation')
    parser.add_argument('--skip-rollout', action='store_true',
                        help='Stop after saving decisions')
    args = parser.parse_args()

    weights = get_weights()
    if weights is None:
        print("Cannot find weight files. Exiting.")
        sys.exit(1)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_file = os.path.join(LOGS_DIR, 'cubeful_benchmark_pr_generation.log')
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')

    bgbot_cpp.init_escape_tables()

    log(log_file, f"=== Cubeful Benchmark PR Generation ===")
    log(log_file, f"Games: {args.n_games}, Seed: {args.seed}")

    # Simulate games
    log(log_file, f"Simulating {args.n_games} games with cube decisions "
        f"({args.sim_workers} workers)...")

    (checker_decisions, cube_decisions, n_turns,
     game_results, sim_elapsed) = simulate_games_parallel(
        weights, args.n_games, args.seed, args.sim_workers)

    # Sort decisions
    checker_decisions.sort(key=lambda d: (d.get('game', 0), d['turn']))
    cube_decisions.sort(key=lambda d: (d.get('game', 0), d['turn']))

    # Game statistics
    n_games = args.n_games
    n_double_pass = sum(1 for g in game_results if g['double_passed'])
    n_completed = sum(1 for g in game_results if g['winner'] >= 0)
    avg_cube = sum(g['cube_value'] for g in game_results) / len(game_results)
    cube_dist = {}
    for g in game_results:
        cv = g['cube_value']
        cube_dist[cv] = cube_dist.get(cv, 0) + 1

    # Cube decision type breakdown
    n_double_decs = sum(1 for d in cube_decisions if d['decision_type'] == 'double')
    n_take_decs = sum(1 for d in cube_decisions if d['decision_type'] == 'take')

    log(log_file, f"Simulation complete: {n_games} games, {n_turns} turns in {sim_elapsed:.0f}s")
    log(log_file, f"  Completed games: {n_completed}/{n_games}")
    log(log_file, f"  Double/Pass endings: {n_double_pass}")
    log(log_file, f"  Avg cube value: {avg_cube:.2f}")
    log(log_file, f"  Cube distribution: {dict(sorted(cube_dist.items()))}")
    log(log_file, f"  Checker-play decisions: {len(checker_decisions)}")
    log(log_file, f"  Cube decisions: {len(cube_decisions)} "
        f"({n_double_decs} double, {n_take_decs} take)")
    log(log_file, f"  Avg turns/game: {n_turns/n_games:.1f}")

    # Collect unique positions
    checker_positions = collect_unique_checker_positions(checker_decisions)
    cube_positions = collect_unique_cube_positions(cube_decisions)

    # Check overlap
    overlap = set(checker_positions.keys()) & set(cube_positions.keys())

    log(log_file, f"  Unique checker-play positions: {len(checker_positions)}")
    log(log_file, f"  Unique cube positions (flipped pre-roll): {len(cube_positions)}")
    log(log_file, f"  Overlap: {len(overlap)}")
    log(log_file, f"  Total unique positions to rollout: "
        f"{len(checker_positions) + len(cube_positions) - len(overlap)}")

    # Save decisions
    save_data = {
        'n_games': n_games,
        'n_turns': n_turns,
        'seed': args.seed,
        'filter_max_moves': FILTER_MAX_MOVES,
        'filter_threshold': FILTER_THRESHOLD,
        'trivial_threshold': TRIVIAL_THRESHOLD,
        'cube_trivial_threshold': CUBE_TRIVIAL_THRESHOLD,
        'checker_decisions': checker_decisions,
        'cube_decisions': cube_decisions,
        'game_results': game_results,
    }

    with open(decisions_file, 'w') as f:
        json.dump(save_data, f)

    file_size_mb = os.path.getsize(decisions_file) / (1024 * 1024)
    log(log_file, f"  Saved decisions to {decisions_file} ({file_size_mb:.1f} MB)")

    if args.skip_rollout:
        log(log_file, "Stopping after decisions. Use cloud_rollout_cubeful.py to rollout.")
        return

    log(log_file, "Done. Run cloud_rollout_cubeful.py to execute rollouts on AWS ECS.")


if __name__ == '__main__':
    main()
