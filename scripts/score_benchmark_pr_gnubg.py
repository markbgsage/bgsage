"""
Score GNUbg against Benchmark PR data.

For each decision, generates ALL legal moves, evaluates each with GNUbg
at the given ply depth, picks the best, and measures equity error against
the rollout-determined best move.

Decisions are scored in parallel using ThreadPoolExecutor. Each worker
processes a batch of decisions, spawning one GNUbg subprocess per decision.

Usage:
  python python/score_benchmark_pr_gnubg.py                  # GNUbg 0-ply
  python python/score_benchmark_pr_gnubg.py --plies 1        # GNUbg 1-ply
  python python/score_benchmark_pr_gnubg.py --plies 0 --plies 1  # Both
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

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

DATA_DIR = os.path.join(project_dir, 'data')
BENCHMARK_DIR = os.path.join(DATA_DIR, 'benchmark_pr')
GNUBG_CLI = r'C:\Program Files (x86)\gnubg\gnubg-cli.exe'

GAME_PLANS = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']


def load_benchmark_data():
    """Load decisions with rollout equities pre-attached."""
    decisions_file = os.path.join(BENCHMARK_DIR, 'decisions.json')
    rollout_file = os.path.join(BENCHMARK_DIR, 'rollouts.jsonl')

    if not os.path.exists(decisions_file):
        print(f"ERROR: {decisions_file} not found.")
        sys.exit(1)

    with open(decisions_file, 'r') as f:
        saved = json.load(f)
    decisions = saved['decisions']

    rollout_equities = {}
    if os.path.exists(rollout_file):
        with open(rollout_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rollout_equities[tuple(rec['board'])] = rec['equity']

    prepared = []
    for dec in decisions:
        candidates = dec['candidates']
        eqs = []
        all_have = True
        for cand in candidates:
            if tuple(cand) not in rollout_equities:
                all_have = False
                break
            eqs.append(rollout_equities[tuple(cand)])
        if not all_have:
            continue
        prepared.append({
            'board': dec['board'],
            'dice': dec['dice'],
            'candidates': candidates,
            'rollout_equities': eqs,
        })

    return prepared, saved


def _flip_board(checkers):
    """Flip board to opponent's perspective."""
    flipped = [0] * 26
    flipped[0] = checkers[25]
    flipped[25] = checkers[0]
    for i in range(1, 25):
        flipped[i] = -checkers[25 - i]
    return flipped


def _build_batch_eval_command(boards, n_plies):
    """Build a GNUbg command that evaluates multiple post-move positions.

    For each board, we flip to opponent's perspective and get cube analytics
    (pre-roll evaluation). The output will contain multiple 'Cube analysis'
    sections that we parse sequentially.
    """
    cmd = 'new session\n'
    cmd += f'set evaluation chequer eval plies {n_plies}\n'
    cmd += f'set evaluation cubedecision eval plies {n_plies}\n'
    cmd += 'set cube value 1\n'

    for board in boards:
        opp_board = _flip_board(board)
        cmd += 'set board simple '
        cmd += str(opp_board[25]) + ' '
        for n in opp_board[1:25]:
            cmd += str(n) + ' '
        cmd += str(opp_board[0]) + '\n'
        cmd += 'set turn 1\n'
        cmd += 'hint\n'

    return cmd


def _parse_batch_cube_analytics(output, n_boards):
    """Parse multiple cube analytics sections from a single GNUbg run.

    Returns list of (equity, probs) tuples, one per board.
    equity is from the mover's perspective (inverted from GNUbg's opponent output).
    """
    results = []
    search_start = 0

    for _ in range(n_boards):
        idx = output.find('Cube analysis', search_start)
        if idx == -1:
            results.append(None)
            continue

        section = output[idx:]
        lines = section.split('\n')

        # Line 1: "N-ply cubeless equity +X.XXX"
        eq_line = lines[1].strip()
        opp_equity = float(eq_line.split()[-1])

        # Line 2: "  0.527 0.148 0.008 - 0.473 0.128 0.005"
        prob_line = lines[2].strip()
        bits = prob_line.split()
        opp_p_win = float(bits[0])
        opp_p_gw = float(bits[1])
        opp_p_bw = float(bits[2])
        opp_p_gl = float(bits[5])
        opp_p_bl = float(bits[6])

        # Invert to mover's perspective
        mover_equity = -opp_equity

        results.append(mover_equity)
        search_start = idx + len('Cube analysis') + 1

    return results


def _run_gnubg(cmd, timeout=300):
    """Run GNUbg CLI with command string, return stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cmd)
        cmd_file = f.name
    try:
        result = subprocess.run(
            [GNUBG_CLI, '-q', '-t', '--no-rc', '-c', cmd_file],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    finally:
        os.remove(cmd_file)


def score_decision_gnubg(dec, n_plies):
    """Score a single decision using GNUbg.

    Generates all legal moves, evaluates each with GNUbg in one batch,
    picks the best, and returns the error vs rollout reference.

    Returns: (game_plan_idx, error, is_outside) or None if skipped.
    """
    board = dec['board']
    dice = dec['dice']
    candidates = dec['candidates']
    rollout_eqs = dec['rollout_equities']

    if len(candidates) < 2:
        return None

    # Classify game plan
    gp = bgbot_cpp.classify_game_plan(board)
    gp_idx = GAME_PLANS.index(gp)

    # Generate all legal moves
    all_moves = bgbot_cpp.possible_moves(board, dice[0], dice[1])

    if len(all_moves) == 0:
        return None

    if len(all_moves) == 1:
        best_move = tuple(all_moves[0])
    else:
        # Batch-evaluate all moves with GNUbg
        cmd = _build_batch_eval_command(all_moves, n_plies)
        output = _run_gnubg(cmd)
        equities = _parse_batch_cube_analytics(output, len(all_moves))

        # Pick best
        best_eq = -1e9
        best_idx = 0
        for i, eq in enumerate(equities):
            if eq is not None and eq > best_eq:
                best_eq = eq
                best_idx = i
        best_move = tuple(all_moves[best_idx])

    # Compare against rollout reference
    best_rollout_eq = max(rollout_eqs)
    worst_rollout_eq = min(rollout_eqs)

    cand_set = {tuple(c): eq for c, eq in zip(candidates, rollout_eqs)}
    if best_move in cand_set:
        chosen_rollout_eq = cand_set[best_move]
        is_outside = False
    else:
        chosen_rollout_eq = worst_rollout_eq
        is_outside = True

    error = best_rollout_eq - chosen_rollout_eq
    return (gp_idx, error, is_outside)


def score_gnubg(decisions, n_plies, n_workers=8):
    """Score GNUbg against benchmark PR decisions in parallel."""
    gp_total_error = [0.0] * 5
    gp_n_decisions = [0] * 5
    gp_n_with_error = [0] * 5
    n_outside = 0
    n_skipped = 0
    n_errors = 0

    t0 = time.perf_counter()
    n_done = 0
    total = len(decisions)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, dec in enumerate(decisions):
            f = executor.submit(score_decision_gnubg, dec, n_plies)
            futures[f] = i

        for future in as_completed(futures):
            n_done += 1
            try:
                result = future.result()
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  WARNING: decision failed: {e}")
                continue

            if result is None:
                n_skipped += 1
                continue

            gp_idx, error, is_outside = result
            gp_total_error[gp_idx] += error
            gp_n_decisions[gp_idx] += 1
            if error > 0:
                gp_n_with_error[gp_idx] += 1
            if is_outside:
                n_outside += 1

            if n_done % 500 == 0:
                elapsed = time.perf_counter() - t0
                rate = n_done / elapsed
                eta = (total - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{total} done ({elapsed:.0f}s, "
                      f"~{eta:.0f}s remaining)...", flush=True)

    elapsed = time.perf_counter() - t0
    total_dec = sum(gp_n_decisions)
    total_err = sum(gp_total_error)
    total_with_err = sum(gp_n_with_error)

    if total_dec == 0:
        print(f"  No decisions scored!")
        return None

    overall_pr = (total_err / total_dec) * 500

    print(f"  GNUbg ({n_plies}-ply): PR = {overall_pr:.2f} "
          f"({total_dec} decisions, {elapsed:.1f}s, "
          f"{total_with_err}/{total_dec} with error, "
          f"{n_outside} outside filtered set)")
    if n_skipped:
        print(f"    ({n_skipped} decisions skipped)")
    if n_errors:
        print(f"    ({n_errors} decisions failed with errors)")

    gp_pr = {}
    parts = []
    for i, gp in enumerate(GAME_PLANS):
        if gp_n_decisions[i] > 0:
            pr = (gp_total_error[i] / gp_n_decisions[i]) * 500
            gp_pr[gp] = pr
            parts.append(f"{gp}={pr:.2f}({gp_n_decisions[i]})")
        else:
            gp_pr[gp] = None
    print(f"    Per plan: {', '.join(parts)}")

    return {
        'overall': overall_pr,
        'per_plan': gp_pr,
        'per_plan_n': {gp: gp_n_decisions[i] for i, gp in enumerate(GAME_PLANS)},
    }


def main():
    parser = argparse.ArgumentParser(
        description='Score GNUbg against Benchmark PR')
    parser.add_argument('--plies', type=int, action='append',
                        help='Ply depth(s) to score (can specify multiple)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel GNUbg workers (0 = CPU count)')
    args = parser.parse_args()

    if args.plies is None:
        args.plies = [0]

    if args.workers <= 0:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()

    bgbot_cpp.init_escape_tables()

    print("Loading benchmark data...")
    decisions, meta = load_benchmark_data()
    print(f"  {len(decisions)} scoreable decisions")
    print(f"  Data from {meta['n_games']} games, {meta['n_turns']} turns")
    print()

    results = {}

    for plies in args.plies:
        label = f"GNUbg ({plies}-ply)"
        print(f"Scoring {label} with {args.workers} workers...")
        result = score_gnubg(decisions, plies, n_workers=args.workers)
        if result is not None:
            results[label] = result
        print()

    if results:
        print("=" * 60)
        print("Benchmark PR Results (lower is better)")
        print("=" * 60)

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
                    row += f"  {'—':>9}"
            print(row)

        print()
        first_res = next(iter(results.values()))
        counts = f"  {'(N decisions)':<22} {'':>7}"
        for gp in GAME_PLANS:
            n = first_res['per_plan_n'].get(gp, 0)
            counts += f"  {n:>9}"
        print(counts)


if __name__ == '__main__':
    main()
