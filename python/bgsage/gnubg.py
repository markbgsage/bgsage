"""
Interface to GNUbg CLI for position evaluation.

Provides post_move_analytics() which evaluates a post-move board position
and returns the five cubeless probabilities matching our NN output format:
  [P(win), P(gammon_win), P(backgammon_win), P(gammon_loss), P(backgammon_loss)]

All evaluations are cubeless (cube locked at 1).

Approach: the post-move probabilities for the player who just moved are the
pre-roll probabilities for the opponent, with win/loss swapped. We flip the
board to the opponent's perspective, ask gnubg for cube analytics (which gives
pre-roll cubeless probabilities), then invert back to the mover's perspective.
"""

import os
import subprocess
import tempfile


GNUBG_CLI = r'C:\Program Files (x86)\gnubg\gnubg-cli.exe'


def _flip_board(checkers):
    """Flip a board to the other player's perspective."""
    flipped = [0] * 26
    flipped[0] = checkers[25]
    flipped[25] = checkers[0]
    for i in range(1, 25):
        flipped[i] = -checkers[25 - i]
    return flipped


def _build_cube_analytics_command(checkers, n_plies=0):
    """Build a gnubg command file to get pre-roll cube analytics for a position.

    The checkers represent the board from the perspective of the player on roll
    (pre-roll). gnubg's 'hint' with 'set turn 1' will give cube analytics
    including cubeless probabilities.

    Args:
        checkers: 26-element list (player on roll's perspective)
        n_plies: evaluation depth (0, 1, 2, or 3)

    Returns:
        Command string for gnubg.
    """
    cmd = 'new session\n'
    cmd += f'set evaluation chequer eval plies {n_plies}\n'
    cmd += f'set evaluation cubedecision eval plies {n_plies}\n'
    # Lock cube at 1 for cubeless evaluation
    cmd += 'set cube value 1\n'

    # set board simple format: player_bar pt1..pt24 opponent_bar
    cmd += 'set board simple '
    cmd += str(checkers[25]) + ' '
    for n in checkers[1:25]:
        cmd += str(n) + ' '
    cmd += str(checkers[0]) + '\n'

    # Ensure it's the player's turn (player = "mghig" = X in gnubg)
    cmd += 'set turn 1\n'
    cmd += 'hint\n'
    return cmd


def _run_gnubg(cmd, timeout=60):
    """Run gnubg-cli with the given command and return stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cmd)
        cmd_file = f.name

    try:
        result = subprocess.run(
            [GNUBG_CLI, '-q', '-t', '--no-rc', '-c', cmd_file],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gnubg-cli failed (rc={result.returncode}):\n"
                f"stderr: {result.stderr}\nstdout: {result.stdout}\n"
                f"command:\n{cmd}")
        return result.stdout
    finally:
        os.remove(cmd_file)


def _parse_cube_analytics(output, n_plies=0):
    """Parse gnubg cube analytics output and return the 5 probabilities.

    The cube analytics section looks like:
        Cube analysis
        N-ply cubeless equity +0.077
          0.527 0.148 0.008 - 0.473 0.128 0.005
        Cubeful equities:
        ...

    Returns:
        dict with keys: p_win, p_gw, p_bw, p_gl, p_bl, equity_cubeless
    """
    if 'Cube analysis' not in output:
        raise ValueError(
            f"No cube analysis found in gnubg output:\n{output}")

    section = output[output.index('Cube analysis'):]
    lines = section.split('\n')

    # Line 1: "N-ply cubeless equity +X.XXX" or "0-ply cubeless equity +X.XXX"
    eq_line = lines[1].strip()
    eq_cubeless = float(eq_line.split()[-1])

    # Line 2: "  0.527 0.148 0.008 - 0.473 0.128 0.005"
    prob_line = lines[2].strip()
    bits = prob_line.split()
    # Format: p_win p_gw p_bw "-" p_loss p_gl p_bl
    p_win = float(bits[0])
    p_gw = float(bits[1])
    p_bw = float(bits[2])
    # bits[3] is "-"
    p_loss = float(bits[4])
    p_gl = float(bits[5])
    p_bl = float(bits[6])

    # Verify cubeless equity matches
    # gnubg's cubeless equity = 2*p_win - 1 + p_gw - p_gl + p_bw - p_bl
    eq_computed = 2.0 * p_win - 1.0 + p_gw - p_gl + p_bw - p_bl

    if abs(eq_computed - eq_cubeless) > 0.005:
        raise ValueError(
            f"Cubeless equity mismatch: computed {eq_computed:.4f} vs "
            f"reported {eq_cubeless:.4f}. "
            f"Probs: Win={p_win}, W(g)={p_gw}, W(bg)={p_bw}, "
            f"L(g)={p_gl}, L(bg)={p_bl}. "
            f"This may indicate incorrect gnubg setup.")

    return {
        'p_win': p_win,
        'p_gw': p_gw,
        'p_bw': p_bw,
        'p_gl': p_gl,
        'p_bl': p_bl,
        'equity_cubeless': eq_cubeless,
    }


def post_move_analytics(checkers, n_plies=0, timeout=60):
    """Evaluate a post-move board position using gnubg.

    Returns cubeless probabilities from the perspective of the player who just
    moved. This matches our NN's evaluation semantics: the NN outputs are
    probabilities assuming the player just moved to this position and it's
    about to switch to the other player's turn.

    Approach:
    1. Flip the board to the opponent's perspective (opponent is about to roll)
    2. Ask gnubg for pre-roll cube analytics from opponent's perspective
    3. Invert the probabilities back to the mover's perspective

    Args:
        checkers: 26-element list in standard board format (mover's perspective):
            [0] = opponent checkers on bar (>= 0)
            [1-24] = board points (positive = mover, negative = opponent)
            [25] = mover checkers on bar (>= 0)
        n_plies: evaluation depth (0, 1, 2, or 3)
        timeout: seconds before killing gnubg process

    Returns:
        dict with:
            probs: [P(win), P(gw), P(bw), P(gl), P(bl)] from mover's perspective
            equity: cubeless equity from mover's perspective
    """
    if len(checkers) != 26:
        raise ValueError(f"checkers must have 26 elements, got {len(checkers)}")

    # Check for game-over: if the mover has no checkers left, they've won
    n_mover = sum(c for c in checkers[1:25] if c > 0) + checkers[25]
    n_opponent = sum(-c for c in checkers[1:25] if c < 0) + checkers[0]

    if n_mover == 0:
        # Mover has borne off all checkers — they win
        # Check if opponent has any borne off
        opp_borne_off = 15 - n_opponent
        if opp_borne_off > 0:
            # Opponent has borne off at least one: single win
            return {'probs': [1.0, 0.0, 0.0, 0.0, 0.0], 'equity': 1.0}
        else:
            # Opponent has borne off none: gammon at least
            # Check for backgammon: opponent has checker in mover's home or on bar
            has_in_home_or_bar = checkers[0] > 0  # opponent on bar
            if not has_in_home_or_bar:
                for i in range(1, 7):  # mover's home board = points 1-6
                    if checkers[i] < 0:
                        has_in_home_or_bar = True
                        break
            if has_in_home_or_bar:
                return {'probs': [1.0, 1.0, 1.0, 0.0, 0.0], 'equity': 3.0}
            else:
                return {'probs': [1.0, 1.0, 0.0, 0.0, 0.0], 'equity': 2.0}

    if n_opponent == 0:
        # Opponent has borne off all checkers — mover loses
        mover_borne_off = 15 - n_mover
        if mover_borne_off > 0:
            return {'probs': [0.0, 0.0, 0.0, 0.0, 0.0], 'equity': -1.0}
        else:
            has_in_home_or_bar = checkers[25] > 0
            if not has_in_home_or_bar:
                for i in range(19, 25):  # opponent's home = points 19-24
                    if checkers[i] > 0:
                        has_in_home_or_bar = True
                        break
            if has_in_home_or_bar:
                return {'probs': [0.0, 0.0, 0.0, 1.0, 1.0], 'equity': -3.0}
            else:
                return {'probs': [0.0, 0.0, 0.0, 1.0, 0.0], 'equity': -2.0}

    # Flip to opponent's perspective (opponent is about to roll)
    opp_board = _flip_board(checkers)

    # Get pre-roll cube analytics from opponent's perspective
    cmd = _build_cube_analytics_command(opp_board, n_plies)
    output = _run_gnubg(cmd, timeout=timeout)
    opp_result = _parse_cube_analytics(output, n_plies)

    # Invert probabilities: opponent's perspective -> mover's perspective
    # Opponent's P(win) = mover's P(loss) = 1 - mover's P(win)
    mover_p_win = 1.0 - opp_result['p_win']
    mover_p_gw = opp_result['p_gl']
    mover_p_bw = opp_result['p_bl']
    mover_p_gl = opp_result['p_gw']
    mover_p_bl = opp_result['p_bw']

    mover_eq = -opp_result['equity_cubeless']

    return {
        'probs': [mover_p_win, mover_p_gw, mover_p_bw, mover_p_gl, mover_p_bl],
        'equity': mover_eq,
    }


def post_move_analytics_many(checkers_list, n_plies=0, max_workers=None, timeout=60):
    """Evaluate multiple post-move positions in parallel.

    Args:
        checkers_list: list of 26-element checker lists (mover's perspective)
        n_plies: evaluation depth
        max_workers: thread pool size (default: CPU count)
        timeout: per-position timeout in seconds

    Returns:
        List of result dicts (same format as post_move_analytics)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(checkers_list))

    results = [None] * len(checkers_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, checkers in enumerate(checkers_list):
            future = executor.submit(post_move_analytics, checkers, n_plies, timeout)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results
