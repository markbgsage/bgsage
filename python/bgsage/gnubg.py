"""
Interface to GNUbg CLI for position evaluation.

Provides:
  - Module-level functions (backward-compatible):
      post_move_analytics()     — cubeless post-move evaluation
      post_move_analytics_many() — parallel batch version

  - GnuBgAnalyzer class (same interface as BgBotAnalyzer):
      cube_action()        — cubeful ND/DT/DP equities + cubeless probs
      post_move_analytics() — cubeless probs + cubeful equity
      checker_play()       — ranked legal moves with equities and probs
"""

import os
import re
import subprocess
import tempfile

from .types import (
    CheckerPlayResult,
    CubeActionResult,
    MoveAnalysis,
    PostMoveAnalysis,
    Probabilities,
)


GNUBG_CLI = r'C:\Program Files (x86)\gnubg\gnubg-cli.exe'

_EVAL_LEVELS = {'1ply': 0, '2ply': 1, '3ply': 2, '4ply': 3}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _flip_board(checkers):
    """Flip a board to the other player's perspective."""
    flipped = [0] * 26
    flipped[0] = checkers[25]
    flipped[25] = checkers[0]
    for i in range(1, 25):
        flipped[i] = -checkers[25 - i]
    return flipped


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


def _board_simple_str(checkers):
    """Format a board as 'set board simple ...' argument string."""
    parts = [str(checkers[25])]
    for n in checkers[1:25]:
        parts.append(str(n))
    parts.append(str(checkers[0]))
    return ' '.join(parts)


def _build_cube_analytics_command(checkers, n_plies=0, jacoby=True):
    """Build a gnubg command file to get pre-roll cube analytics for a position.

    The checkers represent the board from the perspective of the player on roll
    (pre-roll). gnubg's 'hint' with 'set turn 1' will give cube analytics
    including cubeless probabilities.

    Args:
        checkers: 26-element list (player on roll's perspective)
        n_plies: evaluation depth (0, 1, 2, or 3)
        jacoby: if True, enable Jacoby rule (gammons don't count when cube
            is centered)

    Returns:
        Command string for gnubg.
    """
    cmd = 'new session\n'
    cmd += f'set jacoby {"on" if jacoby else "off"}\n'
    cmd += f'set evaluation chequer eval plies {n_plies}\n'
    cmd += f'set evaluation cubedecision eval plies {n_plies}\n'
    # Lock cube at 1 for cubeless evaluation
    cmd += 'set cube value 1\n'

    cmd += f'set board simple {_board_simple_str(checkers)}\n'

    # Ensure it's the player's turn (player = "mghig" = X in gnubg)
    cmd += 'set turn 1\n'
    cmd += 'hint\n'
    return cmd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

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


def _parse_cube_section_full(lines):
    """Parse a single 'Cube analysis' section including cubeful equities.

    Args:
        lines: list of strings starting from the 'Cube analysis' header line.

    Returns:
        dict with cubeless probs/equity + cubeful ND/DT/DP + action info.
    """
    # Line 0: "Cube analysis"
    # Line 1: "N-ply cubeless equity +X.XXX" or "+X.XXX (Money: +Y.YYY)" in match play
    eq_line = lines[1].strip()
    eq_match = re.search(r'equity\s+([+-]?\d+\.\d+)', eq_line)
    eq_cubeless = float(eq_match.group(1))

    # Line 2: "  0.527 0.148 0.008 - 0.473 0.128 0.005"
    prob_line = lines[2].strip()
    bits = prob_line.split()
    p_win = float(bits[0])
    p_gw = float(bits[1])
    p_bw = float(bits[2])
    p_gl = float(bits[5])
    p_bl = float(bits[6])

    # Parse cubeful equities from numbered lines:
    # "1. No double           +0.465"
    # "2. Double, pass        +1.000  (+0.535)"
    # "3. Double, take        +0.394  (-0.071)"
    nd = dt = dp = None
    optimal_action = ""
    for line in lines[3:]:
        line = line.strip()
        m = re.match(r'^\d+\.\s+(.+?)\s+([+-]?\d+\.\d+)', line)
        if m:
            act_name = m.group(1).strip()
            val = float(m.group(2))
            if 'No double' in act_name or 'No dbl' in act_name:
                nd = val
            elif 'Double, take' in act_name:
                dt = val
            elif 'Double, pass' in act_name:
                dp = val
        elif 'Proper cube action' in line:
            if ':' in line:
                optimal_action = line.split(':', 1)[1].strip()
            break

    return {
        'p_win': p_win,
        'p_gw': p_gw,
        'p_bw': p_bw,
        'p_gl': p_gl,
        'p_bl': p_bl,
        'equity_cubeless': eq_cubeless,
        'equity_nd': nd,
        'equity_dt': dt,
        'equity_dp': dp,
        'optimal_action': optimal_action,
    }


def _parse_cube_analytics_full(output, search_start=0):
    """Parse a full cube analytics section including cubeful equities.

    Args:
        output: full gnubg stdout string.
        search_start: character offset to start searching from.

    Returns:
        (result_dict, next_offset) where result_dict has cubeless probs/equity
        plus cubeful ND/DT/DP, and next_offset is the position after this section.
    """
    idx = output.find('Cube analysis', search_start)
    if idx == -1:
        raise ValueError(
            f"No cube analysis found in gnubg output (from offset {search_start})")

    section = output[idx:]
    lines = section.split('\n')
    result = _parse_cube_section_full(lines)

    # Find next offset past this section
    next_offset = idx + len('Cube analysis') + 1
    return result, next_offset


# ---------------------------------------------------------------------------
# Module-level functions (backward-compatible)
# ---------------------------------------------------------------------------

def post_move_analytics(checkers, n_plies=0, timeout=60, jacoby=True):
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
        jacoby: if True, enable Jacoby rule

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
    cmd = _build_cube_analytics_command(opp_board, n_plies, jacoby=jacoby)
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


def post_move_analytics_many(checkers_list, n_plies=0, max_workers=None, timeout=60,
                             jacoby=True):
    """Evaluate multiple post-move positions in parallel.

    Args:
        checkers_list: list of 26-element checker lists (mover's perspective)
        n_plies: evaluation depth
        max_workers: thread pool size (default: CPU count)
        timeout: per-position timeout in seconds
        jacoby: if True, enable Jacoby rule

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
            future = executor.submit(post_move_analytics, checkers, n_plies, timeout,
                                     jacoby=jacoby)
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


# ---------------------------------------------------------------------------
# GnuBgAnalyzer class
# ---------------------------------------------------------------------------

class GnuBgAnalyzer:
    """GNUbg-backed analyzer with the same interface as BgBotAnalyzer.

    Usage::

        from bgsage import GnuBgAnalyzer
        analyzer = GnuBgAnalyzer(eval_level="3ply")
        cube = analyzer.cube_action(board, cube_value=1, cube_owner="centered")
        print(cube.equity_nd, cube.equity_dt, cube.optimal_action)
    """

    def __init__(self, eval_level="1ply", timeout=120, jacoby=True):
        if eval_level not in _EVAL_LEVELS:
            raise ValueError(
                f"eval_level must be one of {list(_EVAL_LEVELS)}, got {eval_level!r}")
        self.n_plies = _EVAL_LEVELS[eval_level]
        # Display label uses our XG convention (1-ply = raw NN)
        self.eval_level_str = f"{self.n_plies + 1}-ply"
        self.timeout = timeout
        self.jacoby = jacoby

    # -- cube_action --------------------------------------------------------

    def cube_action(self, board, cube_value=1, cube_owner="centered",
                    *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate cube decision for a pre-roll position.

        Returns CubeActionResult with cubeful ND/DT/DP equities and cubeless
        probabilities, all from the current player's perspective.

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        j = self.jacoby if jacoby is None else jacoby
        if away1 > 0 or away2 > 0:
            j = False
        cmd = self._build_preroll_command(board, cube_value, cube_owner,
                                         away1=away1, away2=away2,
                                         is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout)
        r, _ = _parse_cube_analytics_full(output)

        probs = Probabilities(r['p_win'], r['p_gw'], r['p_bw'],
                              r['p_gl'], r['p_bl'])

        nd = r['equity_nd'] if r['equity_nd'] is not None else 0.0
        dt = r['equity_dt'] if r['equity_dt'] is not None else 0.0
        dp = r['equity_dp'] if r['equity_dp'] is not None else 1.0

        should_double = min(dt, dp) > nd
        should_take = dt <= dp
        if should_double:
            optimal_action = "Double/Take" if should_take else "Double/Pass"
            optimal_equity = dt if should_take else dp
        else:
            optimal_action = "No Double"
            optimal_equity = nd

        return CubeActionResult(
            probs=probs,
            cubeless_equity=r['equity_cubeless'],
            equity_nd=nd,
            equity_dt=dt,
            equity_dp=dp,
            should_double=should_double,
            should_take=should_take,
            optimal_equity=optimal_equity,
            optimal_action=optimal_action,
            eval_level=self.eval_level_str,
        )

    # -- post_move_analytics ------------------------------------------------

    def post_move_analytics(self, board, cube_owner="centered", cube_value=1,
                            *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate a post-move position (after move, before opponent rolls).

        Returns PostMoveAnalysis with cubeless probs/equity and cubeful equity.
        The cubeful equity is the ND cubeful equity from GNUbg (the value of
        the position without any doubling action).

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        if len(board) != 26:
            raise ValueError(f"board must have 26 elements, got {len(board)}")

        j = self.jacoby if jacoby is None else jacoby
        if away1 > 0 or away2 > 0:
            j = False

        # Flip to opponent's perspective (they are about to roll)
        opp_board = _flip_board(board)

        cmd = self._build_preroll_command(opp_board, cube_value, cube_owner,
                                         away1=away1, away2=away2,
                                         is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout)
        r, _ = _parse_cube_analytics_full(output)

        # Invert to mover's perspective
        probs = Probabilities(
            win=1.0 - r['p_win'],
            gammon_win=r['p_gl'],
            backgammon_win=r['p_bl'],
            gammon_loss=r['p_gw'],
            backgammon_loss=r['p_bw'],
        )
        cubeless_eq = -r['equity_cubeless']
        cubeful_eq = -r['equity_nd'] if r['equity_nd'] is not None else cubeless_eq

        return PostMoveAnalysis(
            probs=probs,
            cubeless_equity=cubeless_eq,
            cubeful_equity=cubeful_eq,
            eval_level=self.eval_level_str,
        )

    # -- checker_play -------------------------------------------------------

    def checker_play(self, board, die1, die2, cube_value=1, cube_owner="centered",
                     *, away1=0, away2=0, is_crawford=False, jacoby=None):
        """Evaluate all legal moves for a position + dice.

        Returns CheckerPlayResult with moves sorted best-first by cubeless equity.
        Uses batch evaluation: generates all legal boards, evaluates each in a
        single GNUbg subprocess (one hint per board from the opponent's perspective).

        Args:
            jacoby: Override the instance default. None = use self.jacoby.
                Auto-disabled for match play.
        """
        from .board import possible_moves

        j = self.jacoby if jacoby is None else jacoby
        if away1 > 0 or away2 > 0:
            j = False

        candidates = possible_moves(board, die1, die2)

        if len(candidates) == 0:
            return CheckerPlayResult(
                moves=[], board=list(board), die1=die1, die2=die2,
                eval_level=self.eval_level_str)

        if len(candidates) == 1:
            # Single legal move — still evaluate it
            b = candidates[0]
            opp_board = _flip_board(b)
            cmd = self._build_preroll_command(opp_board, cube_value, cube_owner,
                                             away1=away1, away2=away2,
                                             is_crawford=is_crawford, jacoby=j)
            output = _run_gnubg(cmd, timeout=self.timeout)
            r, _ = _parse_cube_analytics_full(output)
            probs = Probabilities(
                win=1.0 - r['p_win'], gammon_win=r['p_gl'],
                backgammon_win=r['p_bl'], gammon_loss=r['p_gw'],
                backgammon_loss=r['p_bw'])
            eq = -r['equity_cubeless']
            move = MoveAnalysis(
                board=list(b), equity=eq, cubeless_equity=eq,
                probs=probs, equity_diff=0.0, eval_level=self.eval_level_str)
            return CheckerPlayResult(
                moves=[move], board=list(board), die1=die1, die2=die2,
                eval_level=self.eval_level_str)

        # Batch: evaluate all candidate boards in one GNUbg subprocess
        cmd = self._build_batch_postmove_command(
            candidates, cube_value, cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=j)
        output = _run_gnubg(cmd, timeout=self.timeout)

        move_analyses = []
        search_start = 0
        for cand_board in candidates:
            idx = output.find('Cube analysis', search_start)
            if idx == -1:
                continue
            section_lines = output[idx:].split('\n')
            r = _parse_cube_section_full(section_lines)
            search_start = idx + len('Cube analysis') + 1

            # Invert to mover's perspective
            probs = Probabilities(
                win=1.0 - r['p_win'], gammon_win=r['p_gl'],
                backgammon_win=r['p_bl'], gammon_loss=r['p_gw'],
                backgammon_loss=r['p_bw'])
            eq = -r['equity_cubeless']

            move_analyses.append(MoveAnalysis(
                board=list(cand_board), equity=eq, cubeless_equity=eq,
                probs=probs, equity_diff=0.0,
                eval_level=self.eval_level_str))

        # Sort best-first by equity
        move_analyses.sort(key=lambda m: m.equity, reverse=True)

        # Compute equity_diff relative to best
        if move_analyses:
            best_eq = move_analyses[0].equity
            for m in move_analyses:
                m.equity_diff = m.equity - best_eq

        return CheckerPlayResult(
            moves=move_analyses, board=list(board), die1=die1, die2=die2,
            eval_level=self.eval_level_str)

    # -- command builders ---------------------------------------------------

    @staticmethod
    def _match_prefix(away1, away2, is_crawford):
        """Build GNUbg commands for match context.

        Returns 'new session\\n' for money games (away1==0 and away2==0),
        or 'new match N\\nset score ...\\n[set postcrawford on\\n]' for match play.
        """
        if not away1 and not away2:
            return 'new session\n'
        match_length = max(away1, away2)
        player_score = match_length - away1
        opp_score = match_length - away2
        cmd = f'new match {match_length}\n'
        cmd += f'set score {opp_score} {player_score}\n'
        # Post-Crawford: one player is 1-away and Crawford has already occurred.
        # GNUbg defaults to Crawford when a player first reaches 1-away, so we
        # must explicitly enable post-Crawford play.  During Crawford
        # (is_crawford=True) we don't set this — GNUbg handles it automatically.
        if not is_crawford and (away1 == 1 or away2 == 1):
            cmd += 'set postcrawford on\n'
        return cmd

    def _build_preroll_command(self, checkers, cube_value=1, cube_owner="centered",
                               *, away1=0, away2=0, is_crawford=False, jacoby=True):
        """Build a GNUbg command for a pre-roll position hint (cube analytics)."""
        cmd = self._match_prefix(away1, away2, is_crawford)
        cmd += f'set jacoby {"on" if jacoby else "off"}\n'
        cmd += f'set evaluation chequer eval plies {self.n_plies}\n'
        cmd += f'set evaluation cubedecision eval plies {self.n_plies}\n'
        cmd += f'set cube value {cube_value}\n'

        # Cube owner: "centered" = default, "player" = turn 1 (on roll),
        # "opponent" = turn 0
        if cube_owner == "player":
            cmd += 'set cube owner 1\n'
        elif cube_owner == "opponent":
            cmd += 'set cube owner 0\n'

        cmd += f'set board simple {_board_simple_str(checkers)}\n'
        cmd += 'set turn 1\n'
        cmd += 'hint\n'
        return cmd

    def _build_batch_postmove_command(self, boards, cube_value=1,
                                       cube_owner="centered",
                                       *, away1=0, away2=0, is_crawford=False,
                                       jacoby=True):
        """Build a GNUbg command that evaluates multiple post-move positions.

        For each board, we flip to the opponent's perspective and run hint.
        All boards are evaluated in a single GNUbg subprocess.
        """
        cmd = self._match_prefix(away1, away2, is_crawford)
        cmd += f'set jacoby {"on" if jacoby else "off"}\n'
        cmd += f'set evaluation chequer eval plies {self.n_plies}\n'
        cmd += f'set evaluation cubedecision eval plies {self.n_plies}\n'
        cmd += f'set cube value {cube_value}\n'

        if cube_owner == "player":
            cmd += 'set cube owner 1\n'
        elif cube_owner == "opponent":
            cmd += 'set cube owner 0\n'

        for board in boards:
            opp_board = _flip_board(board)
            cmd += f'set board simple {_board_simple_str(opp_board)}\n'
            cmd += 'set turn 1\n'
            cmd += 'hint\n'

        return cmd
