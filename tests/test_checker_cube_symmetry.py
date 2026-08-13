# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for the symmetry between checker_play and cube_action analytics.

For any post-move position M produced by a checker play, the cubeful equity
reported by `analyzer.checker_play(...).moves[i]` must equal the negative of
`analyzer.cube_action(flip(M), ...).optimal_equity` at the same eval level —
because they're computing the same thing from inverted perspectives. The
multi-ply path enforces this via `cubeful_equity_nply`; the rollout path
enforces it via the C++ `cubeful_rollout_position`, which delegates to
`cubeful_cube_decision` on the flipped position.

Catches regressions where one path forgets to apply the opponent's cube
decision (e.g. returning -ND instead of -optimal for moves that leave a drop
position).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)

import pytest

from bgsage import BgBotAnalyzer, flip_board


# --- Symmetry test positions -----------------------------------------------
#
# Each entry is a checker decision: (board, dice, cube_value, cube_owner). The
# board is from the player-on-roll's perspective; cube_owner is also from that
# POV ("centered" / "player" / "opponent").

# 1) User-reported 1-1 disaster, centered cube. At 3T the opponent's optimal
#    action after most plays is D/P, so a correct checker_play must report
#    cubeful equity = -1.0 for the move 3/1(2).
USER_REPORTED_BOARD = [
    0, 0, 0, 2, -2, 3, 3, 2, 2, 2, -1, 0, 0,
    0, 0, 0, -1, 0, -2, -2, -2, -2, -1, -2, 1, 0,
]
USER_REPORTED_DICE = (1, 1)

# 3/1(2) post-move board (two checkers moved from the 3 to the 1 point).
MOVE_3_1_2_POST_BOARD = [
    0, 2, 0, 0, -2, 3, 3, 2, 2, 2, -1, 0, 0,
    0, 0, 0, -1, 0, -2, -2, -2, -2, -1, -2, 1, 0,
]

# 2) seed_69.xg turn 15 — cube=2 owned by player (mover), dice 3-2. The
#    original SP owns the cube, so opp can't legally double. Pre-fix this
#    saturated every move's cubeful equity to -1.0 in the rollout path
#    because the cube-action collapse in cubeful_rollout_position /
#    cube_decision binding never checked can_double(opp_cube). The bug was
#    invisible to a pure symmetry check (both checker_play and cube_action
#    collapsed identically), so _check_symmetry also asserts should_double
#    is False whenever opp can't legally double.
SEED_69_TURN_15_BOARD = [
    0, 0, 2, 3, -2, 2, 3, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 0,
]
SEED_69_TURN_15_DICE = (3, 2)
# XG's #1 for this position: 6/1 (combined-die move from the 6 point to the
# 1 point). The 2T best after the fix should be this same move.
SEED_69_T15_6_1 = [
    0, 1, 2, 3, -2, 2, 2, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, -1, -2, -2, -3, -3, -2, 0, 1, 0,
]


# Cases used by the parametrized symmetry tests. Adding a new case here
# automatically extends every symmetry test below.
#
# (id, board, dice, cube_value, cube_owner)
SYMMETRY_CASES = [
    ("centered_cube",      USER_REPORTED_BOARD,    USER_REPORTED_DICE,    1, "centered"),
    ("sp_owns_cube_v2",    SEED_69_TURN_15_BOARD,  SEED_69_TURN_15_DICE,  2, "player"),
]


_FLIP_OWNER = {"player": "opponent", "opponent": "player", "centered": "centered"}


def _opp_after(post_move_board):
    """Flip a post-move board to the opponent's pre-roll perspective."""
    return flip_board(post_move_board)


def _check_symmetry(
    analyzer, board, die1, die2, *, tol,
    cube_value=1, cube_owner="centered",
):
    """Best move's cubeful equity must equal -opp_cube_action.optimal_equity.

    Also asserts the bug-catching invariant: when opp can't legally double on
    the post-best-move board (i.e. the original SP owns the cube, so the
    flipped cube_owner from opp's POV is ``"opponent"``), the rollout's noisy
    DT estimate must not collapse cube_action to should_double=True. This is
    the failure mode the pure symmetry assertion can't catch — pre-fix both
    checker_play and cube_action collapsed to -1/+1 identically, so the diff
    was 0 even though both values were wrong.

    Returns the (checker_cubeful, opp_optimal, diff) triple for diagnostics.
    """
    cp = analyzer.checker_play(
        board, die1, die2,
        cube_value=cube_value, cube_owner=cube_owner,
        jacoby=True, beaver=True,
    )
    assert cp.moves, "Expected at least one legal move"
    best = cp.moves[0]

    opp_cube_owner = _FLIP_OWNER[cube_owner]
    ca = analyzer.cube_action(
        _opp_after(best.board),
        cube_value=cube_value, cube_owner=opp_cube_owner,
        jacoby=True, beaver=True,
    )
    diff = abs(best.equity - (-ca.optimal_equity))
    assert diff < tol, (
        f"checker_play cubeful equity ({best.equity:+.4f}) does not match "
        f"-cube_action.optimal_equity ({-ca.optimal_equity:+.4f}); diff={diff:.4f}. "
        f"Best move post-board: {best.board}; opp action: {ca.optimal_action}, "
        f"ND={ca.equity_nd:+.4f} DT={ca.equity_dt:+.4f} DP={ca.equity_dp:+.4f}"
    )
    # Bug-catching: when opp can't legally double (cube owned by original SP),
    # should_double MUST be False regardless of the rollout's DT estimate.
    if opp_cube_owner == "opponent":
        assert not ca.should_double, (
            f"opp cube_action incorrectly reports should_double=True even though "
            f"opp can't legally double (cube owned by original SP). "
            f"ND={ca.equity_nd:+.4f} DT={ca.equity_dt:+.4f} DP={ca.equity_dp:+.4f}"
        )
    return best.equity, ca.optimal_equity, diff


# ---------------------------------------------------------------------------
# Multi-ply paths (already worked pre-fix; included for regression coverage).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["2ply", "3ply"])
@pytest.mark.parametrize(
    "case_id,board,dice,cube_value,cube_owner",
    SYMMETRY_CASES,
    ids=[c[0] for c in SYMMETRY_CASES],
)
def test_multi_ply_best_move_symmetry(
    level, case_id, board, dice, cube_value, cube_owner,
):
    """Multi-ply checker_play's best-move cubeful must match opp cube_action."""
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True)
    _check_symmetry(
        analyzer, board, *dice,
        cube_value=cube_value, cube_owner=cube_owner,
        # Multi-ply uses the same cubeful_equity_nply for both paths, so this
        # is an exact equality up to floating-point noise.
        tol=1e-3,
    )


# ---------------------------------------------------------------------------
# Rollout paths (the regression the bug fix targets).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["truncated2", "truncated3"])
@pytest.mark.parametrize(
    "case_id,board,dice,cube_value,cube_owner",
    SYMMETRY_CASES,
    ids=[c[0] for c in SYMMETRY_CASES],
)
def test_rollout_best_move_symmetry(
    level, case_id, board, dice, cube_value, cube_owner,
):
    """Truncated rollout checker_play's best-move cubeful must match opp cube_action.

    Both paths run the same `cubeful_cube_decision` machinery (cubeful_rollout_position
    is a thin wrapper) with the same seeded dice, so the values should match
    exactly (no Monte Carlo noise across the two calls).
    """
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True)
    _check_symmetry(
        analyzer, board, *dice,
        cube_value=cube_value, cube_owner=cube_owner,
        tol=1e-3,
    )


def test_3T_specific_move_3_1_2_is_drop():
    """User's specific complaint: at 3T, the move 3/1(2) leaves opp with a clear
    D/P. The cubeful equity for 3/1(2) must therefore be exactly -1.0 (equal to
    -DP_money = -1.0). Pre-fix this returned ~-0.79 (the ND value)."""
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True)
    cp = analyzer.checker_play(
        USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    match = next((m for m in cp.moves if m.board == MOVE_3_1_2_POST_BOARD), None)
    assert match is not None, "Expected the 3/1(2) move in the candidate list"
    # Tolerance is loose: cube_decision_1ply's beaver / Janowski math is
    # exact, and DP is exactly +1.0 in money, so -optimal_equity is exactly
    # -1.0. Allow a tiny epsilon for fp arithmetic.
    assert abs(match.equity - (-1.0)) < 1e-3, (
        f"3/1(2) cubeful equity at 3T is {match.equity:+.4f}, expected -1.0000. "
        f"Pre-fix value was ~-0.7880 (= -ND, ignoring opp's D/P). "
        f"Post-move board: {match.board}"
    )


def test_3T_3_1_2_matches_cube_action_on_flipped_board():
    """Direct symmetry check for the specific move 3/1(2) at 3T.

    Independent of which move ends up "best", the cubeful equity for 3/1(2)
    in the checker_play result must equal -cube_action.optimal_equity on its
    flipped post-move board.
    """
    analyzer = BgBotAnalyzer(eval_level="truncated3", cubeful=True)
    cp = analyzer.checker_play(
        USER_REPORTED_BOARD, *USER_REPORTED_DICE,
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    match = next((m for m in cp.moves if m.board == MOVE_3_1_2_POST_BOARD), None)
    assert match is not None, "Expected the 3/1(2) move in the candidate list"

    ca = analyzer.cube_action(
        _opp_after(MOVE_3_1_2_POST_BOARD),
        cube_value=1, cube_owner="centered",
        jacoby=True, beaver=True,
    )
    diff = abs(match.equity - (-ca.optimal_equity))
    assert diff < 1e-3, (
        f"3/1(2) checker_play cubeful = {match.equity:+.4f}, "
        f"-opp cube_action optimal = {-ca.optimal_equity:+.4f}, diff={diff:.4f}"
    )


# ---------------------------------------------------------------------------
# can_double guard regression: SP-owned cube must not saturate to -DP.
# ---------------------------------------------------------------------------
#
# Background: discovered while validating a Sage-vs-XG run on seed_69 turn 15
# (cube=2 owned by user, dice 3-2). Pre-fix, `cubeful_evaluate_board` and
# `cube_decision` collapsed every candidate to optimal=DP=+1.0 (SP cubeful =
# -1.0) because the cube-action collapse logic in `cubeful_rollout_position` /
# the `cube_decision` binding always evaluated "opp doubles + SP takes"
# without checking whether opp could legally double. When the original SP owns
# the cube, opp can't double, so the DT branch represents an illegal cube turn
# and must be ignored — the only valid action is ND. Multi-ply
# (`cube_decision_nply`) always handled this via an explicit `can_double`
# guard; the rollout path was missing it.
#
# The parametrized `test_rollout_best_move_symmetry[sp_owns_cube_v2]` covers
# the symmetry + should_double-vs-can_double invariant. The standalone test
# below additionally checks that the rollout-best move under the fix is XG's
# top recommendation (6/1) — pre-fix the "best" was an arbitrary tie among
# saturated -1.0 entries.

def test_sp_owns_cube_picks_xg_best_2T():
    """At seed_69 turn 15 (cube=2 owned by user, dice 3-2), the 2T best move
    after the can_double-guard fix must be the same as XG's #1 (6/1). Pre-fix,
    every candidate collapsed to cubeful equity = -1.0, so the "best" was
    whichever move happened to sort first among the saturated tie."""
    analyzer = BgBotAnalyzer(eval_level="truncated2", cubeful=True)
    cp = analyzer.checker_play(
        SEED_69_TURN_15_BOARD, *SEED_69_TURN_15_DICE,
        cube_value=2, cube_owner="player",
        jacoby=True, beaver=True,
    )
    assert cp.moves, "Expected legal moves"
    # Sanity: best equity must not be saturated. -1.0 would mean "lose the
    # doubled cube outright", which is only correct for genuine DP positions —
    # and here SP is the cube owner, so opp can't even offer the cube.
    assert cp.moves[0].equity > -0.95, (
        f"Best cubeful equity is {cp.moves[0].equity:+.4f}, saturated at -1.0 "
        f"despite SP owning the cube (opp can't legally double). "
        f"Top 3 moves: {[(m.equity, m.board) for m in cp.moves[:3]]}"
    )
    best_board = cp.moves[0].board
    assert tuple(best_board) == tuple(SEED_69_T15_6_1), (
        f"Expected 6/1 (board {SEED_69_T15_6_1}) as 2T best, got {best_board} "
        f"with equity {cp.moves[0].equity:+.4f}"
    )
