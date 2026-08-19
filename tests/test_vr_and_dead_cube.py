# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Tests for rollout variance reduction and match dead-cube handling.

These lock in three defects that were live together and produced, among other
things, a rolled-out P(backgammon loss) of -45%.

1. VR POLICY MISMATCH. Variance reduction subtracts, per half-move,
   `luck = value(move actually played) - mean over the 21 rolls of
   value(move the policy would play)`. That has zero mean ONLY if both sides
   rank candidates by the same rule. The trial selected moves by CUBEFUL
   equity (cubeful_trial_moves) while the VR mean was built from the
   CUBELESS-best move, so E[luck] != 0 and the correction became an unbounded
   systematic offset. Since cubeless equity contains -P(bl), the cubeless-best
   move is biased toward low backgammon loss, the mean came out too low, luck
   came out positive, and the corrected probability was driven below zero.
   The bias did NOT shrink with trials (stable at -44% from 72 to 20736).

2. MATCH DEAD-CUBE UNITS. The N-ply cubeful recursion carries EQUITY for money
   but MWC for match play. Dead-cube branches short-circuited with money-style
   `cubeless_equity` in both cases, so match values were money equities in an
   MWC slot -- a 2-ply match ND of +10.43 where +0.31 was correct.

3. OWNERSHIP-BLIND MATCH DEADNESS. A cube can only be turned by a player who
   may double it AND gains by doing so. Deadness required BOTH away scores to
   be covered, which is right only for a CENTERED cube; with an owned cube
   only the owner can turn it, so the owner's away score alone decides. e.g.
   3-away/2-away with the cube on 2 owned by the trailer is dead -- their win
   at 2 already takes the match -- but looked live.

KNOWN ORDERING HAZARD (open, not caused by these tests). This module passes in
isolation but four of its match-sensitive cases fail when run after
tests/test_2ply_details.py:

    pytest tests/test_2ply_details.py                 ->  11 passed
    pytest tests/test_vr_and_dead_cube.py             ->  28 passed
    pytest tests/test_2ply_details.py            tests/test_vr_and_dead_cube.py             ->   4 failed

That is an engine-level state leak between top-level evaluations, not a defect
in these assertions -- some per-thread evaluation cache serves a value across a
change of scoring context. Fixing (2) above is what made it visible: before it,
dead-cube branches stored a money equity in BOTH the money and match cases, so
a leaked entry was consistently wrong and nothing could detect it; now the two
genuinely differ. Do NOT relax these assertions to make the suite green -- the
leak is the bug they are reporting.
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

import bgbot_cpp
from bgsage import BgBotAnalyzer
from bgsage.weights import WeightConfig


# A decided race, mover on roll: 15 checkers on board (nothing borne off), 193
# pips against 18, six checkers still on the 19/20 points (inside the
# opponent's home board, so a backgammon is genuinely possible). The opponent
# has 9 checkers left and 6 already off. The mover loses with certainty, which
# pins P(win) and P(gammon loss) at their boundaries and leaves P(backgammon
# loss) as the only component that varies between candidate moves -- so it
# absorbs the whole of any policy mismatch. This is the position that reported
# -45%.
DECIDED_RACE = [0, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 2,
                0, 0, 0, 0, 0, 3, 3, -1, -1, -4, -3, 0]
DECIDED_RACE_DICE = (5, 5)

# A post-move board from that position (after 8/3(2) 6/1(2)) -- the worst
# offender, which rolled out to P(bl) = -45.02% with VR on and +10.4% with VR
# off, both stable to 20736 trials.
DECIDED_RACE_POST_MOVE = [0, 2, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2,
                          0, 0, 0, 0, 0, 3, 3, -1, -1, -4, -3, 0]

# An ordinary contact position (opening 3-1 played 8/5 6/5) for the controls:
# nothing here is degenerate, so these must keep behaving as they always did.
CONTACT = [0, -2, 0, 0, 0, 2, 4, 0, 2, 0, 0, 0, -5, 5,
           0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

OWNERS = {
    "centered": bgbot_cpp.CubeOwner.CENTERED,
    "player": bgbot_cpp.CubeOwner.PLAYER,
    "opponent": bgbot_cpp.CubeOwner.OPPONENT,
}

# The match state the bug was first seen at: 3-point match, score 0-1, so the
# mover is 3-away and the opponent 2-away, with the cube on 2 owned by the
# opponent.
MATCH_3A_2A = dict(away1=3, away2=2, is_crawford=False)


def _rollout_1t(n_trials=360, enable_vr=True):
    """A 1T-configured rollout strategy: 72-trial-style truncated rollout at
    trunc-5 / 1-ply, with `enable_vr` under our control (the analyzer does not
    expose it, and create_rollout defaults it on)."""
    w = WeightConfig.default()
    return bgbot_cpp.create_rollout(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        n_trials=n_trials, truncation_depth=5, decision_ply=1,
        n_threads=1, seed=42, ultra_late_threshold=2, enable_vr=enable_vr)


# ---------------------------------------------------------------------------
# 1. Rolled-out probabilities must be probabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cube_value,cube_owner,match_state", [
    (2, "opponent", MATCH_3A_2A),      # the original report
    (1, "centered", {}),               # money, centered -- also engages cube-aware selection
    (2, "opponent", {}),               # money, owned
])
def test_rolled_out_probabilities_are_valid(cube_value, cube_owner, match_state):
    """Every rolled-out probability must lie in [0, 1] and respect the nesting
    invariants. Pre-fix, 94 of the 105 candidate moves here carried at least
    one out-of-range value, with P(bl) reaching -45%.

    This is deliberately checked across EVERY candidate move rather than the
    best one: the move filter only deep-evaluates a handful, and the corrupt
    values showed up on the moves that keep checkers back.
    """
    analyzer = BgBotAnalyzer(eval_level="truncated1", cubeful=True)
    result = analyzer.checker_play(
        DECIDED_RACE, *DECIDED_RACE_DICE,
        cube_value=cube_value, cube_owner=cube_owner, **match_state)

    # An unbiased Monte-Carlo estimate of a probability pinned AT a boundary
    # can graze past it: this position's true P(bl) is ~0 for the moves that
    # clear the back checkers, and a 72-trial estimate lands a few ten-
    # thousandths either side. That is sampling noise and is expected. The
    # defect this guards against was systematic and three orders of magnitude
    # larger (-0.45, and stable as trials went 72 -> 20736), so the band below
    # separates the two without asserting that noise cannot exist.
    NOISE = 0.01

    assert result.moves, "expected legal moves for 5-5"
    for move in result.moves:
        p = move.probs
        values = {
            "win": p.win, "gammon_win": p.gammon_win,
            "backgammon_win": p.backgammon_win,
            "gammon_loss": p.gammon_loss, "backgammon_loss": p.backgammon_loss,
        }
        for name, value in values.items():
            assert -NOISE <= value <= 1.0 + NOISE, (
                f"P({name}) = {value:.6f} is outside [0, 1] by more than "
                f"sampling noise -- the VR correction has acquired a bias")
        assert p.gammon_win >= p.backgammon_win - 1e-6, (
            "backgammon win cannot exceed gammon win")
        assert p.gammon_loss >= p.backgammon_loss - 1e-6, (
            "backgammon loss cannot exceed gammon loss")


# ---------------------------------------------------------------------------
# 2. Variance reduction must not move the answer
# ---------------------------------------------------------------------------

def test_variance_reduction_is_unbiased():
    """VR is a control variate: it may only reduce variance, never shift the
    estimate. So a VR-on rollout and a VR-off rollout of the same position with
    the same seed must agree.

    This is the direct root-cause test. Pre-fix the two disagreed by 54
    percentage points on P(backgammon loss) (-45% vs +10%), and the gap was a
    genuine bias -- it stayed put as trials went 72 -> 20736 rather than
    shrinking as sampling noise would.
    """
    # Go through cube_decision, not the plain cubeless evaluate_board: the bug
    # only bites when a live cube BRANCH exists, because that is what turns on
    # cube-aware move selection (cubeful_trial_moves && cube_active &&
    # n_branches > 0). A cubeless post-move evaluation has no branches and so
    # never exercised it -- an earlier version of this test used evaluate_board
    # and passed against the broken engine.
    def roll(enable_vr):
        return _rollout_1t(enable_vr=enable_vr).cube_decision(
            DECIDED_RACE, 2, OWNERS["opponent"], 3, 2, False)

    on, off = roll(True), roll(False)

    for i, name in enumerate(["win", "gammon_win", "backgammon_win",
                              "gammon_loss", "backgammon_loss"]):
        assert abs(on["probs"][i] - off["probs"][i]) < 0.05, (
            f"VR shifted P({name}): {on['probs'][i]:.4f} with VR vs "
            f"{off['probs'][i]:.4f} without -- VR must not bias the estimate")
    assert abs(on["cubeless_equity"] - off["cubeless_equity"]) < 0.05, (
        f"VR shifted cubeless equity: {on['cubeless_equity']:+.4f} vs "
        f"{off['cubeless_equity']:+.4f}")


# ---------------------------------------------------------------------------
# 3. Which cubes are dead
# ---------------------------------------------------------------------------

# (label, cube_value, owner, away1, away2, expected_dead)
DEADNESS_CASES = [
    # Owned cubes: only the owner can turn it, so only the owner's away score
    # matters. These are the cases the old both-away-scores test got wrong.
    ("opponent owns, opponent wins match at this cube", 2, "opponent", 3, 2, True),
    ("player owns, player wins match at this cube",     2, "player",   2, 3, True),
    ("opponent owns, opponent still needs more",        2, "opponent", 3, 3, False),
    ("player owns, player still needs more",            2, "player",   3, 3, False),
    # Centered: either may double, so both must be covered. Unchanged behaviour.
    ("centered, both win match at this cube",           2, "centered", 2, 2, True),
    ("centered, live",                                  1, "centered", 5, 5, False),
]


@pytest.mark.parametrize("plies", [1, 2, 3])
@pytest.mark.parametrize("label,cube_value,owner,away1,away2,expect_dead",
                         DEADNESS_CASES)
def test_dead_cube_ignores_cube_life_index(plies, label, cube_value, owner,
                                           away1, away2, expect_dead):
    """A cube that can never be turned again carries no cube leverage, so its
    equity must not depend on the Janowski cube-life index x. A live cube must
    depend on it.

    Driving this through x (rather than asserting a stored number) tests the
    property itself, and covers every ply because the deadness check has to
    hold at the leaves, in the N-ply recursion, and in the shared-cache
    fast path for all-dead nodes -- which is where a 3-ply-only bug hid.
    """
    w = WeightConfig.default()

    def nd(cube_x):
        return bgbot_cpp.cube_decision_nply_unified(
            CONTACT, cube_value, OWNERS[owner], plies,
            w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
            away1=away1, away2=away2, is_crawford=False,
            cube_x_override=cube_x)["equity_nd"]

    normal_x, dead_x = nd(0.68), nd(0.0)
    is_dead = abs(normal_x - dead_x) < 1e-4
    assert is_dead == expect_dead, (
        f"{label} at {plies}-ply: engine says "
        f"{'dead' if is_dead else 'live'}, expected "
        f"{'dead' if expect_dead else 'live'} "
        f"(ND {normal_x:+.4f} at x=0.68 vs {dead_x:+.4f} at x=0)")


# ---------------------------------------------------------------------------
# 4. Dead-cube values must be in the right units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plies", [1, 2, 3])
def test_match_dead_cube_equity_is_normalised(plies):
    """A dead cube in match play must yield a normalised match equity, not a
    money equity.

    The recursion works in MWC space for match play, and the dead-cube
    short-circuit inserted a money equity there. Normalising that produced
    a 2-ply ND of +10.43 and a 3-ply ND of +7.45 for a position worth about
    +0.31. Normalised match equity is anchored so that winning the current
    cube value scores +1, so anything beyond a couple of points is nonsense.
    """
    w = WeightConfig.default()
    for away1, away2 in [(5, 5), (3, 2), (7, 4)]:
        nd = bgbot_cpp.cube_decision_nply_unified(
            CONTACT, 1, OWNERS["centered"], plies,
            w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
            away1=away1, away2=away2, is_crawford=False,
            max_cube_value=1)["equity_nd"]
        assert -3.0 < nd < 3.0, (
            f"{away1}a-{away2}a dead cube at {plies}-ply gave ND={nd:+.4f}; "
            f"a normalised match equity cannot be this large -- a money "
            f"equity has leaked into the MWC-space recursion")


@pytest.mark.parametrize("plies", [1, 2, 3])
def test_money_dead_cube_equals_cubeless_equity(plies):
    """Money play must be untouched by any of this: with the cube dead, the
    cubeful equity is by definition the cubeless equity."""
    w = WeightConfig.default()
    result = bgbot_cpp.cube_decision_nply_unified(
        CONTACT, 1, OWNERS["centered"], plies,
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        max_cube_value=1)
    assert abs(result["equity_nd"] - result["cubeless_equity"]) < 1e-4, (
        f"money dead cube at {plies}-ply: ND={result['equity_nd']:+.6f} "
        f"but cubeless={result['cubeless_equity']:+.6f}")


# ---------------------------------------------------------------------------
# 5. Tie-break on money cubeless equity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["3ply", "4ply", "truncated1", "truncated2",
                                   "truncated3"])
def test_indifferent_score_does_not_invent_backgammons(level):
    """When every legal move scores identically, selection must still prefer
    the move that is better on raw cubeless merit -- otherwise the engine plays
    arbitrarily inside its own lookahead and reports outcomes that cannot
    happen.

    At 3-away/2-away with the cube on 2, a single, a gammon and a backgammon
    all simply lose the match, so all 105 moves for this 5-5 tie at EXACTLY the
    same equity (measured spread 0.000000). After 20/15(3) 19/14 only two
    checkers remain on the 19-point, every point from 1 to 18 is open, and the
    opponent needs at least three more rolls to bear off nine checkers -- so we
    clear on our very next turn and P(backgammon loss) is 0 by construction.

    The engine used to report 3.57% at 3-ply and 99%+ at 2T/3T here, while the
    UNLIMITED game -- same position, same cube, real 0.12 equity spread, so
    selection is forced -- correctly reported 0.00%. The unlimited answer is
    the control: both must agree.
    """
    cube = dict(cube_value=2, cube_owner="opponent")
    analyzer = BgBotAnalyzer(eval_level=level, cubeful=True)

    def backgammon_loss(**extra):
        result = analyzer.checker_play(DECIDED_RACE, *DECIDED_RACE_DICE,
                                       **cube, **extra)
        target = next(m for m in result.moves
                      if _clears_to_two_back_checkers(m.board))
        return target.probs.backgammon_loss

    unlimited = backgammon_loss()
    match = backgammon_loss(**MATCH_3A_2A)

    assert unlimited < 0.01, (
        f"unlimited P(bl)={unlimited:.4f} after 20/15(3) 19/14, but we clear "
        f"the 19-point before the opponent can finish -- true value is 0")
    assert match < 0.01, (
        f"match P(bl)={match:.4f} vs unlimited {unlimited:.4f} for the same "
        f"position and cube. Every move ties at this score, so selection fell "
        f"back to an arbitrary pick and invented a backgammon that cannot "
        f"happen")


def _clears_to_two_back_checkers(board):
    """The post-move board for 20/15(3) 19/14: the 20-point emptied, two
    checkers left on the 19-point, three landed on 15 and one on 14."""
    return (board[20] == 0 and board[19] == 2
            and board[15] == 3 and board[14] == 1)
