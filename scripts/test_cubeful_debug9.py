#!/usr/bin/env python3
"""Narrow down the discrepancy: check if C++ selects different best moves.

The Python trace uses evaluate_board(candidate, opp_pre_roll) for move selection.
The C++ code uses strategy.evaluate_probs(candidates[i], board) where board = opp_pre_roll.
Both should use the (Board, Board) overload and classify from opp_pre_roll.

But wait — let me check: does the C++ code's evaluate_probs(candidates[i], board)
at line 233 actually call the (Board, Board) overload? The `board` parameter is of
type `const Board&`, and `candidates[i]` is also `Board`. So yes, it's (Board, Board).

Actually, I wonder if the issue is that my Python trace and the C++ code handle the
opp_pre_roll position the same way. Let me check by having the C++ do just ONE roll
at 1-ply and see if we match.

Another thought: what if the C++ code has a different `possible_boards` result
than Python's `possible_moves`? They might return different candidate lists.
"""

import sys, os
sys.path.insert(0, "build_macos")
sys.path.insert(0, ".")
import bgbot_cpp

MODELS = "models"
PR_W = os.path.join(MODELS, "sl_s5_purerace.weights.best")
RC_W = os.path.join(MODELS, "sl_s5_racing.weights.best")
AT_W = os.path.join(MODELS, "sl_s5_attacking.weights.best")
PM_W = os.path.join(MODELS, "sl_s5_priming.weights.best")
AN_W = os.path.join(MODELS, "sl_s5_anchoring.weights.best")

strat = bgbot_cpp.GamePlanStrategy(PR_W, RC_W, AT_W, PM_W, AN_W, 200, 400, 400, 400, 400)

POST_MOVE = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 1, 0, 0, -3, 0, -5, 0, 0, 0, 0, 1, 0]
opp_pre_roll = list(bgbot_cpp.flip_board(POST_MOVE))

# Let's think about this differently.
# cubeful_equity_nply(opp_pre_roll, CENTERED, strat, 1) = -0.054
# My manual trace using 0-ply leaf evaluations = -0.0075
#
# The 0-ply leaf evaluations use cubeful_equity_nply(inner_opp_pre, CENTERED, strat, 0)
# which I verified works correctly.
#
# There's a 0.047 gap. Where does it come from?
#
# Wait — could the issue be that in the 1-ply recursion, the code evaluates
# candidates using evaluate_probs(candidates[i], board) which is (Board, Board),
# while at the 0-ply leaf it uses evaluate_probs(flipped, race) which is (Board, bool)?
# These use different game plan classifications (pre-move vs post-move).
# This means the MOVE SELECTION might choose a different best move than what
# the LEAF EVALUATION sees.
#
# But this would only cause different game plan classification at the leaf,
# not different move selection. My Python trace uses the same move selection
# as C++ (both use evaluate_board with (Board, Board)).
#
# Hmm, let me try to verify by checking if the number of candidates matches.
# And let me also try a smaller test: a position where there's only 1 legal move
# per roll, to eliminate move selection as a variable.

# Actually, let me try something simpler: check if the C++ cubeful 1-ply
# matches for a position where the cube CANNOT be doubled (PLAYER owns).
# Then there's no cube decision logic at the leaves, and the 1-ply
# should just be the weighted average of 0-ply leaf PLAYER values.

print("=== Test: 1-ply with PLAYER ownership (no cube decisions at leaves) ===")
eq_1_player = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1)
print(f"C++ 1-ply PLAYER: {eq_1_player:+.6f}")

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

# When owner = PLAYER:
# flip_owner(PLAYER) = OPPONENT → opp_owner = OPPONENT
# opp_can_double = (OPPONENT == CENTERED || OPPONENT == PLAYER) → FALSE!
# So no cube decisions at the leaves.
# player_eq_for_roll = player_eq_nd = -opp_eq_nd

sum_eq = 0.0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)

    # Find best move
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])
    inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

    # opp_owner = OPPONENT → opp can't double → just ND
    opp_nd = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    player_eq = -opp_nd
    sum_eq += weight * player_eq

print(f"Manual 1-ply PLAYER: {sum_eq / 36.0:+.6f}")
print(f"Match? {abs(sum_eq / 36.0 - eq_1_player) < 0.0001}")

# Now test OPPONENT ownership
print("\n=== Test: 1-ply with OPPONENT ownership ===")
eq_1_opp = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.OPPONENT, strat, 1)
print(f"C++ 1-ply OPPONENT: {eq_1_opp:+.6f}")

# When owner = OPPONENT:
# flip_owner(OPPONENT) = PLAYER → opp_owner = PLAYER
# opp_can_double = (PLAYER == CENTERED || PLAYER == PLAYER) → TRUE!
# So the opponent ("opp" in code = "us" in outer frame) CAN double.

sum_eq2 = 0.0
n_double = 0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])
    inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

    # opp_owner = PLAYER → opp can double
    opp_nd = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.PLAYER, strat, 0)
    opp_dt = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)

    player_nd = -opp_nd
    player_dt = -opp_dt * 2.0
    player_dp = -1.0

    opp_dt_scaled = opp_dt * 2.0
    opp_best = min(opp_dt_scaled, 1.0)
    opp_doubles = opp_best > opp_nd

    if opp_doubles:
        n_double += 1
        peq = player_dt if player_dt >= player_dp else player_dp
    else:
        peq = player_nd

    sum_eq2 += weight * peq

print(f"Manual 1-ply OPPONENT: {sum_eq2 / 36.0:+.6f}")
print(f"Match? {abs(sum_eq2 / 36.0 - eq_1_opp) < 0.0001}")
print(f"Doubles: {n_double}/21")

# Now test CENTERED
print("\n=== Test: 1-ply with CENTERED ownership ===")
eq_1_cent = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"C++ 1-ply CENTERED: {eq_1_cent:+.6f}")

# When owner = CENTERED:
# flip_owner(CENTERED) = CENTERED → opp_owner = CENTERED
# opp_can_double = (CENTERED == CENTERED || CENTERED == PLAYER) → TRUE!

sum_eq3 = 0.0
n_double3 = 0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])
    inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

    # opp_owner = CENTERED → opp can double
    opp_nd = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.CENTERED, strat, 0)
    # dt_opp_owner = OPPONENT (from opp's perspective, player owns after double)
    opp_dt = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)

    player_nd = -opp_nd
    player_dt = -opp_dt * 2.0
    player_dp = -1.0

    opp_dt_scaled = opp_dt * 2.0
    opp_best = min(opp_dt_scaled, 1.0)
    opp_doubles = opp_best > opp_nd

    if opp_doubles:
        n_double3 += 1
        peq = player_dt if player_dt >= player_dp else player_dp
        dec = "D/T" if player_dt >= player_dp else "D/P"
        print(f"  {d1}-{d2}: opp_ND={opp_nd:+.5f} opp_DT2x={opp_dt_scaled:+.5f} → {dec}")
    else:
        peq = player_nd

    sum_eq3 += weight * peq

print(f"\nManual 1-ply CENTERED: {sum_eq3 / 36.0:+.6f}")
print(f"C++ 1-ply CENTERED:   {eq_1_cent:+.6f}")
print(f"Match? {abs(sum_eq3 / 36.0 - eq_1_cent) < 0.0001}")
print(f"Doubles: {n_double3}/21")
