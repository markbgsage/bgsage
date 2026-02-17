#!/usr/bin/env python3
"""Test cubeful_equity_nply consistency.

If the 1-ply result is just the weighted average of 0-ply leaf values
(with cube decisions), then we should be able to reproduce it.

Let's check a simpler case: a position with only 1 legal move per roll.
If every roll has exactly 1 candidate, then there's no move selection issue.

But first, let me check: does cubeful_equity_nply(board, owner, strat, 0)
match cl2cf_money on hand-computed pre-roll probs?
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

# Starting position
STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

# Test 1: 0-ply consistency on starting position
print("=== Test 1: 0-ply cubeful consistency on starting position ===")
eq_0 = bgbot_cpp.cubeful_equity_nply(STARTING, bgbot_cpp.CubeOwner.CENTERED, strat, 0)
print(f"cubeful_equity_nply(STARTING, CENTERED, 0) = {eq_0:.6f}")

# Manual: flip → evaluate_probs(flipped, race=is_race(STARTING)) → invert → Janowski
flipped = bgbot_cpp.flip_board(STARTING)
race = bgbot_cpp.is_race(STARTING)
print(f"is_race(STARTING) = {race}")
x = bgbot_cpp.cube_efficiency(STARTING, race)
print(f"cube_efficiency = {x}")

# The C++ uses evaluate_probs(flipped, race) — the (Board, bool) overload
# This classifies game plan from flipped, which is the same as STARTING (symmetric)
print(f"classify(STARTING) = {bgbot_cpp.classify_game_plan(STARTING)}")
print(f"classify(flipped) = {bgbot_cpp.classify_game_plan(list(flipped))}")

# evaluate_board(flipped, STARTING) uses (Board, Board) overload, classifies from STARTING
# evaluate_probs(flipped, race) uses (Board, bool) overload, classifies from flipped
# For the starting position, flipped IS the same as STARTING (symmetric), so they should match
r_bb = strat.evaluate_board(list(flipped), STARTING)
print(f"evaluate_board(flipped, STARTING): eq={r_bb['equity']:.6f}")
print(f"  probs={[f'{p:.4f}' for p in r_bb['probs']]}")

pre_probs = [1.0 - r_bb['probs'][0], r_bb['probs'][3], r_bb['probs'][4],
             r_bb['probs'][1], r_bb['probs'][2]]
jan = bgbot_cpp.cl2cf_money(pre_probs, bgbot_cpp.CubeOwner.CENTERED, x)
print(f"Janowski(pre_roll_probs, CENTERED, x={x:.2f}) = {jan:.6f}")
print(f"C++ cubeful_equity_nply 0-ply: {eq_0:.6f}")
print(f"Match? {abs(jan - eq_0) < 0.001}")

# Test 2: 1-ply cubeful on starting position
print("\n=== Test 2: 1-ply cubeful on starting position ===")
eq_1 = bgbot_cpp.cubeful_equity_nply(STARTING, bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"cubeful_equity_nply(STARTING, CENTERED, 1) = {eq_1:.6f}")

# For the starting position: it's OUR pre-roll.
# For each of our 21 rolls, we find the best move, then the opponent faces a position.
# The opponent might double (cube is CENTERED → can double).
# At 0-ply leaf: evaluate from opponent's perspective.

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

sum_eq = 0.0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(STARTING, d1, d2)

    # Find best move — match C++ code which uses evaluate_probs(candidates[i], board)
    # which is the (Board, Board) overload with board = STARTING
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), STARTING)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])
    opp_pre = bgbot_cpp.flip_board(post_move)

    # opp_owner = flip_owner(CENTERED) = CENTERED
    # opp_can_double = True

    # ND: cubeful_equity_recursive(opp_pre, CENTERED, strat, 0)
    opp_nd = bgbot_cpp.cubeful_equity_nply(list(opp_pre), bgbot_cpp.CubeOwner.CENTERED, strat, 0)

    # DT: cubeful_equity_recursive(opp_pre, OPPONENT, strat, 0)
    opp_dt = bgbot_cpp.cubeful_equity_nply(list(opp_pre), bgbot_cpp.CubeOwner.OPPONENT, strat, 0)

    player_nd = -opp_nd
    player_dt = -opp_dt * 2.0
    player_dp = -1.0

    opp_dt_scaled = opp_dt * 2.0
    opp_best_double = min(opp_dt_scaled, 1.0)
    opp_doubles = opp_best_double > opp_nd

    if opp_doubles:
        if player_dt >= player_dp:
            peq = player_dt
            dec = "D/T"
        else:
            peq = player_dp
            dec = "D/P"
    else:
        peq = player_nd
        dec = "ND "

    sum_eq += weight * peq

    if opp_doubles:
        print(f"{d1}-{d2} (w={weight}): opp_ND={opp_nd:+.5f} opp_DT2x={opp_dt_scaled:+.5f} "
              f"{dec} peq={peq:+.5f}")

print(f"\nManual 1-ply: {sum_eq / 36.0:+.6f}")
print(f"C++ 1-ply:    {eq_1:+.6f}")
print(f"Match? {abs(sum_eq / 36.0 - eq_1) < 0.001}")
