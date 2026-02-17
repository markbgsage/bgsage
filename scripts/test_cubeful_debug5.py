#!/usr/bin/env python3
"""Replicate the C++ cubeful_equity_recursive exactly, using the SAME evaluation
method (evaluate_probs(board, bool)) to match what C++ does."""

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
opp_pre_roll = bgbot_cpp.flip_board(POST_MOVE)

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

# Replicate C++ cubeful_equity_recursive(opp_pre_roll, CENTERED, strat, 1)
# EXACTLY matching the C++ code's behavior.

# owner = CENTERED (input)
# opp_owner = flip_owner(CENTERED) = CENTERED

sum_equity = 0.0
n_double = 0

for d1, d2, weight in ROLLS:
    # Generate roller's legal moves from opp_pre_roll
    candidates = bgbot_cpp.possible_moves(list(opp_pre_roll), d1, d2)

    # Find best move (cubeless) — C++ uses strategy.evaluate_probs(candidates[i], board)
    # where board = opp_pre_roll. This is the (Board, Board) overload.
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), list(opp_pre_roll))
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])

    # After roller moves, opponent's pre-roll = flip(post_move)
    # In C++ this is called "opp_pre_roll" (line 247) — confusing with outer var
    inner_opp_pre_roll = bgbot_cpp.flip_board(post_move)

    # Check game over — skip terminal positions (won't happen from opening)
    # No terminal check needed here — nobody bears off from the opening

    # The opponent (code's "opp" = "us" in outer frame) can double because
    # opp_owner = CENTERED

    # Evaluate ND: cubeful_equity_recursive(inner_opp_pre_roll, CENTERED, strat, 0)
    # At plies=0: flip → evaluate_probs(flipped, race) → invert → Janowski
    opp_eq_nd = bgbot_cpp.cubeful_equity_nply(
        list(inner_opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 0)

    player_eq_nd = -opp_eq_nd

    # Evaluate DT: cubeful_equity_recursive(inner_opp_pre_roll, OPPONENT, strat, 0)
    opp_eq_dt = bgbot_cpp.cubeful_equity_nply(
        list(inner_opp_pre_roll), bgbot_cpp.CubeOwner.OPPONENT, strat, 0)

    player_eq_dt = -opp_eq_dt * 2.0
    player_eq_dp = -1.0

    # Opponent's double decision
    opp_dt_scaled = opp_eq_dt * 2.0
    opp_dp = 1.0
    opp_best_if_double = min(opp_dt_scaled, opp_dp)
    opp_should_double = (opp_best_if_double > opp_eq_nd)

    if opp_should_double:
        n_double += 1
        if player_eq_dt >= player_eq_dp:
            player_eq = player_eq_dt  # take
            dec = "D/T"
        else:
            player_eq = player_eq_dp  # pass
            dec = "D/P"
    else:
        player_eq = player_eq_nd
        dec = "ND "

    sum_equity += weight * player_eq

    print(f"{d1}-{d2} (w={weight}): opp_ND={opp_eq_nd:+.5f} opp_DT={opp_eq_dt:+.5f} "
          f"DT_2x={opp_dt_scaled:+.5f} {dec} peq={player_eq:+.5f}")

print(f"\nTotal equity / 36 = {sum_equity / 36.0:+.6f}")
print(f"C++ cubeful_equity_nply 1-ply = -0.054029")
print(f"Doubles: {n_double}/{len(ROLLS)}")
