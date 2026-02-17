#!/usr/bin/env python3
"""Verify that my Python trace matches the C++ cubeful_equity_nply."""

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

# Post-move board after 24/14
POST_MOVE = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 1, 0, 0, -3, 0, -5, 0, 0, 0, 0, 1, 0]
opp_pre_roll = bgbot_cpp.flip_board(POST_MOVE)

print(f"POST_MOVE:    {POST_MOVE}")
print(f"opp_pre_roll: {list(opp_pre_roll)}")

# C++ cubeful_equity_nply at 1-ply
eq_1 = bgbot_cpp.cubeful_equity_nply(list(opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"\nC++ cubeful_equity_nply(opp_pre_roll, CENTERED, 1-ply) = {eq_1:.6f}")

# What about the CubefulAnalyzer path?
# It does: opp_pre_roll = flip(post_move), opp_owner = flip(CENTERED) = CENTERED
# opp_eq = cubeful_equity_nply(opp_pre_roll, CENTERED, strat_0ply, n_plies=1)
# cf_eq = -opp_eq
print(f"cf_eq from CubefulAnalyzer = {-eq_1:.6f}")

# Now let me also test: what is cubeful_equity_nply at 0-ply?
eq_0 = bgbot_cpp.cubeful_equity_nply(list(opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 0)
print(f"\nC++ cubeful_equity_nply(opp_pre_roll, CENTERED, 0-ply) = {eq_0:.6f}")
print(f"cf_eq from 0-ply = {-eq_0:.6f}")

# Compare with direct Janowski on POST_MOVE probs
r = strat.evaluate_board(POST_MOVE, [0]*26)  # pre_move_board doesn't matter for game plan classification from post-move
print(f"\nDirect evaluate_board(POST_MOVE): eq={r['equity']:.6f}, probs={[f'{p:.4f}' for p in r['probs']]}")

# For 0-ply cubeful:
# The function does: flip→evaluate→invert→Janowski
# flip(opp_pre_roll) = POST_MOVE (undo the flip)
# evaluate(POST_MOVE) = post-move probs from mover's perspective
# invert → pre-roll probs from opp's perspective
# Janowski with CENTERED

flipped = bgbot_cpp.flip_board(list(opp_pre_roll))
print(f"\nflip(opp_pre_roll) = {list(flipped)}")
print(f"POST_MOVE          = {POST_MOVE}")
print(f"Same? {list(flipped) == POST_MOVE}")

race = bgbot_cpp.is_race(list(opp_pre_roll))
print(f"is_race(opp_pre_roll) = {race}")

x = bgbot_cpp.cube_efficiency(list(opp_pre_roll), race)
print(f"cube_efficiency = {x}")

# The evaluate in the 0-ply leaf uses the board's own game plan classification
post_probs = list(strat.evaluate_board(list(flipped), list(opp_pre_roll))["probs"])
pre_roll_probs = [1.0 - post_probs[0], post_probs[3], post_probs[4], post_probs[1], post_probs[2]]
print(f"\npost_probs (from mover perspective):     {[f'{p:.4f}' for p in post_probs]}")
print(f"pre_roll_probs (opp perspective, inverted): {[f'{p:.4f}' for p in pre_roll_probs]}")

jan = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.CENTERED, x)
print(f"Janowski cubeful (opp perspective, centered): {jan:.6f}")
print(f"Expected to match eq_0: {eq_0:.6f}")

# Now try the second method: evaluate using POST_MOVE directly
# The strategy.evaluate_probs(board, pre_move_board) uses pre_move_board for game plan
r2 = strat.evaluate_board(POST_MOVE, POST_MOVE)
print(f"\nevaluate_board(POST_MOVE, POST_MOVE): eq={r2['equity']:.6f}")
r3 = strat.evaluate_board(list(flipped), list(flipped))
print(f"evaluate_board(flipped, flipped): eq={r3['equity']:.6f}")

# The key: evaluate_probs(flipped, race) at the 0-ply leaf
# In the C++ code, line 204: strategy.evaluate_probs(flipped, race)
# This uses the `evaluate_probs(Board, bool)` overload, NOT the (Board, Board) overload
# The bool overload classifies based on is_race flag only? Let me check...
print("\n\n--- Checking game plan classification ---")
# When the cubeful recursion calls strategy.evaluate_probs(flipped, race),
# it passes a boolean `race`. Let's see what game plan this gets.
gp_opp = bgbot_cpp.classify_game_plan(list(opp_pre_roll))
gp_post = bgbot_cpp.classify_game_plan(POST_MOVE)
gp_flipped = bgbot_cpp.classify_game_plan(list(flipped))
print(f"Game plan of opp_pre_roll: {gp_opp}")
print(f"Game plan of POST_MOVE:    {gp_post}")
print(f"Game plan of flipped:      {gp_flipped}")

# Now test with the starting position as pre_move_board
STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]
r4 = strat.evaluate_board(POST_MOVE, STARTING)
print(f"\nevaluate_board(POST_MOVE, STARTING): eq={r4['equity']:.6f}")
print(f"  probs = {[f'{p:.4f}' for p in r4['probs']]}")
