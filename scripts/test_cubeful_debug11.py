#!/usr/bin/env python3
"""Test whether the game plan misclassification in the 0-ply leaf
accounts for the full difference.

For PLAYER ownership (no cube decisions):
  C++ 1-ply = +0.126
  Python 1-ply (using correct game plan) = +0.164
  Diff = 0.038

If I use the WRONG game plan (classify from flipped board instead of
pre-move board), I should get the C++ value.
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

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

# For PLAYER ownership: opp_owner = OPPONENT → no cube decisions
# 1-ply = weighted avg of: -cubeful_0ply(inner_opp_pre, OPPONENT)

sum_python_correct = 0.0  # using evaluate_board (correct game plan)
sum_python_wrong = 0.0    # using cubeful_equity_nply at 0-ply (wrong game plan)

for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)

    # Find best move — use evaluate_board (correct game plan from opp_pre_roll)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    post_move = list(candidates[best_idx])
    inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

    # Method 1: C++ 0-ply leaf (uses evaluate_probs(flipped, race) — wrong game plan)
    opp_leaf_wrong = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    sum_python_wrong += weight * (-opp_leaf_wrong)

    # Method 2: Compute leaf manually with correct game plan
    inner_flipped = list(bgbot_cpp.flip_board(inner_opp_pre))
    race = bgbot_cpp.is_race(inner_opp_pre)
    x = bgbot_cpp.cube_efficiency(inner_opp_pre, race)

    # Correct: classify from inner_opp_pre (pre-roll board)
    pp = strat.evaluate_board(inner_flipped, inner_opp_pre)
    probs = list(pp['probs'])
    pre_probs = [1.0 - probs[0], probs[3], probs[4], probs[1], probs[2]]
    opp_leaf_correct = bgbot_cpp.cl2cf_money(pre_probs, bgbot_cpp.CubeOwner.OPPONENT, x)
    sum_python_correct += weight * (-opp_leaf_correct)

eq_1_player_cpp = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1)

print("1-ply PLAYER ownership:")
print(f"  C++ cubeful_equity_nply: {eq_1_player_cpp:+.6f}")
print(f"  Manual (wrong GP, matching C++): {sum_python_wrong / 36.0:+.6f}")
print(f"  Manual (correct GP):             {sum_python_correct / 36.0:+.6f}")
print()

# If manual_wrong matches C++, then the issue IS the game plan classification.
# But we also need to check if the C++ selects the SAME best moves as Python.
# If the move selection also has game plan issues, they could differ.

# Actually the C++ code's move selection at line 233:
#   strategy.evaluate_probs(candidates[i], board)
# This IS the (Board, Board) overload. So move selection uses the correct
# game plan (classifying from board = opp_pre_roll). Only the 0-ply LEAF
# uses the wrong overload.

# So the manual_wrong result should match the C++ 1-ply result if:
# 1. Move selection is the same (both use correct game plan)
# 2. Leaf evaluation is the same (both use wrong game plan via (Board, bool))

# Wait, but my "manual wrong" uses cubeful_equity_nply(inner_opp_pre, OPPONENT, 0)
# which internally calls evaluate_probs(flipped, race) — the (Board, bool) overload.
# So it should match what C++ does at the leaf.

# But the question is: does C++ select the same best move?
# Both C++ and Python use evaluate_probs(candidates[i], board=opp_pre_roll).
# For Python, this goes through evaluate_board → evaluate_probs(Board, Board).
# For C++, this calls evaluate_probs(Board, Board) directly.
# They should call the same code path.

# Unless there's a bug where the C++ cubeful code uses a different
# evaluate_probs overload for move selection...

# Let me re-read cube.cpp line 233 one more time:
# auto p = strategy.evaluate_probs(candidates[i], board);
# candidates[i] is Board, board is Board.
# This calls Strategy::evaluate_probs(const Board&, const Board&)
# which GamePlanStrategy overrides to classify from pre_move_board.
# So move selection uses the correct game plan. ✓

# And the 0-ply leaf (line 204):
# auto post_probs = strategy.evaluate_probs(flipped, race);
# flipped is Board, race is bool.
# This calls Strategy::evaluate_probs(const Board&, bool)
# which GamePlanStrategy overrides to classify from the board itself (flipped).
# So the leaf uses the wrong game plan. ✗

# If both facts are true, then sum_python_wrong should match C++.
# Let me see...
