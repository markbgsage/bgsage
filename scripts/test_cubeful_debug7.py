#!/usr/bin/env python3
"""Debug the specific mismatch for the opp_pre_roll position (after 24/14).

Since the starting position matches perfectly but opp_pre_roll doesn't,
the issue must be in how we trace the opp_pre_roll case.
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
opp_pre_roll = bgbot_cpp.flip_board(POST_MOVE)

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

print(f"opp_pre_roll: {list(opp_pre_roll)}")
eq_1_cpp = bgbot_cpp.cubeful_equity_nply(list(opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"C++ 1-ply: {eq_1_cpp:+.6f}")

# The key difference from the starting position test (which matched) might be
# in how possible_moves works. Let me check if possible_boards and possible_moves
# return the same set of candidates.

# Actually wait — in the C++ code, the function uses `possible_boards(board, d1, d2, candidates)`
# which appends to a vector. Let me check if Python's possible_moves matches.

# Also, the C++ code at line 220 does: possible_boards(board, roll.d1, roll.d2, candidates)
# and candidates is re-used (reserve 32, then appended to). Does possible_boards CLEAR first?

print("\n--- Checking possible_boards behavior ---")
# In the C++ code, candidates is declared outside the loop and NEVER CLEARED!
# possible_boards APPENDS to the vector.
# This means candidates grows with each roll iteration!

# Wait, let me re-read the code more carefully...

# Line 212-213:
# std::vector<Board> candidates;
# candidates.reserve(32);
#
# Line 220:
# possible_boards(board, roll.d1, roll.d2, candidates);

# Does possible_boards clear the vector first? Let me check...

# If it doesn't clear, then each iteration appends MORE candidates,
# and best_idx would be wrong because it indexes into the growing vector!
# This would be a serious bug!

print("Let me check if possible_boards clears the output vector...")
