#!/usr/bin/env python3
"""Minimal test: verify that cubeful_equity_nply at 1-ply matches manual computation.

The issue: C++ gives -0.054 but manual Python trace gives -0.011.
Let me verify with a simpler approach.
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

# Post-move board after 24/14
POST_MOVE = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 1, 0, 0, -3, 0, -5, 0, 0, 0, 0, 1, 0]
opp_pre_roll = bgbot_cpp.flip_board(POST_MOVE)

print(f"opp_pre_roll: {list(opp_pre_roll)}")

# C++ results
eq_0 = bgbot_cpp.cubeful_equity_nply(list(opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 0)
eq_1 = bgbot_cpp.cubeful_equity_nply(list(opp_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"C++ cubeful_equity_nply 0-ply: {eq_0:.6f}")
print(f"C++ cubeful_equity_nply 1-ply: {eq_1:.6f}")

# Manual 0-ply computation to verify match:
flipped_of_opp = bgbot_cpp.flip_board(list(opp_pre_roll))
print(f"\nflipped_of_opp_pre_roll (= POST_MOVE): {list(flipped_of_opp)}")
race = bgbot_cpp.is_race(list(opp_pre_roll))
x = bgbot_cpp.cube_efficiency(list(opp_pre_roll), race)

# The 0-ply path in cubeful_equity_recursive:
# strategy.evaluate_probs(flipped, race)  -- uses (Board, bool) overload!
# This classifies game plan based on flipped (= POST_MOVE), not opp_pre_roll
gp_post_move = bgbot_cpp.classify_game_plan(POST_MOVE)
gp_opp_pre_roll = bgbot_cpp.classify_game_plan(list(opp_pre_roll))
print(f"Game plan of POST_MOVE (what 0-ply leaf uses): {gp_post_move}")
print(f"Game plan of opp_pre_roll: {gp_opp_pre_roll}")

# The evaluate_probs(flipped, race=False) call at the 0-ply leaf
# classifies_game_plan(flipped) = classify(POST_MOVE) = racing
# Then uses racing NN to evaluate POST_MOVE

# But in my Python trace, I used:
# strat.evaluate_board(our_flipped, our_pre_roll) which calls evaluate_probs(Board, Board)
# and classifies based on our_pre_roll, not our_flipped.

# Let me verify: at the 0-ply leaf of the 1-ply call, the code evaluates each
# leaf position. Let me trace one specific roll to see if the C++ and Python
# select different moves.

ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

# The 1-ply cubeful_equity_recursive(opp_pre_roll, CENTERED, strat, 1):
# For each roll of the "roller" (which is the opponent in our outer frame):
# - generate candidates from opp_pre_roll
# - find best move using strategy.evaluate_probs(candidates[i], board=opp_pre_roll)
#   This uses the (Board, Board) overload: classify game plan from opp_pre_roll
# - Then evaluate the leaf at 0-ply

# The KEY difference between my Python trace and the C++ code:
# In my Python trace (debug2.py), I evaluated moves using:
#   strat.evaluate_board(c, opp_pre_roll)
# which calls evaluate_probs(c, opp_pre_roll) → classifies from opp_pre_roll
#
# In the C++ code, move selection uses:
#   strategy.evaluate_probs(candidates[i], board)   [line 233]
# where board = opp_pre_roll. Same thing. So move selection should match.
#
# But the LEAF evaluation uses:
#   strategy.evaluate_probs(flipped, race)   [line 204]
# This uses the (Board, bool) overload, classifying from flipped.
# In my Python trace, I did:
#   strat.evaluate_board(our_flipped, our_pre_roll)
# which uses (Board, Board) overload, classifying from our_pre_roll.

# Let me test one specific leaf to see the difference.

print("\n--- Testing leaf evaluation differences ---")
# Take roll 1-3 (which gave cubeless ~ -0.006 in my trace)
candidates = bgbot_cpp.possible_moves(list(opp_pre_roll), 1, 3)
print(f"Roll 1-3: {len(candidates)} candidates")

# Find best move the way C++ does it
best_eq_cpp = -999.0
best_idx_cpp = 0
for i, c in enumerate(candidates):
    r = strat.evaluate_board(list(c), list(opp_pre_roll))
    if r["equity"] > best_eq_cpp:
        best_eq_cpp = r["equity"]
        best_idx_cpp = i

print(f"Best move index: {best_idx_cpp}, equity: {best_eq_cpp:.6f}")

opp_post_move = list(candidates[best_idx_cpp])
our_pre_roll = bgbot_cpp.flip_board(opp_post_move)

# Now evaluate the leaf two ways:
# Way 1: C++ code's approach: evaluate_probs(flipped, race)
# We can't call evaluate_probs directly from Python, but we can see the effect
# via cubeful_equity_nply at 0-ply from our_pre_roll
leaf_eq_cpp = bgbot_cpp.cubeful_equity_nply(list(our_pre_roll), bgbot_cpp.CubeOwner.CENTERED, strat, 0)
print(f"Leaf equity (C++ 0-ply cubeful from our_pre_roll): {leaf_eq_cpp:.6f}")

# Way 2: My Python trace approach
our_flipped = bgbot_cpp.flip_board(list(our_pre_roll))
race_leaf = bgbot_cpp.is_race(list(our_pre_roll))
x_leaf = bgbot_cpp.cube_efficiency(list(our_pre_roll), race_leaf)

# Using (Board, Board) overload (my trace):
post_probs_bb = strat.evaluate_board(list(our_flipped), list(our_pre_roll))
print(f"evaluate_board(our_flipped, our_pre_roll): eq={post_probs_bb['equity']:.6f}")
print(f"  probs={[f'{p:.4f}' for p in post_probs_bb['probs']]}")
print(f"  game_plan of our_pre_roll: {bgbot_cpp.classify_game_plan(list(our_pre_roll))}")

# Using (Board, bool) overload (C++ leaf code):
# We can approximate by checking game plan of our_flipped
print(f"  game_plan of our_flipped:  {bgbot_cpp.classify_game_plan(list(our_flipped))}")

# The actual leaf evaluation in C++ cubeful_equity_recursive:
# strategy.evaluate_probs(flipped, race)
# where flipped = flip(board) = flip(our_pre_roll) = our_flipped = opp_post_move
# race = is_race(board) = is_race(our_pre_roll) = race_leaf
#
# evaluate_probs(flipped, race=False) → classifies game plan from flipped
# classify_game_plan(flipped) = classify(our_flipped)
# Then evaluates flipped with that NN → inverts → Janowski

# The difference: (Board, bool) classifies from the board being evaluated (post-move),
# while (Board, Board) classifies from the pre-move board.
# For the 5-NN architecture, the pre-move board determines which NN to use.
# Using the post-move board for classification could pick a different NN!

# Let's verify by manually computing what the C++ would get at this leaf:
# classify_game_plan(our_flipped) gives the "wrong" game plan
# classify_game_plan(our_pre_roll) gives the "right" game plan

gp_wrong = bgbot_cpp.classify_game_plan(list(our_flipped))
gp_right = bgbot_cpp.classify_game_plan(list(our_pre_roll))

if gp_wrong != gp_right:
    print(f"\n  *** GAME PLAN MISMATCH: C++ leaf uses {gp_wrong}, should use {gp_right} ***")
else:
    print(f"\n  Game plans match: {gp_right}")

# Now let me check ALL rolls to see how often game plans mismatch
print("\n\n--- Game plan mismatches across all rolls ---")
n_mismatch = 0
sum_eq_cpp_leaves = 0.0
sum_eq_python_leaves = 0.0

for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(list(opp_pre_roll), d1, d2)

    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), list(opp_pre_roll))
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    opp_post = list(candidates[best_idx])
    our_pre = bgbot_cpp.flip_board(opp_post)
    our_flip = bgbot_cpp.flip_board(list(our_pre))

    gp_leaf_cpp = bgbot_cpp.classify_game_plan(list(our_flip))  # what C++ uses
    gp_leaf_correct = bgbot_cpp.classify_game_plan(list(our_pre))  # what should be used

    # Get leaf equity the C++ way (cubeful_equity_nply at 0-ply)
    leaf_cpp = bgbot_cpp.cubeful_equity_nply(list(our_pre), bgbot_cpp.CubeOwner.CENTERED, strat, 0)

    # Get leaf equity my way (using correct game plan via evaluate_board)
    race_l = bgbot_cpp.is_race(list(our_pre))
    x_l = bgbot_cpp.cube_efficiency(list(our_pre), race_l)
    pp = strat.evaluate_board(list(our_flip), list(our_pre))["probs"]
    pre_probs = [1.0 - pp[0], pp[3], pp[4], pp[1], pp[2]]
    leaf_python = bgbot_cpp.cl2cf_money(pre_probs, bgbot_cpp.CubeOwner.CENTERED, x_l)

    sum_eq_cpp_leaves += weight * leaf_cpp
    sum_eq_python_leaves += weight * leaf_python

    mismatch = "MISMATCH" if gp_leaf_cpp != gp_leaf_correct else ""
    if gp_leaf_cpp != gp_leaf_correct:
        n_mismatch += 1

    if mismatch or abs(leaf_cpp - leaf_python) > 0.001:
        print(f"{d1}-{d2} (w={weight}): C++_gp={gp_leaf_cpp:10s} correct_gp={gp_leaf_correct:10s} "
              f"leaf_cpp={leaf_cpp:+.5f} leaf_py={leaf_python:+.5f} diff={leaf_cpp-leaf_python:+.5f} {mismatch}")

print(f"\nGame plan mismatches: {n_mismatch}/{len(ROLLS)}")
print(f"Avg leaf eq (C++ way):    {sum_eq_cpp_leaves / 36:.6f}")
print(f"Avg leaf eq (Python way): {sum_eq_python_leaves / 36:.6f}")
