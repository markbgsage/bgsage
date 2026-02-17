#!/usr/bin/env python3
"""Check if the best-move selection matters: what if we use ALL candidates
(not just the best) and weight them? No, that doesn't make sense.

Instead, let me compare the leaf values for EVERY roll using both methods:
1. cubeful_equity_nply(inner_opp_pre, OPPONENT, strat, 0) — what C++ 1-ply uses
2. My manual computation

If they match for all 21 rolls, the issue must be in move selection.
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

# Test PLAYER ownership (no cube decisions)
# C++ 1-ply PLAYER: +0.126452
# My Python: +0.164102

# The C++ 1-ply cubeful with PLAYER ownership:
# 1. For each roll, generates candidates
# 2. Evaluates each candidate with evaluate_probs(cand, board=opp_pre_roll) — (Board, Board) overload
# 3. Picks best by compute_equity(probs) = cubeless equity
# 4. For the best move, opponent's pre-roll = flip(best_move)
# 5. Evaluates leaf: cubeful_equity_recursive(opp_pre_roll, opp_owner=OPPONENT, strat, 0)
# 6. opp_owner = flip_owner(PLAYER) = OPPONENT → no cube decisions
# 7. player_eq = -opp_eq

# Let me verify step by step for roll 1-1:
print("=== Roll 1-1 detailed trace ===")
d1, d2 = 1, 1
candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
print(f"Candidates: {len(candidates)}")

# Step 1-3: Find best move
# In C++, this iterates all candidates, evaluates with evaluate_probs(cand, board).
# For GamePlanStrategy, evaluate_probs(cand, board) classifies game plan from board=opp_pre_roll.
# Let's see what game plan that is:
gp_opp_pre = bgbot_cpp.classify_game_plan(opp_pre_roll)
print(f"Game plan of opp_pre_roll: {gp_opp_pre}")

best_eq = -999.0
best_idx = 0
all_eqs = []
for i, c in enumerate(candidates):
    bl = list(c)
    r = strat.evaluate_board(bl, opp_pre_roll)
    all_eqs.append(r["equity"])
    if r["equity"] > best_eq:
        best_eq = r["equity"]
        best_idx = i

print(f"Best move: idx={best_idx}, eq={best_eq:+.6f}")
print(f"2nd best: {sorted(all_eqs, reverse=True)[1]:+.6f}" if len(all_eqs) > 1 else "")

# Step 4-5: Get leaf value
post_move = list(candidates[best_idx])
inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

leaf = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
player_eq = -leaf
print(f"inner_opp_pre: {inner_opp_pre}")
print(f"leaf 0-ply OPPONENT: {leaf:+.6f}")
print(f"player_eq: {player_eq:+.6f}")

# Now let me check: what does the C++ code ACTUALLY get for this leaf?
# The C++ code evaluates the leaf at 0-ply with:
#   flipped = flip(inner_opp_pre)
#   race = is_race(inner_opp_pre)
#   post_probs = strategy.evaluate_probs(flipped, race)  [← (Board, bool) overload!]
#   pre_roll_probs = invert_probs(post_probs)
#   x = cube_efficiency(inner_opp_pre, race)
#   return cl2cf_money(pre_roll_probs, OPPONENT, x)

inner_flipped = list(bgbot_cpp.flip_board(inner_opp_pre))
race = bgbot_cpp.is_race(inner_opp_pre)
x = bgbot_cpp.cube_efficiency(inner_opp_pre, race)

# Method A: (Board, bool) — what C++ 0-ply leaf does
# evaluate_probs(flipped, race) → classifies from flipped
gp_wrong = bgbot_cpp.classify_game_plan(inner_flipped)
# We can't call evaluate_probs(Board, bool) directly from Python.
# But cubeful_equity_nply at 0-ply DOES call it. So `leaf` above already uses it.

# Method B: (Board, Board) — correct game plan
r_correct = strat.evaluate_board(inner_flipped, inner_opp_pre)
probs_correct = list(r_correct['probs'])
pre_probs_correct = [1.0 - probs_correct[0], probs_correct[3], probs_correct[4],
                      probs_correct[1], probs_correct[2]]
leaf_correct = bgbot_cpp.cl2cf_money(pre_probs_correct, bgbot_cpp.CubeOwner.OPPONENT, x)

gp_correct = bgbot_cpp.classify_game_plan(inner_opp_pre)
print(f"\nGame plan (C++ uses, from flipped): {gp_wrong}")
print(f"Game plan (correct, from inner_opp_pre): {gp_correct}")
print(f"leaf via cubeful_equity_nply(0-ply): {leaf:+.6f}")
print(f"leaf via correct game plan:          {leaf_correct:+.6f}")
print(f"Difference: {leaf - leaf_correct:+.6f}")

# I wonder: could the issue be that on macOS, the build uses a different
# evaluate_probs overload resolution than expected?
# Let me verify by testing cubeful_equity_nply(STARTING, CENTERED, 0) manually:
STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]
eq_start = bgbot_cpp.cubeful_equity_nply(STARTING, bgbot_cpp.CubeOwner.CENTERED, strat, 0)
print(f"\n=== Starting position 0-ply leaf ===")
print(f"cubeful_equity_nply(STARTING, CENTERED, 0): {eq_start:+.6f}")

# Manual with correct GP:
start_flipped = list(bgbot_cpp.flip_board(STARTING))
r_start = strat.evaluate_board(start_flipped, STARTING)
p_start = list(r_start['probs'])
pre_start = [1.0 - p_start[0], p_start[3], p_start[4], p_start[1], p_start[2]]
race_start = bgbot_cpp.is_race(STARTING)
x_start = bgbot_cpp.cube_efficiency(STARTING, race_start)
jan_start = bgbot_cpp.cl2cf_money(pre_start, bgbot_cpp.CubeOwner.CENTERED, x_start)
print(f"Manual Janowski: {jan_start:+.6f}")
print(f"Match: {abs(eq_start - jan_start) < 0.0001}")
# For starting position, both overloads should give same result because it's symmetric.

# Let me now test a DIFFERENT non-symmetric position to see if the 0-ply
# leaf gives different values between Python and C++
print("\n\n=== Testing 0-ply leaf discrepancy ===")
# Use the inner_opp_pre from roll 1-1
print(f"Test position: {inner_opp_pre}")
leaf_cpp = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
print(f"C++ 0-ply OPPONENT: {leaf_cpp:+.6f}")

# Replicate manually:
test_flipped = list(bgbot_cpp.flip_board(inner_opp_pre))
test_race = bgbot_cpp.is_race(inner_opp_pre)
test_x = bgbot_cpp.cube_efficiency(inner_opp_pre, test_race)

# The C++ calls evaluate_probs(test_flipped, test_race) — (Board, bool) overload
# This classifies from test_flipped, not inner_opp_pre
gp_from_flipped = bgbot_cpp.classify_game_plan(test_flipped)
gp_from_preroll = bgbot_cpp.classify_game_plan(inner_opp_pre)
print(f"GP from flipped (what C++ uses): {gp_from_flipped}")
print(f"GP from preroll (correct):       {gp_from_preroll}")

# I can't call evaluate_probs(Board, bool) from Python directly.
# But I CAN test by using evaluate_board with different pre_move_board:
# evaluate_board(test_flipped, test_flipped) should use the same GP as
# evaluate_probs(test_flipped, race) because evaluate_probs(board, board)
# classifies from board. Wait, no — evaluate_probs(Board, Board) classifies
# from the SECOND argument (pre_move_board), and evaluate_probs(Board, bool)
# classifies from the FIRST argument (the board itself).

# So to replicate evaluate_probs(test_flipped, race), I need:
# evaluate_board(test_flipped, ???) where ??? classifies to the same GP as test_flipped
# That would be: evaluate_board(test_flipped, test_flipped)
r_wrong_gp = strat.evaluate_board(test_flipped, test_flipped)
p_wrong = list(r_wrong_gp['probs'])
pre_wrong = [1.0 - p_wrong[0], p_wrong[3], p_wrong[4], p_wrong[1], p_wrong[2]]
leaf_wrong_gp = bgbot_cpp.cl2cf_money(pre_wrong, bgbot_cpp.CubeOwner.OPPONENT, test_x)

r_right_gp = strat.evaluate_board(test_flipped, inner_opp_pre)
p_right = list(r_right_gp['probs'])
pre_right = [1.0 - p_right[0], p_right[3], p_right[4], p_right[1], p_right[2]]
leaf_right_gp = bgbot_cpp.cl2cf_money(pre_right, bgbot_cpp.CubeOwner.OPPONENT, test_x)

print(f"\nLeaf with wrong GP (flipped→flipped):   {leaf_wrong_gp:+.6f}")
print(f"Leaf with correct GP (flipped→preroll):  {leaf_right_gp:+.6f}")
print(f"C++ cubeful_equity_nply 0-ply:           {leaf_cpp:+.6f}")

# But wait — evaluate_probs(Board, bool) doesn't classify from the Board
# in the same way as evaluate_probs(Board, Board) with pre_move=board.
# evaluate_probs(Board, bool) ignores the bool and classifies from board.
# evaluate_probs(Board, Board) classifies from the SECOND arg (pre_move_board).
# When I call evaluate_board(flipped, flipped), it classifies from flipped (second arg).
# When C++ calls evaluate_probs(flipped, race), it classifies from flipped (the board itself).
# Both should classify from flipped → SAME RESULT.
# So leaf_wrong_gp SHOULD match leaf_cpp.

print(f"\nDoes wrong GP match C++? {abs(leaf_wrong_gp - leaf_cpp) < 0.0001}")
if abs(leaf_wrong_gp - leaf_cpp) > 0.0001:
    print(f"MISMATCH! Diff = {leaf_wrong_gp - leaf_cpp:+.6f}")
    print("The issue is NOT just game plan classification!")
