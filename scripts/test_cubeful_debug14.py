#!/usr/bin/env python3
"""Direct comparison: compute leaf sums for each roll and compare.
If individual rolls differ, we know move selection is different.
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

# Idea: to isolate the move selection, let me check what happens if I evaluate
# ALL candidates for each roll and compare the max equity.
# The C++ code picks the candidate with the highest compute_equity(evaluate_probs(cand, board)).
# My Python picks the candidate with the highest evaluate_board(cand, board)["equity"].
# These should be the same.

# But let me verify: could the C++ code's evaluate_probs() be different from
# what evaluate_board() calls? Let me check by computing evaluate_board for
# every candidate and seeing if the max matches.

# Actually, let me try a completely different approach. What if the problem is
# that the cubeful code's move selection at the 1-ply level accidentally uses
# the (Board, bool) overload instead of (Board, Board)?
#
# In C++: evaluate_probs(candidates[i], board)
# candidates[i] is Board, board is Board.
# But what if the compiler sees `board` as something else? Let me check the
# function signature again...

# No, the Board type is std::array<int,26>. Both parameters are Board.
# The (Board, Board) overload should be selected.

# Let me try yet another approach: add a temporary C++ function that does
# exactly what cubeful_equity_recursive does at 1-ply for ONE ROLL and
# returns the best move index and leaf value.

# Actually, let me try the SIMPLEST test: make a dummy position where there's
# only ONE legal move for a given roll. Then move selection can't differ.

# Or better yet: let me just compute the 1-ply value using a different method.
# Create a MultiPlyStrategy at 1-ply and use it to evaluate opp_pre_roll.
# The MultiPlyStrategy also evaluates at N-ply but uses cubeless evaluation.
# Compare the cubeless 1-ply equity to see if THAT matches my manual trace.

multipy = bgbot_cpp.create_multipy_5nn(PR_W, RC_W, AT_W, PM_W, AN_W,
                                        200, 400, 400, 400, 400, n_plies=1)

# evaluate_board at 1-ply cubeless (MultiPlyStrategy)
# We need to evaluate the flipped board (post-move of previous player)
# Actually, for the CubefulAnalyzer's use case:
# It calls multipy.evaluate_board(target_board, STARTING) to get cubeless probs.

# For opp_pre_roll, the cubeless 1-ply equity at the pre-roll:
# This is the opponent's pre-roll equity from their perspective.
# To get it: flip → evaluate_probs_nply(flipped, opp_pre_roll) → invert → equity

# Actually, the multipy.evaluate_board(flipped, opp_pre_roll) should give
# the post-move probs from the mover's perspective. Let me check.

flipped = list(bgbot_cpp.flip_board(opp_pre_roll))
r_multipy = multipy.evaluate_board(flipped, opp_pre_roll)
print(f"MultiPly 1-ply evaluate_board(flipped, opp_pre_roll):")
print(f"  equity: {r_multipy['equity']:+.6f}")
print(f"  probs: {[f'{p:.4f}' for p in r_multipy['probs']]}")

# This is the post-move evaluation at 1-ply from the mover's perspective.
# The mover just moved to flipped. From the mover's perspective, this is their
# post-move equity. Inverting gives opp's pre-roll equity.
pp = list(r_multipy['probs'])
pre_probs = [1.0 - pp[0], pp[3], pp[4], pp[1], pp[2]]
cubeless_preroll = bgbot_cpp.cubeless_equity(pre_probs)
print(f"  Pre-roll cubeless equity (inverted): {cubeless_preroll:+.6f}")

# Compare with my manual cubeless 1-ply
sum_cl = 0.0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    pm = list(candidates[best_idx])

    # Leaf: 0-ply cubeless equity from mover's (roller's) perspective
    leaf_eq = strat.evaluate_board(pm, opp_pre_roll)["equity"]
    # This is the roller's post-move equity
    # The roller's 1-ply equity = weighted avg of opponent's best response
    # But at 1-ply, there IS no opponent response — we just evaluate the leaf
    # Wait, no. At 1-ply:
    # - Roller rolls → picks best move → leaf
    # - At the leaf: evaluate post-move position = roller's post-move equity
    # - This is NOT the same as the 1-ply pre-roll equity
    #
    # Actually, the 1-ply pre-roll equity as computed by MultiPlyStrategy is:
    # For each roll: generate candidates, evaluate each at 0-ply
    # → pick best → that candidate's equity = roller's post-move equity
    # → but then for the REAL 1-ply, you need to go one more level:
    #   flip to opponent, generate OPPONENT's candidates, evaluate at 0-ply
    #   → opponent picks best → invert → this gives the 1-ply equity
    #
    # The multipy evaluate_board does the full 1-ply evaluation internally.
    # My manual loop above just does 0-ply on the roller's moves.

    # Never mind the multipy comparison. Let me focus on the core issue.
    sum_cl += weight * leaf_eq

print(f"\nManual 0-ply best-move equity (weighted avg): {sum_cl / 36.0:+.6f}")

# Let me just directly compare: what position does C++ cubeful see after
# each roll? If I can compare the post-move board the C++ picks vs what
# Python picks, I'll know if move selection differs.

# Since I can't add debug output to C++, let me try to make the two
# computations converge by using the C++ 0-ply leaf evaluation.

# For PLAYER ownership, 1-ply:
# C++ cubeful_equity_recursive(opp_pre_roll, PLAYER, strat, 1):
#   for each roll:
#     possible_boards(opp_pre_roll, d1, d2, candidates)
#     pick best: evaluate_probs(cand, opp_pre_roll) → compute_equity
#     opp_pre = flip(best)
#     opp_owner = flip(PLAYER) = OPPONENT → no cube decisions
#     opp_eq_nd = cubeful_equity_recursive(opp_pre, OPPONENT, strat, 0)
#     player_eq = -opp_eq_nd
#   sum / 36

# My code does exactly this but calls cubeful_equity_nply(opp_pre, OPPONENT, strat, 0).
# cubeful_equity_nply just calls cubeful_equity_recursive.
# And I verified the leaf values match.
# So the ONLY way the total can differ is if different best moves are picked.

# Let me check by brute force: for EACH candidate of EACH roll, compute the
# leaf value, and check if my Python "best" gives the same leaf as would C++.

print("\n\n=== Brute force: check all candidates for roll 6-6 ===")
d1, d2 = 6, 6
candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
print(f"Roll {d1}-{d2}: {len(candidates)} candidates")

best_leaf_eq = -999.0
best_leaf_idx = -1

for i, c in enumerate(candidates):
    bl = list(c)
    # Move selection equity (what C++ uses to pick best)
    r = strat.evaluate_board(bl, opp_pre_roll)
    move_eq = r["equity"]

    # Leaf equity (what the 1-ply result uses)
    ip = list(bgbot_cpp.flip_board(bl))
    leaf = bgbot_cpp.cubeful_equity_nply(ip, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    player_eq = -leaf

    if move_eq == max([strat.evaluate_board(list(c2), opp_pre_roll)["equity"] for c2 in candidates]):
        marker = " <-- BEST (move selection)"
    else:
        marker = ""

    if i < 5 or abs(move_eq - best_eq) < 0.01 or marker:
        print(f"  {i}: move_eq={move_eq:+.6f}  leaf_player_eq={player_eq:+.6f}{marker}")

    if move_eq > best_leaf_eq:
        best_leaf_eq = move_eq
        best_leaf_idx = i

pm = list(candidates[best_leaf_idx])
ip = list(bgbot_cpp.flip_board(pm))
final_leaf = bgbot_cpp.cubeful_equity_nply(ip, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
print(f"\nBest move idx={best_leaf_idx}, move_eq={best_leaf_eq:+.6f}, leaf player_eq={-final_leaf:+.6f}")

multipy.clear_cache()
