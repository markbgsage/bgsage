#!/usr/bin/env python3
"""Pin down the mismatch for PLAYER ownership (no cube decisions).

The 1-ply PLAYER result should be:
  sum over 21 rolls of: weight * (-leaf_0ply_OPPONENT(flip(best_move)))
  / 36

The leaf_0ply_OPPONENT is cubeful_equity_nply(inner_opp_pre, OPPONENT, strat, 0)
which does: flip → evaluate_probs(flipped, race) → invert → Janowski(OPPONENT)

Let me trace each step for one roll and compare to what C++ does.
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

# Focus on roll 1-1 with PLAYER ownership
print("=== Roll 1-1 with PLAYER ownership ===")
d1, d2 = 1, 1
candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
print(f"Number of candidates: {len(candidates)}")

# Find best move
best_eq = -999.0
best_idx = 0
for i, c in enumerate(candidates):
    bl = list(c)
    r = strat.evaluate_board(bl, opp_pre_roll)
    if r["equity"] > best_eq:
        best_eq = r["equity"]
        best_idx = i
    if i < 5:
        print(f"  candidate {i}: eq={r['equity']:+.6f}")

print(f"Best: idx={best_idx}, eq={best_eq:+.6f}")
post_move = list(candidates[best_idx])
inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

# 0-ply leaf for OPPONENT (since outer owner=PLAYER → opp_owner=OPPONENT)
leaf = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
print(f"0-ply leaf OPPONENT: {leaf:+.6f}")
print(f"player_eq = -leaf = {-leaf:+.6f}")

# Now let me check what the C++ code ACTUALLY uses for move selection.
# In cubeful_equity_recursive, line 233:
#   strategy.evaluate_probs(candidates[i], board)
# This is evaluate_probs(Board, Board) overload.
# For GamePlanStrategy, this classifies from board (= opp_pre_roll).

# But wait — there might be a subtle issue: the C++ code uses
# NeuralNetwork::compute_equity(p) on the probs to get equity for comparison.
# Let me check what compute_equity does.

# The evaluate method uses evaluate_probs then compute_equity.
# But does the Python evaluate_board use the same thing?

# Let me check with one candidate:
bl = list(candidates[0])
r = strat.evaluate_board(bl, opp_pre_roll)
print(f"\nCandidate 0 via evaluate_board: eq={r['equity']:+.6f}")
print(f"  probs={[f'{p:.4f}' for p in r['probs']]}")
# Compute equity manually from probs:
p = list(r['probs'])
manual_eq = 2*p[0] - 1 + p[1] - p[3] + p[2] - p[4]
print(f"  manual cubeless eq from probs = {manual_eq:+.6f}")

# Actually, I realize the issue might be simpler.
# Let me check if `possible_moves` returns the SAME boards as `possible_boards`.
# The Python `possible_moves` might be a different function!

# Let me check a specific candidate:
print(f"\n--- Checking all candidate boards for 1-1 ---")
for i, c in enumerate(candidates):
    bl = list(c)
    diffs = [(j, opp_pre_roll[j], bl[j]) for j in range(26) if bl[j] != opp_pre_roll[j]]
    print(f"  {i}: {diffs}")

# Now let me check: what if Python and C++ have different evaluation of
# the same board? Let me compare evaluate_board with manual probs computation.
print("\n--- evaluate_board consistency check ---")
for i in range(min(3, len(candidates))):
    bl = list(candidates[i])
    r = strat.evaluate_board(bl, opp_pre_roll)
    p = list(r['probs'])
    eq_from_probs = 2*p[0] - 1 + p[1] - p[3] + p[2] - p[4]
    print(f"  {i}: evaluate_board eq={r['equity']:+.6f}  eq_from_probs={eq_from_probs:+.6f}  "
          f"diff={r['equity'] - eq_from_probs:+.8f}")

# Let me also try the full 1-ply PLAYER trace with detailed per-roll output
print("\n\n=== Full 1-ply PLAYER trace ===")
ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

sum_python = 0.0
sum_cpp_leaves = 0.0

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

    # My Python leaf (using cubeful_equity_nply at 0-ply, same as C++ does internally)
    opp_leaf = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    player_eq = -opp_leaf

    # The C++ leaf should be the same, but let me also check what the C++ code
    # would compute at the 0-ply leaf.
    # In cubeful_equity_recursive at 0-ply:
    #   flipped = flip(board)  where board = inner_opp_pre
    #   race = is_race(board)
    #   post_probs = strategy.evaluate_probs(flipped, race)
    #   pre_roll_probs = invert_probs(post_probs)
    #   x = cube_efficiency(board, race)
    #   return cl2cf_money(pre_roll_probs, owner, x)

    sum_python += weight * player_eq

print(f"\nManual 1-ply PLAYER: {sum_python / 36.0:+.6f}")
eq_1_player_cpp = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1)
print(f"C++ 1-ply PLAYER:   {eq_1_player_cpp:+.6f}")
print(f"Difference: {sum_python / 36.0 - eq_1_player_cpp:+.6f}")

# The Python trace uses the exact same C++ function (cubeful_equity_nply at 0-ply)
# for leaf evaluation. So if they differ, it must be because C++ selects different
# best moves at the 1-ply level.
#
# But how? Both use evaluate_board → evaluate_probs(Board, Board) → classify from opp_pre_roll.
#
# Unless... the C++ cubeful code doesn't use evaluate_probs(Board, Board)!
# Let me re-read cube.cpp line 233:
#   auto p = strategy.evaluate_probs(candidates[i], board);
#
# `candidates[i]` is a Board, `board` is a Board. This calls evaluate_probs(Board, Board).
# But `strategy` is a `const Strategy&` — a reference to the base class.
# Does virtual dispatch work correctly here? It should, since GamePlanStrategy
# overrides evaluate_probs(Board, Board).
#
# Actually wait — `strategy.evaluate_probs(candidates[i], board)` in the C++ code
# calls with (Board, Board). But what does line 234 do?
#   eq = NeuralNetwork::compute_equity(p);
# compute_equity is the same formula as cubeless_equity.
# And my Python uses evaluate_board which returns {"equity": ..., "probs": ...}
# where equity = compute_equity(probs).
# So they should match.

# Let me try something: what if the candidate lists differ?
print("\n\n=== Comparing candidate lists between Python and C++ ===")
# Actually both use the same C++ function (possible_moves/possible_boards).
# Let me verify by checking candidate count per roll.
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
    print(f"  {d1}-{d2}: {len(candidates)} candidates")
