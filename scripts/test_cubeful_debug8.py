#!/usr/bin/env python3
"""Focused: test one specific roll from opp_pre_roll to find where the mismatch is."""

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

print(f"opp_pre_roll: {opp_pre_roll}")

# C++ full 1-ply result
eq_1_cpp = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.CENTERED, strat, 1)
print(f"C++ 1-ply CENTERED: {eq_1_cpp:+.6f}")

# Also test with PLAYER and OPPONENT to see the pattern
eq_1_player = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1)
eq_1_opp = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.OPPONENT, strat, 1)
print(f"C++ 1-ply PLAYER:   {eq_1_player:+.6f}")
print(f"C++ 1-ply OPPONENT: {eq_1_opp:+.6f}")

# Now let me trace just 3-4 (one specific roll) step by step
print("\n=== Tracing roll 3-4 from opp_pre_roll ===")
d1, d2 = 3, 4
candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)
print(f"Number of candidates: {len(candidates)}")

# Find best move
best_eq = -999.0
best_idx = 0
equities = []
for i, c in enumerate(candidates):
    bl = list(c)
    r = strat.evaluate_board(bl, opp_pre_roll)
    equities.append(r["equity"])
    if r["equity"] > best_eq:
        best_eq = r["equity"]
        best_idx = i

print(f"Best move index: {best_idx}, equity: {best_eq:.6f}")
print(f"Top 3 equities: {sorted(equities, reverse=True)[:3]}")

post_move = list(candidates[best_idx])
inner_opp_pre = list(bgbot_cpp.flip_board(post_move))

print(f"\npost_move (roller's perspective):   {post_move}")
print(f"inner_opp_pre (opp's pre-roll):     {inner_opp_pre}")

# 0-ply leaf evaluation
leaf_nd = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.CENTERED, strat, 0)
leaf_dt = bgbot_cpp.cubeful_equity_nply(inner_opp_pre, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
print(f"\nLeaf ND (CENTERED): {leaf_nd:+.6f}")
print(f"Leaf DT (OPPONENT): {leaf_dt:+.6f}")
print(f"Leaf DT * 2:        {leaf_dt * 2:+.6f}")

# Now let me check: what if the C++ code evaluates candidates differently?
# The C++ uses strategy.evaluate_probs(candidates[i], board) where board = opp_pre_roll
# strategy is a GamePlanStrategy. evaluate_probs(Board, Board) classifies from pre_move_board.
# But wait: the Python evaluate_board also uses (Board, Board). Let me verify they pick the same move.

# Actually, let me try something: what if the issue is that possible_moves in Python
# returns boards in a different order or set than possible_boards in C++?
# The C++ cube code uses possible_boards() directly. The Python binding might use
# a different function.

# Let me check by printing all candidate boards:
print(f"\n--- All candidates for roll {d1}-{d2} from opp_pre_roll ---")
for i, c in enumerate(candidates):
    bl = list(c)
    diffs = [(j, opp_pre_roll[j], bl[j]) for j in range(26) if bl[j] != opp_pre_roll[j]]
    r = strat.evaluate_board(bl, opp_pre_roll)
    marker = " <-- BEST" if i == best_idx else ""
    print(f"  {i}: eq={r['equity']:+.6f}  changes={diffs}{marker}")

# Let me try a completely different approach: test with various owners
# on the SAME position to see if the cubeful equity values make sense.
print("\n\n=== Owner comparison for opp_pre_roll ===")
for ply in range(3):
    for owner_name, owner in [("CENTERED", bgbot_cpp.CubeOwner.CENTERED),
                               ("PLAYER", bgbot_cpp.CubeOwner.PLAYER),
                               ("OPPONENT", bgbot_cpp.CubeOwner.OPPONENT)]:
        eq = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, owner, strat, ply)
        print(f"  {ply}-ply {owner_name:10s}: {eq:+.6f}")
    print()

# Expected relationship: with centered cube, the roller has the most advantage
# because they can double. PLAYER (roller owns) > CENTERED > OPPONENT.
# At 0-ply this comes from Janowski piecewise linear.
