#!/usr/bin/env python3
"""Compare leaf values: mine vs C++ 0-ply, for ALL rolls.

Since individual 0-ply leaves match when using wrong GP,
the difference must come from DIFFERENT MOVE SELECTION.

Hypothesis: the C++ code at line 233 uses evaluate_probs(candidates[i], board)
which is the (Board, Board) overload. But maybe there's a nuance I'm missing.

Let me check the C++ code more carefully. The function signature is:
  strategy.evaluate_probs(candidates[i], board)
where strategy is `const Strategy&`.

But wait — GamePlanStrategy has evaluate_probs(Board, bool) AND evaluate_probs(Board, Board).
With strategy being `const Strategy&` and two Board arguments, it should call
evaluate_probs(const Board&, const Board&).

Unless... the compiler resolves the overload at the base class level and there's
an issue with virtual dispatch? No, both overloads are virtual in Strategy and
overridden in GamePlanStrategy. The resolution depends on the STATIC type of
the arguments.

Actually, let me check: what if `board` in the for loop shadows something?
Or what if the wrong overload is being called?

Let me just add a per-roll comparison with the exact same leaf evaluation.
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

# For PLAYER ownership: compute what C++ 1-ply should give.
# Use cubeful_equity_nply at 0-ply for leaves (which we verified matches C++).

sum_eq = 0.0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)

    # Python best move selection
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    pm = list(candidates[best_idx])
    ip = list(bgbot_cpp.flip_board(pm))

    # Leaf evaluation (verified to match C++)
    leaf = bgbot_cpp.cubeful_equity_nply(ip, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    peq = -leaf

    # Also compute what the C++ WOULD get with best_move_index
    # Let me try using best_move_index directly instead of my manual loop
    bmi = strat.evaluate_board(opp_pre_roll, opp_pre_roll)  # dummy, just to check API

    sum_eq += weight * peq

print(f"Manual PLAYER 1-ply: {sum_eq / 36.0:+.6f}")
print(f"C++ PLAYER 1-ply:    {bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1):+.6f}")

# Let me try: use the C++ best_move_index instead of my Python loop
print("\n=== Using best_move_index ===")
sum_eq2 = 0.0

for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)

    # Use C++ best_move_index
    bmi = strat.best_move_index(candidates, opp_pre_roll)

    pm = list(candidates[bmi])
    ip = list(bgbot_cpp.flip_board(pm))
    leaf = bgbot_cpp.cubeful_equity_nply(ip, bgbot_cpp.CubeOwner.OPPONENT, strat, 0)
    peq = -leaf
    sum_eq2 += weight * peq

print(f"Using best_move_index: {sum_eq2 / 36.0:+.6f}")
print(f"C++ PLAYER 1-ply:     {bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.PLAYER, strat, 1):+.6f}")

# Check if best_move_index picks different moves than evaluate_board loop
print("\n=== Comparing move selection ===")
n_diff = 0
for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(opp_pre_roll, d1, d2)

    # Python loop
    best_eq = -999.0
    best_idx_py = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), opp_pre_roll)
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx_py = i

    # C++ best_move_index
    best_idx_cpp = strat.best_move_index(candidates, opp_pre_roll)

    if best_idx_py != best_idx_cpp:
        n_diff += 1
        print(f"  {d1}-{d2}: Python={best_idx_py} C++={best_idx_cpp}")
        # Show equities of both
        r_py = strat.evaluate_board(list(candidates[best_idx_py]), opp_pre_roll)
        r_cpp = strat.evaluate_board(list(candidates[best_idx_cpp]), opp_pre_roll)
        print(f"    Python best: eq={r_py['equity']:+.6f}")
        print(f"    C++ best:    eq={r_cpp['equity']:+.6f}")

print(f"\nDifferent move selections: {n_diff}/21")
