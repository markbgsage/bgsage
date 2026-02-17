#!/usr/bin/env python3
"""Debug cubeful equity calculations for the 24/14 move (starting position, 6-4).

Compares:
1. N-ply cubeless probs + simple Janowski conversion
2. Full N-ply cubeful recursion (what the webapp uses)
3. Shows the discrepancy and helps identify if there's a bug
"""

import sys
import os

# Add build dir to path
sys.path.insert(0, "build_macos")
sys.path.insert(0, ".")

import bgbot_cpp

# Model paths (Stage 5)
MODELS = "models"
PR_W = os.path.join(MODELS, "sl_s5_purerace.weights.best")
RC_W = os.path.join(MODELS, "sl_s5_racing.weights.best")
AT_W = os.path.join(MODELS, "sl_s5_attacking.weights.best")
PM_W = os.path.join(MODELS, "sl_s5_priming.weights.best")
AN_W = os.path.join(MODELS, "sl_s5_anchoring.weights.best")

NH_PR = 200
NH_RC = 400
NH_AT = 400
NH_PM = 400
NH_AN = 400

# Starting position
STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

def fmt_probs(probs):
    """Format probabilities for display."""
    p = list(probs)
    return (f"P(win)={p[0]:.4f}  P(gw)={p[1]:.4f}  P(bw)={p[2]:.4f}  "
            f"P(gl)={p[3]:.4f}  P(bl)={p[4]:.4f}")

def janowski_cubeful(probs, owner, x):
    """Pure Python Janowski conversion for verification."""
    p = list(probs)
    p_win = p[0]
    p_gw, p_bw = p[1], p[2]
    p_gl, p_bl = p[3], p[4]

    W = 1.0 + (p_gw + p_bw) / p_win if p_win > 1e-7 else 1.0
    L = 1.0 + (p_gl + p_bl) / (1.0 - p_win) if (1.0 - p_win) > 1e-7 else 1.0

    e_dead = 2.0 * p_win - 1.0 + p_gw - p_gl + p_bw - p_bl

    TP = (L - 0.5) / (W + L + 0.5)
    CP = (L + 1.0) / (W + L + 0.5)

    # Live equity for centered cube
    if p_win < TP:
        e_live = -L + (-1.0 + L) * p_win / TP
    elif p_win < CP:
        e_live = -1.0 + 2.0 * (p_win - TP) / (CP - TP)
    else:
        e_live = 1.0 + (W - 1.0) * (p_win - CP) / (1.0 - CP)

    e_cf = e_dead * (1.0 - x) + e_live * x
    return e_cf, e_dead, e_live, W, L, TP, CP

def main():
    print("=" * 80)
    print("Cubeful Equity Debug — Starting Position, 6-4 roll, 24/14 move")
    print("=" * 80)

    # Create strategies
    strat_0ply = bgbot_cpp.GamePlanStrategy(PR_W, RC_W, AT_W, PM_W, AN_W,
                                             NH_PR, NH_RC, NH_AT, NH_PM, NH_AN)
    strat_1ply = bgbot_cpp.create_multipy_5nn(PR_W, RC_W, AT_W, PM_W, AN_W,
                                               NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
                                               n_plies=1)
    strat_2ply = bgbot_cpp.create_multipy_5nn(PR_W, RC_W, AT_W, PM_W, AN_W,
                                               NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
                                               n_plies=2)

    # Generate candidates for starting position with 6-4
    candidates = bgbot_cpp.possible_moves(STARTING, 6, 4)
    print(f"\nNumber of legal moves: {len(candidates)}")

    # Find the 24/14 move — the checker on point 24 goes to point 14
    # Starting position: point 24 has 2 white checkers.
    # After 24/14: point 24 has 1, point 14 has 1 new checker (was 0)
    target_board = None
    for c in candidates:
        b = list(c)
        # 24/14 means: point 24 goes from 2 to 1, point 14 goes from 0 to 1
        if b[24] == 1 and b[14] == 1:
            # Also verify nothing else changed oddly
            target_board = b
            break

    if target_board is None:
        # Try another encoding - point 24 drops by 1, point 14 gains 1
        for c in candidates:
            b = list(c)
            if b[24] == STARTING[24] - 1 and b[14] == STARTING[14] + 1:
                target_board = b
                break

    if target_board is None:
        print("Could not find 24/14 move! Listing all candidates:")
        for i, c in enumerate(candidates):
            b = list(c)
            diffs = [(j, STARTING[j], b[j]) for j in range(26) if b[j] != STARTING[j]]
            print(f"  Move {i}: {diffs}")
        return

    print(f"\n24/14 board: {target_board}")

    # Check race status
    race = bgbot_cpp.is_race(target_board)
    print(f"Is race: {race}")

    # Get cube efficiency
    x = bgbot_cpp.cube_efficiency(target_board, race)
    print(f"Cube efficiency x: {x}")

    # =======================================================================
    # 0-ply evaluation
    # =======================================================================
    print("\n" + "=" * 80)
    print("0-PLY EVALUATION")
    print("=" * 80)

    r0 = strat_0ply.evaluate_board(target_board, STARTING)
    probs_0 = list(r0["probs"])
    eq_0 = r0["equity"]
    print(f"Post-move probs (0-ply): {fmt_probs(probs_0)}")
    print(f"Cubeless equity (0-ply): {eq_0:.6f}")

    # Janowski on these post-move probs (centered cube)
    cf_jan_0, e_dead_0, e_live_0, W0, L0, TP0, CP0 = janowski_cubeful(probs_0, "centered", x)
    print(f"Janowski cubeful (x={x}): {cf_jan_0:.6f}")
    print(f"  W={W0:.4f}  L={L0:.4f}  TP={TP0:.4f}  CP={CP0:.4f}")
    print(f"  E_dead={e_dead_0:.6f}  E_live={e_live_0:.6f}")

    # C++ Janowski
    cf_cpp_0 = bgbot_cpp.cl2cf_money(probs_0, bgbot_cpp.CubeOwner.CENTERED, x)
    print(f"C++ cl2cf_money:         {cf_cpp_0:.6f}")

    # =======================================================================
    # 1-ply evaluation
    # =======================================================================
    print("\n" + "=" * 80)
    print("1-PLY EVALUATION")
    print("=" * 80)

    r1 = strat_1ply.evaluate_board(target_board, STARTING)
    probs_1 = list(r1["probs"])
    eq_1 = r1["equity"]
    print(f"Post-move probs (1-ply cubeless): {fmt_probs(probs_1)}")
    print(f"Cubeless equity (1-ply):          {eq_1:.6f}")

    # Janowski on 1-ply probs
    cf_jan_1, e_dead_1, e_live_1, W1, L1, TP1, CP1 = janowski_cubeful(probs_1, "centered", x)
    print(f"Janowski cubeful on 1-ply probs:  {cf_jan_1:.6f}")
    print(f"  W={W1:.4f}  L={L1:.4f}  TP={TP1:.4f}  CP={CP1:.4f}")
    print(f"  E_dead={e_dead_1:.6f}  E_live={e_live_1:.6f}")

    # Full 1-ply cubeful recursion (what the webapp does)
    # CubefulAnalyzer._cubeful_equity does:
    #   opp_pre_roll = flip(post_move_board)
    #   opp_eq = cubeful_equity_nply(opp_pre_roll, opp_owner, strategy_0ply, n_plies)
    #   return -opp_eq
    opp_pre_roll = bgbot_cpp.flip_board(target_board)
    # Centered cube, from player's perspective = CENTERED
    # From opponent's perspective after flip = CENTERED
    opp_eq_1 = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.CENTERED,
                                              strat_0ply, 1)
    cf_recursive_1 = -opp_eq_1
    print(f"Recursive cubeful (1-ply):        {cf_recursive_1:.6f}")
    print(f"  opp_eq = {opp_eq_1:.6f}")

    # What about using the cubeful_equity_nply directly on the pre-roll?
    # The starting position IS the pre-roll for this player.
    # But we want post-move equity, not pre-roll equity.
    # After the move, the opponent faces opp_pre_roll.

    # =======================================================================
    # 2-ply evaluation
    # =======================================================================
    print("\n" + "=" * 80)
    print("2-PLY EVALUATION")
    print("=" * 80)

    r2 = strat_2ply.evaluate_board(target_board, STARTING)
    probs_2 = list(r2["probs"])
    eq_2 = r2["equity"]
    print(f"Post-move probs (2-ply cubeless): {fmt_probs(probs_2)}")
    print(f"Cubeless equity (2-ply):          {eq_2:.6f}")

    # Janowski on 2-ply probs
    cf_jan_2, e_dead_2, e_live_2, W2, L2, TP2, CP2 = janowski_cubeful(probs_2, "centered", x)
    print(f"Janowski cubeful on 2-ply probs:  {cf_jan_2:.6f}")
    print(f"  W={W2:.4f}  L={L2:.4f}  TP={TP2:.4f}  CP={CP2:.4f}")

    # Full 2-ply cubeful recursion
    opp_eq_2 = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.CENTERED,
                                              strat_0ply, 2)
    cf_recursive_2 = -opp_eq_2
    print(f"Recursive cubeful (2-ply):        {cf_recursive_2:.6f}")
    print(f"  opp_eq = {opp_eq_2:.6f}")

    # =======================================================================
    # 3-ply evaluation (for completeness)
    # =======================================================================
    print("\n" + "=" * 80)
    print("3-PLY EVALUATION (may be slow)")
    print("=" * 80)

    opp_eq_3 = bgbot_cpp.cubeful_equity_nply(opp_pre_roll, bgbot_cpp.CubeOwner.CENTERED,
                                              strat_0ply, 3)
    cf_recursive_3 = -opp_eq_3
    print(f"Recursive cubeful (3-ply):        {cf_recursive_3:.6f}")
    print(f"  opp_eq = {opp_eq_3:.6f}")

    # =======================================================================
    # Summary comparison
    # =======================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<45s} {'Cubeful Eq':>12s} {'Cubeless Eq':>12s}")
    print("-" * 70)
    print(f"{'0-ply Janowski (x='+str(x)+')':<45s} {cf_jan_0:>12.6f} {eq_0:>12.6f}")
    print(f"{'1-ply Janowski on cubeless probs':<45s} {cf_jan_1:>12.6f} {eq_1:>12.6f}")
    print(f"{'1-ply recursive cubeful':<45s} {cf_recursive_1:>12.6f} {'':>12s}")
    print(f"{'2-ply Janowski on cubeless probs':<45s} {cf_jan_2:>12.6f} {eq_2:>12.6f}")
    print(f"{'2-ply recursive cubeful':<45s} {cf_recursive_2:>12.6f} {'':>12s}")
    print(f"{'3-ply recursive cubeful':<45s} {cf_recursive_3:>12.6f} {'':>12s}")

    print("\n" + "=" * 80)
    print("XG COMPARISON")
    print("=" * 80)
    print(f"XG 2-ply (=our 1-ply): cubeless=+0.002, cubeful=-0.008")
    print(f"XG 3-ply (=our 2-ply): cubeless=+0.005, cubeful=+0.184")
    print(f"Our 1-ply cubeless:    {eq_1:.6f}")
    print(f"Our 2-ply cubeless:    {eq_2:.6f}")
    print(f"Our 1-ply recursive cf: {cf_recursive_1:.6f}")
    print(f"Our 2-ply recursive cf: {cf_recursive_2:.6f}")

    print("\nNote: XG cubeful equity for opening moves should be close to cubeless")
    print("because the cube is centered and the position is nearly even.")
    print("Large divergence between cubeful and cubeless suggests an issue.")

    strat_1ply.clear_cache()
    strat_2ply.clear_cache()


if __name__ == "__main__":
    main()
