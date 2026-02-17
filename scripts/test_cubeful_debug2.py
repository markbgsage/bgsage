#!/usr/bin/env python3
"""Debug cubeful equity — trace what's happening at 1-ply.

For each of the 21 opponent rolls after 24/14, show:
- The opponent's best move
- The 0-ply leaf evaluation (ND and DT)
- Whether the opponent "should double" at the leaf
- The resulting equity contributed to the sum
"""

import sys
import os
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

# Starting position and 24/14 move result
STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]
POST_MOVE = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 1, 0, 0, -3, 0, -5, 0, 0, 0, 0, 1, 0]

# The opponent's pre-roll position (what cubeful_equity_recursive receives)
opp_pre_roll = bgbot_cpp.flip_board(POST_MOVE)

# 21 unique rolls with weights
ROLLS = [
    (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
    (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
    (2,3,2), (2,4,2), (2,5,2), (2,6,2),
    (3,4,2), (3,5,2), (3,6,2),
    (4,5,2), (4,6,2),
    (5,6,2)
]

def invert_probs(probs):
    """Invert probs: P(win)->1-P(win), P(gw)<->P(gl), P(bw)<->P(bl)"""
    return [1.0 - probs[0], probs[3], probs[4], probs[1], probs[2]]

print("Tracing 1-ply cubeful recursion for 24/14 move")
print("Opponent's pre-roll board:", opp_pre_roll)
print()

# The function is called as cubeful_equity_recursive(opp_pre_roll, CENTERED, strat, 1)
# opp_owner = CENTERED, so opp_can_double = true
# flip_owner(CENTERED) = CENTERED → our_owner (player's owner) = CENTERED

sum_equity = 0.0
total_weight = 0

for d1, d2, weight in ROLLS:
    # Generate opponent's legal moves (opponent is on roll at opp_pre_roll)
    candidates = bgbot_cpp.possible_moves(list(opp_pre_roll), d1, d2)

    # Find best move at 0-ply (cubeless)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), list(opp_pre_roll))
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    opp_post_move = list(candidates[best_idx])

    # After opponent moves, it's our turn
    # our_pre_roll = flip(opp_post_move)
    our_pre_roll = bgbot_cpp.flip_board(opp_post_move)

    # Now at 0-ply leaf:
    # Evaluate our_pre_roll with CENTERED ownership (ND case)
    our_flipped = bgbot_cpp.flip_board(list(our_pre_roll))
    race = bgbot_cpp.is_race(list(our_pre_roll))
    x = bgbot_cpp.cube_efficiency(list(our_pre_roll), race)

    post_probs = strat.evaluate_board(list(our_flipped), list(our_pre_roll))["probs"]
    pre_roll_probs = invert_probs(list(post_probs))

    # ND equity from our perspective (0-ply Janowski with CENTERED)
    our_eq_nd = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.CENTERED, x)

    # DT evaluation: after opponent doubles and we take, we own the cube
    # From our perspective: PLAYER owns. From opp perspective (which the recursive
    # call uses): OPPONENT owns.
    # But wait — the recursive call evaluates at OUR pre-roll with dt_opp_owner.
    # dt_opp_owner = OPPONENT (from opp perspective, meaning we own it)
    # The recursive call is cubeful_equity_recursive(our_pre_roll, OPPONENT, ...)
    # At the 0-ply leaf, this does cl2cf_money(pre_roll_probs, OPPONENT, x)
    # OPPONENT from whose perspective? From the "roller's" perspective in the leaf call.
    # The roller at the leaf IS us. So OPPONENT = the other guy owns it.
    # But after we take the double, WE own it! This seems wrong!

    # Let me compute both to see:
    our_eq_dt_opp_owns = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.OPPONENT, x)
    our_eq_dt_we_own = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.PLAYER, x)

    # What the code actually does:
    # cubeful_equity_recursive(opp_pre_roll, opp_owner=CENTERED, strat, plies=1)
    # opp_owner = CENTERED → flip_owner(CENTERED) = CENTERED
    # Wait no — let me re-read the code...
    #
    # Line 216: CubeOwner opp_owner = flip_owner(owner);
    # owner = CENTERED (input), flip_owner(CENTERED) = CENTERED
    # So opp_owner = CENTERED
    #
    # But this is confusing. Let me trace very carefully:
    #
    # cubeful_equity_recursive(opp_pre_roll, CENTERED, strat, 1):
    #   owner = CENTERED  (the "roller" = the opponent in our original frame)
    #   flip_owner(CENTERED) = CENTERED → this is called "opp_owner" in the code
    #   But "opp" of the opponent = US!
    #
    # For each roll of the "roller" (opponent in our frame):
    #   Find best move → opp_post_move
    #   our_pre_roll = flip(opp_post_move)
    #
    #   opp_can_double: opp_owner = CENTERED, so "can the opp of the roller double?"
    #   Wait, the naming is confusing. Let me use the code's frame.
    #
    # In the code's frame, "the roller" at the top level is the OPPONENT in our frame.
    # The code's "opp" of the roller is US.
    # opp_owner = CENTERED means: from the code's-opp (= us) perspective, cube is CENTERED.
    #
    # opp_can_double checks if opp_owner == CENTERED or PLAYER
    #   CENTERED → true. So "we" can double.
    #
    # But wait — this is the WRONG question! The code is asking whether the code's "opp"
    # (= us) can double at their pre-roll. But we already made our move (24/14). It's the
    # OPPONENT's turn. The opponent rolled, made their best move, and now it's back to us.
    # At OUR pre-roll, can WE double? Yes, cube is centered, we can.
    #
    # But should WE be modeled as doubling here? At 1-ply, we're at the leaf.
    # The question is: what equity do we get at this leaf?
    #
    # ND: we don't double → cl2cf_money(our_probs, opp_owner=CENTERED, x)
    #   Wait, the code does: cubeful_equity_recursive(our_pre_roll, opp_owner=CENTERED, plies=0)
    #   At plies=0: cl2cf_money(pre_roll_probs, owner=CENTERED, x)
    #   owner here = opp_owner = CENTERED. This is OUR ownership from OUR perspective = CENTERED. Correct!
    #
    # DT: dt_opp_owner = OPPONENT → cubeful_equity_recursive(our_pre_roll, OPPONENT, plies=0)
    #   At plies=0: cl2cf_money(pre_roll_probs, owner=OPPONENT, x)
    #   This means: from our (the roller's) perspective, OPPONENT owns the cube.
    #   After the code's "opp" (us) doubles and the roller (opponent) takes,
    #   the ROLLER (opponent) owns the cube. From the roller's perspective = PLAYER.
    #   But we're evaluating at OUR pre-roll, where WE are the roller!
    #   So OPPONENT ownership = opponent owns cube = correct IF opponent took our double.
    #
    # Wait, I'm getting confused by the layers. Let me think about it differently.
    #
    # The question at this node: "we" (the code's "opp") might double before rolling.
    # If we double and opponent takes: opponent owns cube → from OUR perspective: OPPONENT owns.
    # So the DT evaluation uses cl2cf_money(our_probs, OPPONENT, x).
    # This gives OUR equity when OPPONENT owns the cube. That's correct!
    #
    # But then opp_eq_dt (from the outer function's perspective, which is the OPPONENT's
    # perspective) = cubeful_equity_recursive(our_pre_roll, OPPONENT, plies=0)
    # This returns equity from OUR perspective (the "roller" of that leaf call).
    # The outer function then does: player_eq_dt = -opp_eq_dt * 2
    # "player" in the outer function = the top-level roller = OPPONENT in our frame.
    # So player_eq_dt = -(our equity when opp owns cube) * 2
    #
    # Hmm, this should be right... Let me just compute the numbers.

    # In the code: for the 1-ply call cubeful_equity_recursive(opp_pre_roll, CENTERED, strat, 1):
    # At each roll:
    #   opp_eq_nd = cubeful_equity_recursive(our_pre_roll, CENTERED, strat, 0)
    #             = cl2cf_money(our_pre_roll_probs, CENTERED, x)  [from our/roller perspective]
    #   player_eq_nd = -opp_eq_nd  [player = top-level roller = opp in our orig frame]
    #
    #   opp_eq_dt = cubeful_equity_recursive(our_pre_roll, OPPONENT, strat, 0)
    #            = cl2cf_money(our_pre_roll_probs, OPPONENT, x)  [opp owns from our perspective]
    #   player_eq_dt = -opp_eq_dt * 2
    #
    #   opp_should_double: check if min(opp_eq_dt*2, 1) > opp_eq_nd
    #   = min(our_eq_when_opp_owns * 2, 1) > our_eq_centered

    # Wait, that's checking from the "opp" (= our) perspective whether we should double.
    # opp_dt_scaled = opp_eq_dt * 2 = cl2cf_money(OPPONENT) * 2
    # This is the equity WE get if we double (from our perspective, at 2x stakes).
    # opp_dp = +1.0 = the equity WE get if we double and opponent passes.
    # opp_best_if_double = min(our DT equity at 2x, +1) = min of what WE get
    # But this is wrong! min gives what's WORST for us. The opponent would choose
    # the response that's worst for us. If we double, opp picks min(DT, DP) from OUR view.
    # So opp_best_if_double = min(our_DT_2x, +1).
    #
    # Wait, "opp_best_if_double" is named from the opp's perspective but computed
    # from our numbers? Let me re-read...
    #
    # opp_dt_scaled = opp_eq_dt * 2. opp_eq_dt was computed as
    # cubeful_equity_recursive(our_pre_roll, OPPONENT, 0) which returns from
    # OUR (the recursive call's roller's) perspective. So opp_eq_dt = our equity.
    # opp_dt_scaled = our equity * 2.
    #
    # BUT the comment says "DT from opp perspective, 2x stakes". That's WRONG.
    # It's from OUR perspective!
    #
    # Then: opp_should_double = (min(our_eq*2, 1) > our_eq_nd)
    # This asks: should WE double if the worst outcome for us (min of take/pass)
    # is better than not doubling? That logic is correct for whether WE should double.
    # But the variable is named "opp_should_double"!
    #
    # Hmm wait — in the code's frame, "opp" IS us. Let me re-read with the code's naming.
    # The function was called with "board = opp_pre_roll" which is the position from
    # the perspective of the top-level roller (the opponent in our real frame).
    # In the code, "the roller" = the opponent. The code's "opp" = us.
    # opp_owner = CENTERED (from the code's opp perspective = from OUR perspective)
    # opp_can_double = true (we can double)
    #
    # After the top-level roller (opponent) moves:
    # opp_pre_roll [in the loop] = our_pre_roll = flip(opp_post_move)
    #
    # opp_eq_nd = cubeful_equity_recursive(our_pre_roll, opp_owner=CENTERED, 0)
    #   This call: roller = us, owner = CENTERED (from our perspective)
    #   Returns: our cubeful equity, centered cube.
    #
    # player_eq_nd = -opp_eq_nd = -(our equity) = opponent's equity for ND
    #
    # opp_eq_dt = cubeful_equity_recursive(our_pre_roll, OPPONENT, 0)
    #   This call: roller = us, owner = OPPONENT (from our perspective, opp owns cube)
    #   Returns: our cubeful equity when opponent owns cube.
    #
    # Now: "opp_dt_scaled = opp_eq_dt * 2" = (our eq when opp owns) * 2
    # "opp_dp = 1.0" = ... this is the DP value but from whose perspective?
    #
    # The comment says "DP from opp perspective". In the code's frame, "opp" = us.
    # If WE double and opponent passes, WE get +1.0. So this is correct from OUR perspective.
    #
    # "opp_best_if_double = min(opp_dt_scaled, opp_dp)" = min(our_DT*2, +1)
    # If WE double, the OPPONENT picks between take and pass.
    # From OUR perspective: take gives us opp_dt_scaled, pass gives us +1.
    # Opponent picks what gives US less → min. Correct!
    #
    # "opp_should_double = (opp_best_if_double > opp_eq_nd)"
    # WE should double if our worst case when doubling > our equity when not doubling.
    # = min(our_DT*2, +1) > our_ND. Correct!
    #
    # If we double and opponent takes:
    # player_eq_dt = -opp_eq_dt * 2 = -(our eq when opp owns) * 2
    # This is the OPPONENT's equity at doubled stakes. Correct!
    #
    # player_eq_dp = -1.0 = opponent's equity if they pass. Correct!
    #
    # If we double: opponent takes if player_eq_dt >= player_eq_dp
    # i.e., -(our eq opp owns)*2 >= -1 → (our eq opp owns)*2 <= 1
    # i.e., opponent takes if our equity at doubled stakes ≤ 1 (the pass value).
    # Opponent takes if taking is better for them than passing.
    # Opponent's take equity = -(our eq opp owns)*2, pass equity = -1.
    # Takes if -(our)*2 >= -1 → our*2 <= 1. Correct!

    # So the logic seems correct. Let me just compute the actual numbers.

    cubeless_eq = bgbot_cpp.cubeless_equity(pre_roll_probs)

    # What the code computes (ND from our perspective, centered cube):
    our_nd = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.CENTERED, x)

    # What the code computes (DT - after "we" double, opp takes, opp owns from our perspective):
    our_dt_opp_owns = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.OPPONENT, x)

    # The code's decision for whether "we" (code's "opp") should double:
    opp_dt_scaled = our_dt_opp_owns * 2.0
    opp_dp = 1.0
    opp_best_if_double = min(opp_dt_scaled, opp_dp)
    should_we_double = opp_best_if_double > our_nd

    # If we double, opponent's response:
    player_eq_dt = -our_dt_opp_owns * 2.0  # opponent's DT equity
    player_eq_dp = -1.0                       # opponent's DP equity

    if should_we_double:
        if player_eq_dt >= player_eq_dp:
            player_eq = player_eq_dt  # take
            decision = "WE DOUBLE, OPP TAKES"
        else:
            player_eq = player_eq_dp  # pass
            decision = "WE DOUBLE, OPP PASSES"
    else:
        player_eq = -our_nd
        decision = "WE DON'T DOUBLE"

    total_weight += weight
    sum_equity += weight * player_eq

    if should_we_double:
        print(f"Roll {d1}-{d2} (w={weight}): cubeless={cubeless_eq:+.4f}  "
              f"our_ND={our_nd:+.4f}  our_DT_2x={opp_dt_scaled:+.4f}  "
              f"→ {decision}  player_eq={player_eq:+.4f}  x={x:.3f}")

print(f"\nTotal weight: {total_weight}")
print(f"Sum equity / 36 = {sum_equity / 36.0:+.6f}")
print(f"(This should match opp_eq from the 1-ply call)")

# Also count decisions
print("\n\nNow showing ALL rolls:")
sum_equity2 = 0.0
n_double = 0
n_no_double = 0

for d1, d2, weight in ROLLS:
    candidates = bgbot_cpp.possible_moves(list(opp_pre_roll), d1, d2)
    best_eq = -999.0
    best_idx = 0
    for i, c in enumerate(candidates):
        r = strat.evaluate_board(list(c), list(opp_pre_roll))
        if r["equity"] > best_eq:
            best_eq = r["equity"]
            best_idx = i

    opp_post_move = list(candidates[best_idx])
    our_pre_roll = bgbot_cpp.flip_board(opp_post_move)
    our_flipped = bgbot_cpp.flip_board(list(our_pre_roll))
    race = bgbot_cpp.is_race(list(our_pre_roll))
    x = bgbot_cpp.cube_efficiency(list(our_pre_roll), race)
    post_probs = strat.evaluate_board(list(our_flipped), list(our_pre_roll))["probs"]
    pre_roll_probs = invert_probs(list(post_probs))
    cubeless_eq = bgbot_cpp.cubeless_equity(pre_roll_probs)

    our_nd = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.CENTERED, x)
    our_dt_opp_owns = bgbot_cpp.cl2cf_money(pre_roll_probs, bgbot_cpp.CubeOwner.OPPONENT, x)

    opp_dt_scaled = our_dt_opp_owns * 2.0
    opp_best_if_double = min(opp_dt_scaled, 1.0)
    should_we_double = opp_best_if_double > our_nd

    if should_we_double:
        player_eq_dt = -our_dt_opp_owns * 2.0
        player_eq_dp = -1.0
        if player_eq_dt >= player_eq_dp:
            player_eq = player_eq_dt
            dec = "D/T"
            n_double += 1
        else:
            player_eq = player_eq_dp
            dec = "D/P"
            n_double += 1
    else:
        player_eq = -our_nd
        dec = "ND "
        n_no_double += 1

    sum_equity2 += weight * player_eq

    print(f"{d1}-{d2} (w={weight}): cl={cubeless_eq:+.5f}  ND={our_nd:+.5f}  "
          f"DT_2x={opp_dt_scaled:+.5f}  {dec}  peq={player_eq:+.5f}")

print(f"\nDoubles: {n_double}/{len(ROLLS)}  No-doubles: {n_no_double}/{len(ROLLS)}")
print(f"Weighted sum / 36 = {sum_equity2 / 36.0:+.6f}")
