#include "bgbot/cube.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <array>

namespace bgbot {

void compute_WL(const std::array<float, NUM_OUTPUTS>& probs, float& W, float& L) {
    float p_win = probs[0];
    float p_gw = probs[1];
    float p_bw = probs[2];
    float p_gl = probs[3];
    float p_bl = probs[4];

    // W = average value of wins: 1 (single) + fraction that are gammon/bg
    if (p_win > 1e-7f) {
        W = 1.0f + (p_gw + p_bw) / p_win;
    } else {
        W = 1.0f;  // No wins — W doesn't matter, but avoid division by zero
    }

    // L = average value of losses
    float p_lose = 1.0f - p_win;
    if (p_lose > 1e-7f) {
        L = 1.0f + (p_gl + p_bl) / p_lose;
    } else {
        L = 1.0f;  // No losses
    }
}

float money_live(float W, float L, float p_win, CubeOwner owner) {
    // Take point and cash point (live cube)
    float TP = (L - 0.5f) / (W + L + 0.5f);
    float CP = (L + 1.0f) / (W + L + 0.5f);

    float p = p_win;

    switch (owner) {
        case CubeOwner::CENTERED: {
            // Interpolate through (0, -L), (TP, -1), (CP, +1), (1, +W)
            if (p < TP) {
                // Segment: (0, -L) → (TP, -1)
                return -L + (-1.0f + L) * p / TP;
            } else if (p < CP) {
                // Segment: (TP, -1) → (CP, +1)
                return -1.0f + 2.0f * (p - TP) / (CP - TP);
            } else {
                // Segment: (CP, +1) → (1, +W)
                return 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::PLAYER: {
            // Player owns cube: interpolate through (0, -L), (CP, +1), (1, +W)
            if (p < CP) {
                return -L + (1.0f + L) * p / CP;
            } else {
                return 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
            }
        }
        case CubeOwner::OPPONENT: {
            // Opponent owns cube: interpolate through (0, -L), (TP, -1), (1, +W)
            if (p < TP) {
                return -L + (-1.0f + L) * p / TP;
            } else {
                return -1.0f + (W + 1.0f) * (p - TP) / (1.0f - TP);
            }
        }
    }
    return 0.0f;  // unreachable
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x) {
    float W, L;
    compute_WL(probs, W, L);

    float e_dead = cubeless_equity(probs);
    float e_live = money_live(W, L, probs[0], owner);

    return e_dead * (1.0f - cube_x) + e_live * cube_x;
}

// ---------------------------------------------------------------------------
// Match play cubeful evaluation (Janowski in MWC space)
// ---------------------------------------------------------------------------

// Check if the cube is "dead" — no useful cube actions possible.
// This happens when the cube value is enough for both players to win the match
// on any win, or during Crawford game.
static bool is_dead_cube(const CubeInfo& cube) {
    if (cube.is_money()) return false;
    if (cube.match.is_crawford) return true;
    // If both players win the match with a normal win at current cube value
    if (cube.match.away1 <= cube.cube_value && cube.match.away2 <= cube.cube_value)
        return true;
    return false;
}

// Compute MWC-space take point and cash point for match play.
// These are the P(win) thresholds analogous to money game TP/CP.
//
// For the player's perspective:
// TP = opponent's take point (P(win) below which opponent should pass)
//    = (MWC_lose_cube - MWC_dp) / (MWC_lose_cube - MWC_win_cube)
// CP = player's cash point (P(win) above which player should play on for gammon)
//    = (MWC_dp - MWC_win_cube_lose) / (MWC_win_cube_win - MWC_win_cube_lose)
//
// But we use the Janowski formulation which works in MWC space directly.
// The take point and cash point in MWC space map to specific P(win) values.

// Compute cubeful MWC for match play — centered cube.
// Piecewise-linear interpolation with 3 regions:
//   p < opponent's TG: opponent too good to double
//   opponent's TG < p < player's TG: in doubling window
//   p > player's TG: player too good to double
// Compute dead-cube cash point (take point) in P(win) space using MET values
// at the doubled cube value. This is where the opponent is indifferent between
// taking and passing the double. Following GNUbg's GetPoints approach.
//
// CP = (rDTL - rDP) / (rDTL - rDTW)
// where:
//   rDTW = weighted MWC if double-taken and doubler wins (at doubled cv)
//   rDTL = weighted MWC if double-taken and doubler loses (at doubled cv)
//   rDP  = MWC if opponent passes (doubler wins cv points)
static float match_cash_point(
    int away1, int away2, int cv, bool craw,
    float rG, float rBG,          // gammon/bg ratios for the doubler's wins
    float rG_opp, float rBG_opp)  // gammon/bg ratios for the doubler's losses
{
    // MWC when opponent passes (doubler wins cv points)
    float rDP = get_met_after(away1, away2, cv, true, craw);

    // MWC outcomes at doubled cube value (2*cv)
    int dcv = 2 * cv;
    float dtw_s = get_met_after(away1, away2, dcv, true, craw);
    float dtw_g = get_met_after(away1, away2, 2*dcv, true, craw);
    float dtw_b = get_met_after(away1, away2, 3*dcv, true, craw);
    float dtl_s = get_met_after(away1, away2, dcv, false, craw);
    float dtl_g = get_met_after(away1, away2, 2*dcv, false, craw);
    float dtl_b = get_met_after(away1, away2, 3*dcv, false, craw);

    // Weighted MWC for double-take outcomes
    float rDTW = (1.0f - rG - rBG) * dtw_s + rG * dtw_g + rBG * dtw_b;
    float rDTL = (1.0f - rG_opp - rBG_opp) * dtl_s + rG_opp * dtl_g + rBG_opp * dtl_b;

    float denom = rDTL - rDTW;
    if (std::abs(denom) < 1e-10f) return 0.5f;
    return std::clamp((rDTL - rDP) / denom, 0.0f, 1.0f);
}

static float cl2cf_match_centered(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC
    float eq_dead = cubeless_equity(probs);
    float mwc_dead = eq2mwc(eq_dead, away1, away2, cv, craw);

    // MET lookups at current cube value (for no-double outcomes)
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    // D/P MWC (player cashes cv points)
    float mwc_cash = mwc_win_s;
    // Opponent's D/P MWC (player loses cv points) — from player's perspective
    float mwc_opp_cash = mwc_lose_s;

    // Weighted MWCs for all win/loss types
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Compute take/cash points using MET at doubled cube value (GNUbg approach).
    // Player's cash point: P(win) above which player plays on for gammon
    float player_cp = match_cash_point(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1);
    // Opponent's cash point: compute from opponent's perspective (flip away/gammon ratios)
    float opp_cp = match_cash_point(away2, away1, cv, craw, rG1, rBG1, rG0, rBG0);

    // Convert to the thresholds used in the piecewise interpolation
    float opp_tg = 1.0f - opp_cp;   // Below this P(win), opponent is too good
    float player_tg = player_cp;      // Above this P(win), player is too good

    // Piecewise-linear live cube MWC (3 regions, following GNUbg Cl2CfMatchCentered)
    float mwc_live;

    if (p_win <= opp_tg) {
        // Region 1: Opponent too good to double
        // Linear from (0, mwc_lose_all) to (opp_tg, mwc_opp_cash)
        if (opp_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else if (p_win < player_tg) {
        // Region 2: In doubling window
        // Linear from (opp_tg, mwc_opp_cash) to (player_tg, mwc_cash)
        float range = player_tg - opp_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_opp_cash + (mwc_cash - mwc_opp_cash) * (p_win - opp_tg) / range;
        } else {
            mwc_live = mwc_opp_cash;
        }
    } else {
        // Region 3: Player too good to double (play on for gammon)
        // Linear from (player_tg, mwc_cash) to (1, mwc_win_all)
        float range = 1.0f - player_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    // Janowski interpolation in MWC space
    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Compute cubeful MWC for match play — player owns cube.
// Piecewise-linear interpolation with 2 regions:
//   p < player's TG: below cash point (only player can double)
//   p >= player's TG: above cash point (player plays on for gammon)
static float cl2cf_match_owned(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC
    float eq_dead = cubeless_equity(probs);
    float mwc_dead = eq2mwc(eq_dead, away1, away2, cv, craw);

    // MET lookups at current cube value
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_cash = mwc_win_s;

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Player's cash point using MET at doubled cube value
    float player_tg = match_cash_point(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1);

    float mwc_live;
    if (p_win <= player_tg) {
        // Region 1: Below cash point
        // Linear from (0, mwc_lose_all) to (player_tg, mwc_cash)
        if (player_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_cash - mwc_lose_all) * p_win / player_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else {
        // Region 2: Player too good to double (play on for gammon)
        // Linear from (player_tg, mwc_cash) to (1, mwc_win_all)
        float range = 1.0f - player_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_cash + (mwc_win_all - mwc_cash) * (p_win - player_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

// Compute cubeful MWC for match play — opponent owns cube (unavailable to player).
// Piecewise-linear interpolation with 2 regions:
//   p < opponent's TG: opponent too good to double
//   p >= opponent's TG: above opponent's cash point
static float cl2cf_match_unavailable(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    float p_win = probs[0];

    // Gammon/backgammon ratios
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;
        rBG0 = probs[2] / p_win;
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);
        rBG1 = probs[4] / (1.0f - p_win);
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC
    float eq_dead = cubeless_equity(probs);
    float mwc_dead = eq2mwc(eq_dead, away1, away2, cv, craw);

    // MET lookups at current cube value
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_opp_cash = mwc_lose_s;

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Opponent's cash point using MET at doubled cube value (from opponent's perspective)
    float opp_cp = match_cash_point(away2, away1, cv, craw, rG1, rBG1, rG0, rBG0);
    float opp_tg = 1.0f - opp_cp;

    float mwc_live;
    if (p_win <= opp_tg) {
        // Region 1: Opponent too good to double
        // Linear from (0, mwc_lose_all) to (opp_tg, mwc_opp_cash)
        if (opp_tg > 1e-10f) {
            mwc_live = mwc_lose_all + (mwc_opp_cash - mwc_lose_all) * p_win / opp_tg;
        } else {
            mwc_live = mwc_lose_all;
        }
    } else {
        // Region 2: Above opponent's cash point
        // Linear from (opp_tg, mwc_opp_cash) to (1, mwc_win_all)
        float range = 1.0f - opp_tg;
        if (range > 1e-10f) {
            mwc_live = mwc_opp_cash + (mwc_win_all - mwc_opp_cash) * (p_win - opp_tg) / range;
        } else {
            mwc_live = mwc_win_all;
        }
    }

    return mwc_dead * (1.0f - cube_x) + mwc_live * cube_x;
}

float cl2cf_match(const std::array<float, NUM_OUTPUTS>& probs,
                  const CubeInfo& cube, float cube_x)
{
    // Dead cube: return cubeless MWC directly
    if (is_dead_cube(cube)) {
        float eq_dead = cubeless_equity(probs);
        return eq2mwc(eq_dead, cube.match.away1, cube.match.away2,
                      cube.cube_value, cube.match.is_crawford);
    }

    // Dispatch by cube ownership
    switch (cube.owner) {
        case CubeOwner::CENTERED:
            return cl2cf_match_centered(probs, cube, cube_x);
        case CubeOwner::PLAYER:
            return cl2cf_match_owned(probs, cube, cube_x);
        case CubeOwner::OPPONENT:
            return cl2cf_match_unavailable(probs, cube, cube_x);
    }
    return 0.0f;  // unreachable
}

float cl2cf(const std::array<float, NUM_OUTPUTS>& probs,
            const CubeInfo& cube, float cube_x)
{
    if (cube.is_money()) {
        return cl2cf_money(probs, cube.owner, cube_x);
    }
    // Match play: cl2cf_match returns MWC, convert to equity
    float mwc = cl2cf_match(probs, cube, cube_x);
    return mwc2eq(mwc, cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

float cube_efficiency(const Board& board, bool is_race_pos) {
    if (!is_race_pos) {
        return 0.68f;  // Contact/crashed
    }
    // Race: linear in roller's pip count, clamped to [0.6, 0.7]
    auto [player_pips, opponent_pips] = pip_counts(board);
    float x = 0.55f + 0.00125f * static_cast<float>(player_pips);
    return std::clamp(x, 0.6f, 0.7f);
}

// Money game cube decision (0-ply).
static CubeDecision cube_decision_0ply_money(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    CubeDecision result;
    result.equity_dp = 1.0f;  // Double/Pass: always +1.0 for money games

    // No Double equity: cubeful equity with current cube state
    result.equity_nd = cl2cf_money(probs, cube.owner, cube_x);

    // Double/Take equity: cubeful equity if cube is doubled and opponent takes.
    // After doubling, the opponent owns the cube at 2x the current value.
    // The cubeful equity at the new cube state, normalized to the new cube value,
    // is cl2cf_money(probs, OPPONENT, cube_x). We multiply by 2 to normalize
    // back to the original cube value (since doubling doubles the stakes).
    result.equity_dt = 2.0f * cl2cf_money(probs, CubeOwner::OPPONENT, cube_x);

    // Decision logic
    // If we double, the opponent picks the response that gives us LESS equity.
    // So the effective equity of doubling = min(DT, DP).
    // We should double if min(DT, DP) > ND.
    float best_double = std::min(result.equity_dt, result.equity_dp);
    result.should_double = (best_double > result.equity_nd);

    // Should opponent take? Take if DT < DP (from doubler's perspective,
    // opponent prefers to give us less equity)
    result.should_take = (result.equity_dt <= result.equity_dp);

    // Optimal equity after both sides play optimally
    if (result.should_double) {
        // We double; opponent picks the option that gives us less equity
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        result.optimal_equity = result.equity_nd;
    }

    return result;
}

// Match play cube decision (0-ply).
// Computes ND/DT/DP in MWC space, then converts to equity at original cube value.
static CubeDecision cube_decision_0ply_match(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    CubeDecision result;

    // DP MWC: opponent passes, player wins cv points
    float dp_m = dp_mwc(away1, away2, cv, craw);

    // ND MWC: cubeful MWC with current cube state
    float nd_m = cl2cf_match(probs, cube, cube_x);

    // DT MWC: cube is doubled and opponent takes.
    // After doubling: cube_value = 2*cv, opponent owns the cube.
    CubeInfo dt_cube;
    dt_cube.cube_value = 2 * cv;
    dt_cube.owner = CubeOwner::OPPONENT;
    dt_cube.match = cube.match;
    float dt_m = cl2cf_match(probs, dt_cube, cube_x);

    // Convert all three MWC values to equity at the original cube value.
    // This normalizes them to a common scale for comparison.
    result.equity_nd = mwc2eq(nd_m, away1, away2, cv, craw);
    result.equity_dt = mwc2eq(dt_m, away1, away2, cv, craw);
    result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);

    // Decision logic (same structure as money, but can_double matters for match)
    bool player_can_double = can_double(cube);

    if (!player_can_double) {
        // Player cannot double (Crawford, dead cube, etc.)
        result.should_double = false;
        result.should_take = true;  // Irrelevant since no double
        result.optimal_equity = result.equity_nd;
    } else {
        // Standard cube decision logic
        float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }
    }

    return result;
}

CubeDecision cube_decision_0ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    if (cube.is_money()) {
        return cube_decision_0ply_money(probs, cube, cube_x);
    }
    return cube_decision_0ply_match(probs, cube, cube_x);
}

CubeDecision cube_decision_0ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    float x = cube_efficiency(board, is_race_pos);
    return cube_decision_0ply(probs, cube, x);
}

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

// Flip cube ownership when switching to opponent's perspective.
// PLAYER ↔ OPPONENT, CENTERED stays CENTERED.
static CubeOwner flip_owner(CubeOwner owner) {
    switch (owner) {
        case CubeOwner::PLAYER:   return CubeOwner::OPPONENT;
        case CubeOwner::OPPONENT: return CubeOwner::PLAYER;
        default:                  return CubeOwner::CENTERED;
    }
}

// The 21 unique dice rolls with weights (same as MultiPlyStrategy::ALL_ROLLS).
struct DiceRoll { int d1, d2, weight; };
static const DiceRoll ALL_ROLLS[21] = {
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
};

// ---------------------------------------------------------------------------
// Unified N-ply cubeful evaluation (money game & match play)
// ---------------------------------------------------------------------------
//
// Computes cubeful equity for a PRE-ROLL position from the roller's perspective.
//
// Algorithm (same for money and match):
//   0-ply (leaf): NN evaluation → Janowski cubeful conversion.
//   N-ply: for each of 21 rolls:
//     1. Generate legal moves.
//     2. Evaluate each candidate's post-move position with 0-ply Janowski cubeful.
//     3. TINY-filter candidates by cubeful equity.
//     4. For survivors with plies>1: quick 0-ply cube action on opponent's
//        pre-roll — if clear D/P, shortcut to DP value.
//     5. Recurse at plies-1 on remaining candidates for refined cubeful equity.
//     6. Pick best candidate by cubeful equity.
//   Average best-candidate equities across 21 rolls.
//
// Move selection uses cubeful equity throughout. Cube decisions (ND vs DT vs DP)
// are only computed at the top level in cube_decision_nply.
//
// For money: operates in equity space, returns cubeful equity (cube=1 normalized).
// For match: operates in MWC space, returns MWC from roller's perspective.

// Helper: evaluate a post-move board at 0-ply Janowski cubeful.
// Returns value from the MOVER's perspective (the player who just moved).
// For money: returns cubeful equity. For match: returns cubeful MWC.
static float eval_post_move_cubeful_0ply(
    const Board& post_move,    // post-move board (mover's perspective)
    const CubeInfo& cube,      // cube state from mover's perspective
    const Strategy& strategy)
{
    // post_move is from the mover's perspective — that's what the NN evaluates.
    bool race = is_race(post_move);
    auto probs = strategy.evaluate_probs(post_move, race);
    // probs are from mover's perspective (post-move semantics). Good.
    float x = cube_efficiency(post_move, race);

    if (cube.is_money()) {
        return cl2cf_money(probs, cube.owner, x);
    } else {
        return cl2cf_match(probs, cube, x);
    }
}

// Helper: evaluate a pre-roll board at 0-ply Janowski cubeful.
// Returns value from the ROLLER's perspective.
static float eval_pre_roll_cubeful_0ply(
    const Board& board,        // pre-roll board (roller's perspective)
    const CubeInfo& cube,      // cube state from roller's perspective
    const Strategy& strategy)
{
    Board flipped = flip(board);
    bool race = is_race(board);

    // Terminal check
    GameResult result = check_game_over(flipped);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = invert_probs(terminal_probs(result));
        if (cube.is_money()) {
            return cubeless_equity(t_probs);
        } else {
            return cubeless_mwc(t_probs, cube.match.away1, cube.match.away2,
                                cube.cube_value, cube.match.is_crawford);
        }
    }

    auto post_probs = strategy.evaluate_probs(flipped, race);
    auto pre_roll_probs = invert_probs(post_probs);
    float x = cube_efficiency(board, race);

    if (cube.is_money()) {
        return cl2cf_money(pre_roll_probs, cube.owner, x);
    } else {
        return cl2cf_match(pre_roll_probs, cube, x);
    }
}

// Helper: get the D/P value from the perspective of the player who would pass.
// For money: opponent passes = player gets +1.0.
// For match: opponent passes = player wins cube_value points.
static float dp_value(const CubeInfo& cube) {
    if (cube.is_money()) {
        return 1.0f;
    } else {
        return dp_mwc(cube.match.away1, cube.match.away2,
                       cube.cube_value, cube.match.is_crawford);
    }
}

// Helper: flip a value from opponent's perspective to player's perspective.
// Money: negate. Match MWC: 1 - x.
static float flip_value(float val, bool is_money) {
    return is_money ? -val : (1.0f - val);
}

// Helper: make a CubeInfo for the opponent's perspective.
static CubeInfo make_opp_cube(const CubeInfo& cube) {
    CubeInfo opp;
    opp.cube_value = cube.cube_value;
    opp.owner = flip_owner(cube.owner);
    opp.match = cube.match.flip();
    return opp;
}

// Unified recursive cubeful evaluation.
// Returns cubeful equity (money) or cubeful MWC (match) from roller's perspective.
static float cubeful_recursive(
    const Board& board,        // pre-roll, roller's perspective
    const CubeInfo& cube,      // from roller's perspective
    const Strategy& strategy,
    int plies,
    const MoveFilter& filter,
    int n_threads,
    bool allow_parallel)
{
    bool is_money = cube.is_money();

    // Terminal check
    Board flipped = flip(board);
    GameResult result = check_game_over(flipped);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = invert_probs(terminal_probs(result));
        if (is_money) {
            return cubeless_equity(t_probs);
        } else {
            return cubeless_mwc(t_probs, cube.match.away1, cube.match.away2,
                                cube.cube_value, cube.match.is_crawford);
        }
    }

    if (plies <= 0) {
        return eval_pre_roll_cubeful_0ply(board, cube, strategy);
    }

    // Cube info from the opponent's perspective (after player moves, before opp rolls)
    CubeInfo opp_cube = make_opp_cube(cube);

    // Lambda to evaluate a single roll. Returns weighted value contribution.
    auto evaluate_roll = [&](int roll_idx) -> double {
        const auto& roll = ALL_ROLLS[roll_idx];

        std::vector<Board> candidates;
        candidates.reserve(32);

        // Generate the roller's legal moves
        possible_boards(board, roll.d1, roll.d2, candidates);
        int n_cand = static_cast<int>(candidates.size());

        if (n_cand == 0) {
            // No legal moves (shouldn't happen in backgammon, but safety)
            return roll.weight * static_cast<double>(
                eval_pre_roll_cubeful_0ply(board, cube, strategy));
        }

        // Step 1: Evaluate all candidates at 0-ply cubeful for ranking/filtering.
        // Use the post-move Janowski estimate as a cheap heuristic.
        // These values are only used for relative ranking, not as final values.
        std::vector<float> cand_values(n_cand);

        for (int i = 0; i < n_cand; ++i) {
            GameResult post_result = check_game_over(candidates[i]);
            if (post_result != GameResult::NOT_OVER) {
                auto tp = terminal_probs(post_result);
                cand_values[i] = is_money ? cubeless_equity(tp)
                    : cubeless_mwc(tp, cube.match.away1, cube.match.away2,
                                   cube.cube_value, cube.match.is_crawford);
            } else {
                cand_values[i] = eval_post_move_cubeful_0ply(
                    candidates[i], cube, strategy);
            }
        }

        // Step 2: Find best 0-ply value and TINY-filter.
        float best_0ply = *std::max_element(cand_values.begin(), cand_values.end());

        float threshold = filter.threshold;
        int max_moves = filter.max_moves;

        // Collect indices sorted by 0-ply value (descending)
        std::vector<std::pair<float, int>> scored;
        scored.reserve(n_cand);
        for (int i = 0; i < n_cand; ++i) {
            scored.push_back({cand_values[i], i});
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Filter: keep up to max_moves within threshold of best
        int n_survivors = 0;
        for (int i = 0; i < n_cand && n_survivors < max_moves; ++i) {
            if (best_0ply - scored[i].first > threshold && n_survivors > 0) break;
            ++n_survivors;
        }

        // Step 3: For each survivor, model the opponent's cube decision, then
        // recurse at plies-1 with the correct post-cube-decision cube state.
        //
        // At the opponent's pre-roll position, the opponent may double:
        //   - If opp doubles and we PASS: our value = my_dp_val (fixed)
        //   - If opp doubles and we TAKE: recurse with doubled cube, we own it
        //   - If opp doesn't double: recurse with current cube state
        //
        // We use a quick 0-ply cube check to decide whether the opponent doubles,
        // then recurse at plies-1 with the appropriate cube state.

        // Value we get if opponent doubles and we pass
        float my_dp_val;
        if (is_money) {
            my_dp_val = -1.0f;  // Opponent D/P means we lose 1.0 at current cube
        } else {
            my_dp_val = 1.0f - dp_mwc(opp_cube.match.away1, opp_cube.match.away2,
                                        opp_cube.cube_value, opp_cube.match.is_crawford);
        }

        // Cube state if opponent doubles and we take (from opponent's perspective)
        CubeInfo opp_dt_cube;
        opp_dt_cube.cube_value = 2 * opp_cube.cube_value;
        opp_dt_cube.owner = CubeOwner::OPPONENT;  // We own it (OPPONENT from opp's view)
        opp_dt_cube.match = opp_cube.match;

        bool opp_can_dbl = can_double(opp_cube);

        float best_val = -1e30f;
        for (int si = 0; si < n_survivors; ++si) {
            int idx = scored[si].second;
            const Board& post_move = candidates[idx];

            // Check terminal
            GameResult post_result = check_game_over(post_move);
            if (post_result != GameResult::NOT_OVER) {
                float v = cand_values[idx];  // Terminal — value is exact
                best_val = std::max(best_val, v);
                continue;
            }

            Board opp_pre_roll = flip(post_move);

            // Determine the cube state for the recursion by modeling
            // the opponent's cube decision at their pre-roll position.
            if (opp_can_dbl) {
                // Quick 0-ply check: would the opponent double here?
                float opp_nd_0ply = eval_pre_roll_cubeful_0ply(
                    opp_pre_roll, opp_cube, strategy);
                float opp_dt_0ply = eval_pre_roll_cubeful_0ply(
                    opp_pre_roll, opp_dt_cube, strategy);
                float opp_dp_val = dp_value(opp_cube);

                // Opponent doubles if min(dt, dp) > nd (from their perspective)
                float opp_best_double = std::min(opp_dt_0ply, opp_dp_val);
                if (opp_best_double > opp_nd_0ply) {
                    // Opponent would double.
                    if (opp_dt_0ply > opp_dp_val) {
                        // We would pass (DT worse than DP for us = better for opp)
                        best_val = std::max(best_val, my_dp_val);
                        continue;
                    }
                    // We would take — recurse with the DOUBLED cube state
                    float opp_val = cubeful_recursive(
                        opp_pre_roll, opp_dt_cube, strategy, plies - 1, filter,
                        n_threads, /*allow_parallel=*/false);
                    float our_val = flip_value(opp_val, is_money);
                    best_val = std::max(best_val, our_val);
                    continue;
                }
            }

            // Opponent doesn't double (or can't) — recurse with current cube
            float opp_val = cubeful_recursive(
                opp_pre_roll, opp_cube, strategy, plies - 1, filter,
                n_threads, /*allow_parallel=*/false);
            float our_val = flip_value(opp_val, is_money);
            best_val = std::max(best_val, our_val);
        }

        return roll.weight * static_cast<double>(best_val);
    };

    double sum_val = 0.0;

    if (allow_parallel && n_threads > 1 && plies > 1) {
        std::array<double, 21> roll_vals{};
        multipy_parallel_for(21, n_threads, [&](int idx) {
            roll_vals[idx] = evaluate_roll(idx);
        });
        for (int i = 0; i < 21; ++i) {
            sum_val += roll_vals[i];
        }
    } else {
        for (int i = 0; i < 21; ++i) {
            sum_val += evaluate_roll(i);
        }
    }

    return static_cast<float>(sum_val / 36.0);
}

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads)
{
    CubeInfo cube;
    cube.cube_value = 1;
    cube.owner = owner;
    // match defaults to {0,0,false} = money game
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    return cubeful_recursive(board, cube, strategy, n_plies, filter,
                             n_threads, allow_parallel);
}

// Match play overload: returns cubeful equity (NOT MWC).
float cubeful_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads)
{
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    float val = cubeful_recursive(board, cube, strategy, n_plies, filter,
                                   n_threads, allow_parallel);
    if (cube.is_money()) {
        return val;  // Already equity
    }
    return mwc2eq(val, cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads)
{
    if (n_plies <= 0) {
        // Use 0-ply path: get pre-roll probs, apply Janowski
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float x = cube_efficiency(board, race);
        return cube_decision_0ply(pre_roll_probs, cube, x);
    }

    bool is_money = cube.is_money();
    bool allow_parallel = (n_threads > 1 && n_plies > 1);

    CubeDecision result;

    // ND: cubeful value with current cube state
    float nd_val = cubeful_recursive(board, cube, strategy, n_plies, filter,
                                      n_threads, allow_parallel);

    // DT: cubeful value with doubled cube, opponent owns
    CubeInfo dt_cube;
    dt_cube.cube_value = 2 * cube.cube_value;
    dt_cube.owner = CubeOwner::OPPONENT;
    dt_cube.match = cube.match;
    float dt_val = cubeful_recursive(board, dt_cube, strategy, n_plies, filter,
                                      n_threads, allow_parallel);

    if (is_money) {
        result.equity_nd = nd_val;
        result.equity_dt = 2.0f * dt_val;  // Normalize DT to cube=1 scale
        result.equity_dp = 1.0f;
    } else {
        int away1 = cube.match.away1;
        int away2 = cube.match.away2;
        int cv = cube.cube_value;
        bool craw = cube.match.is_crawford;

        // DP MWC: opponent passes, player wins cv points
        float dp_m = dp_mwc(away1, away2, cv, craw);

        // Convert all three MWC values to equity at the original cube value
        result.equity_nd = mwc2eq(nd_val, away1, away2, cv, craw);
        result.equity_dt = mwc2eq(dt_val, away1, away2, cv, craw);
        result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);
    }

    // Decision logic
    bool player_can_double = can_double(cube);

    if (!player_can_double) {
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = result.equity_nd;
    } else {
        float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }
    }

    return result;
}

} // namespace bgbot
