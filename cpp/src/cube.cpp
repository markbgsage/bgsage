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
    return cl2cf_money(probs, owner, cube_x, false);
}

float cl2cf_money(const std::array<float, NUM_OUTPUTS>& probs,
                  CubeOwner owner, float cube_x, bool jacoby_active) {
    float W, L;
    if (jacoby_active) {
        // Jacoby: gammons/backgammons worth nothing with centered cube
        W = 1.0f;
        L = 1.0f;
    } else {
        compute_WL(probs, W, L);
    }

    float e_dead = jacoby_active
        ? (2.0f * probs[0] - 1.0f)  // Gammon terms zeroed
        : cubeless_equity(probs);
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
// Automatic redouble detection (GNUbg's GetCubePrimeValue).
// When the doubler (away_i) is close enough that losing at cv or 2*cv costs
// the same, they always take a recube — making the effective cube 2*cv.
// Uses 1-indexed away values (GNUbg uses 0-indexed).
static int get_cube_prime_value(int away_i, int away_j, int cv) {
    if ((away_i - 1) < 2 * cv && (away_j - 1) >= 2 * cv)
        return 2 * cv;
    return cv;
}

// Compute dead-cube cash point (take point) in P(win) space using MET values
// at the doubled cube value. This is where the opponent is indifferent between
// taking and passing the double. Following GNUbg's GetPoints approach.
//
// CP = (rDTL - rDP) / (rDTL - rDTW)
// where:
//   rDTW = weighted MWC if double-taken and doubler wins (at effective cube)
//   rDTL = weighted MWC if double-taken and doubler loses (at effective cube)
//   rDP  = MWC if opponent passes (doubler wins cv points)
//
// Effective cube accounts for automatic redoubles: when a player is close
// enough (away < 2*dcv), losing at dcv or 2*dcv is equivalent, so they
// always take a recube. This changes the DTW/DTL MET lookups.
static float match_cash_point(
    int away1, int away2, int cv, bool craw,
    float rG, float rBG,          // gammon/bg ratios for the doubler's wins
    float rG_opp, float rBG_opp)  // gammon/bg ratios for the doubler's losses
{
    // MWC when opponent passes (doubler wins cv points)
    float rDP = get_met_after(away1, away2, cv, true, craw);

    // Nominal doubled cube value
    int dcv = 2 * cv;

    // Auto-redouble: effective cube may be higher than dcv
    int prime_doubler = get_cube_prime_value(away1, away2, dcv);
    int prime_taker = get_cube_prime_value(away2, away1, dcv);

    // DTW: doubler wins at prime_doubler effective cube
    float dtw_s = get_met_after(away1, away2, prime_doubler, true, craw);
    float dtw_g = get_met_after(away1, away2, 2*prime_doubler, true, craw);
    float dtw_b = get_met_after(away1, away2, 3*prime_doubler, true, craw);

    // DTL: doubler loses at prime_taker effective cube
    float dtl_s = get_met_after(away1, away2, prime_taker, false, craw);
    float dtl_g = get_met_after(away1, away2, 2*prime_taker, false, craw);
    float dtl_b = get_met_after(away1, away2, 3*prime_taker, false, craw);

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

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

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

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

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

    // Dead cube MWC (uses MET-weighted outcome values, not linear approximation)
    float mwc_dead = cubeless_mwc(probs, away1, away2, cv, craw);

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
        return cubeless_mwc(probs, cube.match.away1, cube.match.away2,
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
        return cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());
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
    // Race: linear in average pip count (both players), clamped to [0.6, 0.7].
    // Using the average instead of just the roller's pips makes the Janowski
    // cubeful conversion perspective-independent, eliminating a systematic
    // bias at odd ply levels in N-ply match play evaluation.
    auto [player_pips, opponent_pips] = pip_counts(board);
    float avg_pips = static_cast<float>(player_pips + opponent_pips) / 2.0f;
    float x = 0.55f + 0.00125f * avg_pips;
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

    // No Double equity: cubeful equity with current cube state.
    // With Jacoby and centered cube, gammons are zeroed (W=1, L=1).
    result.equity_nd = cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());

    // Double/Take equity: cubeful equity if cube is doubled and opponent takes.
    // After doubling, the opponent owns the cube at 2x the current value.
    // Jacoby is NOT active here — the cube has been turned, gammons count.
    float actual_dt = 2.0f * cl2cf_money(probs, CubeOwner::OPPONENT, cube_x, false);

    // Beaver check: if beavers are allowed and DT < 0, the opponent beavers.
    // DB = 2 * DT (beaver doubles the cube value, opponent retains ownership).
    // This is exact for money games because cl2cf_money is cube-value-independent.
    if (cube.beaver && actual_dt < 0.0f) {
        result.equity_dt = 2.0f * actual_dt;  // DB equity
        result.is_beaver = true;
    } else {
        result.equity_dt = actual_dt;
        result.is_beaver = false;
    }

    // Decision logic
    // If we double, the opponent picks the response that gives us LESS equity.
    // With beaver: opponent already chose min(DT, DB, DP) — result.equity_dt
    // reflects the effective response (DB if beaver, DT otherwise).
    float best_double = std::min(result.equity_dt, result.equity_dp);
    result.should_double = (best_double > result.equity_nd);

    // Should opponent take? Take if effective DT < DP (beaver counts as take)
    result.should_take = (result.equity_dt <= result.equity_dp);

    // Optimal equity after both sides play optimally
    if (result.should_double) {
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

    // Convert MWC values to equity for display.
    // All three use the SAME normalization (original cube value) so they're
    // directly comparable — matching GNUbg's FindCubeDecision convention.
    result.equity_nd = mwc2eq(nd_m, away1, away2, cv, craw);
    result.equity_dt = mwc2eq(dt_m, away1, away2, cv, craw);
    result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);

    // Decision logic in MWC space for correctness (equities are all on the
    // same scale now, but MWC decisions are the canonical approach for match play).
    bool player_can_double = can_double(cube);

    // Post-Crawford automatic double: the trailing player (away1 > 1,
    // opponent at 1-away) should always double — losing costs the match
    // regardless of cube value, but winning scores more points.
    bool auto_double = (!craw && player_can_double && away1 > 1 && away2 == 1);

    if (!player_can_double) {
        // Player cannot double (Crawford, dead cube, etc.)
        result.should_double = false;
        result.should_take = true;  // Irrelevant since no double
        result.optimal_equity = result.equity_nd;
    } else if (auto_double) {
        result.should_double = true;
        result.should_take = (dt_m <= dp_m);
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        float best_double_mwc = std::min(dt_m, dp_m);
        result.should_double = (best_double_mwc > nd_m);
        result.should_take = (dt_m <= dp_m);

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
    float x = (cube.cube_x_override >= 0.0f) ? cube.cube_x_override
                                               : cube_efficiency(board, is_race_pos);
    return cube_decision_0ply(probs, cube, x);
}

// ---------------------------------------------------------------------------
// N-ply cubeful evaluation
// ---------------------------------------------------------------------------

// flip_owner() is now an inline function in cube.h.

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
// Evaluate-all-and-decide N-ply cubeful evaluation (money game & match play)
// ---------------------------------------------------------------------------
//
// Carries an array of cube states ("cci" = count of cube infos) through the
// entire recursive search. At each level, states are expanded (cci → 2*cci)
// into ND + DT branches, evaluated recursively, then collapsed (2*cci → cci)
// by making optimal cube decisions using the full recursive values.
//
// This matches gnubg's EvaluatePositionCubeful4 approach: cube decisions
// emerge from the tree values rather than from 0-ply heuristic predictions.
//
// For money: operates in equity space (normalized to cube=1).
// For match: operates in MWC space internally.

// Max cube states supported. Doubles per ply level, so 64 supports 6+ ply.
static constexpr int MAX_CCI = 64;

// Resolve cube efficiency: use override if set, otherwise auto-detect.
static float resolve_cube_x(const CubeInfo& cube, const Board& board, bool race) {
    if (cube.cube_x_override >= 0.0f) return cube.cube_x_override;
    return cube_efficiency(board, race);
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
            if (cube.jacoby_active()) {
                return 2.0f * t_probs[0] - 1.0f;  // Jacoby: gammons worth nothing
            }
            return cubeless_equity(t_probs);
        } else {
            return cubeless_mwc(t_probs, cube.match.away1, cube.match.away2,
                                cube.cube_value, cube.match.is_crawford);
        }
    }

    auto post_probs = strategy.evaluate_probs(flipped, race);
    auto pre_roll_probs = invert_probs(post_probs);
    float x = resolve_cube_x(cube, board, race);

    if (cube.is_money()) {
        return cl2cf_money(pre_roll_probs, cube.owner, x, cube.jacoby_active());
    } else {
        return cl2cf_match(pre_roll_probs, cube, x);
    }
}

// Helper: flip a CubeInfo to the opponent's perspective.
// PLAYER ↔ OPPONENT, CENTERED stays. Match away values swapped.
static CubeInfo flip_cube_perspective(const CubeInfo& cube) {
    CubeInfo opp;
    opp.cube_value = cube.cube_value;
    opp.owner = flip_owner(cube.owner);
    opp.match = cube.match.flip();
    opp.cube_x_override = cube.cube_x_override;
    opp.jacoby = cube.jacoby;
    opp.beaver = cube.beaver;
    return opp;
}

// Expand cci cube states into 2*cci states (ND + DT pairs).
//
// For each input state i:
//   output[2*i]   = ND branch (same cube state)
//   output[2*i+1] = DT branch (doubled, opponent owns) or unavailable
//
// fTop: if true, never create DT branches (top-level already provides ND+DT).
// fInvert: if true, flip perspective to opponent's view (for recursion prep).
static void make_cube_pos(
    const CubeInfo input[],   // cci input states
    int cci,
    bool fTop,
    CubeInfo output[],        // 2*cci output states
    bool fInvert)
{
    for (int ici = 0, i = 0; ici < cci; ici++) {
        // ND branch (even index): same cube state
        if (input[ici].cube_value > 0) {
            output[i] = fInvert ? flip_cube_perspective(input[ici]) : input[ici];
        } else {
            output[i].cube_value = -1;  // unavailable sentinel
        }
        i++;

        // DT branch (odd index): doubled, opponent owns
        if (!fTop && input[ici].cube_value > 0 && can_double(input[ici])) {
            CubeInfo dt = input[ici];
            dt.cube_value = 2 * input[ici].cube_value;
            dt.owner = CubeOwner::OPPONENT;  // taker owns
            output[i] = fInvert ? flip_cube_perspective(dt) : dt;
        } else {
            output[i].cube_value = -1;  // unavailable sentinel
        }
        i++;
    }
}

// Collapse 2*cci expanded equities back to cci by making optimal cube decisions.
//
// For each pair (ND at 2*i, DT at 2*i+1):
//   - If DT is available: compare ND vs DT vs DP, pick optimal
//   - If DT unavailable: result = ND
//
// Money: DT equity scaled by 2x (evaluated at doubled cube, normalize to 1x).
// Match: DT in MWC space, no scaling needed.
static void get_ecf3(
    float arCubeful[],         // output: cci optimal equities
    int cci,
    const float arCf[],        // input: 2*cci expanded equities
    const CubeInfo aci[])      // input: 2*cci expanded cube states
{
    for (int ici = 0, i = 0; ici < cci; ici++, i += 2) {
        if (aci[i + 1].cube_value > 0) {
            // DT branch available: make cube decision
            float rND = arCf[i];
            float rDT;
            bool is_money = aci[i].is_money();

            if (is_money) {
                rDT = 2.0f * arCf[i + 1];  // Scale DT to cube=1
            } else {
                rDT = arCf[i + 1];          // MWC space, no scaling
            }

            // Beaver: if enabled and DT < 0, opponent beavers (DB = 2*DT).
            // Exact for money games (cl2cf_money is cube-value-independent).
            if (is_money && aci[i].beaver && rDT < 0.0f) {
                rDT = 2.0f * rDT;  // DB equity
            }

            // D/P value from the ND branch's cube state
            float rDP;
            if (is_money) {
                rDP = 1.0f;
            } else {
                rDP = dp_mwc(aci[i].match.away1, aci[i].match.away2,
                             aci[i].cube_value, aci[i].match.is_crawford);
            }

            // Decision: double if min(DT, DP) > ND
            if (rDT >= rND && rDP >= rND) {
                // Should double; opponent picks their best response
                arCubeful[ici] = (rDT >= rDP) ? rDP : rDT;
            } else {
                arCubeful[ici] = rND;
            }
        } else {
            // No cube available: always no double
            arCubeful[ici] = arCf[i];
        }
    }
}

// Core recursive cubeful evaluation carrying multiple cube states.
//
// Evaluates a pre-roll position for all cci cube states simultaneously.
// At leaves, applies Janowski. At internal nodes, loops over 21 rolls,
// picks the best move by cubeless equity, recurses with all cube states,
// then collapses via get_ecf3.
//
// board: pre-roll, from the roller's perspective.
// aciCubePos: cci cube states, all from the roller's perspective.
// arCubeful: output array of cci equities/MWC values.
static void cubeful_recursive_multi(
    const Board& board,
    const CubeInfo aciCubePos[],
    int cci,
    const Strategy& strategy,
    int plies,
    const MoveFilter& filter,
    int n_threads,
    bool allow_parallel,
    bool fTop,
    float arCubeful[])
{
    bool is_money = (cci > 0 && aciCubePos[0].cube_value > 0)
                    ? aciCubePos[0].is_money() : true;

    // Terminal check
    Board flipped = flip(board);
    GameResult result = check_game_over(flipped);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = invert_probs(terminal_probs(result));
        for (int ici = 0; ici < cci; ici++) {
            if (aciCubePos[ici].cube_value <= 0) {
                arCubeful[ici] = 0.0f;
                continue;
            }
            if (aciCubePos[ici].is_money()) {
                if (aciCubePos[ici].jacoby_active()) {
                    arCubeful[ici] = 2.0f * t_probs[0] - 1.0f;  // Jacoby: gammons = 0
                } else {
                    arCubeful[ici] = cubeless_equity(t_probs);
                }
            } else {
                arCubeful[ici] = cubeless_mwc(t_probs,
                    aciCubePos[ici].match.away1, aciCubePos[ici].match.away2,
                    aciCubePos[ici].cube_value, aciCubePos[ici].match.is_crawford);
            }
        }
        return;
    }

    // --- Leaf node (0-ply): NN evaluation + Janowski ---
    if (plies <= 0) {
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float default_x = resolve_cube_x(
            (cci > 0 && aciCubePos[0].cube_value > 0) ? aciCubePos[0] : CubeInfo{},
            board, race);

        // Expand cci → 2*cci (fInvert=false: evaluate from current player's perspective)
        CubeInfo aci[MAX_CCI * 2];
        make_cube_pos(aciCubePos, cci, fTop, aci, false);

        float arCf[MAX_CCI * 2];
        for (int i = 0; i < 2 * cci; i++) {
            if (aci[i].cube_value <= 0) {
                arCf[i] = 0.0f;
                continue;
            }
            float x = (aci[i].cube_x_override >= 0.0f)
                       ? aci[i].cube_x_override : default_x;
            if (aci[i].is_money()) {
                arCf[i] = cl2cf_money(pre_roll_probs, aci[i].owner, x,
                                       aci[i].jacoby_active());
            } else {
                arCf[i] = cl2cf_match(pre_roll_probs, aci[i], x);
            }
        }

        // Collapse 2*cci → cci
        get_ecf3(arCubeful, cci, arCf, aci);
        return;
    }

    // --- Internal node (plies > 0): recurse over 21 rolls ---

    // Expand cube states for recursion (fInvert=true: flip to opponent's perspective)
    CubeInfo aci[MAX_CCI * 2];
    make_cube_pos(aciCubePos, cci, fTop, aci, true);
    int expanded_cci = 2 * cci;

    // Accumulators for weighted cubeful equities
    float arCf[MAX_CCI * 2] = {};

    // Lambda to evaluate a single dice roll
    auto evaluate_roll = [&](int roll_idx, float arCfLocal[]) {
        const auto& roll = ALL_ROLLS[roll_idx];

        // Generate legal moves
        std::vector<Board> candidates;
        candidates.reserve(32);
        possible_boards(board, roll.d1, roll.d2, candidates);
        int n_cand = static_cast<int>(candidates.size());

        if (n_cand == 0) {
            // No legal moves: evaluate standing pat (flip board = opponent's turn)
            Board opp_board = flip(flipped);  // same as board, but flip(board) = flipped
            float arCfTemp[MAX_CCI * 2];
            cubeful_recursive_multi(opp_board, aci, expanded_cci,
                                    strategy, plies - 1, filter,
                                    n_threads, false, false, arCfTemp);
            for (int i = 0; i < expanded_cci; i++)
                arCfLocal[i] = roll.weight * arCfTemp[i];
            return;
        }

        // Pick best move by cubeless 0-ply equity (shared across all cube states)
        int best_idx = 0;
        float best_eq = -1e30f;
        for (int c = 0; c < n_cand; c++) {
            float eq;
            GameResult gr = check_game_over(candidates[c]);
            if (gr != GameResult::NOT_OVER) {
                eq = cubeless_equity(terminal_probs(gr));
            } else {
                auto probs = strategy.evaluate_probs(candidates[c],
                                                      is_race(candidates[c]));
                eq = cubeless_equity(probs);
            }
            if (eq > best_eq) { best_eq = eq; best_idx = c; }
        }

        // Check if best move is terminal
        GameResult post_result = check_game_over(candidates[best_idx]);
        if (post_result != GameResult::NOT_OVER) {
            // terminal_probs gives mover's perspective, but accumulated values
            // are from the opponent of the mover's perspective (matching the
            // recursive case). Invert to get the correct perspective.
            auto tp = invert_probs(terminal_probs(post_result));
            for (int i = 0; i < expanded_cci; i++) {
                if (aci[i].cube_value <= 0) {
                    arCfLocal[i] = 0.0f;
                    continue;
                }
                if (aci[i].is_money()) {
                    if (aci[i].jacoby_active()) {
                        arCfLocal[i] = roll.weight * (2.0f * tp[0] - 1.0f);
                    } else {
                        arCfLocal[i] = roll.weight * cubeless_equity(tp);
                    }
                } else {
                    arCfLocal[i] = roll.weight * cubeless_mwc(tp,
                        aci[i].match.away1, aci[i].match.away2,
                        aci[i].cube_value, aci[i].match.is_crawford);
                }
            }
            return;
        }

        // Flip to opponent's perspective and recurse
        Board opp_pre_roll = flip(candidates[best_idx]);

        float arCfTemp[MAX_CCI * 2];
        cubeful_recursive_multi(opp_pre_roll, aci, expanded_cci,
                                strategy, plies - 1, filter,
                                n_threads, false, false, arCfTemp);

        for (int i = 0; i < expanded_cci; i++)
            arCfLocal[i] = roll.weight * arCfTemp[i];
    };

    // Execute rolls (serial or parallel)
    if (allow_parallel && n_threads > 1 && plies > 1) {
        std::array<std::array<float, MAX_CCI * 2>, 21> roll_results{};
        multipy_parallel_for(21, n_threads, [&](int idx) {
            evaluate_roll(idx, roll_results[idx].data());
        });
        for (int r = 0; r < 21; r++)
            for (int i = 0; i < expanded_cci; i++)
                arCf[i] += roll_results[r][i];
    } else {
        float arCfLocal[MAX_CCI * 2];
        for (int r = 0; r < 21; r++) {
            std::fill(arCfLocal, arCfLocal + expanded_cci, 0.0f);
            evaluate_roll(r, arCfLocal);
            for (int i = 0; i < expanded_cci; i++)
                arCf[i] += arCfLocal[i];
        }
    }

    // Average over 36 and flip perspective back to current player
    for (int i = 0; i < expanded_cci; i++) {
        if (is_money) {
            arCf[i] = -arCf[i] / 36.0f;          // Negate for opponent → player
        } else {
            arCf[i] = 1.0f - arCf[i] / 36.0f;    // MWC complement
        }
    }

    // Un-invert the cube states back to current player's perspective
    // (make_cube_pos with fInvert=true flipped them to opponent's view)
    for (int i = 0; i < expanded_cci; i++) {
        if (aci[i].cube_value > 0) {
            aci[i] = flip_cube_perspective(aci[i]);
        }
    }

    // Collapse 2*cci → cci via optimal cube decisions
    get_ecf3(arCubeful, cci, arCf, aci);
}

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads)
{
    if (n_plies <= 0) {
        CubeInfo cube;
        cube.cube_value = 1;
        cube.owner = owner;
        return eval_pre_roll_cubeful_0ply(board, cube, strategy);
    }

    CubeInfo aciCubePos[1];
    aciCubePos[0].cube_value = 1;
    aciCubePos[0].owner = owner;
    // match defaults to {0,0,false} = money game

    float arCubeful[1];
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    cubeful_recursive_multi(board, aciCubePos, 1, strategy, n_plies, filter,
                            n_threads, allow_parallel, false, arCubeful);
    return arCubeful[0];
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
    if (n_plies <= 0) {
        float val = eval_pre_roll_cubeful_0ply(board, cube, strategy);
        if (cube.is_money()) return val;
        return mwc2eq(val, cube.match.away1, cube.match.away2,
                      cube.cube_value, cube.match.is_crawford);
    }

    CubeInfo aciCubePos[1] = {cube};
    float arCubeful[1];
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    cubeful_recursive_multi(board, aciCubePos, 1, strategy, n_plies, filter,
                            n_threads, allow_parallel, false, arCubeful);

    if (cube.is_money()) return arCubeful[0];
    return mwc2eq(arCubeful[0], cube.match.away1, cube.match.away2,
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
        float x = resolve_cube_x(cube, board, race);
        return cube_decision_0ply(pre_roll_probs, cube, x);
    }

    bool is_money = cube.is_money();
    bool allow_parallel = (n_threads > 1 && n_plies > 1);

    // Two initial cube states: ND (current) and DT (doubled, opponent owns)
    CubeInfo aciCubePos[2];
    aciCubePos[0] = cube;                                // ND state
    aciCubePos[1] = cube;                                // DT state
    aciCubePos[1].cube_value = 2 * cube.cube_value;
    aciCubePos[1].owner = CubeOwner::OPPONENT;

    float arCubeful[2];
    cubeful_recursive_multi(board, aciCubePos, 2, strategy, n_plies, filter,
                            n_threads, allow_parallel, /*fTop=*/true, arCubeful);

    // arCubeful[0] = ND value, arCubeful[1] = DT value (at doubled cube scale)
    CubeDecision result;

    if (is_money) {
        result.equity_nd = arCubeful[0];
        float actual_dt = 2.0f * arCubeful[1];  // Scale DT to cube=1
        result.equity_dp = 1.0f;

        // Beaver: if enabled and DT < 0, opponent beavers (DB = 2*DT).
        if (cube.beaver && actual_dt < 0.0f) {
            result.equity_dt = 2.0f * actual_dt;  // DB equity
            result.is_beaver = true;
        } else {
            result.equity_dt = actual_dt;
        }
    } else {
        int away1 = cube.match.away1;
        int away2 = cube.match.away2;
        int cv = cube.cube_value;
        bool craw = cube.match.is_crawford;

        float dp_m = dp_mwc(away1, away2, cv, craw);

        result.equity_nd = mwc2eq(arCubeful[0], away1, away2, cv, craw);
        result.equity_dt = mwc2eq(arCubeful[1], away1, away2, cv, craw);
        result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);
    }

    // Decision logic
    bool player_can_double = can_double(cube);
    // Post-Crawford automatic double: trailing player should always double
    bool auto_double = (!is_money && !cube.match.is_crawford &&
                        player_can_double &&
                        cube.match.away1 > 1 && cube.match.away2 == 1);

    if (!player_can_double) {
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = result.equity_nd;
    } else if (is_money) {
        // Money: compare equities (all at same scale)
        float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }
    } else if (auto_double) {
        // Post-Crawford trailer: always double, opponent decides take/pass
        float nd_m = arCubeful[0];
        float dt_m = arCubeful[1];
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                 cube.cube_value, cube.match.is_crawford);
        result.should_double = true;
        result.should_take = (dt_m <= dp_m_val);
        result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
    } else {
        // Match: compare in MWC space (canonical for match play decisions)
        float nd_m = arCubeful[0];
        float dt_m = arCubeful[1];
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                 cube.cube_value, cube.match.is_crawford);
        float best_double_mwc = std::min(dt_m, dp_m_val);
        result.should_double = (best_double_mwc > nd_m);
        result.should_take = (dt_m <= dp_m_val);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }
    }

    return result;
}

} // namespace bgbot
