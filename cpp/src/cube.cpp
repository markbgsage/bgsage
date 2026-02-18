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

// Temporary debug flag for cubeful equity tracing
static bool g_cubeful_debug = false;
static int g_cubeful_debug_depth = 0;

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
    // GNUbg also checks -2,-2 score (both 2-away) as dead
    if (cube.match.away1 == 2 && cube.match.away2 == 2)
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

    // Gammon/backgammon ratios (same decomposition as money game)
    float rG0, rBG0, rG1, rBG1;
    if (p_win > 1e-7f) {
        rG0 = (probs[1] - probs[2]) / p_win;   // gammon ratio of wins
        rBG0 = probs[2] / p_win;                // backgammon ratio of wins
    } else {
        rG0 = 0.0f;
        rBG0 = 0.0f;
    }
    if (p_win < 1.0f - 1e-7f) {
        rG1 = (probs[3] - probs[4]) / (1.0f - p_win);  // gammon ratio of losses
        rBG1 = probs[4] / (1.0f - p_win);               // backgammon ratio of losses
    } else {
        rG1 = 0.0f;
        rBG1 = 0.0f;
    }

    // Dead cube MWC
    float eq_dead = cubeless_equity(probs);
    float mwc_dead = eq2mwc(eq_dead, away1, away2, cv, craw);

    // MET lookups for various outcomes at current cube value
    // Win outcomes: single, gammon, backgammon (player wins cv, 2cv, 3cv points)
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    // Loss outcomes
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    // Player's cash point (MWC when player wins cv points = D/P from player's side)
    float mwc_cash = mwc_win_s;

    // Opponent's cash point (MWC when opponent wins cv points = D/P from opp's side)
    float mwc_opp_cash = mwc_lose_s;

    // MWC when player wins ALL types (weighted by gammon ratios)
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;

    // MWC when player loses ALL types (weighted)
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Compute take points in P(win) space using MWC anchor points:
    // Opponent's take/too-good point: below this P(win), opponent should pass
    // (same as money game TP, but in MWC-space terms)
    float opp_tg, player_tg;

    // Opponent's take point: from opponent's perspective, they pass when
    // the doubler's equity is better than D/P. Using MWC:
    // opp_tg = P(win) where opponent is indifferent between taking and passing
    float denom_opp = mwc_cash - mwc_lose_all;
    if (std::abs(denom_opp) > 1e-10f) {
        opp_tg = (mwc_opp_cash - mwc_lose_all) / denom_opp;
        // Clamp to sensible range
        opp_tg = std::clamp(opp_tg, 0.0f, 1.0f);
    } else {
        opp_tg = 0.0f;
    }

    // Player's too-good point: above this P(win), player should play on for gammon
    float denom_player = mwc_win_all - mwc_lose_all;
    if (std::abs(denom_player) > 1e-10f) {
        player_tg = (mwc_cash - mwc_lose_all) / denom_player;
        player_tg = std::clamp(player_tg, 0.0f, 1.0f);
    } else {
        player_tg = 1.0f;
    }

    // Piecewise-linear live cube MWC (3 regions, same as GNUbg Cl2CfMatchCentered)
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

    // MET lookups
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_cash = mwc_win_s;  // Player cashes = wins cv points

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Player's cash/too-good point
    float denom = mwc_win_all - mwc_lose_all;
    float player_tg;
    if (std::abs(denom) > 1e-10f) {
        player_tg = (mwc_cash - mwc_lose_all) / denom;
        player_tg = std::clamp(player_tg, 0.0f, 1.0f);
    } else {
        player_tg = 1.0f;
    }

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

    // MET lookups
    float mwc_win_s  = get_met_after(away1, away2, cv, true, craw);
    float mwc_win_g  = get_met_after(away1, away2, 2*cv, true, craw);
    float mwc_win_b  = get_met_after(away1, away2, 3*cv, true, craw);
    float mwc_lose_s = get_met_after(away1, away2, cv, false, craw);
    float mwc_lose_g = get_met_after(away1, away2, 2*cv, false, craw);
    float mwc_lose_b = get_met_after(away1, away2, 3*cv, false, craw);

    float mwc_opp_cash = mwc_lose_s;  // Opponent cashes = player loses cv points

    // Weighted MWCs
    float mwc_win_all = (1.0f - rG0 - rBG0) * mwc_win_s
                      + rG0 * mwc_win_g + rBG0 * mwc_win_b;
    float mwc_lose_all = (1.0f - rG1 - rBG1) * mwc_lose_s
                       + rG1 * mwc_lose_g + rBG1 * mwc_lose_b;

    // Opponent's take/too-good point (from player's perspective)
    float denom = mwc_opp_cash - mwc_lose_all;
    float opp_tg;
    // opp_tg = P(win) below which opponent should cash (pass a double they would give)
    // Using same formula as centered case for the opponent's region:
    float denom2 = mwc_win_s - mwc_lose_all;  // cash - lose
    if (std::abs(denom2) > 1e-10f) {
        opp_tg = (mwc_opp_cash - mwc_lose_all) / denom2;
        opp_tg = std::clamp(opp_tg, 0.0f, 1.0f);
    } else {
        opp_tg = 0.0f;
    }

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

// Internal recursive cubeful equity evaluation (MONEY GAME).
//
// Computes cubeful equity for a PRE-ROLL position from the roller's perspective.
// `owner` is cube ownership from the roller's perspective.
//
// At 0-ply: get cubeless pre-roll probs (flip → evaluate → invert), then Janowski.
// At N-ply: for each roll, find best checkerplay move (cubeless), then the
//   opponent has a pre-roll position where they may double. Evaluate the
//   opponent's position at (N-1) ply for both ND and DT cube states. The
//   opponent picks the cube action that maximizes their equity.
//
// Returns cubeful equity normalized to cube value 1, from roller's perspective.
static float cubeful_equity_recursive(
    const Board& board,       // pre-roll, roller's perspective
    CubeOwner owner,          // from roller's perspective
    const Strategy& strategy,
    int plies,
    const MoveFilter& filter,
    int n_threads,
    bool allow_parallel)
{
    // Get cubeless pre-roll probs: flip → evaluate (post-move semantics) → invert
    Board flipped = flip(board);
    bool race = is_race(board);

    // Terminal check on the flipped board (did previous mover already win?)
    GameResult result = check_game_over(flipped);
    if (result != GameResult::NOT_OVER) {
        // Terminal from previous mover's perspective → invert for current roller
        auto t_probs = invert_probs(terminal_probs(result));
        return cubeless_equity(t_probs);  // dead cube at terminal
    }

    if (plies <= 0) {
        // 0-ply: Janowski conversion
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float x = cube_efficiency(board, race);
        return cl2cf_money(pre_roll_probs, owner, x);
    }

    // N-ply: loop over 21 rolls for the player on roll
    // Ownership from opponent's perspective after the player moves
    CubeOwner opp_owner = flip_owner(owner);

    // Lambda to evaluate a single roll. Returns weighted equity contribution.
    auto evaluate_roll = [&](int roll_idx) -> double {
        const auto& roll = ALL_ROLLS[roll_idx];

        thread_local std::vector<Board> candidates;
        candidates.clear();
        if (candidates.capacity() < 32) candidates.reserve(32);

        // Generate the roller's legal moves
        possible_boards(board, roll.d1, roll.d2, candidates);

        // Find best move (cubeless) — simple: evaluate each, pick highest equity
        int best_idx = 0;
        if (candidates.size() > 1) {
            double best_eq = -1e30;
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                auto p = strategy.evaluate_probs(candidates[i], board);
                double eq = NeuralNetwork::compute_equity(p);
                if (eq > best_eq) {
                    best_eq = eq;
                    best_idx = i;
                }
            }
        }

        Board post_move = candidates[best_idx];

        // Check if the game is over (player won by bearing off all)
        GameResult post_result = check_game_over(post_move);
        if (post_result != GameResult::NOT_OVER) {
            auto tp = terminal_probs(post_result);
            return roll.weight * static_cast<double>(cubeless_equity(tp));
        }

        // After the player moves, it's the opponent's turn.
        Board opp_pre_roll = flip(post_move);

        // Can the opponent double?
        bool opp_can_double = (opp_owner == CubeOwner::CENTERED ||
                               opp_owner == CubeOwner::PLAYER);

        // Evaluate ND: opponent doesn't double, plays with current cube state
        // Recursive calls are always serial (parallelism only at top level)
        float opp_eq_nd = cubeful_equity_recursive(
            opp_pre_roll, opp_owner, strategy, plies - 1, filter,
            n_threads, /*allow_parallel=*/false);

        float player_eq_nd = -opp_eq_nd;
        float player_eq_for_roll;

        if (opp_can_double) {
            CubeOwner dt_opp_owner = CubeOwner::OPPONENT;
            float opp_eq_dt = cubeful_equity_recursive(
                opp_pre_roll, dt_opp_owner, strategy, plies - 1, filter,
                n_threads, /*allow_parallel=*/false);

            float player_eq_dt = -opp_eq_dt * 2.0f;
            float player_eq_dp = -1.0f;

            float opp_dt_scaled = opp_eq_dt * 2.0f;
            float opp_dp = 1.0f;
            float opp_best_if_double = std::min(opp_dt_scaled, opp_dp);
            bool opp_should_double = (opp_best_if_double > opp_eq_nd);

            if (opp_should_double) {
                if (player_eq_dt >= player_eq_dp) {
                    player_eq_for_roll = player_eq_dt;
                } else {
                    player_eq_for_roll = player_eq_dp;
                }
            } else {
                player_eq_for_roll = player_eq_nd;
            }
        } else {
            player_eq_for_roll = player_eq_nd;
        }

        return roll.weight * static_cast<double>(player_eq_for_roll);
    };

    double sum_equity = 0.0;

    if (allow_parallel && n_threads > 1 && plies > 1) {
        // Parallel: fan out across 21 rolls using the shared thread pool
        std::array<double, 21> roll_equities{};
        multipy_parallel_for(21, n_threads, [&](int idx) {
            roll_equities[idx] = evaluate_roll(idx);
        });
        for (int i = 0; i < 21; ++i) {
            sum_equity += roll_equities[i];
        }
    } else {
        // Serial
        for (int i = 0; i < 21; ++i) {
            sum_equity += evaluate_roll(i);
        }
    }

    return static_cast<float>(sum_equity / 36.0);
}

// Internal recursive cubeful evaluation (MATCH PLAY).
//
// Same structure as the money game version, but operates in MWC (Match Winning
// Chance) space instead of equity space.
//
// `cube` has cube ownership from the roller's perspective and match state
// (away1=roller's away, away2=opponent's away).
//
// Returns cubeful MWC from the roller's perspective.
static float cubeful_mwc_recursive(
    const Board& board,        // pre-roll, roller's perspective
    const CubeInfo& cube,      // from roller's perspective
    const Strategy& strategy,
    int plies,
    const MoveFilter& filter,
    int n_threads,
    bool allow_parallel)
{
    Board flipped = flip(board);
    bool race = is_race(board);

    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    // Terminal check
    GameResult result = check_game_over(flipped);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = invert_probs(terminal_probs(result));
        // Terminal: compute MWC from terminal probs (cubeless, dead cube)
        return cubeless_mwc(t_probs, away1, away2, cv, craw);
    }

    if (plies <= 0) {
        // 0-ply: Janowski conversion in MWC space
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float x = cube_efficiency(board, race);
        return cl2cf_match(pre_roll_probs, cube, x);
    }

    // Opponent's cube info (flipped perspective)
    CubeInfo opp_cube;
    opp_cube.cube_value = cv;
    opp_cube.owner = flip_owner(cube.owner);
    opp_cube.match = cube.match.flip();  // Swap away1/away2

    // Lambda to evaluate a single roll. Returns weighted MWC contribution.
    auto evaluate_roll = [&](int roll_idx) -> double {
        const auto& roll = ALL_ROLLS[roll_idx];

        thread_local std::vector<Board> candidates;
        candidates.clear();
        if (candidates.capacity() < 32) candidates.reserve(32);

        // Generate the roller's legal moves
        possible_boards(board, roll.d1, roll.d2, candidates);

        // Find best move (cubeless)
        int best_idx = 0;
        if (candidates.size() > 1) {
            double best_eq = -1e30;
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                auto p = strategy.evaluate_probs(candidates[i], board);
                double eq = NeuralNetwork::compute_equity(p);
                if (eq > best_eq) {
                    best_eq = eq;
                    best_idx = i;
                }
            }
        }

        Board post_move = candidates[best_idx];

        // Check if game is over
        GameResult post_result = check_game_over(post_move);
        if (post_result != GameResult::NOT_OVER) {
            auto tp = terminal_probs(post_result);
            // Terminal MWC from player's perspective
            return roll.weight * static_cast<double>(
                cubeless_mwc(tp, away1, away2, cv, craw));
        }

        // Opponent's turn
        Board opp_pre_roll = flip(post_move);

        // Can opponent double?
        bool opp_can_double = can_double(opp_cube);

        // Opponent's ND MWC (from opponent's perspective)
        float opp_mwc_nd = cubeful_mwc_recursive(
            opp_pre_roll, opp_cube, strategy, plies - 1, filter,
            n_threads, /*allow_parallel=*/false);

        // Player's MWC if opponent doesn't double
        float player_mwc_nd = 1.0f - opp_mwc_nd;
        float player_mwc_for_roll;

        if (opp_can_double) {
            // Opponent DT: doubled cube, opponent's opponent (= player) owns
            CubeInfo opp_dt_cube;
            opp_dt_cube.cube_value = 2 * cv;
            opp_dt_cube.owner = CubeOwner::OPPONENT;  // Player owns from opp's perspective
            opp_dt_cube.match = opp_cube.match;

            float opp_mwc_dt = cubeful_mwc_recursive(
                opp_pre_roll, opp_dt_cube, strategy, plies - 1, filter,
                n_threads, /*allow_parallel=*/false);

            // Opponent's DP MWC (opponent wins cv points)
            float opp_mwc_dp = dp_mwc(opp_cube.match.away1, opp_cube.match.away2,
                                       cv, craw);

            // Opponent decides: double if max(dt, dp) > nd (maximizing MWC)
            float opp_best_if_double = std::max(opp_mwc_dt, opp_mwc_dp);
            bool opp_should_double = (opp_best_if_double > opp_mwc_nd);

            if (opp_should_double) {
                // Player's response to opponent's double:
                // Player's MWC if taking = 1 - opp_mwc_dt
                // Player's MWC if passing = 1 - opp_mwc_dp
                float player_mwc_dt = 1.0f - opp_mwc_dt;
                float player_mwc_dp = 1.0f - opp_mwc_dp;
                // Player maximizes their MWC
                player_mwc_for_roll = std::max(player_mwc_dt, player_mwc_dp);
            } else {
                player_mwc_for_roll = player_mwc_nd;
            }
        } else {
            player_mwc_for_roll = player_mwc_nd;
        }

        return roll.weight * static_cast<double>(player_mwc_for_roll);
    };

    double sum_mwc = 0.0;

    if (allow_parallel && n_threads > 1 && plies > 1) {
        std::array<double, 21> roll_mwcs{};
        multipy_parallel_for(21, n_threads, [&](int idx) {
            roll_mwcs[idx] = evaluate_roll(idx);
        });
        for (int i = 0; i < 21; ++i) {
            sum_mwc += roll_mwcs[i];
        }
    } else {
        for (int i = 0; i < 21; ++i) {
            sum_mwc += evaluate_roll(i);
        }
    }

    return static_cast<float>(sum_mwc / 36.0);
}

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads)
{
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    return cubeful_equity_recursive(board, owner, strategy, n_plies, filter,
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
    if (cube.is_money()) {
        return cubeful_equity_nply(board, cube.owner, strategy, n_plies,
                                   filter, n_threads);
    }
    bool allow_parallel = (n_threads > 1 && n_plies > 1);
    float mwc = cubeful_mwc_recursive(board, cube, strategy, n_plies, filter,
                                       n_threads, allow_parallel);
    return mwc2eq(mwc, cube.match.away1, cube.match.away2,
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

    if (cube.is_money()) {
        // Money game N-ply (existing logic)
        CubeDecision result;
        result.equity_dp = 1.0f;

        result.equity_nd = cubeful_equity_nply(board, cube.owner, strategy,
                                                n_plies, filter, n_threads);
        result.equity_dt = 2.0f * cubeful_equity_nply(
            board, CubeOwner::OPPONENT, strategy, n_plies, filter, n_threads);

        float best_double = std::min(result.equity_dt, result.equity_dp);
        result.should_double = (best_double > result.equity_nd);
        result.should_take = (result.equity_dt <= result.equity_dp);

        if (result.should_double) {
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            result.optimal_equity = result.equity_nd;
        }

        return result;
    }

    // Match play N-ply: work in MWC space, convert to equity at the end
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    bool allow_parallel = (n_threads > 1 && n_plies > 1);

    CubeDecision result;

    // DP MWC: opponent passes, player wins cv points
    float dp_m = dp_mwc(away1, away2, cv, craw);

    // ND MWC: cubeful MWC with current cube state
    float nd_m = cubeful_mwc_recursive(board, cube, strategy, n_plies, filter,
                                        n_threads, allow_parallel);

    // DT MWC: cube is doubled and opponent takes
    CubeInfo dt_cube;
    dt_cube.cube_value = 2 * cv;
    dt_cube.owner = CubeOwner::OPPONENT;
    dt_cube.match = cube.match;
    float dt_m = cubeful_mwc_recursive(board, dt_cube, strategy, n_plies, filter,
                                        n_threads, allow_parallel);

    // Convert all three MWC values to equity at the original cube value
    result.equity_nd = mwc2eq(nd_m, away1, away2, cv, craw);
    result.equity_dt = mwc2eq(dt_m, away1, away2, cv, craw);
    result.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);

    // Decision logic respecting can_double
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
