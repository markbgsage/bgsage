#include "bgbot/cube.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <array>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace bgbot {

// Profiling counters for cubeful recursion (for diagnostics)
static std::atomic<int64_t> g_leaf_count{0};
static std::atomic<int64_t> g_internal_count{0};
static std::atomic<int64_t> g_cache_hit_count{0};
static std::atomic<int64_t> g_move_gen_count{0};
static std::atomic<int64_t> g_total_candidates{0};

void reset_cubeful_counters() {
    g_leaf_count = 0;
    g_internal_count = 0;
    g_cache_hit_count = 0;
    g_move_gen_count = 0;
    g_total_candidates = 0;
}

void print_cubeful_counters() {
    printf("  Leaf evaluations: %lld\n", (long long)g_leaf_count.load());
    printf("  Internal nodes:   %lld\n", (long long)g_internal_count.load());
    printf("  Cache hits:       %lld\n", (long long)g_cache_hit_count.load());
    printf("  Move gen calls:   %lld\n", (long long)g_move_gen_count.load());
    printf("  Total candidates: %lld (avg %.1f)\n",
           (long long)g_total_candidates.load(),
           g_move_gen_count.load() > 0
               ? (double)g_total_candidates.load() / g_move_gen_count.load()
               : 0.0);
}

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

float money_live(float W, float L, float p_win, CubeOwner owner,
                 bool jacoby) {
    // Take point and cash point (live cube)
    float TP = (L - 0.5f) / (W + L + 0.5f);
    float CP = (L + 1.0f) / (W + L + 0.5f);

    float p = p_win;

    switch (owner) {
        case CubeOwner::CENTERED: {
            // Interpolate through (0, -L), (TP, -1), (CP, +1), (1, +W)
            //
            // Under Jacoby, the extreme segments are clamped to ±1.0
            // (matching GNUbg's MoneyLive).  Rationale: above the cash
            // point, the player doubles and the opponent passes → +1.0;
            // below the take point, the opponent doubles and the player
            // passes → -1.0.  There is no "too good" / "too weak" premium
            // because gammons are worthless while the cube is centered.
            // The middle segment (doubling window) is unchanged because
            // once the cube is turned, gammons count — the real W/L apply.
            if (p < TP) {
                // Segment: (0, -L) → (TP, -1)
                return jacoby ? -1.0f
                              : -L + (-1.0f + L) * p / TP;
            } else if (p < CP) {
                // Segment: (TP, -1) → (CP, +1)
                return -1.0f + 2.0f * (p - TP) / (CP - TP);
            } else {
                // Segment: (CP, +1) → (1, +W)
                return jacoby ? 1.0f
                              : 1.0f + (W - 1.0f) * (p - CP) / (1.0f - CP);
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
    // Always compute real W/L from the full gammon/backgammon rates.
    // The live-cube equity uses real W/L because once the cube is turned,
    // gammons count (Jacoby deactivates).
    compute_WL(probs, W, L);

    // Dead-cube equity: always full cubeless equity (gammons/backgammons
    // included).  "Dead cube" in Janowski means the cube has been turned
    // but can never be turned again — after the double, Jacoby deactivates
    // and gammons count at the cube value.
    float e_dead = cubeless_equity(probs);

    // Live-cube equity: uses real W/L but with Jacoby clamping on the
    // extreme segments (p < TP → -1, p > CP → +1).  Matching GNUbg's
    // MoneyLive: the middle segment (doubling window) is unchanged because
    // doubling activates gammons; the extreme segments are clamped because
    // there is no "too good/too weak" premium under Jacoby (you'd just
    // double to cash or get doubled out).
    float e_live = money_live(W, L, probs[0], owner, jacoby_active);

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

// Compute live-cube cash points for both players simultaneously using
// GNUbg's recursive GetPoints algorithm.
//
// The algorithm works from the "dead" cube level (highest level where at
// least one player can't survive a recube) DOWN to the current cube level,
// computing cash points recursively.
//
// At dead cube levels: CP = (DTL - DP) / (DTL - DTW)
//   with the doubler's own prime for BOTH DTW and DTL.
// At live cube levels: CP = 1 - opp_CP_above * (DP - DTW) / (rRDP - DTW)
//   where rRDP = MWC after losing at the doubled value (recube-drop payoff).
//
// Parameters use 1-indexed away values. Gammon ratios are from the player's
// perspective: rG0/rBG0 for player's wins, rG1/rBG1 for player's losses.
static void get_match_points(
    int away1, int away2, int cv, bool craw,
    float rG0, float rBG0,        // gammon/bg ratios for player's wins
    float rG1, float rBG1,        // gammon/bg ratios for player's losses
    float& player_cp,             // out: player's cash point
    float& opp_cp)                // out: opponent's cash point
{
    // 0-indexed away values (GNUbg convention)
    int idx0 = away1 - 1;
    int idx1 = away2 - 1;

    // Gammon/bg ratios per side (0=player, 1=opponent)
    float arG[2] = {rG0, rG1};
    float arBG[2] = {rBG0, rBG1};

    // Find the dead cube level: highest cube where both can survive a recube
    int nDead = cv;
    int nMax = 0;
    while (idx0 >= 2 * nDead && idx1 >= 2 * nDead) {
        nMax++;
        nDead *= 2;
    }

    // Cash points at each level (k=0: player, k=1: opponent)
    constexpr int MAX_LEVELS = 16;
    float arCPLive[2][MAX_LEVELS] = {};

    for (int nCubeValue = nDead, n = nMax; nCubeValue >= cv;
         nCubeValue >>= 1, n--) {
        // Auto-redouble primes at this cube level (0-indexed comparisons)
        int prime[2];
        prime[0] = (idx0 < 2*nCubeValue && idx1 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;
        prime[1] = (idx1 < 2*nCubeValue && idx0 >= 2*nCubeValue)
                   ? 2*nCubeValue : nCubeValue;

        for (int k = 0; k < 2; k++) {
            // k=0: player doubles, k=1: opponent doubles
            int away_d = (k == 0) ? away1 : away2;
            int away_t = (k == 0) ? away2 : away1;
            int id = (k == 0) ? idx0 : idx1;
            int it = (k == 0) ? idx1 : idx0;

            float gW = arG[k], bgW = arBG[k];       // doubler's wins
            float gL = arG[1-k], bgL = arBG[1-k];   // doubler's losses

            if (id < 2*nCubeValue || it < 2*nCubeValue) {
                // Dead cube: at least one player can't survive a recube.
                // Use doubler's own prime for BOTH DTW and DTL.
                // The effective cube after DT (with possible auto-redouble) is
                // 2*prime[k], so outcomes are at 2*p, 4*p, 6*p (not p, 2p, 3p).
                int p = prime[k];
                int dp = 2 * p;  // effective cube level after DT + auto-redouble

                float rDP = get_met_after(away_d, away_t, nCubeValue, true, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dp, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dp, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dp, true, craw);

                float rDTL = (1.0f - gL - bgL)
                                * get_met_after(away_d, away_t, dp, false, craw)
                           + gL * get_met_after(away_d, away_t, 2*dp, false, craw)
                           + bgL * get_met_after(away_d, away_t, 3*dp, false, craw);

                float denom = rDTL - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        (rDTL - rDP) / denom, 0.0f, 1.0f);
            } else {
                // Live cube: both players can survive a recube.
                // Use standard doubled value, recursive formula.
                int dcv = 2 * nCubeValue;

                float rDP = get_met_after(away_d, away_t,
                                          nCubeValue, true, craw);
                float rRDP = get_met_after(away_d, away_t,
                                           dcv, false, craw);

                float rDTW = (1.0f - gW - bgW)
                                * get_met_after(away_d, away_t, dcv, true, craw)
                           + gW * get_met_after(away_d, away_t, 2*dcv, true, craw)
                           + bgW * get_met_after(away_d, away_t, 3*dcv, true, craw);

                float denom = rRDP - rDTW;
                if (std::abs(denom) < 1e-10f)
                    arCPLive[k][n] = 0.5f;
                else
                    arCPLive[k][n] = std::clamp(
                        1.0f - arCPLive[1-k][n+1] * (rDP - rDTW) / denom,
                        0.0f, 1.0f);
            }
        }
    }

    player_cp = arCPLive[0][0];
    opp_cp = arCPLive[1][0];
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

    // Compute live-cube cash points using recursive algorithm (GNUbg GetPoints).
    float player_cp, opp_cp;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp, opp_cp);

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

    // Player's cash point using recursive algorithm (GNUbg GetPoints)
    float player_cp_val, opp_cp_val;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp_val, opp_cp_val);
    float player_tg = player_cp_val;

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

    // Opponent's cash point using recursive algorithm (GNUbg GetPoints)
    float player_cp_val, opp_cp_val;
    get_match_points(away1, away2, cv, craw, rG0, rBG0, rG1, rBG1,
                     player_cp_val, opp_cp_val);
    float opp_tg = 1.0f - opp_cp_val;

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
    // Dead cube: return cubeless equity directly (no Janowski interpolation)
    if (cube_is_dead(cube)) {
        if (cube.is_money()) {
            return cubeless_equity(probs);
        }
        // Match play: return cubeless MWC converted to equity
        float mwc = cubeless_mwc(probs, cube.match.away1, cube.match.away2,
                                  cube.cube_value, cube.match.is_crawford);
        return mwc2eq(mwc, cube.match.away1, cube.match.away2,
                      cube.cube_value, cube.match.is_crawford);
    }

    if (cube.is_money()) {
        return cl2cf_money(probs, cube.owner, cube_x, cube.jacoby_active());
    }
    // Match play: cl2cf_match returns MWC, convert to equity
    float mwc = cl2cf_match(probs, cube, cube_x);
    return mwc2eq(mwc, cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

float cube_efficiency(
    const std::array<float, NUM_OUTPUTS>& /*probs*/,
    bool is_race_pos,
    int player_pips,
    int opponent_pips)
{
    // Current implementation ignores probs; reserved for a future ML formula
    // trained on (probs, is_race, pips) -> implied Janowski cube life index.
    if (!is_race_pos) {
        return 0.68f;  // Contact/crashed
    }
    // Race: linear in average pip count (both players), clamped to [0.6, 0.7].
    // Using the average instead of just the roller's pips makes the Janowski
    // cubeful conversion perspective-independent, eliminating a systematic
    // bias at odd ply levels in N-ply match play evaluation.
    float avg_pips = static_cast<float>(player_pips + opponent_pips) * 0.5f;
    float x = 0.55f + 0.00125f * avg_pips;
    return std::clamp(x, 0.6f, 0.7f);
}

// Money game cube decision (1-ply / raw NN).
static CubeDecision cube_decision_1ply_money(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    CubeDecision result;
    result.equity_dp = 1.0f;  // Double/Pass: always +1.0 for money games

    // When cube is dead (max_cube_value reached), no cube actions possible.
    // ND = cubeless equity, DT/DP are degenerate, should_double = false.
    if (cube_is_dead(cube)) {
        float cl_eq = cubeless_equity(probs);
        result.equity_nd = cl_eq;
        result.equity_dt = cl_eq;  // Degenerate — can't double
        result.equity_dp = 1.0f;
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = cl_eq;
        result.is_beaver = false;
        return result;
    }

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

// Match play cube decision (1-ply / raw NN).
// Computes ND/DT/DP in MWC space, then converts to equity at original cube value.
static CubeDecision cube_decision_1ply_match(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    int away1 = cube.match.away1;
    int away2 = cube.match.away2;
    int cv = cube.cube_value;
    bool craw = cube.match.is_crawford;

    CubeDecision result;

    // When cube is dead via max_cube_value, return cubeless MWC as equity.
    if (cube_is_dead(cube)) {
        float mwc = cubeless_mwc(probs, away1, away2, cv, craw);
        float eq = mwc2eq(mwc, away1, away2, cv, craw);
        result.equity_nd = eq;
        result.equity_dt = eq;
        result.equity_dp = mwc2eq(dp_mwc(away1, away2, cv, craw), away1, away2, cv, craw);
        result.should_double = false;
        result.should_take = true;
        result.optimal_equity = eq;
        result.is_beaver = false;
        return result;
    }

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

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    float cube_x)
{
    if (cube.is_money()) {
        return cube_decision_1ply_money(probs, cube, cube_x);
    }
    return cube_decision_1ply_match(probs, cube, cube_x);
}

CubeDecision cube_decision_1ply(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube,
    const Board& board,
    bool is_race_pos)
{
    float x;
    if (cube.cube_x_override >= 0.0f) {
        x = cube.cube_x_override;
    } else {
        auto [pp, op] = pip_counts(board);
        x = cube_efficiency(probs, is_race_pos, pp, op);
    }
    return cube_decision_1ply(probs, cube, x);
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
// emerge from the tree values rather than from 1-ply heuristic predictions.
//
// For money: operates in equity space (normalized to cube=1).
// For match: operates in MWC space internally.

// Max cube states supported. Doubles per ply level, so 64 supports 6+ ply.
static constexpr int MAX_CCI = 64;
static constexpr bool kEnableCubefulCache = true;
static constexpr int MAX_CUBEFUL_CACHE_CCI = 16;

static inline std::size_t hash_combine(std::size_t seed, std::size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// Open-addressing cubeful cache: fixed-size table with linear probing.
// Much faster than unordered_map for this workload.
//
// Uses a global epoch counter to invalidate all thread-local caches at once.
// begin_cubeful_cache_epoch() increments the global epoch; each cache entry
// stores the epoch it was written at. Lookups reject entries from stale epochs.
// This solves the cross-thread invalidation problem: worker threads in the
// persistent thread pool retain their thread_local caches across calls, but
// stale entries are automatically rejected when the epoch changes.
struct CubefulCacheEntry {
    Board board;
    int plies;
    int cci;
    bool fTop;
    bool occupied;
    uint64_t epoch;
    float values[MAX_CCI];
};

static constexpr int CUBEFUL_CACHE_SIZE = 8192;   // must be power of 2
static constexpr int CUBEFUL_CACHE_MASK = CUBEFUL_CACHE_SIZE - 1;
static_assert((CUBEFUL_CACHE_SIZE & (CUBEFUL_CACHE_SIZE - 1)) == 0, "Must be power of 2");

// Global epoch counter — incremented by begin_cubeful_cache_epoch() to
// invalidate all thread-local cubeful caches without clearing them.
static std::atomic<uint64_t> g_cubeful_cache_epoch{1};

struct CubefulOpenCache {
    CubefulCacheEntry entries[CUBEFUL_CACHE_SIZE];

    void clear() {
        // Zero-fill sets all occupied=false
        std::memset(entries, 0, sizeof(entries));
    }

    static std::size_t hash_board(const Board& board, int plies, int cci, bool fTop) {
        // Fast hash: XOR board values with rotations + plies/cci/fTop
        std::size_t h = static_cast<std::size_t>(plies) * 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<std::size_t>(cci) * 0x517cc1b727220a95ULL;
        h ^= static_cast<std::size_t>(fTop) * 0x6c62272e07bb0142ULL;
        for (int i = 0; i < 26; ++i) {
            h ^= static_cast<std::size_t>(static_cast<unsigned int>(board[i]))
                 * (0x9e3779b97f4a7c15ULL + static_cast<std::size_t>(i) * 7);
            h = (h << 7) | (h >> 57);  // rotate
        }
        return h;
    }

    bool get(const Board& board, int plies, int cci, bool fTop, uint64_t epoch, float* out) const {
        std::size_t idx = hash_board(board, plies, cci, fTop) & CUBEFUL_CACHE_MASK;
        // Linear probe up to 4 slots
        for (int probe = 0; probe < 4; ++probe) {
            const auto& e = entries[(idx + probe) & CUBEFUL_CACHE_MASK];
            if (!e.occupied || e.epoch != epoch) continue;
            if (e.plies == plies && e.cci == cci && e.fTop == fTop && e.board == board) {
                for (int i = 0; i < cci; ++i) out[i] = e.values[i];
                return true;
            }
        }
        return false;
    }

    void put(const Board& board, int plies, int cci, bool fTop, uint64_t epoch, const float* values) {
        std::size_t idx = hash_board(board, plies, cci, fTop) & CUBEFUL_CACHE_MASK;
        // Linear probe up to 4 slots, replace first empty/stale or matching slot
        for (int probe = 0; probe < 4; ++probe) {
            auto& e = entries[(idx + probe) & CUBEFUL_CACHE_MASK];
            if (!e.occupied || e.epoch != epoch ||
                (e.plies == plies && e.cci == cci && e.fTop == fTop && e.board == board)) {
                e.board = board;
                e.plies = plies;
                e.cci = cci;
                e.fTop = fTop;
                e.occupied = true;
                e.epoch = epoch;
                for (int i = 0; i < cci; ++i) e.values[i] = values[i];
                return;
            }
        }
        // All 4 probe slots occupied by current-epoch entries: evict first
        auto& e = entries[idx & CUBEFUL_CACHE_MASK];
        e.board = board;
        e.plies = plies;
        e.cci = cci;
        e.fTop = fTop;
        e.occupied = true;
        e.epoch = epoch;
        for (int i = 0; i < cci; ++i) e.values[i] = values[i];
    }
};

static thread_local CubefulOpenCache g_cubeful_cache;

static void begin_cubeful_cache_epoch() {
    g_cubeful_cache_epoch.fetch_add(1, std::memory_order_relaxed);
}

static bool get_cached_cubeful(
    const Board& board,
    const CubeInfo aciCubePos[],
    int cci,
    const Strategy& strategy,
    int plies,
    bool fTop,
    float arCubeful[])
{
    if (!kEnableCubefulCache || cci <= 0) return false;
    uint64_t epoch = g_cubeful_cache_epoch.load(std::memory_order_relaxed);
    return g_cubeful_cache.get(board, plies, cci, fTop, epoch, arCubeful);
}

static void put_cached_cubeful(
    const Board& board,
    const CubeInfo aciCubePos[],
    int cci,
    const Strategy& strategy,
    int plies,
    bool fTop,
    const float arCubeful[])
{
    if (!kEnableCubefulCache || cci <= 0) return;
    uint64_t epoch = g_cubeful_cache_epoch.load(std::memory_order_relaxed);
    g_cubeful_cache.put(board, plies, cci, fTop, epoch, arCubeful);
}

// Resolve cube efficiency: use override if set, otherwise auto-detect.
static float resolve_cube_x(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube, const Board& board, bool race)
{
    if (cube.cube_x_override >= 0.0f) return cube.cube_x_override;
    auto [pp, op] = pip_counts(board);
    return cube_efficiency(probs, race, pp, op);
}

// Helper: evaluate a pre-roll board at 1-ply Janowski cubeful.
// Returns value from the ROLLER's perspective.
static float eval_pre_roll_cubeful_1ply(
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
    float x = resolve_cube_x(pre_roll_probs, cube, board, race);

    if (cube.is_money()) {
        return cl2cf_money(pre_roll_probs, cube.owner, x, cube.jacoby_active());
    } else {
        return cl2cf_match(pre_roll_probs, cube, x);
    }
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
    const GamePlanStrategy* base_gps,
    int plies,
    const MoveFilter& filter,
    int n_threads,
    bool allow_parallel,
    bool fTop,
    float arCubeful[],
    const Strategy* move_filter = nullptr)
{
    bool is_money = (cci > 0 && aciCubePos[0].cube_value > 0)
                    ? aciCubePos[0].is_money() : true;

    if (get_cached_cubeful(board, aciCubePos, cci, strategy, plies, fTop, arCubeful)) {
        g_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    // Terminal check
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = terminal_probs(result);
        for (int ici = 0; ici < cci; ici++) {
            if (aciCubePos[ici].cube_value <= 0) {
                arCubeful[ici] = 0.0f;
                continue;
            }
            // Dead cube (max_cube_value reached): pure cubeless equity, gammons count
            if (cube_is_dead(aciCubePos[ici])) {
                arCubeful[ici] = cubeless_equity(t_probs);
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
        put_cached_cubeful(board, aciCubePos, cci, strategy, plies, fTop, arCubeful);
        return;
    }

    // --- Leaf node (1-ply): NN evaluation + Janowski ---
    g_leaf_count.fetch_add(1, std::memory_order_relaxed);
    if (plies <= 1) {
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float default_x = resolve_cube_x(
            pre_roll_probs,
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
            // Dead cube: skip Janowski, return pure cubeless equity
            if (cube_is_dead(aci[i])) {
                arCf[i] = cubeless_equity(pre_roll_probs);
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
        put_cached_cubeful(board, aciCubePos, cci, strategy, plies, fTop, arCubeful);
        return;
    }

    // --- Internal node (plies > 0): recurse over 21 rolls ---
    g_internal_count.fetch_add(1, std::memory_order_relaxed);

    // Expand cube states for recursion (fInvert=true: flip to opponent's perspective)
    CubeInfo aci[MAX_CCI * 2];
    make_cube_pos(aciCubePos, cci, fTop, aci, true);
    int expanded_cci = 2 * cci;

    // Accumulators for weighted cubeful equities
    float arCf[MAX_CCI * 2] = {};

    // Lambda to evaluate a single dice roll
    auto evaluate_roll = [&](int roll_idx, float arCfLocal[]) {
        const auto& roll = ALL_ROLLS[roll_idx];
        const bool child_allow_parallel = allow_parallel && (plies - 1 > 2);

        // Generate legal moves
        thread_local std::vector<Board> candidates;
        candidates.clear();
        if (candidates.capacity() < 32) candidates.reserve(32);
        possible_boards(board, roll.d1, roll.d2, candidates);
        int n_cand = static_cast<int>(candidates.size());
        g_move_gen_count.fetch_add(1, std::memory_order_relaxed);
        g_total_candidates.fetch_add(n_cand, std::memory_order_relaxed);

        if (n_cand == 0) {
            // No legal moves: evaluate standing pat (flip board = opponent's turn)
            Board opp_board = flip(board);
            float arCfTemp[MAX_CCI * 2];
            cubeful_recursive_multi(opp_board, aci, expanded_cci,
                                    strategy, base_gps, plies - 1, filter,
                                    n_threads, child_allow_parallel, false,
                                    arCfTemp, move_filter);
            for (int i = 0; i < expanded_cci; i++)
                arCfLocal[i] = roll.weight * arCfTemp[i];
            return;
        }

        // Special case: forced move, so no move-choice work is needed.
        if (n_cand == 1) {
            const Board& best_board = candidates[0];
            GameResult post_result = check_game_over(best_board);
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
                    // Dead cube: pure cubeless equity, gammons count
                    if (cube_is_dead(aci[i])) {
                        arCfLocal[i] = roll.weight * cubeless_equity(tp);
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

            Board opp_pre_roll = flip(best_board);
            float arCfTemp[MAX_CCI * 2];
            cubeful_recursive_multi(opp_pre_roll, aci, expanded_cci,
                                    strategy, base_gps, plies - 1, filter,
                                    n_threads, child_allow_parallel, false,
                                    arCfTemp, move_filter);
            for (int i = 0; i < expanded_cci; i++) {
                arCfLocal[i] = roll.weight * arCfTemp[i];
            }
            return;
        }

        // Pick best move by cubeless 1-ply equity (shared across all cube states).
        // Two-stage filtering: if move_filter is set and we have enough candidates,
        // use the cheap filter (e.g. PubEval) to narrow to top K, then evaluate
        // survivors with the full model.
        static constexpr int MOVE_FILTER_THRESHOLD = 16;
        static constexpr int MOVE_FILTER_KEEP = 15;

        int best_idx = 0;
        const std::vector<Board>* eval_candidates = &candidates;
        thread_local std::vector<Board> filtered;
        thread_local std::vector<int> orig_indices;

        // Stage 1: cheap pre-filter (PubEval or similar)
        if (move_filter && n_cand > MOVE_FILTER_THRESHOLD) {
            thread_local std::vector<std::pair<double, int>> filter_scores;
            filter_scores.clear();
            filter_scores.reserve(n_cand);
            bool pre_move_race = is_race(board);
            for (int c = 0; c < n_cand; c++) {
                GameResult gr = check_game_over(candidates[c]);
                double eq = (gr != GameResult::NOT_OVER)
                    ? 1e30 // terminals always survive
                    : move_filter->evaluate(candidates[c], pre_move_race);
                filter_scores.push_back({-eq, c});  // negate for ascending sort
            }
            int keep = std::min(MOVE_FILTER_KEEP, n_cand);
            std::partial_sort(filter_scores.begin(), filter_scores.begin() + keep,
                              filter_scores.end());

            filtered.clear();
            filtered.reserve(keep);
            orig_indices.clear();
            orig_indices.reserve(keep);
            for (int k = 0; k < keep; k++) {
                filtered.push_back(candidates[filter_scores[k].second]);
                orig_indices.push_back(filter_scores[k].second);
            }
            eval_candidates = &filtered;
        }

        // Stage 2: full model evaluation on survivors
        if (base_gps) {
            int local_best = base_gps->batch_evaluate_candidates_best_prob(
                *eval_candidates, board, nullptr, nullptr);
            best_idx = (eval_candidates == &candidates)
                ? local_best : orig_indices[local_best];
        } else {
            float best_eq = -1e30f;
            int local_best = 0;
            for (int c = 0; c < static_cast<int>(eval_candidates->size()); c++) {
                float eq;
                GameResult gr = check_game_over((*eval_candidates)[c]);
                if (gr != GameResult::NOT_OVER) {
                    eq = cubeless_equity(terminal_probs(gr));
                } else {
                    auto probs = strategy.evaluate_probs((*eval_candidates)[c],
                                                       is_race((*eval_candidates)[c]));
                    eq = cubeless_equity(probs);
                }
                if (eq > best_eq) {
                    best_eq = eq;
                    local_best = c;
                }
            }
            best_idx = (eval_candidates == &candidates)
                ? local_best : orig_indices[local_best];
        }

        // Check if best move is terminal
        const Board& best_board = candidates[best_idx];
        GameResult post_result = check_game_over(best_board);
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
                // Dead cube: pure cubeless equity, gammons count
                if (cube_is_dead(aci[i])) {
                    arCfLocal[i] = roll.weight * cubeless_equity(tp);
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
        Board opp_pre_roll = flip(best_board);

        float arCfTemp[MAX_CCI * 2];
        cubeful_recursive_multi(opp_pre_roll, aci, expanded_cci,
                                strategy, base_gps, plies - 1, filter,
                                n_threads, child_allow_parallel, false,
                                arCfTemp, move_filter);

        for (int i = 0; i < expanded_cci; i++)
            arCfLocal[i] = roll.weight * arCfTemp[i];
    };

    // Execute rolls (serial or parallel)
    if (allow_parallel && n_threads > 1 && plies > 1) {
        std::array<std::array<float, MAX_CCI * 2>, 21> roll_results;
        multipy_parallel_for(21, n_threads, [&](int idx) {
            evaluate_roll(idx, roll_results[idx].data());
        });
        for (int r = 0; r < 21; r++)
            for (int i = 0; i < expanded_cci; i++)
                arCf[i] += roll_results[r][i];
    } else {
        float arCfLocal[MAX_CCI * 2];
        for (int r = 0; r < 21; r++) {
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
    put_cached_cubeful(board, aciCubePos, cci, strategy, plies, fTop, arCubeful);
}

float cubeful_equity_nply(
    const Board& board,
    CubeOwner owner,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    if (n_plies <= 1) {
        CubeInfo cube;
        cube.cube_value = 1;
        cube.owner = owner;
        return eval_pre_roll_cubeful_1ply(board, cube, strategy);
    }

    begin_cubeful_cache_epoch();

    CubeInfo aciCubePos[1];
    aciCubePos[0].cube_value = 1;
    aciCubePos[0].owner = owner;
    // match defaults to {0,0,false} = money game

    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);
    float arCubeful[1];
    bool allow_parallel = (n_threads > 1 && n_plies > 2);
    cubeful_recursive_multi(board, aciCubePos, 1, strategy, base_gps, n_plies, filter,
                            n_threads, allow_parallel, false, arCubeful, move_filter);
    return arCubeful[0];
}

// Match play overload: returns cubeful equity (NOT MWC).
float cubeful_equity_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    if (n_plies <= 1) {
        float val = eval_pre_roll_cubeful_1ply(board, cube, strategy);
        if (cube.is_money()) return val;
        return mwc2eq(val, cube.match.away1, cube.match.away2,
                      cube.cube_value, cube.match.is_crawford);
    }

    begin_cubeful_cache_epoch();

    CubeInfo aciCubePos[1] = {cube};
    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);
    float arCubeful[1];
    bool allow_parallel = (n_threads > 1 && n_plies > 2);
    cubeful_recursive_multi(board, aciCubePos, 1, strategy, base_gps, n_plies, filter,
                            n_threads, allow_parallel, false, arCubeful, move_filter);

    if (cube.is_money()) return arCubeful[0];
    return mwc2eq(arCubeful[0], cube.match.away1, cube.match.away2,
                  cube.cube_value, cube.match.is_crawford);
}

// Batched version: evaluate multiple cube states against the same board in a
// single recursion.  Writes one cubeful equity per cube state into `out`.
//
// The cubeful recursion already supports multiple simultaneous cube states
// (up to MAX_CCI=64) via the `aciCubePos[]` / `cci` interface, so this just
// forwards `cubes` into a single call to `cubeful_recursive_multi` with
// cci=n_cubes.  Move selection (cubeless 1-ply) and the recursive tree are
// shared across all cube states; only the Janowski leaf conversions and the
// `get_ecf3` cube-decision collapses differ per state.  Numerically
// equivalent to calling `cubeful_equity_nply` separately on each cube.
void cubeful_equity_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    float* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool fTop)
{
    if (n_cubes <= 0) return;
    if (n_cubes > MAX_CCI) n_cubes = MAX_CCI;   // defensive cap

    if (n_plies <= 1) {
        // 1-ply fast path: evaluate pre-roll board ONCE, then apply Janowski
        // per cube state.  Mirrors eval_pre_roll_cubeful_1ply but shares the
        // NN evaluation across cubes.
        for (int i = 0; i < n_cubes; ++i) {
            float val = eval_pre_roll_cubeful_1ply(board, cubes[i], strategy);
            if (cubes[i].is_money()) {
                out[i] = val;
            } else {
                out[i] = mwc2eq(val, cubes[i].match.away1, cubes[i].match.away2,
                                cubes[i].cube_value, cubes[i].match.is_crawford);
            }
        }
        return;
    }

    begin_cubeful_cache_epoch();

    // Copy cubes into a stack buffer for cubeful_recursive_multi.
    CubeInfo aciCubePos[MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) aciCubePos[i] = cubes[i];

    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);
    float arCubeful[MAX_CCI];
    bool allow_parallel = (n_threads > 1 && n_plies > 2);
    cubeful_recursive_multi(board, aciCubePos, n_cubes, strategy, base_gps, n_plies,
                            filter, n_threads, allow_parallel, fTop,
                            arCubeful, move_filter);

    // Convert MWC -> equity for match-play states; leave money-game states as-is.
    for (int i = 0; i < n_cubes; ++i) {
        if (cubes[i].is_money()) {
            out[i] = arCubeful[i];
        } else {
            out[i] = mwc2eq(arCubeful[i], cubes[i].match.away1, cubes[i].match.away2,
                            cubes[i].cube_value, cubes[i].match.is_crawford);
        }
    }
}

// Batched cube decision.  For each of n_cubes input cube states, compute the
// same ND/DT/DP equities and decision that the single-branch cube_decision_nply
// would.  Runs a single cubeful_recursive_multi call with cci = 2*n_cubes and
// fTop=true: pairs [cubes[i], cubes[i]_doubled_opp] are laid out consecutively
// so that make_cube_pos expansion at the top level is skipped (via fTop), and
// the shared recursive tree handles move selection and NN evaluations once for
// all branches.  Output[i] contains the decision for cubes[i].
void cube_decision_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    CubeDecision* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    if (n_cubes <= 0) return;
    // Cap to MAX_CCI / 2 (since we expand to 2*n_cubes states).
    if (n_cubes > MAX_CCI / 2) n_cubes = MAX_CCI / 2;

    if (n_plies <= 1) {
        // 1-ply path: get pre-roll probs ONCE (shared), then Janowski per cube.
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        for (int i = 0; i < n_cubes; ++i) {
            float x = resolve_cube_x(pre_roll_probs, cubes[i], board, race);
            out[i] = cube_decision_1ply(pre_roll_probs, cubes[i], x);
        }
        return;
    }

    begin_cubeful_cache_epoch();

    // Build 2 * n_cubes states: for each input cube, [ND variant, DT variant].
    // Using fTop=true suppresses make_cube_pos's top-level DT expansion so the
    // caller-supplied DT variants are used directly.
    const int n_states = 2 * n_cubes;
    CubeInfo aciCubePos[MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) {
        aciCubePos[2 * i]     = cubes[i];                                // ND
        aciCubePos[2 * i + 1] = cubes[i];                                // DT
        aciCubePos[2 * i + 1].cube_value = 2 * cubes[i].cube_value;
        aciCubePos[2 * i + 1].owner      = CubeOwner::OPPONENT;
    }

    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);
    float arCubeful[MAX_CCI];
    bool allow_parallel = (n_threads > 1 && n_plies > 2);
    cubeful_recursive_multi(board, aciCubePos, n_states, strategy, base_gps, n_plies,
                            filter, n_threads, allow_parallel, /*fTop=*/true,
                            arCubeful, move_filter);

    // Unpack per-branch decisions.  Mirrors the decision logic in
    // cube_decision_nply(single).  Each branch consumes arCubeful[2i..2i+1].
    for (int i = 0; i < n_cubes; ++i) {
        const CubeInfo& cube = cubes[i];
        float nd_raw = arCubeful[2 * i];
        float dt_raw = arCubeful[2 * i + 1];
        bool is_money = cube.is_money();

        CubeDecision result = {};

        if (is_money) {
            result.equity_nd = nd_raw;
            float actual_dt = 2.0f * dt_raw;   // Scale doubled-cube value back to cube=1
            result.equity_dp = 1.0f;

            if (cube.beaver && actual_dt < 0.0f) {
                result.equity_dt = 2.0f * actual_dt;   // DB equity
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
            result.equity_nd = mwc2eq(nd_raw, away1, away2, cv, craw);
            result.equity_dt = mwc2eq(dt_raw, away1, away2, cv, craw);
            result.equity_dp = mwc2eq(dp_m,   away1, away2, cv, craw);
        }

        bool player_can_double = can_double(cube);
        bool auto_double = (!is_money && !cube.match.is_crawford &&
                            player_can_double &&
                            cube.match.away1 > 1 && cube.match.away2 == 1);

        if (!player_can_double) {
            result.should_double = false;
            result.should_take = true;
            result.optimal_equity = result.equity_nd;
        } else if (is_money) {
            float best_double = std::min(result.equity_dt, result.equity_dp);
            result.should_double = (best_double > result.equity_nd);
            result.should_take = (result.equity_dt <= result.equity_dp);
            result.optimal_equity = result.should_double
                ? std::min(result.equity_dt, result.equity_dp)
                : result.equity_nd;
        } else if (auto_double) {
            float dt_m = dt_raw;
            float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                    cube.cube_value, cube.match.is_crawford);
            result.should_double = true;
            result.should_take = (dt_m <= dp_m_val);
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            float nd_m = nd_raw;
            float dt_m = dt_raw;
            float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                    cube.cube_value, cube.match.is_crawford);
            float best_double = std::min(dt_m, dp_m_val);
            result.should_double = (best_double > nd_m);
            result.should_take = (dt_m <= dp_m_val);
            result.optimal_equity = result.should_double
                ? std::min(result.equity_dt, result.equity_dp)
                : result.equity_nd;
        }

        out[i] = result;
    }
}

// ---------------------------------------------------------------------------
// Helper: pick the best move for a single dice roll (used by detail functions).
// Returns the index of the best candidate in `candidates`, or -1 if no legal moves.
// ---------------------------------------------------------------------------
static int pick_best_move_for_roll(
    const Board& board,
    int die1, int die2,
    const Strategy& strategy,
    const GamePlanStrategy* base_gps,
    std::vector<Board>& candidates,
    const Strategy* move_filter = nullptr)
{
    candidates.clear();
    if (candidates.capacity() < 32) candidates.reserve(32);
    possible_boards(board, die1, die2, candidates);
    int n_cand = static_cast<int>(candidates.size());

    if (n_cand == 0) return -1;
    if (n_cand == 1) return 0;

    static constexpr int PREFILTER_THRESHOLD = 16;
    static constexpr int PREFILTER_KEEP = 15;

    // Pre-filter with cheap evaluator (e.g. PubEval) if available
    const std::vector<Board>* eval_candidates = &candidates;
    thread_local std::vector<Board> filtered;
    thread_local std::vector<int> orig_indices;

    if (move_filter && n_cand > PREFILTER_THRESHOLD) {
        thread_local std::vector<std::pair<double, int>> filter_scores;
        filter_scores.clear();
        filter_scores.reserve(n_cand);
        bool pre_move_race = is_race(board);
        for (int c = 0; c < n_cand; c++) {
            GameResult gr = check_game_over(candidates[c]);
            double eq = (gr != GameResult::NOT_OVER)
                ? 1e30 : move_filter->evaluate(candidates[c], pre_move_race);
            filter_scores.push_back({-eq, c});
        }
        int keep = std::min(PREFILTER_KEEP, n_cand);
        std::partial_sort(filter_scores.begin(), filter_scores.begin() + keep,
                          filter_scores.end());
        filtered.clear();
        filtered.reserve(keep);
        orig_indices.clear();
        orig_indices.reserve(keep);
        for (int k = 0; k < keep; k++) {
            filtered.push_back(candidates[filter_scores[k].second]);
            orig_indices.push_back(filter_scores[k].second);
        }
        eval_candidates = &filtered;
    }

    int best_idx = 0;
    if (base_gps) {
        int local_best = base_gps->batch_evaluate_candidates_best_prob(
            *eval_candidates, board, nullptr, nullptr);
        best_idx = (eval_candidates == &candidates)
            ? local_best : orig_indices[local_best];
    } else {
        float best_eq = -1e30f;
        int local_best = 0;
        for (int c = 0; c < static_cast<int>(eval_candidates->size()); c++) {
            float eq;
            GameResult gr = check_game_over((*eval_candidates)[c]);
            if (gr != GameResult::NOT_OVER) {
                eq = cubeless_equity(terminal_probs(gr));
            } else {
                auto probs = strategy.evaluate_probs((*eval_candidates)[c],
                                                     is_race((*eval_candidates)[c]));
                eq = cubeless_equity(probs);
            }
            if (eq > best_eq) {
                best_eq = eq;
                local_best = c;
            }
        }
        best_idx = (eval_candidates == &candidates)
            ? local_best : orig_indices[local_best];
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// Compute terminal equity for a game-over position from the OPPONENT of the
// mover's perspective (matching the convention in cubeful_recursive_multi).
// ---------------------------------------------------------------------------
static float terminal_equity_for_cube(
    GameResult result, const CubeInfo& ci, bool invert_perspective)
{
    auto tp = invert_perspective ? invert_probs(terminal_probs(result))
                                : terminal_probs(result);
    if (ci.cube_value <= 0) return 0.0f;
    if (cube_is_dead(ci)) return cubeless_equity(tp);
    if (ci.is_money()) {
        if (ci.jacoby_active()) return 2.0f * tp[0] - 1.0f;
        return cubeless_equity(tp);
    }
    return cubeless_mwc(tp, ci.match.away1, ci.match.away2,
                         ci.cube_value, ci.match.is_crawford);
}

// ---------------------------------------------------------------------------
// Helper: extract per-opponent-roll cubeful equities for one L2 cube state pair.
//
// detail.opponent_rolls must already be resized to 21 with dice/boards set.
// nd_idx/dt_idx: L2 expansion indices for the ND/DT of this branch.
// branch_cv: effective cube value for the ND state of this branch.
// redouble_cv: effective cube value if the opponent redoubles (DT of this branch).
// initial_cv: the original cube value (for scaling to per-initial-cube units).
// ---------------------------------------------------------------------------
static void extract_opp_roll_equities(
    PlayerRollDetail& detail,
    const std::array<std::array<float, MAX_CCI * 2>, 21>& opp_roll_results,
    const float arCf_L2[],
    const CubeInfo aci_L2_uninv[],
    int nd_idx, int dt_idx,
    int branch_cv,
    int redouble_cv,
    int initial_cv,
    int away1, int away2, bool is_crawford,
    bool is_money)
{
    bool opp_dt_available = (aci_L2_uninv[dt_idx].cube_value > 0);

    if (opp_dt_available) {
        float rND_opp = arCf_L2[nd_idx];
        float rDT_opp, rDP_opp;
        if (is_money) {
            rDT_opp = 2.0f * arCf_L2[dt_idx];
            rDP_opp = 1.0f;
        } else {
            rDT_opp = arCf_L2[dt_idx];
            rDP_opp = dp_mwc(aci_L2_uninv[nd_idx].match.away1,
                              aci_L2_uninv[nd_idx].match.away2,
                              aci_L2_uninv[nd_idx].cube_value,
                              aci_L2_uninv[nd_idx].match.is_crawford);
        }
        // Beaver check
        if (is_money && aci_L2_uninv[nd_idx].beaver && rDT_opp < 0.0f) {
            rDT_opp = 2.0f * rDT_opp;
        }
        bool opp_should_double = (rDT_opp >= rND_opp && rDP_opp >= rND_opp);
        bool opp_should_take = opp_should_double ? (rDT_opp <= rDP_opp) : true;

        if (opp_should_double && !opp_should_take) {
            // D/P: opponent doubles (or redoubles), player passes
            detail.opponent_dp = true;
            detail.opponent_rolls.clear();
        } else if (opp_should_double) {
            // D/T: use DT branch per-roll values, scaled to per-initial-cube
            float scale = static_cast<float>(redouble_cv) / static_cast<float>(initial_cv);
            for (int opp_r = 0; opp_r < 21; opp_r++) {
                float raw = opp_roll_results[opp_r][dt_idx];
                float eq;
                if (is_money) {
                    eq = scale * raw;
                } else {
                    eq = mwc2eq(raw, away1, away2, redouble_cv, is_crawford);
                }
                detail.opponent_rolls[opp_r].cubeful_equity = eq;
            }
        } else {
            // ND: use ND branch per-roll values, scaled to per-initial-cube
            float scale = static_cast<float>(branch_cv) / static_cast<float>(initial_cv);
            for (int opp_r = 0; opp_r < 21; opp_r++) {
                float raw = opp_roll_results[opp_r][nd_idx];
                float eq;
                if (is_money) {
                    eq = scale * raw;
                } else {
                    eq = mwc2eq(raw, away1, away2, branch_cv, is_crawford);
                }
                detail.opponent_rolls[opp_r].cubeful_equity = eq;
            }
        }
    } else {
        // No DT branch available: ND only
        float scale = static_cast<float>(branch_cv) / static_cast<float>(initial_cv);
        for (int opp_r = 0; opp_r < 21; opp_r++) {
            float raw = opp_roll_results[opp_r][nd_idx];
            float eq;
            if (is_money) {
                eq = scale * raw;
            } else {
                eq = mwc2eq(raw, away1, away2, branch_cv, is_crawford);
            }
            detail.opponent_rolls[opp_r].cubeful_equity = eq;
        }
    }
}

// ---------------------------------------------------------------------------
// cube_decision_nply_with_details: N-ply cube decision with per-roll details.
// Manually implements the top two levels of the cubeful recursion to capture
// per-roll data, delegating to cubeful_recursive_multi for deeper levels.
// ---------------------------------------------------------------------------
CubeDecision cube_decision_nply_with_details(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    TwoPlyDetails& details,
    const Strategy* move_filter)
{
    begin_cubeful_cache_epoch();

    bool is_money = cube.is_money();
    bool allow_parallel = (n_threads > 1 && n_plies > 2);

    // Two initial cube states: ND (current) and DT (doubled, opponent owns)
    CubeInfo aciCubePos[2];
    aciCubePos[0] = cube;
    aciCubePos[1] = cube;
    aciCubePos[1].cube_value = 2 * cube.cube_value;
    aciCubePos[1].owner = CubeOwner::OPPONENT;

    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);

    // Expand cube states for level 1 (fTop=true, fInvert=true)
    CubeInfo aci_L1[MAX_CCI * 2];
    make_cube_pos(aciCubePos, 2, /*fTop=*/true, aci_L1, /*fInvert=*/true);
    int expanded_cci_L1 = 4;  // 2 * 2

    // Accumulators for weighted cubeful equities (level 1)
    float arCf_L1[MAX_CCI * 2] = {};

    // Prepare details output
    details.nd_player_rolls.clear();
    details.nd_player_rolls.resize(21);
    details.dt_player_rolls.clear();
    details.dt_player_rolls.resize(21);

    // --- Level 1: loop over 21 player rolls ---
    // Per-roll storage for later accumulation
    struct L1RollResult {
        float arCfLocal[MAX_CCI * 2] = {};
    };
    std::array<L1RollResult, 21> l1_results;

    auto evaluate_player_roll = [&](int roll_idx) {
        const auto& roll = ALL_ROLLS[roll_idx];
        auto& nd_detail = details.nd_player_rolls[roll_idx];
        auto& dt_detail = details.dt_player_rolls[roll_idx];
        auto& l1r = l1_results[roll_idx];
        nd_detail.die1 = roll.d1;
        nd_detail.die2 = roll.d2;
        dt_detail.die1 = roll.d1;
        dt_detail.die2 = roll.d2;

        // Generate legal moves and pick best
        thread_local std::vector<Board> candidates;
        int best_idx = pick_best_move_for_roll(board, roll.d1, roll.d2,
                                                strategy, base_gps, candidates, move_filter);

        if (best_idx < 0) {
            // No legal moves (dancing): standing pat
            nd_detail.post_move_board = board;
            dt_detail.post_move_board = board;
            Board opp_board = flip(board);

            // --- Run level 2 for this player roll (standing pat) ---
            // Expand cube states for level 2
            CubeInfo aci_L2[MAX_CCI * 2];
            make_cube_pos(aci_L1, expanded_cci_L1, /*fTop=*/false, aci_L2, /*fInvert=*/true);
            int expanded_cci_L2 = 2 * expanded_cci_L1;

            // Per-opponent-roll storage
            std::array<std::array<float, MAX_CCI * 2>, 21> opp_roll_results;
            std::array<Board, 21> opp_boards;
            std::array<bool, 21> opp_terminal;
            nd_detail.opponent_rolls.resize(21);
            dt_detail.opponent_rolls.resize(21);

            float arCf_L2[MAX_CCI * 2] = {};

            for (int opp_r = 0; opp_r < 21; opp_r++) {
                const auto& opp_roll = ALL_ROLLS[opp_r];
                // Populate dice/boards for both ND and DT details
                nd_detail.opponent_rolls[opp_r].die1 = opp_roll.d1;
                nd_detail.opponent_rolls[opp_r].die2 = opp_roll.d2;
                dt_detail.opponent_rolls[opp_r].die1 = opp_roll.d1;
                dt_detail.opponent_rolls[opp_r].die2 = opp_roll.d2;
                opp_terminal[opp_r] = false;

                thread_local std::vector<Board> opp_candidates;
                int opp_best = pick_best_move_for_roll(opp_board, opp_roll.d1, opp_roll.d2,
                                                        strategy, base_gps, opp_candidates, move_filter);

                Board opp_post_move;
                if (opp_best < 0) {
                    opp_post_move = opp_board;
                } else {
                    opp_post_move = opp_candidates[opp_best];
                }

                // Record board from PLAYER's perspective
                opp_boards[opp_r] = flip(opp_post_move);
                nd_detail.opponent_rolls[opp_r].post_move_board = opp_boards[opp_r];
                dt_detail.opponent_rolls[opp_r].post_move_board = opp_boards[opp_r];

                // Check terminal
                GameResult post_result = check_game_over(opp_post_move);
                if (post_result != GameResult::NOT_OVER) {
                    opp_terminal[opp_r] = true;
                    for (int i = 0; i < expanded_cci_L2; i++) {
                        opp_roll_results[opp_r][i] = terminal_equity_for_cube(
                            post_result, aci_L2[i], /*invert_perspective=*/true);
                        arCf_L2[i] += opp_roll.weight * opp_roll_results[opp_r][i];
                    }
                    continue;
                }

                // Flip to player's perspective and recurse at plies-2
                Board player_pre_roll = flip(opp_post_move);
                float arCfTemp2[MAX_CCI * 2];
                cubeful_recursive_multi(player_pre_roll, aci_L2, expanded_cci_L2,
                                        strategy, base_gps, n_plies - 2, filter,
                                        n_threads, false, false, arCfTemp2, move_filter);
                for (int i = 0; i < expanded_cci_L2; i++) {
                    opp_roll_results[opp_r][i] = arCfTemp2[i];
                    arCf_L2[i] += opp_roll.weight * arCfTemp2[i];
                }
            }

            // Perspective flip (level 2): opponent perspective
            for (int i = 0; i < expanded_cci_L2; i++) {
                if (is_money) {
                    arCf_L2[i] = -arCf_L2[i] / 36.0f;
                } else {
                    arCf_L2[i] = 1.0f - arCf_L2[i] / 36.0f;
                }
            }

            // Un-invert cube states back to level 1's perspective
            CubeInfo aci_L2_uninv[MAX_CCI * 2];
            for (int i = 0; i < expanded_cci_L2; i++) {
                aci_L2_uninv[i] = aci_L2[i];
                if (aci_L2[i].cube_value > 0) {
                    aci_L2_uninv[i] = flip_cube_perspective(aci_L2[i]);
                }
            }

            // Collapse via get_ecf3 (opponent's cube decision)
            float arCubeful_L2[MAX_CCI * 2];
            get_ecf3(arCubeful_L2, expanded_cci_L1, arCf_L2, aci_L2_uninv);

            int cv = cube.cube_value;
            int a1 = cube.match.away1, a2 = cube.match.away2;
            bool craw = cube.match.is_crawford;

            // ND opponent-roll equities (L2 indices 0,1)
            extract_opp_roll_equities(nd_detail, opp_roll_results,
                arCf_L2, aci_L2_uninv, 0, 1, cv, 2*cv, cv, a1, a2, craw, is_money);

            // DT opponent-roll equities (L2 indices 4,5)
            extract_opp_roll_equities(dt_detail, opp_roll_results,
                arCf_L2, aci_L2_uninv, 4, 5, 2*cv, 4*cv, cv, a1, a2, craw, is_money);

            // ND player-roll equity: arCubeful_L2[0] from opponent's perspective
            if (is_money) {
                nd_detail.cubeful_equity = -arCubeful_L2[0];
            } else {
                nd_detail.cubeful_equity = mwc2eq(1.0f - arCubeful_L2[0], a1, a2, cv, craw);
            }

            // DT player-roll equity: arCubeful_L2[2] from opponent's perspective, per-2x-cube
            if (is_money) {
                dt_detail.cubeful_equity = -2.0f * arCubeful_L2[2];
            } else {
                dt_detail.cubeful_equity = mwc2eq(1.0f - arCubeful_L2[2], a1, a2, 2*cv, craw);
            }

            // Accumulate for level 1
            for (int i = 0; i < expanded_cci_L1; i++)
                l1r.arCfLocal[i] = roll.weight * arCubeful_L2[i];
            return;
        }

        const Board& best_board = candidates[best_idx];
        nd_detail.post_move_board = best_board;
        dt_detail.post_move_board = best_board;

        // Check if player's move is terminal
        GameResult post_result = check_game_over(best_board);
        if (post_result != GameResult::NOT_OVER) {
            nd_detail.is_terminal = true;
            dt_detail.is_terminal = true;
            // Terminal equity from opponent's perspective (for level 1 accumulation)
            for (int i = 0; i < expanded_cci_L1; i++) {
                l1r.arCfLocal[i] = roll.weight * terminal_equity_for_cube(
                    post_result, aci_L1[i], /*invert_perspective=*/true);
            }
            // ND player's equity (per initial cube)
            auto tp = terminal_probs(post_result);
            nd_detail.cubeful_equity = cubeless_equity(tp);
            if (is_money && cube.jacoby_active()) {
                nd_detail.cubeful_equity = 2.0f * tp[0] - 1.0f;
            }
            if (!is_money) {
                nd_detail.cubeful_equity = mwc2eq(
                    cubeless_mwc(tp, cube.match.away1, cube.match.away2,
                                  cube.cube_value, cube.match.is_crawford),
                    cube.match.away1, cube.match.away2,
                    cube.cube_value, cube.match.is_crawford);
            }
            // DT player's equity (per initial cube, at doubled cube level)
            // Jacoby is never active in DT (cube is turned)
            if (is_money) {
                dt_detail.cubeful_equity = 2.0f * cubeless_equity(tp);
            } else {
                dt_detail.cubeful_equity = mwc2eq(
                    cubeless_mwc(tp, cube.match.away1, cube.match.away2,
                                  2 * cube.cube_value, cube.match.is_crawford),
                    cube.match.away1, cube.match.away2,
                    2 * cube.cube_value, cube.match.is_crawford);
            }
            return;
        }

        // --- Level 2: opponent's turn ---
        Board opp_pre_roll = flip(best_board);

        // Expand cube states for level 2
        CubeInfo aci_L2[MAX_CCI * 2];
        make_cube_pos(aci_L1, expanded_cci_L1, /*fTop=*/false, aci_L2, /*fInvert=*/true);
        int expanded_cci_L2 = 2 * expanded_cci_L1;

        // Per-opponent-roll storage
        std::array<std::array<float, MAX_CCI * 2>, 21> opp_roll_results;
        std::array<Board, 21> opp_boards;
        std::array<bool, 21> opp_terminal;
        nd_detail.opponent_rolls.resize(21);
        dt_detail.opponent_rolls.resize(21);

        float arCf_L2[MAX_CCI * 2] = {};

        for (int opp_r = 0; opp_r < 21; opp_r++) {
            const auto& opp_roll = ALL_ROLLS[opp_r];
            // Populate dice/boards for both ND and DT details
            nd_detail.opponent_rolls[opp_r].die1 = opp_roll.d1;
            nd_detail.opponent_rolls[opp_r].die2 = opp_roll.d2;
            dt_detail.opponent_rolls[opp_r].die1 = opp_roll.d1;
            dt_detail.opponent_rolls[opp_r].die2 = opp_roll.d2;
            opp_terminal[opp_r] = false;

            thread_local std::vector<Board> opp_candidates;
            int opp_best = pick_best_move_for_roll(opp_pre_roll, opp_roll.d1, opp_roll.d2,
                                                    strategy, base_gps, opp_candidates, move_filter);

            Board opp_post_move;
            if (opp_best < 0) {
                opp_post_move = opp_pre_roll;
            } else {
                opp_post_move = opp_candidates[opp_best];
            }

            opp_boards[opp_r] = flip(opp_post_move);
            nd_detail.opponent_rolls[opp_r].post_move_board = opp_boards[opp_r];
            dt_detail.opponent_rolls[opp_r].post_move_board = opp_boards[opp_r];

            GameResult opp_post_result = check_game_over(opp_post_move);
            if (opp_post_result != GameResult::NOT_OVER) {
                opp_terminal[opp_r] = true;
                for (int i = 0; i < expanded_cci_L2; i++) {
                    opp_roll_results[opp_r][i] = terminal_equity_for_cube(
                        opp_post_result, aci_L2[i], /*invert_perspective=*/true);
                    arCf_L2[i] += opp_roll.weight * opp_roll_results[opp_r][i];
                }
                continue;
            }

            Board player_pre_roll = flip(opp_post_move);
            float arCfTemp2[MAX_CCI * 2];
            cubeful_recursive_multi(player_pre_roll, aci_L2, expanded_cci_L2,
                                    strategy, base_gps, n_plies - 2, filter,
                                    n_threads, false, false, arCfTemp2, move_filter);
            for (int i = 0; i < expanded_cci_L2; i++) {
                opp_roll_results[opp_r][i] = arCfTemp2[i];
                arCf_L2[i] += opp_roll.weight * arCfTemp2[i];
            }
        }

        // Perspective flip (level 2)
        for (int i = 0; i < expanded_cci_L2; i++) {
            if (is_money) {
                arCf_L2[i] = -arCf_L2[i] / 36.0f;
            } else {
                arCf_L2[i] = 1.0f - arCf_L2[i] / 36.0f;
            }
        }

        // Un-invert cube states
        CubeInfo aci_L2_uninv[MAX_CCI * 2];
        for (int i = 0; i < expanded_cci_L2; i++) {
            aci_L2_uninv[i] = aci_L2[i];
            if (aci_L2[i].cube_value > 0) {
                aci_L2_uninv[i] = flip_cube_perspective(aci_L2[i]);
            }
        }

        // Collapse via get_ecf3 (opponent's cube decision)
        float arCubeful_L2[MAX_CCI * 2];
        get_ecf3(arCubeful_L2, expanded_cci_L1, arCf_L2, aci_L2_uninv);

        int cv = cube.cube_value;
        int a1 = cube.match.away1, a2 = cube.match.away2;
        bool craw = cube.match.is_crawford;

        // ND opponent-roll equities (L2 indices 0,1)
        extract_opp_roll_equities(nd_detail, opp_roll_results,
            arCf_L2, aci_L2_uninv, 0, 1, cv, 2*cv, cv, a1, a2, craw, is_money);

        // DT opponent-roll equities (L2 indices 4,5)
        extract_opp_roll_equities(dt_detail, opp_roll_results,
            arCf_L2, aci_L2_uninv, 4, 5, 2*cv, 4*cv, cv, a1, a2, craw, is_money);

        // ND player-roll equity: arCubeful_L2[0] from opponent's perspective
        if (is_money) {
            nd_detail.cubeful_equity = -arCubeful_L2[0];
        } else {
            nd_detail.cubeful_equity = mwc2eq(1.0f - arCubeful_L2[0], a1, a2, cv, craw);
        }

        // DT player-roll equity: arCubeful_L2[2] from opponent's perspective, per-2x-cube
        if (is_money) {
            dt_detail.cubeful_equity = -2.0f * arCubeful_L2[2];
        } else {
            dt_detail.cubeful_equity = mwc2eq(1.0f - arCubeful_L2[2], a1, a2, 2*cv, craw);
        }

        // Accumulate for level 1
        for (int i = 0; i < expanded_cci_L1; i++)
            l1r.arCfLocal[i] = roll.weight * arCubeful_L2[i];
    };

    // Execute player rolls (serial or parallel)
    if (allow_parallel && n_threads > 1) {
        multipy_parallel_for(21, n_threads, [&](int idx) {
            evaluate_player_roll(idx);
        });
    } else {
        for (int r = 0; r < 21; r++) {
            evaluate_player_roll(r);
        }
    }

    // Accumulate level 1 results
    for (int r = 0; r < 21; r++)
        for (int i = 0; i < expanded_cci_L1; i++)
            arCf_L1[i] += l1_results[r].arCfLocal[i];

    // Perspective flip (level 1)
    for (int i = 0; i < expanded_cci_L1; i++) {
        if (is_money) {
            arCf_L1[i] = -arCf_L1[i] / 36.0f;
        } else {
            arCf_L1[i] = 1.0f - arCf_L1[i] / 36.0f;
        }
    }

    // Un-invert cube states
    CubeInfo aci_L1_uninv[MAX_CCI * 2];
    for (int i = 0; i < expanded_cci_L1; i++) {
        aci_L1_uninv[i] = aci_L1[i];
        if (aci_L1[i].cube_value > 0) {
            aci_L1_uninv[i] = flip_cube_perspective(aci_L1[i]);
        }
    }

    // Collapse via get_ecf3
    float arCubeful[2];
    get_ecf3(arCubeful, 2, arCf_L1, aci_L1_uninv);

    // Build CubeDecision (same logic as cube_decision_nply)
    CubeDecision cd;
    if (is_money) {
        cd.equity_nd = arCubeful[0];
        float actual_dt = 2.0f * arCubeful[1];
        cd.equity_dp = 1.0f;
        if (cube.beaver && actual_dt < 0.0f) {
            cd.equity_dt = 2.0f * actual_dt;
            cd.is_beaver = true;
        } else {
            cd.equity_dt = actual_dt;
        }
    } else {
        int away1 = cube.match.away1;
        int away2 = cube.match.away2;
        int cv = cube.cube_value;
        bool craw = cube.match.is_crawford;
        float dp_m = dp_mwc(away1, away2, cv, craw);
        cd.equity_nd = mwc2eq(arCubeful[0], away1, away2, cv, craw);
        cd.equity_dt = mwc2eq(arCubeful[1], away1, away2, cv, craw);
        cd.equity_dp = mwc2eq(dp_m, away1, away2, cv, craw);
    }

    // Decision logic
    bool player_can_double = can_double(cube);
    bool auto_double = (!is_money && !cube.match.is_crawford &&
                        player_can_double &&
                        cube.match.away1 > 1 && cube.match.away2 == 1);

    if (!player_can_double) {
        cd.should_double = false;
        cd.should_take = true;
        cd.optimal_equity = cd.equity_nd;
    } else if (is_money) {
        float best_double = std::min(cd.equity_dt, cd.equity_dp);
        cd.should_double = (best_double > cd.equity_nd);
        cd.should_take = (cd.equity_dt <= cd.equity_dp);
        cd.optimal_equity = cd.should_double ? std::min(cd.equity_dt, cd.equity_dp)
                                              : cd.equity_nd;
    } else if (auto_double) {
        float nd_m = arCubeful[0];
        float dt_m = arCubeful[1];
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                 cube.cube_value, cube.match.is_crawford);
        cd.should_double = true;
        cd.should_take = (dt_m <= dp_m_val);
        cd.optimal_equity = std::min(cd.equity_dt, cd.equity_dp);
    } else {
        float nd_m = arCubeful[0];
        float dt_m = arCubeful[1];
        float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                 cube.cube_value, cube.match.is_crawford);
        float best_double_mwc = std::min(dt_m, dp_m_val);
        cd.should_double = (best_double_mwc > nd_m);
        cd.should_take = (dt_m <= dp_m_val);
        cd.optimal_equity = cd.should_double ? std::min(cd.equity_dt, cd.equity_dp)
                                              : cd.equity_nd;
    }

    return cd;
}

CubeDecision cube_decision_nply(
    const Board& board,
    const CubeInfo& cube,
    const Strategy& strategy,
    int n_plies,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter)
{
    if (n_plies <= 1) {
        // Use 1-ply path: get pre-roll probs, apply Janowski
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strategy.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float x = resolve_cube_x(pre_roll_probs, cube, board, race);
        return cube_decision_1ply(pre_roll_probs, cube, x);
    }

    begin_cubeful_cache_epoch();

    bool is_money = cube.is_money();
    bool allow_parallel = (n_threads > 1 && n_plies > 2);

    // Two initial cube states: ND (current) and DT (doubled, opponent owns)
    CubeInfo aciCubePos[2];
    aciCubePos[0] = cube;                                // ND state
    aciCubePos[1] = cube;                                // DT state
    aciCubePos[1].cube_value = 2 * cube.cube_value;
    aciCubePos[1].owner = CubeOwner::OPPONENT;

    const auto* base_gps = dynamic_cast<const GamePlanStrategy*>(&strategy);
    float arCubeful[2];
    cubeful_recursive_multi(board, aciCubePos, 2, strategy, base_gps, n_plies, filter,
                            n_threads, allow_parallel, /*fTop=*/true, arCubeful, move_filter);

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
