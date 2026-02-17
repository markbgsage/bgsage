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

float cube_efficiency(const Board& board, bool is_race_pos) {
    if (!is_race_pos) {
        return 0.68f;  // Contact/crashed
    }
    // Race: linear in roller's pip count, clamped to [0.6, 0.7]
    auto [player_pips, opponent_pips] = pip_counts(board);
    float x = 0.55f + 0.00125f * static_cast<float>(player_pips);
    return std::clamp(x, 0.6f, 0.7f);
}

CubeDecision cube_decision_0ply(
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

// Internal recursive cubeful equity evaluation.
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

    CubeDecision result;
    result.equity_dp = 1.0f;

    // No Double: cubeful equity with current cube state
    result.equity_nd = cubeful_equity_nply(board, cube.owner, strategy, n_plies, filter, n_threads);

    // Double/Take: cubeful equity with opponent owning cube at 2x value.
    // After doubling, opponent owns cube → from player's perspective, OPPONENT owns.
    // Equity at doubled stakes: multiply by 2.
    result.equity_dt = 2.0f * cubeful_equity_nply(
        board, CubeOwner::OPPONENT, strategy, n_plies, filter, n_threads);

    // Decision logic (same as 0-ply)
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

} // namespace bgbot
