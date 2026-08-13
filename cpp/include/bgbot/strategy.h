// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#pragma once

#include "types.h"
#include "board.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace bgbot {

// Forward declaration to avoid including cube.h (which includes strategy.h).
// The cube-aware Strategy methods below take CubeInfo by reference, so the
// header only needs the type name. strategy.cpp includes cube.h for the
// default implementations.
struct CubeInfo;

constexpr int NUM_OUTPUTS = 5;  // P(win), P(gw), P(bw), P(gl), P(bl)

// Compute equity from 5 output probabilities.
// equity = 2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)
inline double compute_equity(const std::array<float, NUM_OUTPUTS>& probs) {
    return 2.0 * probs[0] - 1.0
         + probs[1] - probs[3]
         + probs[2] - probs[4];
}

// Probabilities for terminal (game-over) positions from the perspective
// of the player who is "player 1" on the board.
// WIN means player 1 won; LOSS means player 1 lost.
inline std::array<float, NUM_OUTPUTS> terminal_probs(GameResult result) {
    switch (result) {
        case GameResult::WIN_SINGLE:      return {1, 0, 0, 0, 0};
        case GameResult::WIN_GAMMON:      return {1, 1, 0, 0, 0};
        case GameResult::WIN_BACKGAMMON:  return {1, 1, 1, 0, 0};
        case GameResult::LOSS_SINGLE:     return {0, 0, 0, 0, 0};
        case GameResult::LOSS_GAMMON:     return {0, 0, 0, 1, 0};
        case GameResult::LOSS_BACKGAMMON: return {0, 0, 0, 1, 1};
        default:                          return {0.5f, 0, 0, 0, 0};
    }
}

// Invert probabilities from one player's perspective to the other's.
// If probs are from the opponent's viewpoint, this gives player 1's probs.
inline std::array<float, NUM_OUTPUTS> invert_probs(
    const std::array<float, NUM_OUTPUTS>& p)
{
    return {
        1.0f - p[0],   // P(win) = 1 - P(opp_win)
        p[3],           // P(gw)  = P(opp_gl)
        p[4],           // P(bw)  = P(opp_bl)
        p[1],           // P(gl)  = P(opp_gw)
        p[2]            // P(bl)  = P(opp_bw)
    };
}

// Clamp cubeless probabilities to enforce sanity invariants implied by the
// board state. `probs` are from the perspective of the player whose checkers
// are positive on `board` (the standard "current player" / mover convention).
//
// Invariants enforced (NN may emit small non-zero values for impossible
// outcomes; this zeros them out exactly):
//
//   1. If the player has at least one checker borne off, gammon and
//      backgammon LOSS are impossible -> P(gl)=0, P(bl)=0.
//   2. If the opponent has at least one checker borne off, gammon and
//      backgammon WIN are impossible -> P(gw)=0, P(bw)=0.
//   3. If contact is broken AND the player has no checkers in the
//      opponent's home board (points 19-24) or on the bar (index 25),
//      P(bl)=0 (the player cannot be backgammoned).
//   4. If contact is broken AND the opponent has no checkers in the
//      player's home board (points 1-6) or on the bar (index 0),
//      P(bw)=0 (the player cannot win a backgammon).
//
// The contact-broken precondition (rules 3/4) ensures the "no checker in
// the danger zone" status is stable: under contact, a checker can be hit
// and sent to the bar (re-entering in opponent's home), so the property
// could be re-violated; once contact is broken, no re-entry is possible.
//
// This is a pure post-hoc clamp on cubeless probabilities — it does not
// adjust P(win) or P(gw)/P(gl). Equity remains a linear function of the
// (possibly clamped) probs.
inline void clamp_probs_to_board(
    std::array<float, NUM_OUTPUTS>& probs, const Board& board)
{
    // Invariant 1: player has borne off at least one checker.
    if (player_borne_off(board) > 0) {
        probs[3] = 0.0f;  // P(gl)
        probs[4] = 0.0f;  // P(bl)
    }
    // Invariant 2: opponent has borne off at least one checker.
    if (opponent_borne_off(board) > 0) {
        probs[1] = 0.0f;  // P(gw)
        probs[2] = 0.0f;  // P(bw)
    }
    // Invariants 3 and 4 require contact to be broken.
    if ((probs[4] != 0.0f || probs[2] != 0.0f) && is_race(board)) {
        if (probs[4] != 0.0f) {
            // Player checkers in opponent's home (19-24) or on bar (25)?
            int p_in_danger = board[25];
            for (int i = 19; i <= 24; ++i) {
                if (board[i] > 0) p_in_danger += board[i];
            }
            if (p_in_danger == 0) probs[4] = 0.0f;
        }
        if (probs[2] != 0.0f) {
            // Opponent checkers in player's home (1-6) or on bar (0)?
            int o_in_danger = board[0];
            for (int i = 1; i <= 6; ++i) {
                if (board[i] < 0) o_in_danger -= board[i];
            }
            if (o_in_danger == 0) probs[2] = 0.0f;
        }
    }
}

// Abstract strategy interface.
// A strategy evaluates board positions and selects the best move.
class Strategy {
public:
    virtual ~Strategy() = default;

    // Identity of the underlying evaluator, used to key shared evaluation
    // caches (the cubeful evaluation engine's per-thread cache). Wrappers
    // that don't change evaluation results for non-intercepted positions
    // (e.g. BearoffStrategy) forward to the wrapped strategy so equivalent
    // evaluators share cache entries. The id is a process-unique monotonic
    // counter (NOT the object address — a freed strategy's address can be
    // reused by a different model's strategy, which would silently serve
    // the old model's cached values), so distinct strategy objects never
    // collide even across construct/destroy cycles.
    virtual uint64_t eval_identity() const { return eval_id_; }

    // Evaluate a post-move board position. Higher = better for player 1.
    // The board is from player 1's perspective.
    // `pre_move_is_race` is the race classification of the board BEFORE
    // the move was applied (some strategies use this for weight selection).
    virtual double evaluate(const Board& board, bool pre_move_is_race) const = 0;

    // Returns the 5 NN output probabilities for a position.
    // [P(win), P(gw), P(bw), P(gl), P(bl)]
    // Default implementation synthesizes from evaluate() equity.
    // Subclasses with actual NN should override for accuracy.
    virtual std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const;

    // Overload that takes the full pre-move board.
    // Default calls evaluate_probs(board, is_race(pre_move_board)).
    virtual std::array<float, NUM_OUTPUTS> evaluate_probs(
        const Board& board, const Board& pre_move_board) const;

    // Select the best post-move board from a list of candidates.
    // Returns the index into `candidates`. Default implementation calls
    // evaluate() on each and picks the highest.
    virtual int best_move_index(const std::vector<Board>& candidates,
                                bool pre_move_is_race) const;

    // Overload that takes the full pre-move board for strategies that need it
    // (e.g., GamePlanStrategy needs to classify the pre-move game plan).
    // Default implementation calls best_move_index(candidates, is_race(pre_move_board)).
    virtual int best_move_index(const std::vector<Board>& candidates,
                                const Board& pre_move_board) const;

    // ----- Cube-aware move selection -----
    //
    // Cube-aware variant of best_move_index: returns the candidate with the
    // highest cubeful equity under the given cube state. The cube state may
    // be money or match; the dispatcher cl2cf handles both. `cube_x` is the
    // Janowski cube efficiency for the pre-move board, computed once by the
    // caller and reused per cube state to avoid redundant work.
    //
    // The default implementation evaluates cubeless probs for each candidate
    // via evaluate_probs (clamped against the candidate board), applies cl2cf,
    // and returns the argmax. Strategies with batch evaluation (NNStrategy)
    // or deeper search (MultiPlyStrategy, RolloutStrategy) override this for
    // speed and accuracy.
    virtual int best_move_index_cubeful(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        const CubeInfo& ci,
        float cube_x) const;

    // Batched variant: pick the best candidate independently for each cube
    // state. Returns one index per cube into out_indices[0..n_cubes-1]. The
    // default implementation evaluates cubeless probs for each candidate
    // exactly once (the expensive part — NN forward pass) and then loops
    // cl2cf per cube state (essentially free). Strategies that share work
    // across cubes more aggressively (MultiPlyStrategy via
    // cubeful_equity_nply_multi) can override.
    virtual void best_move_index_cubeful_multi(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        const CubeInfo* cubes,
        int n_cubes,
        float cube_x,
        int* out_indices) const;

    // ----- Batch evaluation methods -----
    // These evaluate multiple candidate post-move boards efficiently.
    // Default implementations loop over candidates calling evaluate_probs().
    // Concrete strategies (GamePlanStrategy, etc.) override with optimized
    // batch encoding + forward pass implementations.

    // Evaluate all candidates, fill equities[0..n-1], return best index.
    // Handles terminal positions (check_game_over) automatically.
    virtual int evaluate_candidates_equity(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities) const;

    // Batch encoding + forward pass variant of evaluate_candidates_equity.
    // For the default implementation, identical to evaluate_candidates_equity.
    // Concrete strategies amortize encoding and use batch forward passes.
    virtual int batch_evaluate_candidates_equity(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities) const;

    // Like batch_evaluate_candidates_equity, but also stores the full
    // 5-output probabilities for each candidate in probs_out.
    virtual int batch_evaluate_candidates_equity_probs(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities,
        std::array<float, NUM_OUTPUTS>* probs_out) const;

    // Like batch_evaluate_candidates_equity, but returns only the
    // best candidate's probabilities in best_probs_out.
    // If equities is nullptr, equity outputs are not stored.
    virtual int batch_evaluate_candidates_best_prob(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities,
        std::array<float, NUM_OUTPUTS>* best_probs_out) const;

private:
    static uint64_t next_eval_id() {
        static std::atomic<uint64_t> counter{1};
        return counter.fetch_add(1, std::memory_order_relaxed);
    }
    const uint64_t eval_id_ = next_eval_id();
};

} // namespace bgbot
