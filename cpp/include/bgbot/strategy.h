#pragma once

#include "types.h"
#include <array>
#include <vector>

namespace bgbot {

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

// Abstract strategy interface.
// A strategy evaluates board positions and selects the best move.
class Strategy {
public:
    virtual ~Strategy() = default;

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
};

} // namespace bgbot
