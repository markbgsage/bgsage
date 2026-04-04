#include "bgbot/strategy.h"
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include <limits>
#include <algorithm>

namespace bgbot {

std::array<float, NUM_OUTPUTS> Strategy::evaluate_probs(
    const Board& board, bool pre_move_is_race) const
{
    // Default: approximate from equity. Non-NN strategies (e.g. PubEval)
    // only produce a single equity value; we map it to a win probability.
    double eq = evaluate(board, pre_move_is_race);
    float p_win = static_cast<float>(std::clamp((eq + 1.0) / 2.0, 0.0, 1.0));
    return {p_win, 0.0f, 0.0f, 0.0f, 0.0f};
}

std::array<float, NUM_OUTPUTS> Strategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    return evaluate_probs(board, is_race(pre_move_board));
}

int Strategy::best_move_index(const std::vector<Board>& candidates,
                              bool pre_move_is_race) const {
    int best_idx = 0;
    double best_val = -1e30;

    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        double val = evaluate(candidates[i], pre_move_is_race);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::best_move_index(const std::vector<Board>& candidates,
                              const Board& pre_move_board) const {
    return best_move_index(candidates, is_race(pre_move_board));
}

// ----- Default batch evaluation implementations -----
// These loop over candidates individually. Concrete strategies override
// with optimized batch encoding + forward pass implementations.

int Strategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        double eq;
        if (r != GameResult::NOT_OVER) {
            eq = compute_equity(terminal_probs(r));
        } else {
            eq = compute_equity(evaluate_probs(candidates[i], pre_move_board));
        }
        if (equities) equities[i] = eq;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    return evaluate_candidates_equity(candidates, pre_move_board, equities);
}

int Strategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* probs_out) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = compute_equity(probs);
        } else {
            probs = evaluate_probs(candidates[i], pre_move_board);
            eq = compute_equity(probs);
        }
        if (equities) equities[i] = eq;
        if (probs_out) probs_out[i] = probs;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
        }
    }
    return best_idx;
}

int Strategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    int best_idx = 0;
    double best_eq = -1e30;
    std::array<float, NUM_OUTPUTS> best_probs{};
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        GameResult r = check_game_over(candidates[i]);
        std::array<float, NUM_OUTPUTS> probs;
        double eq;
        if (r != GameResult::NOT_OVER) {
            probs = terminal_probs(r);
            eq = compute_equity(probs);
        } else {
            probs = evaluate_probs(candidates[i], pre_move_board);
            eq = compute_equity(probs);
        }
        if (equities) equities[i] = eq;
        if (eq > best_eq) {
            best_eq = eq;
            best_idx = i;
            best_probs = probs;
        }
    }
    if (best_probs_out) *best_probs_out = best_probs;
    return best_idx;
}

} // namespace bgbot
