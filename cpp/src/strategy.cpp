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

} // namespace bgbot
