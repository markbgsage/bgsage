// SPDX-License-Identifier: MPL-2.0
#pragma once

#include "bearoff.h"
#include "cube.h"
#include "neural_net.h"
#include <algorithm>
#include <typeinfo>

namespace bgbot {

inline uint32_t rollout_board_hash(const Board& board) {
    uint32_t hash = 2166136261u;
    for (int point : board) hash = (hash ^ static_cast<uint32_t>(point)) * 16777619u;
    return hash ^ (hash >> 16);
}

// All 21 rolls have the same pre-move routing board. Static cube-aware play
// follows the first live cube, and VR needs that winner's probabilities.
inline bool try_rollout_vr_rolls(
    const Strategy& base, const BearoffDB* db, const Board& board,
    const std::array<std::vector<Board>, 21>& candidates,
    const CubeInfo* cubes, int n_cubes, float cube_x, std::array<int, 21>& picks,
    std::array<std::array<float, NUM_OUTPUTS>, 21>& probs)
{
    if (!cubes || n_cubes == 0 || typeid(base) != typeid(BackgameAwarePairStrategy))
        return false;
    if (check_game_over(board) != GameResult::NOT_OVER) return false;
    // The opponent cannot move home on our roll. If already home, delegate
    // the entire fan, including mixed NN/database boundary rolls.
    if (db && board[0] == 0 &&
        std::none_of(board.begin() + 1, board.begin() + 19, [](int n) { return n < 0; }))
        return false;
    const auto& model = static_cast<const BackgameAwarePairStrategy&>(base);
    const int net = model.nn_index_for(board);
    if (net >= BLENDED_SENTINEL_BASE) return false;
    const auto& nn = *model.nn(net);

    thread_local std::vector<float> hidden, inputs, values;
    thread_local std::vector<std::array<float, NUM_OUTPUTS>> outputs;
    hidden.resize(nn.n_hidden());
    inputs.resize(nn.n_inputs());
    const auto save = [&](const auto& encoded) {
        return nn.forward_save_base(encoded.data(), hidden.data(), inputs.data());
    };
    auto pass = net == 0 ? save(compute_tesauro_inputs(board))
                         : save(compute_extended_contact_inputs(board));
    clamp_probs_to_board(pass, board);
    // Different dice can reach the same endpoint. The generating vectors
    // stay immutable for this entire fan, so entries can refer to their boards.
    struct EvaluatedMove {
        const Board* board = nullptr;
        std::array<float, NUM_OUTPUTS> probabilities;
        float value;
    };
    std::array<EvaluatedMove, 256> evaluated;
    for (size_t roll = 0; roll < candidates.size(); ++roll) {
        const auto& moves = candidates[roll];
        if (moves.empty()) {
            picks[roll] = -1;
            probs[roll] = pass;
            continue;
        }
        values.resize(moves.size());
        outputs.resize(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) {
            auto& entry = evaluated[rollout_board_hash(moves[i]) & (evaluated.size() - 1)];
            if (entry.board && *entry.board == moves[i]) {
                outputs[i] = entry.probabilities;
                values[i] = entry.value;
                continue;
            }
            const auto result = check_game_over(moves[i]);
            auto& p = outputs[i];
            p = result == GameResult::NOT_OVER
                ? nn.forward_board_from_base(moves[i], board, hidden.data(), inputs.data())
                : terminal_probs(result);
            clamp_probs_to_board(p, moves[i]);
            values[i] = cl2cf(p, cubes[0], cube_x);
            entry = {&moves[i], p, values[i]};
        }
        const int best = moves.size() == 1 ? 0 :
            pick_best_with_tiebreak(values.data(), outputs.data(), static_cast<int>(moves.size()));
        picks[roll] = best;
        probs[roll] = outputs[best];
    }
    return true;
}

} // namespace bgbot
