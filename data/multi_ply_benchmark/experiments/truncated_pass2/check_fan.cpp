// SPDX-License-Identifier: MPL-2.0
// Independent check of the optimized variance-reduction winners against
// the existing per-roll selector, including derived-model and DB fallbacks.
#include "bgbot/rollout_vr.h"
#include "bgbot/moves.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace bgbot;
using Fan = std::array<std::vector<Board>, 21>;
using Outputs = std::array<std::array<float, NUM_OUTPUTS>, 21>;

int main(int argc, char** argv) {
    if (argc != 4) throw std::invalid_argument("expected weight manifest, boards, bearoff database");
    std::ifstream manifest(argv[1]), positions(argv[2]);
    std::vector<std::string> weights;
    std::vector<int> sizes;
    std::string path;
    int width;
    while (manifest >> width >> path) {
        sizes.push_back(width);
        weights.push_back(path);
    }
    BackgameAwarePairStrategy model(weights, sizes, true);
    auto db = std::make_unique<BearoffDB>();
    if (!db->load(argv[3])) throw std::runtime_error("bearoff data missing");
    const CubeInfo money{1, CubeOwner::CENTERED, {}, -1, true, true};
    const CubeInfo match{2, CubeOwner::OPPONENT, {3, 5, false}};
    const CubeInfo dmp{2, CubeOwner::PLAYER, {1, 1, true}};

    int visited = 0, fallback = 0, rolls = 0, passes = 0, changed = 0;
    double max_prob = 0, max_loss = 0;
    Board board;
    Fan fan;
    Outputs output;
    std::array<int, 21> picks;
    while (positions >> board[0]) {
        for (size_t i = 1; i < board.size(); ++i)
            if (!(positions >> board[i])) throw std::runtime_error("incomplete board");
        ++visited;
        int roll = 0;
        for (int a = 1; a <= 6; ++a)
            for (int b = a; b <= 6; ++b)
                possible_boards_unsorted(board, a, b, fan[roll++]);
        for (const auto& driver : {money, match, dmp}) {
            const CubeInfo branches[] = {driver, money};
            const float x = 0.68f;
            if (!try_rollout_vr_rolls(model, db.get(), board, fan, branches, 2, x, picks, output)) {
                ++fallback;
                continue;
            }
            for (size_t r = 0; r < fan.size(); ++r) {
                const auto& moves = fan[r];
                int reference[2]{};
                if (!moves.empty())
                    model.best_move_index_cubeful_multi(moves, board, branches, 2, x, reference);
                const int expected = moves.empty() ? -1 : reference[0];
                if (picks[r] < (moves.empty() ? -1 : 0) || picks[r] >= static_cast<int>(moves.size()))
                    throw std::runtime_error("invalid endpoint");
                const auto& chosen = moves.empty() ? board : moves[picks[r]];
                auto p = model.evaluate_probs(chosen, board);
                clamp_probs_to_board(p, chosen);
                for (int k = 0; k < NUM_OUTPUTS; ++k)
                    max_prob = std::max(max_prob, static_cast<double>(std::abs(p[k] - output[r][k])));
                if (picks[r] != expected) {
                    ++changed;
                    auto ref = model.evaluate_probs(moves[expected], board);
                    clamp_probs_to_board(ref, moves[expected]);
                    max_loss = std::max(max_loss, static_cast<double>(cl2cf(ref, driver, x) - cl2cf(p, driver, x)));
                }
                ++rolls;
                if (moves.empty() || (moves.size() == 1 && moves[0] == board)) ++passes;
            }
        }
    }
    Board home{};
    home[1] = 3;
    home[24] = -3;
    if (try_rollout_vr_rolls(model, db.get(), home, fan, &money, 1, 0.68f, picks, output) ||
        try_rollout_vr_rolls(model, nullptr, STARTING_BOARD, fan, nullptr, 0, 0.68f, picks, output))
        throw std::runtime_error("database/cubeless fallback failed");
    Board terminal{};
    terminal[24] = -7;
    if (try_rollout_vr_rolls(model, nullptr, terminal, fan, &money, 1, 0.68f, picks, output))
        throw std::runtime_error("terminal fallback failed");
    Fan empty;
    if (!try_rollout_vr_rolls(model, nullptr, STARTING_BOARD, empty, &money, 1, 0.68f, picks, output))
        throw std::runtime_error("unexpected empty-roll fallback");
    auto pass = model.evaluate_probs(STARTING_BOARD, STARTING_BOARD);
    clamp_probs_to_board(pass, STARTING_BOARD);
    for (size_t r = 0; r < empty.size(); ++r) {
        if (picks[r] != -1) throw std::runtime_error("empty roll returned an endpoint index");
        for (int k = 0; k < NUM_OUTPUTS; ++k)
            if (std::abs(pass[k] - output[r][k]) > 1.0e-5f) throw std::runtime_error("empty roll probabilities");
    }
    struct Derived : BackgameAwarePairStrategy { using BackgameAwarePairStrategy::BackgameAwarePairStrategy; };
    Derived derived(weights, sizes, true);
    if (try_rollout_vr_rolls(derived, nullptr, STARTING_BOARD, fan, &money, 1, 0.68f, picks, output))
        throw std::runtime_error("derived model bypassed");
    std::cout << "{\"boards\":" << visited << ",\"delegated_fans\":" << fallback
              << ",\"rolls\":" << rolls << ",\"forced_passes\":" << passes << ",\"different_winners\":"
              << changed << ",\"max_probability_delta_same_board\":" << max_prob
              << ",\"max_reference_equity_loss\":" << max_loss << "}\n";
    return visited && rolls && passes && max_prob < 0.0025 && max_loss < 0.005 ? 0 : 1;
}
