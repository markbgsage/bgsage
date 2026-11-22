// SPDX-License-Identifier: MPL-2.0
// Run with build.ps1 -Checks. Paths are supplied by that script.
#include "bgbot/bearoff.h"
#include "bgbot/cube.h"
#include "bgbot/moves.h"
#include "bgbot/neural_net.h"
#include <iostream>
#include <stdexcept>

using namespace bgbot;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// A downstream derived model may implement a distinct evaluator.
struct Custom : BackgameAwarePairStrategy {
    using BackgameAwarePairStrategy::BackgameAwarePairStrategy;
    mutable size_t calls = 0;
    std::array<float, NUM_OUTPUTS> evaluate_probs(const Board& b, const Board& pre) const override {
        ++calls;
        return BackgameAwarePairStrategy::evaluate_probs(b, pre);
    }
};

int main(int argc, char** argv) {
    check(argc == 2, "pass bearoff database path");
    NeuralNetwork contact(25, EXTENDED_CONTACT_INPUTS), race(25, TESAURO_INPUTS);
    std::vector<float> base(EXTENDED_CONTACT_INPUTS), hidden(25), saved(base.size());
    contact.forward_save_base(base.data(), hidden.data(), saved.data());
    for (const int changed : {64, 65, 128, EXTENDED_CONTACT_INPUTS}) {
        auto other = base;
        std::fill_n(other.begin(), changed, 1.0f);
        check(contact.forward(other.data()) ==
              contact.forward_from_base(other.data(), hidden.data(), saved.data()), "dense fallback");
    }
    check(contact.save_weights("contact.weights") && race.save_weights("race.weights"), "fixture weights");
    std::vector<std::string> weights(NUM_BACKGAME_PAIR_NNS, "contact.weights");
    weights[0] = "race.weights";
    std::vector<int> hidden_sizes(weights.size(), 25);
    auto pair = std::make_shared<BackgameAwarePairStrategy>(weights, hidden_sizes);
    const Board board = {0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0};
    const std::array<CubeInfo, 2> cubes = {CubeInfo{1, CubeOwner::CENTERED},
                                            CubeInfo{2, CubeOwner::OPPONENT, {3, 5, false}}};
    std::array<int,2> selected{};
    const float x = 0.65f;
    for (int d1 = 1; d1 <= 6; ++d1) {
        for (int d2 = 1; d2 <= d1; ++d2) {
            const auto moves = possible_boards(board, d1, d2);
            pair->best_move_index_cubeful_multi(moves, board, cubes.data(), 2, x, selected.data());
            for (int j = 0; j < 2; ++j) {
                std::vector<std::array<float, NUM_OUTPUTS>> probs;
                std::vector<float> cf;
                for (const auto& endpoint : moves) {
                    auto p = pair->evaluate_probs(endpoint, board);
                    clamp_probs_to_board(p, endpoint);
                    probs.push_back(p);
                    cf.push_back(cl2cf(p, cubes[j], x));
                }
                const int expected = pick_best_with_tiebreak(cf.data(), probs.data(), int(cf.size()));
                check(selected[j] == expected, "dense vs batch winner");
                check(pair->best_move_index_cubeful(moves, board, cubes[j], x) == expected,
                      "single cube winner");
            }
        }
    }
    auto moves = possible_boards(board, 2, 1);
    Custom custom(weights, hidden_sizes);
    custom.best_move_index_cubeful_multi(moves, board, cubes.data(), 2, x, selected.data());
    check(custom.calls == moves.size(), "custom scalar evaluator");

    Board won{};
    won[24] = -1;
    moves.push_back(won);
    pair->best_move_index_cubeful_multi(moves, board, cubes.data(), 2, x, selected.data());
    check(selected[0] == int(moves.size())-1 && selected[1] == selected[0], "terminal winner");

    auto db = std::make_unique<BearoffDB>();
    check(db->load(argv[1]), "load database");
    BearoffStrategy wrapped(pair, db.get());
    Board near{};
    near[6] = 6;
    near[19] = -6;
    moves = possible_boards(near, 2, 1);
    wrapped.best_move_index_cubeful_multi(moves, near, cubes.data(), 2, x, selected.data());
    for (int j = 0; j < 2; ++j) {
        auto cf = [&](const Board& b) { return cl2cf(db->lookup_probs(b, true), cubes[j], x); };
        int best = 0;
        for (int i = 1; i < int(moves.size()); ++i)
            if (cf(moves[i]) > cf(moves[best])) best = i;
        check(selected[j] == best, "all-bearoff exact winner");
        check(wrapped.evaluate_probs(moves[best], near) == db->lookup_probs(moves[best], true),
              "bearoff winner probabilities");
    }
    moves.push_back(board);
    std::array<int,2> plain{};
    pair->best_move_index_cubeful_multi(moves, near, cubes.data(), 2, x, plain.data());
    wrapped.best_move_index_cubeful_multi(moves, near, cubes.data(), 2, x, selected.data());
    check(selected == plain, "mixed DB boundary selection delegates to base");
    std::cout << "Passed dense fallback, 42 dense/one/multi-cube index comparisons, "
                 "custom evaluator, terminal, all-bearoff and mixed-boundary checks.\n";
}
