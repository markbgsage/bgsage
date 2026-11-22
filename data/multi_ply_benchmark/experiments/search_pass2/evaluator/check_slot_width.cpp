// SPDX-License-Identifier: MPL-2.0
// Check the C++ prebuilt pair constructor through the public cubeful tree.
#include "bgbot/cube.h"
#include "bgbot/neural_net.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace bgbot;

// The generic strategy path performs a full, slot-appropriate encoding for
// every candidate. Its network weights and routing are shared with the batch.
class DenseReference final : public Strategy {
public:
    explicit DenseReference(const GamePlanPairStrategy& model) : model_(model) {}
    double evaluate(const Board& board, bool race) const override {
        return model_.evaluate(board, race);
    }
    std::array<float, NUM_OUTPUTS> evaluate_probs(const Board& board, bool race) const override {
        return model_.evaluate_probs(board, race);
    }
private:
    const GamePlanPairStrategy& model_;
};

using Nets = std::array<std::shared_ptr<NeuralNetwork>, NUM_PAIR_NNS>;

Nets make_nets(int race_width, int contact_width) {
    Nets nets;
    nets.fill(std::make_shared<NeuralNetwork>(27, contact_width, 0.2f, 29));
    nets[0] = std::make_shared<NeuralNetwork>(27, race_width, 0.2f, 13);
    return nets;
}

bool compare(const Board& board, int race_width, int contact_width) {
    const GamePlanPairStrategy model(make_nets(race_width, contact_width));
    const DenseReference dense(model);
    const auto expected = cubeful_probs_and_equity_nply(board, CubeInfo{}, dense, 2);
    try {
        const auto actual = cubeful_probs_and_equity_nply(board, CubeInfo{}, model, 2);
        float error = std::abs(actual.equity - expected.equity);
        for (int i = 0; i < NUM_OUTPUTS; ++i)
            error = std::max(error, std::abs(actual.probs[i] - expected.probs[i]));
        std::cout << race_width << '/' << contact_width << " max_abs_error=" << error << '\n';
        return std::isfinite(error) && error <= 0.000002f;
    } catch (const std::exception& ex) {
        std::cout << race_width << '/' << contact_width << " exception=" << ex.what() << '\n';
        return false;
    }
}

int main() {
    Board race{};
    race[1] = 2; race[3] = 3; race[5] = 5; race[7] = 5;
    race[18] = -5; race[20] = -5; race[22] = -3; race[24] = -2;
    bool ok = compare(STARTING_BOARD, 196, 196);
    ok &= compare(STARTING_BOARD, 196, 183);
    ok &= compare(race, 167, 244);
    ok &= compare(STARTING_BOARD, 196, 244);
    ok &= compare(race, 196, 244);
    for (auto widths : {std::pair{197, 244}, std::pair{196, 245}}) {
        try {
            const GamePlanPairStrategy invalid(make_nets(widths.first, widths.second));
            std::cout << "Oversized network accepted\n";
            ok = false;
        } catch (const std::invalid_argument&) {
            std::cout << widths.first << '/' << widths.second << " rejected\n";
        }
    }
    return ok ? 0 : 1;
}
