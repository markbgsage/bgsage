#pragma once

#include "strategy.h"
#include <array>

namespace bgbot {

// PubEval: Tesauro's published linear evaluation function for backgammon.
// Uses 122 inputs and separate weight vectors for race and contact positions.
class PubEval : public Strategy {
public:
    enum class WeightSource { TESAURO, LISTNET };

    explicit PubEval(WeightSource src = WeightSource::TESAURO);

    double evaluate(const Board& board, bool pre_move_is_race) const override;

private:
    // Compute the 122 PubEval inputs from a board position
    void compute_inputs(const Board& board, double* inputs) const;

    std::array<double, 122> weights_contact_;
    std::array<double, 122> weights_race_;
};

} // namespace bgbot
