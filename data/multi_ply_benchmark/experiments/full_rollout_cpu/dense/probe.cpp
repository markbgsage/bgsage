// SPDX-License-Identifier: MPL-2.0
#include "bgbot/neural_net.h"
#include <algorithm>
#include <cstring>
#include <vector>

// This shim resolves the public C++ methods from the engine loaded by check.py.
// It uses the headers from the same engine export as the binary under test.
extern "C" int probe_hidden(int width, int hidden, int seed, const float* inputs,
                            const float* changed, float* result) {
    bgbot::NeuralNetwork net(hidden, width, 0.1f, seed);
    float* sums = result;
    float* saved_inputs = sums + hidden;
    const auto prediction = net.forward_save_base(inputs, sums, saved_inputs);
    const auto delta_prediction = net.forward_from_base(changed, sums, saved_inputs);
    float* outputs = saved_inputs + width;
    std::copy(prediction.begin(), prediction.end(), outputs);
    std::copy(delta_prediction.begin(), delta_prediction.end(), outputs + prediction.size());

#ifdef CHECK_DIRECT_PREPARATION
    std::vector<float> direct_sums(hidden);
    std::vector<float> direct_inputs(width);
    net.save_hidden_base(inputs, direct_sums.data(), direct_inputs.data());
    const auto direct_result = net.forward_from_base(changed, direct_sums.data(), direct_inputs.data());
    if (std::memcmp(sums, direct_sums.data(), hidden * sizeof(float)) != 0) return 1;
    if (std::memcmp(saved_inputs, direct_inputs.data(), width * sizeof(float)) != 0) return 2;
    if (std::memcmp(delta_prediction.data(), direct_result.data(), direct_result.size() * sizeof(float)) != 0) return 3;
#endif
    return 0;
}
