// SPDX-License-Identifier: MPL-2.0
// Directly validate sparse features, saturation and NN results on sibling and unrelated boards.
#include "bgbot/neural_net.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace bgbot;

struct Counts {
    int boards = 0, saturated = 0, output_floats = 0;
    double max_shift = 0;
};

template<size_t Count, class Encode>
void verify(const Board& board, const Board& base, const NeuralNetwork& nn, Encode encode, Counts& stats) {
    const auto a = encode(base), b = encode(board);
    int indices[64];
    float amounts[64];
    const int count = board_input_deltas(board, base, a.data(), Count == TESAURO_INPUTS, indices, amounts);
    int found = 0;
    for (size_t i = 0; i < Count; ++i) {
        const float expected = b[i] - a[i];
        if (expected == 0) continue;
        if (found < count && (indices[found] != int(i) || std::abs(amounts[found]-expected) > 1e-7))
            throw std::runtime_error("wrong feature index or delta");
        ++found;
    }
    if (std::min(found, 64) != count) throw std::runtime_error("incomplete feature changes");
    float hidden[27];
    std::array<float,Count> saved;
    nn.forward_save_base(a.data(), hidden, saved.data());
    const auto direct = nn.forward_board_from_base(board, base, hidden, saved.data());
    const auto dense = nn.forward_from_base(b.data(), hidden, saved.data());
    for (int k = 0; k < NN_OUTPUTS; ++k) {
        const double d = std::abs(direct[k]-dense[k]);
        stats.max_shift = std::max(stats.max_shift, d);
        if (d > 1e-5) throw std::runtime_error("wrong NN output");
    }
    ++stats.boards;
    stats.output_floats += NN_OUTPUTS;
    stats.saturated += count == 64;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    NeuralNetwork race(27, TESAURO_INPUTS), contact(27, EXTENDED_CONTACT_INPUTS);
    Counts stats;
    auto check = [&](const Board& board, const Board& base) {
        verify<TESAURO_INPUTS>(board, base, race, compute_tesauro_inputs, stats);
        verify<EXTENDED_CONTACT_INPUTS>(board, base, contact, compute_extended_contact_inputs, stats);
    };
    std::ifstream corpus(argv[1]);
    std::string id, previous;
    Board board, base;
    while (corpus >> id) {
        for (int& n : board) corpus >> n;
        if (id != previous) base = board;
        check(board, base);
        previous = id;
    }
    std::ifstream random(argv[2], std::ios::binary);
    bool first = true;
    while (random.read(reinterpret_cast<char*>(board.data()), sizeof(board))) {
        if (first) base = board;
        check(board, base);
        base = board;
        first = false;
    }
    if (stats.boards < 100000 || !stats.saturated) return 3;
    std::cout << "{\"encodings\":" << stats.boards << ",\"at_sparse_limit\":" << stats.saturated
              << ",\"nn_outputs\":" << stats.output_floats << ",\"max_nn_shift\":" << stats.max_shift << "}\n";
}
