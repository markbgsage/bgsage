// SPDX-License-Identifier: MPL-2.0
// Profile neural inference on the legal sibling corpus shared by the CPU studies.
#include "bgbot/neural_net.h"
#include "bgbot/cube.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace bgbot;

// Only the neural evaluator is timed. Make accidental use of cube ranking fail.
float bgbot::cl2cf(const std::array<float, NUM_OUTPUTS>&, const CubeInfo&, float) {
    std::abort();
}

struct Sibling {
    bool first;
    Board board;
};

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[1]);
    std::string key, previous;
    std::vector<Sibling> siblings;
    while (input >> key) {
        Sibling item{key != previous, {}};
        for (int& point : item.board) input >> point;
        siblings.push_back(item);
        previous = key;
    }
    NeuralNetwork net(400, EXTENDED_CONTACT_INPUTS);
    if (siblings.empty() || !net.load_weights(argv[2])) return 3;
    std::ofstream outputs(argv[3], std::ios::binary);
    std::array<float, 400> hidden;
    std::array<float, EXTENDED_CONTACT_INPUTS> inputs;
    Board base;
    for (int direct : {0, 1}) {
        for (int rep = 0; rep < 4; ++rep) {
            double checksum = 0;
            auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < 32; ++iteration) {
                for (const auto& sibling : siblings) {
                    std::array<float, NN_OUTPUTS> p;
                    if (sibling.first) {
                        base = sibling.board;
                        auto in = compute_extended_contact_inputs(base);
                        p = net.forward_save_base(in.data(), hidden.data(), inputs.data());
                    } else if (direct) {
                        p = net.forward_board_from_base(sibling.board, base, hidden.data(), inputs.data());
                    } else {
                        auto in = compute_extended_contact_inputs(sibling.board);
                        p = net.forward_from_base(in.data(), hidden.data(), inputs.data());
                    }
                    if (rep == 0 && iteration == 0)
                        outputs.write(reinterpret_cast<const char*>(p.data()), sizeof(p));
                    checksum += p[iteration % NN_OUTPUTS];
                }
            }
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
            std::cout << direct << " " << rep << " " << ns / (32 * siblings.size()) << " " << checksum << "\n";
        }
    }
    return outputs ? 0 : 3;
}
