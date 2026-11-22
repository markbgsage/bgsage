// SPDX-License-Identifier: MPL-2.0
#include "bgbot/moves.h"
#include "bgbot/pubeval.h"
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace bgbot;

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    std::ifstream input(argv[1]);
    std::ofstream boards(argv[2], std::ios::binary);
    std::ofstream scores(argv[3], std::ios::binary);
    if (!input || !boards || !scores) return 2;
    PubEval tesauro(PubEval::WeightSource::TESAURO);
    PubEval listnet(PubEval::WeightSource::LISTNET);
    Board board;
    std::vector<Board> output;
    uint64_t calls = 0, endpoints = 0;
    while (input >> board[0]) {
        for (int i = 1; i < 26; ++i) if (!(input >> board[i])) return 3;
        for (int d1 = 1; d1 <= 6; ++d1) {
            for (int d2 = 1; d2 <= 6; ++d2) {
                possible_boards_unsorted(board, d1, d2, output);
                const uint32_t count = static_cast<uint32_t>(output.size());
                boards.write(reinterpret_cast<const char*>(&count), sizeof(count));
                for (const auto& child : output) {
                    boards.write(reinterpret_cast<const char*>(child.data()), 26 * sizeof(int));
                    const double values[] = {
                        tesauro.evaluate(child, false), tesauro.evaluate(child, true),
                        listnet.evaluate(child, false), listnet.evaluate(child, true)
                    };
                    scores.write(reinterpret_cast<const char*>(values), sizeof(values));
                }
                ++calls;
                endpoints += count;
            }
        }
    }
    std::cout << calls << " ordered dice cases, " << endpoints << " endpoints\n";
    return boards && scores ? 0 : 4;
}
