// SPDX-License-Identifier: MPL-2.0
#include "bgbot/moves.h"
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace bgbot;

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    std::ifstream input(argv[1]);
    if (!input) return 2;
    const int repeats = std::stoi(argv[2]);
    std::vector<Board> boards, results;
    Board board;
    while (input >> board[0]) {
        for (int i = 1; i < 26; ++i) if (!(input >> board[i])) return 3;
        boards.push_back(board);
    }
    for (bool doubles : {false, true}) {
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t endpoints = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            for (const auto& position : boards) {
                for (int d1 = 1; d1 <= 6; ++d1) {
                    for (int d2 = d1; d2 <= 6; ++d2) {
                        if ((d1 == d2) != doubles) continue;
                        possible_boards_unsorted(position, d1, d2, results);
                        endpoints += results.size();
                    }
                }
            }
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        std::cout << (doubles ? "doubles " : "mixed ") << endpoints << " "
                  << std::chrono::duration<double>(elapsed).count() << "\n";
    }
}
