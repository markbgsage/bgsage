// SPDX-License-Identifier: MPL-2.0
#include "bgbot/moves.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace bgbot {
void reference_sorted(const Board&, int, int, std::vector<Board>&);
void reference_unsorted(const Board&, int, int, std::vector<Board>&);
}
using namespace bgbot;

void add_near_home_positions(std::vector<Board>& boards) {
    // Exhaust four active checkers spread over 1-9, with and without a blot
    // or an opposing made point. This includes immediate and mid-roll bearoff.
    for (int a = 1; a <= 9; ++a)
        for (int b = a; b <= 9; ++b)
            for (int c = b; c <= 9; ++c)
                for (int d = c; d <= 9; ++d) {
                    Board board{};
                    for (int point : {a, b, c, d}) ++board[point];
                    for (int j = 1; j <= 9; ++j) {
                        if (board[j]) continue;
                        board[j] = -1 - ((a + j) % 3 == 0);
                        boards.push_back(board);
                        board[j] = 0;
                    }
                    boards.push_back(board);
                }
}

void add_random_positions(std::vector<Board>& boards) {
    std::mt19937 rng(1234567);
    for (int i = 0; i < 12000; ++i) {
        Board board{};
        for (int sign : {-1, 1}) {
            const int count = 1 + rng() % 15;
            const int bar = sign > 0 ? 25 : 0;
            for (int j = 0; j < count; ++j) {
                if (rng() % 18 == 0) {
                    ++board[bar];
                    continue;
                }
                int point;
                do { point = 1 + rng() % 24; } while (board[point] * sign < 0);
                board[point] += sign;
            }
        }
        boards.push_back(board);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    std::ifstream input(argv[1]);
    if (!input) return 2;
    std::vector<Board> boards, expected, actual;
    Board board{};
    while (input >> board[0]) {
        for (int i = 1; i < 26; ++i) if (!(input >> board[i])) return 2;
        boards.push_back(board);
    }
    add_near_home_positions(boards);
    add_random_positions(boards);
    std::uint64_t cases = 0, endpoints = 0;
    for (const Board& start : boards) {
        for (int d1 = 1; d1 <= 6; ++d1) {
            for (int d2 = 1; d2 <= 6; ++d2) {
                reference_unsorted(start, d1, d2, expected);
                possible_boards_unsorted(start, d1, d2, actual);
                ++cases;
                endpoints += actual.size();
                if (actual == expected) {
                    reference_sorted(start, d1, d2, expected);
                    possible_boards(start, d1, d2, actual);
                    if (actual == expected) continue;
                    std::cout << "Sorted ";
                }
                std::cout << "mismatch at case " << cases << ", dice " << d1 << " " << d2
                          << ", baseline " << expected.size() << ", candidate " << actual.size() << "\n";
                for (int count : start) std::cout << count << " ";
                std::cout << "\n";
                return 3;
            }
        }
    }
    std::cout << "verified " << boards.size() << " boards, " << cases << " ordered cases, "
              << endpoints << " endpoints per API\n";
}
