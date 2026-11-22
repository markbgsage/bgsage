// SPDX-License-Identifier: MPL-2.0
// Regression case for the unsorted generator's original 2,048-slot table.
#include "bgbot/moves.h"
#include <algorithm>
#include <iostream>

int main() {
    const bgbot::Board board = {
        0, -15, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
        1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0};
    std::vector<bgbot::Board> actual;
    for (int d1 = 1; d1 <= 6; ++d1) {
        for (int d2 = d1; d2 <= 6; ++d2) {
            bgbot::possible_boards_unsorted(board, d1, d2, actual);
            std::sort(actual.begin(), actual.end());
            if (actual != bgbot::possible_boards(board, d1, d2)) return 1;
            if (d1 == 1 && d2 == 1 && actual.size() != 2054) return 2;
        }
    }
    std::cout << "All 21 rolls match the sorted generator, including 2,054 endpoints on 1-1.\n";
}
