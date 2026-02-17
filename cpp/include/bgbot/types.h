#pragma once

#include <array>
#include <vector>
#include <cstdint>

namespace bgbot {

// Board is a 26-element array of ints.
// Index 0:    player 2 checkers on the bar (>= 0)
// Index 1-24: board points. Positive = player 1, negative = player 2.
// Index 25:   player 1 checkers on the bar (>= 0)
using Board = std::array<int, 26>;

// Starting position for a new game
inline constexpr Board STARTING_BOARD = {
    0,         // [0]  p2 bar
    -2, 0, 0, 0, 0, 5,   // [1-6]   p1 home board / p2 outer
    0, 3, 0, 0, 0, -5,   // [7-12]  p1 outer / p2 home board
    5, 0, 0, 0, -3, 0,   // [13-18]
    -5, 0, 0, 0, 0, 2,   // [19-24]
    0          // [25] p1 bar
};

// Game outcome from the perspective of player 1.
// Positive = player 1 wins, negative = player 2 wins.
// Magnitude: 1 = single, 2 = gammon, 3 = backgammon.
// 0 = game not over.
enum class GameResult : int {
    NOT_OVER        =  0,
    WIN_SINGLE      =  1,
    WIN_GAMMON      =  2,
    WIN_BACKGAMMON  =  3,
    LOSS_SINGLE     = -1,
    LOSS_GAMMON     = -2,
    LOSS_BACKGAMMON = -3,
};

} // namespace bgbot
