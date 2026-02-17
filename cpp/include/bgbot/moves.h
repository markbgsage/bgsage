#pragma once

#include "types.h"

namespace bgbot {

// Generate all boards reachable by moving one checker with a single die value.
// Appends results to `out`. Caller must clear `out` first if desired.
// Board is from player 1's perspective (player 1 is on roll).
void possible_boards_one_die(const Board& b, int die, std::vector<Board>& out);

// Generate all possible resulting board positions after applying a dice roll.
// Returns a vector of unique post-move boards. If no moves are possible,
// returns a vector containing just the input board (player loses their turn).
// The board is always from player 1's perspective (player 1 is on roll).
std::vector<Board> possible_boards(const Board& board, int die1, int die2);

// Same as above, but writes results into a pre-allocated vector.
// The vector is cleared first, then filled. Avoids heap allocation
// when called in a loop (caller reuses the vector).
void possible_boards(const Board& board, int die1, int die2,
                     std::vector<Board>& results);

} // namespace bgbot
