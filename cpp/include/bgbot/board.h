#pragma once

#include "types.h"

namespace bgbot {

// Flip the board to the opponent's perspective.
// After player 1 moves, we flip so player 2 sees the board from their POV.
Board flip(const Board& b);

// Check if the game is over. Returns GameResult (0 if not over).
GameResult check_game_over(const Board& b);

// Is this a race position? (No contact between players' checkers)
bool is_race(const Board& b);

// Is this a crashed position? One side has all checkers in their home board
// plus opponent's side (no outfield checkers), with at least one checker
// on the opponent's side, and no checkers on their 6-point.
bool is_crashed(const Board& b);

// Count how many checkers player 1 has on the board (including bar).
int player_checkers_on_board(const Board& b);

// Count how many checkers player 1 has borne off.
int player_borne_off(const Board& b);

// Count how many checkers player 2 has borne off.
int opponent_borne_off(const Board& b);

// Compute pip counts for both players.
// Returns (player_pips, opponent_pips).
std::pair<int, int> pip_counts(const Board& b);

} // namespace bgbot
