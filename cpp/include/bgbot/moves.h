// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
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

// Same legality rules as possible_boards, but duplicate boards are rejected
// at insertion time via a generation-stamped hash table instead of the
// insertion-sorted dedup pass, and the output is in first-seen generation
// order rather than board-sorted order. Several times faster on rolls with
// many candidates (no O(n^2) 104-byte board shifting). Downstream argmax
// consumers treat candidates symmetrically, so only exact-tie picks can
// differ from the sorted variant. Used by the evaluation hot paths (rollout
// trials, the N-ply cubeful recursion).
void possible_boards_unsorted(const Board& board, int die1, int die2,
                              std::vector<Board>& results);

} // namespace bgbot
