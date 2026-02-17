#include "bgbot/moves.h"
#include <algorithm>

namespace bgbot {

void possible_boards_one_die(const Board& b, int die, std::vector<Board>& out) {
    // If player has checkers on the bar, must enter them first
    if (b[25] > 0) {
        int target = 25 - die;
        if (b[target] >= -1) {
            Board nb = b;
            if (nb[target] == -1) {
                nb[0] += 1;
                nb[target] = 0;
            }
            nb[25] -= 1;
            nb[target] += 1;
            out.push_back(nb);
        }
        return; // Must enter from bar
    }

    // Can player bear off?
    bool can_bear_off = true;
    for (int i = 7; i <= 24; ++i) {
        if (b[i] > 0) { can_bear_off = false; break; }
    }

    // Home board moves (points 1-6)
    for (int i = 1; i <= 6; ++i) {
        if (b[i] <= 0) continue;

        int target = i - die;
        if (target > 0) {
            if (b[target] >= -1) {
                Board nb = b;
                if (nb[target] == -1) {
                    nb[0] += 1;
                    nb[target] = 0;
                }
                nb[i] -= 1;
                nb[target] += 1;
                out.push_back(nb);
            }
        } else if (can_bear_off) {
            if (target == 0) {
                Board nb = b;
                nb[i] -= 1;
                out.push_back(nb);
            } else {
                bool highest = true;
                for (int j = i + 1; j <= 6; ++j) {
                    if (b[j] > 0) { highest = false; break; }
                }
                if (highest) {
                    Board nb = b;
                    nb[i] -= 1;
                    out.push_back(nb);
                }
            }
        }
    }

    // Outer board moves (points 7-24)
    for (int i = 7; i <= 24; ++i) {
        if (b[i] <= 0) continue;

        int target = i - die;
        if (target >= 1 && b[target] >= -1) {
            Board nb = b;
            if (nb[target] == -1) {
                nb[0] += 1;
                nb[target] = 0;
            }
            nb[i] -= 1;
            nb[target] += 1;
            out.push_back(nb);
        }
    }
}

namespace {

// Check if player 1 has any checkers on the board
bool has_checkers(const Board& b) {
    if (b[25] > 0) return true;
    for (int i = 1; i <= 24; ++i) {
        if (b[i] > 0) return true;
    }
    return false;
}

// Recursively generate all boards reachable with the given sequence of dice.
// Uses a shared scratch vector to avoid per-recursion heap allocation.
void generate_boards(const Board& b, const int* dice, int n_dice,
                     std::vector<Board>& results,
                     std::vector<Board>& scratch) {
    if (!has_checkers(b)) {
        results.push_back(b);
        return;
    }

    scratch.clear();
    possible_boards_one_die(b, dice[0], scratch);

    if (n_dice == 1) {
        if (!scratch.empty()) {
            results.insert(results.end(), scratch.begin(), scratch.end());
        }
        return;
    }

    if (scratch.empty()) {
        return;
    }

    // Save intermediate results to avoid scratch being overwritten in recursion.
    // Use a local copy since recursion will reuse scratch.
    // For 2-die non-doubles: max depth is 2, so one level of copying.
    // For doubles: max depth is 4, but each level has few boards.
    auto intermediates = scratch;  // copy

    for (const auto& nb : intermediates) {
        size_t before = results.size();
        generate_boards(nb, dice + 1, n_dice - 1, results, scratch);
        if (results.size() == before) {
            // Couldn't use remaining dice from this position
            results.push_back(nb);
        }
    }
}

} // anonymous namespace

std::vector<Board> possible_boards(const Board& board, int die1, int die2) {
    std::vector<Board> results;
    results.reserve(32);
    std::vector<Board> scratch;
    scratch.reserve(16);

    if (die1 == die2) {
        int dice[4] = {die1, die1, die1, die1};
        generate_boards(board, dice, 4, results, scratch);
    } else {
        int dice_a[2] = {die1, die2};
        int dice_b[2] = {die2, die1};
        generate_boards(board, dice_a, 2, results, scratch);
        generate_boards(board, dice_b, 2, results, scratch);
    }

    if (results.empty()) {
        results.push_back(board);
        return results;
    }

    // Deduplicate
    std::sort(results.begin(), results.end());
    results.erase(std::unique(results.begin(), results.end()), results.end());

    return results;
}

void possible_boards(const Board& board, int die1, int die2,
                     std::vector<Board>& results) {
    results.clear();
    thread_local std::vector<Board> scratch;
    scratch.clear();
    scratch.reserve(16);

    if (die1 == die2) {
        int dice[4] = {die1, die1, die1, die1};
        generate_boards(board, dice, 4, results, scratch);
    } else {
        int dice_a[2] = {die1, die2};
        int dice_b[2] = {die2, die1};
        generate_boards(board, dice_a, 2, results, scratch);
        generate_boards(board, dice_b, 2, results, scratch);
    }

    if (results.empty()) {
        results.push_back(board);
        return;
    }

    std::sort(results.begin(), results.end());
    results.erase(std::unique(results.begin(), results.end()), results.end());
}

} // namespace bgbot
