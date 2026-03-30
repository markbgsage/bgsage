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

struct BoardLess {
    bool operator()(const Board& a, const Board& b) const noexcept {
        for (int i = 0; i < 26; ++i) {
            if (a[i] != b[i]) return a[i] < b[i];
        }
        return false;
    }
};

void sort_unique_boards(std::vector<Board>& results) {
    if (results.size() <= 1) return;

    constexpr BoardLess board_less{};

    // Candidate lists are usually small; maintaining a sorted unique vector
    // avoids the heavier general-purpose introsort path in the hot move-gen loop.
    if (results.size() <= 48) {
        std::vector<Board> sorted_unique;
        sorted_unique.reserve(results.size());
        for (const auto& board : results) {
            auto it = std::lower_bound(sorted_unique.begin(), sorted_unique.end(), board, board_less);
            if (it == sorted_unique.end() || *it != board) {
                sorted_unique.insert(it, board);
            }
        }
        results.swap(sorted_unique);
        return;
    }

    std::sort(results.begin(), results.end(), board_less);
    results.erase(std::unique(results.begin(), results.end()), results.end());
}

// Strict recursive generator: only adds boards where ALL n_dice are used.
// Does NOT add partial (fewer-dice) fallback results.
// This enforces the backgammon rule: must use maximum number of dice possible.
//
// Each recursion depth gets its own scratch vector so we don't have to copy the
// current level's candidate list before descending.
void generate_boards_strict(const Board& b, const int* dice, int n_dice,
                            std::vector<Board>& results,
                            std::vector<Board>* scratch_by_depth,
                            int depth = 0) {
    // All checkers borne off mid-sequence — game over, remaining dice irrelevant
    if (!has_checkers(b)) {
        results.push_back(b);
        return;
    }

    std::vector<Board>& scratch = scratch_by_depth[depth];
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

    for (const auto& nb : scratch) {
        // Only recurse — do NOT add intermediate if remaining dice can't be used.
        generate_boards_strict(nb, dice + 1, n_dice - 1, results, scratch_by_depth, depth + 1);
    }
}

} // anonymous namespace

std::vector<Board> possible_boards(const Board& board, int die1, int die2) {
    std::vector<Board> results;
    results.reserve(32);
    std::array<std::vector<Board>, 4> scratch_by_depth;
    for (auto& scratch : scratch_by_depth) {
        scratch.reserve(16);
    }

    if (die1 == die2) {
        // Doubles: must use maximum number of dice possible (4, then 3, ...)
        int dice[4] = {die1, die1, die1, die1};
        for (int n = 4; n >= 1; --n) {
            generate_boards_strict(board, dice, n, results, scratch_by_depth.data());
            if (!results.empty()) break;
        }
    } else {
        // Non-doubles: must use both dice if possible
        int dice_a[2] = {die1, die2};
        int dice_b[2] = {die2, die1};
        generate_boards_strict(board, dice_a, 2, results, scratch_by_depth.data());
        generate_boards_strict(board, dice_b, 2, results, scratch_by_depth.data());

        if (results.empty()) {
            // Can't use both dice. Must use the larger one if possible.
            int large = std::max(die1, die2);
            int small = std::min(die1, die2);
            possible_boards_one_die(board, large, results);
            if (results.empty()) {
                possible_boards_one_die(board, small, results);
            }
        }
    }

    if (results.empty()) {
        results.push_back(board);
        return results;
    }

    sort_unique_boards(results);

    return results;
}

void possible_boards(const Board& board, int die1, int die2,
                     std::vector<Board>& results) {
    results.clear();
    thread_local std::array<std::vector<Board>, 4> scratch_by_depth;
    for (auto& scratch : scratch_by_depth) {
        if (scratch.capacity() < 16) {
            scratch.reserve(16);
        }
    }

    if (die1 == die2) {
        // Doubles: must use maximum number of dice possible (4, then 3, ...)
        int dice[4] = {die1, die1, die1, die1};
        for (int n = 4; n >= 1; --n) {
            generate_boards_strict(board, dice, n, results, scratch_by_depth.data());
            if (!results.empty()) break;
        }
    } else {
        // Non-doubles: must use both dice if possible
        int dice_a[2] = {die1, die2};
        int dice_b[2] = {die2, die1};
        generate_boards_strict(board, dice_a, 2, results, scratch_by_depth.data());
        generate_boards_strict(board, dice_b, 2, results, scratch_by_depth.data());

        if (results.empty()) {
            // Can't use both dice. Must use the larger one if possible.
            int large = std::max(die1, die2);
            int small = std::min(die1, die2);
            possible_boards_one_die(board, large, results);
            if (results.empty()) {
                possible_boards_one_die(board, small, results);
            }
        }
    }

    if (results.empty()) {
        results.push_back(board);
        return;
    }

    sort_unique_boards(results);
}

} // namespace bgbot
