// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/moves.h"
#include <algorithm>
#include <cstdint>

namespace bgbot {

void possible_boards_one_die(const Board& b, int die, std::vector<Board>& out) {
    // Push-then-mutate-in-place: each legal move pushes a copy of `b` into the
    // output vector and mutates it there, instead of copying to a stack `nb`
    // and then copying that into the vector (two 104-byte copies -> one).
    // Bit-identical output, same order. out.back() stays valid after push_back
    // (any realloc happens during the push), and no prior reference is held
    // across iterations.

    // If player has checkers on the bar, must enter them first
    if (b[25] > 0) {
        int target = 25 - die;
        if (b[target] >= -1) {
            out.push_back(b);
            Board& nb = out.back();
            if (nb[target] == -1) {
                nb[0] += 1;
                nb[target] = 0;
            }
            nb[25] -= 1;
            nb[target] += 1;
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
                out.push_back(b);
                Board& nb = out.back();
                if (nb[target] == -1) {
                    nb[0] += 1;
                    nb[target] = 0;
                }
                nb[i] -= 1;
                nb[target] += 1;
            }
        } else if (can_bear_off) {
            if (target == 0) {
                out.push_back(b);
                out.back()[i] -= 1;
            } else {
                bool highest = true;
                for (int j = i + 1; j <= 6; ++j) {
                    if (b[j] > 0) { highest = false; break; }
                }
                if (highest) {
                    out.push_back(b);
                    out.back()[i] -= 1;
                }
            }
        }
    }

    // Outer board moves (points 7-24)
    for (int i = 7; i <= 24; ++i) {
        if (b[i] <= 0) continue;

        int target = i - die;
        if (target >= 1 && b[target] >= -1) {
            out.push_back(b);
            Board& nb = out.back();
            if (nb[target] == -1) {
                nb[0] += 1;
                nb[target] = 0;
            }
            nb[i] -= 1;
            nb[target] += 1;
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

namespace {

uint32_t move_board_hash(const Board& b) {
    // A dot product over points vectorizes. Board equality below still
    // decides every collision; hashing never changes first-seen order.
    static constexpr uint32_t coefficients[26] = {
        767017, 541951, 709901, 987211, 774359, 690293, 936359, 539347,
        930547, 913217, 869233, 795203, 960593, 580031, 642977, 544897,
        686057, 787837, 998617, 982273, 609859, 849677, 600337, 791291,
        544331, 855011
    };
    uint32_t h = 0;
    for (int i = 0; i < 26; ++i) {
        h += static_cast<uint32_t>(b[i]) * coefficients[i];
    }
    return h ^ (h >> 16);
}

// Generation-stamped open-addressing table used to reject duplicate boards
// at insertion time. begin() invalidates all entries in O(1) by bumping the
// generation counter.
struct MoveDedup {
    static constexpr uint32_t TBL = 2048;     // power of 2
    std::vector<int32_t> head = std::vector<int32_t>(TBL, -1);
    std::vector<uint32_t> stamp = std::vector<uint32_t>(TBL, 0);
    uint32_t gen = 0;
    uint32_t mask = TBL - 1;

    void grow(const std::vector<Board>& boards) {
        const size_t size = head.size() * 2;
        head.resize(size);
        stamp.assign(size, 0u);
        mask = static_cast<uint32_t>(size - 1);
        for (size_t i = 0; i < boards.size(); ++i) {
            uint32_t slot = move_board_hash(boards[i]) & mask;
            while (stamp[slot] == gen) slot = (slot + 1) & mask;
            stamp[slot] = gen;
            head[slot] = static_cast<int32_t>(i);
        }
    }

    void begin() {
        ++gen;
        if (gen == 0) {
            std::fill(stamp.begin(), stamp.end(), 0u);
            gen = 1;
        }
    }

    // Returns true if `b` is new; records `idx` (the board's position in the
    // output vector) as its slot value.
    bool insert(const Board& b, const std::vector<Board>& v, int idx) {
        // A legal double can have over 2,048 distinct results. Always
        // leave an empty slot so probing terminates, including on a miss.
        if (static_cast<uint32_t>(idx) >= mask - mask / 4) grow(v);
        uint32_t slot = move_board_hash(b) & mask;
        for (;;) {
            if (stamp[slot] != gen) {
                stamp[slot] = gen;
                head[slot] = idx;
                return true;
            }
            if (v[head[slot]] == b) return false;
            slot = (slot + 1) & mask;
        }
    }
};

thread_local MoveDedup g_move_dedup;

// Moving checkers in a different order can reach exactly the same board
// after two or three dice of a double. From there the remaining dice are
// identical, so the first visit emitted every reachable leaf. A hash
// collision here can evict an entry and cost work, but cannot discard a play.
struct DoublesPrefixes {
    static constexpr int slots = 256;
    uint32_t generation = 0;
    uint32_t stamps[2][slots]{};
    Board boards[2][slots];

    void begin() {
        if (++generation == 0) {
            for (auto& row : stamps) std::fill_n(row, slots, 0u);
            ++generation;
        }
    }

    bool already_visited(const Board& board, int depth) {
        const auto slot = move_board_hash(board) & (slots - 1);
        auto& stamp = stamps[depth - 2][slot];
        auto& previous = boards[depth - 2][slot];
        if (stamp == generation && previous == board) return true;
        stamp = generation;
        previous = board;
        return false;
    }
};
thread_local DoublesPrefixes g_doubles_prefixes;

// Strict generator: only emits boards where ALL n_dice are used (or the game
// ended mid-sequence). Leaves are dedup-checked at insertion, so duplicate
// boards are never copied into the output.
void generate_strict_dedup(const Board& b, const int* dice, int n_dice,
                           std::vector<Board>& results,
                           std::vector<Board>* scratch_by_depth,
                           MoveDedup& dedup,
                           int depth = 0)
{
    if (depth >= 2 && g_doubles_prefixes.already_visited(b, depth)) return;
    if (!has_checkers(b)) {
        if (dedup.insert(b, results, static_cast<int>(results.size()))) {
            results.push_back(b);
        }
        return;
    }

    std::vector<Board>& scratch = scratch_by_depth[depth];
    scratch.clear();
    possible_boards_one_die(b, dice[0], scratch);

    if (n_dice == 1) {
        for (const auto& nb : scratch) {
            if (dedup.insert(nb, results, static_cast<int>(results.size()))) {
                results.push_back(nb);
            }
        }
        return;
    }

    for (const auto& nb : scratch) {
        generate_strict_dedup(nb, dice + 1, n_dice - 1, results,
                              scratch_by_depth, dedup, depth + 1);
    }
}

} // namespace

void possible_boards_unsorted(const Board& board, int die1, int die2,
                              std::vector<Board>& results)
{
    results.clear();
    thread_local std::array<std::vector<Board>, 4> scratch_by_depth;
    for (auto& scratch : scratch_by_depth) {
        if (scratch.capacity() < 16) scratch.reserve(16);
    }
    MoveDedup& dedup = g_move_dedup;
    dedup.begin();

    if (die1 == die2) {
        int dice[4] = {die1, die1, die1, die1};
        for (int n = 4; n >= 1; --n) {
            g_doubles_prefixes.begin();
            generate_strict_dedup(board, dice, n, results,
                                  scratch_by_depth.data(), dedup);
            if (!results.empty()) break;
        }
    } else {
        int dice_a[2] = {die1, die2};
        int dice_b[2] = {die2, die1};
        generate_strict_dedup(board, dice_a, 2, results,
                              scratch_by_depth.data(), dedup);
        generate_strict_dedup(board, dice_b, 2, results,
                              scratch_by_depth.data(), dedup);

        if (results.empty()) {
            int large = std::max(die1, die2);
            int small = std::min(die1, die2);
            possible_boards_one_die(board, large, results);
            if (results.empty()) {
                possible_boards_one_die(board, small, results);
            }
            // One-die fallback bypasses the dedup table; duplicates are
            // impossible there (each origin point yields a distinct board).
        }
    }

    if (results.empty()) {
        results.push_back(board);
    }
}

} // namespace bgbot
