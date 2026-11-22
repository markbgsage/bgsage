// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/moves.h"
#include <algorithm>
#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif

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

static constexpr uint32_t coefficients[26] = {
        767017, 541951, 709901, 987211, 774359, 690293, 936359, 539347,
        930547, 913217, 869233, 795203, 960593, 580031, 642977, 544897,
        686057, 787837, 998617, 982273, 609859, 849677, 600337, 791291,
        544331, 855011
    };

uint32_t raw_board_hash(const Board& b) {
    uint32_t h = 0;
    for (int i = 0; i < 26; ++i) {
        h += static_cast<uint32_t>(b[i]) * coefficients[i];
    }
    return h;
}

uint32_t move_board_hash(const Board& b) {
    const auto h = raw_board_hash(b);
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
    bool insert(const Board& b, const std::vector<Board>& v, int idx, uint32_t hash) {
        // A legal double can have over 2,048 distinct results. Always
        // leave an empty slot so probing terminates, including on a miss.
        if (static_cast<uint32_t>(idx) >= mask - mask / 4) grow(v);
        uint32_t slot = hash & mask;
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

    bool already_visited(const Board& board, int depth, uint32_t hash) {
        const auto slot = hash & (slots - 1);
        auto& stamp = stamps[depth - 2][slot];
        auto& previous = boards[depth - 2][slot];
        if (stamp == generation && previous == board) return true;
        stamp = generation;
        previous = board;
        return false;
    }
};
thread_local DoublesPrefixes g_doubles_prefixes;

// Each ply changes at most an origin, a destination and the opponent's bar.
// Mutate one board while descending, then restore it before emitting siblings.
struct RollMoves {
    Board board;
    std::vector<Board>& results;
    uint32_t occupied = 0;
    uint32_t hash;

    RollMoves(const Board& start, std::vector<Board>& out)
        : board(start), results(out), hash(raw_board_hash(start)) {
        for (int i = 1; i <= 25; ++i)
            if (board[i] > 0) occupied |= 1u << i;
    }

    void generate(const int* dice, int left, uint32_t occupied_now, uint32_t raw_hash,
                  int depth = 0) {
        const uint32_t mixed = raw_hash ^ (raw_hash >> 16);
        if (left && depth >= 2 && g_doubles_prefixes.already_visited(board, depth, mixed))
            return;
        if (!left || !occupied_now) {
            if (g_move_dedup.insert(board, results, static_cast<int>(results.size()), mixed))
                results.push_back(board);
            return;
        }
        constexpr uint32_t bar = 1u << 25;
        constexpr uint32_t home = (1u << 7) - 1;
        const int die = *dice;
        uint32_t origins = (occupied_now & bar) ? bar : occupied_now;
        while (origins) {
#ifdef _MSC_VER
            unsigned long bit;
            _BitScanForward(&bit, origins);
            const int from = static_cast<int>(bit);
#else
            const int from = __builtin_ctz(origins);
#endif
            origins &= origins - 1;
            const int to = from - die;
            if (to > 0) {
                if (board[to] < -1) continue;
            } else {
                if (occupied_now & ~home) continue;
                if (to < 0 && (occupied_now >> (from + 1))) continue;
            }
            const uint32_t from_bit = 1u << from;
            uint32_t next_occupied = board[from] == 1 ? (occupied_now & ~from_bit) : occupied_now;
            uint32_t next_hash = raw_hash - coefficients[from];
            --board[from];
            const int old_target = to > 0 ? board[to] : 0;
            if (to > 0) {
                next_occupied |= 1u << to;
                if (old_target == -1) {
                    ++board[0];
                    board[to] = 1;
                    next_hash += coefficients[0] + 2u * coefficients[to];
                } else {
                    ++board[to];
                    next_hash += coefficients[to];
                }
            }
            generate(dice + 1, left - 1, next_occupied, next_hash, depth + 1);
            if (to > 0) {
                board[to] = old_target;
                if (old_target == -1) --board[0];
            }
            ++board[from];
        }
    }
};

} // namespace

void possible_boards_unsorted(const Board& board, int die1, int die2,
                              std::vector<Board>& results) {
    results.clear();
    g_move_dedup.begin();
    RollMoves walker(board, results);
    if (die1 == die2) {
        const int dice[] = {die1, die1, die1, die1};
        for (int count = 4; count >= 1 && results.empty(); --count) {
            g_doubles_prefixes.begin();
            walker.generate(dice, count, walker.occupied, walker.hash);
        }
    } else {
        const int forward[] = {die1, die2};
        const int reverse[] = {die2, die1};
        walker.generate(forward, 2, walker.occupied, walker.hash);
        walker.generate(reverse, 2, walker.occupied, walker.hash);
        if (results.empty()) {
            possible_boards_one_die(board, std::max(die1, die2), results);
            if (results.empty()) possible_boards_one_die(board, std::min(die1, die2), results);
        }
    }
    if (results.empty()) results.push_back(board);
}

} // namespace bgbot
