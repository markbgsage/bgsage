// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "bgbot/moves.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace bgbot {

void validate_die(int die) {
    if (die < 1 || die > 6) throw std::invalid_argument("Die must be between 1 and 6");
}

void possible_boards_one_die(const Board& b, int die, std::vector<Board>& out) {
    validate_die(die);
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

std::vector<Board> possible_boards(const Board& board, int die1, int die2) {
    std::vector<Board> results;
    results.reserve(32);
    possible_boards(board, die1, die2, results);
    return results;
}

void possible_boards(const Board& board, int die1, int die2,
                     std::vector<Board>& results) {
    possible_boards_unsorted(board, die1, die2, results);
    std::sort(results.begin(), results.end());
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
        // Keep an empty slot so probing always terminates.
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

struct RollState {
    Board board;
    uint32_t occupied = 0;
    uint32_t blocked = 0;

    explicit RollState(const Board& start) : board(start) {
        for (int i = 1; i <= 25; ++i) {
            if (board[i] > 0) occupied |= 1u << i;
            if (board[i] < -1) blocked |= 1u << i;
        }
    }

    // Made points cannot change while the player is on roll. All regular
    // moves with this die can use the same destination availability mask.
    uint32_t allowed(int die) const {
        return ~(blocked << die) & (~1u << die);
    }

    static uint32_t legal(uint32_t men, uint32_t allowed, int die) {
        constexpr uint32_t bar = 1u << 25;
        if (men & bar) return bar & allowed;
        uint32_t origins = men & allowed;
        if (men && men < (1u << 7)) {
            const uint32_t exact = 1u << die;
            if (men >= exact) origins |= men & exact;
            else {
                // An oversized die can only take the farthest checker.
                uint32_t highest = men;
                while (highest & (highest - 1)) highest &= highest - 1;
                origins |= highest;
            }
        }
        return origins;
    }

    // Changes are local to the callback, including hits to the opponent bar.
    // Ascending origin order is part of the unsorted API's tie behavior.
    template<class Visit>
    void each(int die, uint32_t men, uint32_t origins, Visit visit) {
        while (origins) {
#ifdef _MSC_VER
            unsigned long bit;
            _BitScanForward(&bit, origins);
            const int from = static_cast<int>(bit);
#else
            const int from = __builtin_ctz(origins);
#endif
            origins &= origins - 1;
            uint32_t next = --board[from] ? men : men & ~(1u << from);
            const int to = from - die;
            const int target = to > 0 ? board[to] : 0;
            const bool hit = target == -1;
            if (to > 0) {
                board[to] = hit ? 1 : target + 1;
                if (hit) ++board[0];
                next |= 1u << to;
            }
            visit(from, to, hit, next);
            if (to > 0) board[to] = target;
            if (hit) --board[0];
            ++board[from];
        }
    }

    // A double's endpoint determines how often each origin was used. Start
    // at the bar and work down: starting men plus arrivals minus men left
    // equals departures. Emit the earliest legal ordering of this multiset.
    // Other origins cannot disable a legal move, so an earlier sibling can
    // always play its own origin first if a later sequence ever uses it.
    void doubles(int die, int left, uint32_t men, uint32_t allowed,
                 uint32_t passed, std::vector<Board>& results) {
        if (!left || !men) {
            results.push_back(board);
            return;
        }
        each(die, men, legal(men, allowed, die) & ~passed,
             [&](int from, int, bool, uint32_t next) {
                 doubles(die, left - 1, next, allowed, passed, results);
                 passed |= 1u << from;
             });
    }

    static uint32_t moved_hash(uint32_t hash, int from, int to, bool hit) {
        hash -= coefficients[from];
        if (to > 0) hash += coefficients[to];
        if (hit) hash += coefficients[0] + coefficients[to];
        return hash;
    }

    void mixed(int die1, int die2, uint32_t first, uint32_t allowed_second,
               uint32_t skip_second, uint32_t hash, std::vector<Board>& results) {
        auto emit = [&](uint32_t raw) {
            const uint32_t hash_key = raw ^ (raw >> 16);
            if (g_move_dedup.insert(board, results, static_cast<int>(results.size()), hash_key))
                results.push_back(board);
        };
        each(die1, occupied, first, [&](int from, int to, bool hit, uint32_t next) {
            const auto after_first = moved_hash(hash, from, to, hit);
            if (!next) {
                emit(after_first);
                return;
            }
            const auto second = legal(next, allowed_second, die2) & ~skip_second;
            each(die2, next, second, [&](int from2, int to2, bool hit2, uint32_t) {
                emit(moved_hash(after_first, from2, to2, hit2));
            });
        });
    }
};

} // namespace

void possible_boards_unsorted(const Board& board, int die1, int die2,
                              std::vector<Board>& results) {
    validate_die(die1);
    validate_die(die2);
    results.clear();
    RollState roll(board);
    const uint32_t allowed_first = roll.allowed(die1);
    if (die1 == die2) {
        for (int count = 4; count >= 1 && results.empty(); --count)
            roll.doubles(die1, count, roll.occupied, allowed_first, 0, results);
    } else {
        g_move_dedup.begin();
        const uint32_t hash = raw_board_hash(board);
        const uint32_t allowed_second = roll.allowed(die2);
        const auto first = roll.legal(roll.occupied, allowed_first, die1);
        const auto second = roll.legal(roll.occupied, allowed_second, die2);
        roll.mixed(die1, die2, first, allowed_second, 0, hash, results);
        // If both steps were legal initially, the first order already emitted
        // their endpoint. The second order needs only newly enabled origins.
        roll.mixed(die2, die1, second, allowed_first, first, hash, results);
        if (results.empty()) {
            possible_boards_one_die(board, std::max(die1, die2), results);
            if (results.empty()) possible_boards_one_die(board, std::min(die1, die2), results);
        }
    }
    if (results.empty()) results.push_back(board);
}

} // namespace bgbot
