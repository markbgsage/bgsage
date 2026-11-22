// SPDX-License-Identifier: MPL-2.0
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include "bgbot/moves.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace bgbot;
int main() {
    std::mt19937 rng(31737);
    uint64_t hash = 14695981039346656037ULL;
    unsigned boards = 0, pairs = 0, saturated = 0;
    auto features = [&](const Board& b) {
        const auto f = compute_extended_contact_inputs(b);
        const Board other = flip(b);
        for (int side = 0; side < 2; ++side) {
            const Board& half = side ? other : b;
            if (f[side * 122 + 104] != back_escapes(half) / 36.0f ||
                f[side * 122 + 116] != enter_loss(half) / (36.0f * (49.0f / 6.0f)))
                throw std::runtime_error("Scalar escape or bar entry mismatch");
        }
        for (float x : f) {
            uint32_t bits;
            std::memcpy(&bits, &x, sizeof(bits));
            hash = (hash ^ bits) * 1099511628211ULL;
        }
        ++boards;
        return f;
    };
    auto check_deltas = [&](const Board& before, const Board& after, const auto& base,
                            const auto& end, bool tesauro) {
        int indices[INPUT_DELTA_LIMIT];
        float values[INPUT_DELTA_LIMIT];
        const int count = board_input_deltas(after, before, base.data(), tesauro, indices, values);
        int expected = 0;
        for (int i = 0; i < static_cast<int>(base.size()); ++i) {
            const float delta = end[i] - base[i];
            if (!delta) continue;
            if (expected < INPUT_DELTA_LIMIT &&
                (expected >= count || indices[expected] != i || values[expected] != delta))
                throw std::runtime_error("Incremental encoder differs from dense");
            ++expected;
        }
        if (count != std::min(expected, INPUT_DELTA_LIMIT))
            throw std::runtime_error("Incorrect delta count");
        if (!tesauro) saturated += count == INPUT_DELTA_LIMIT;
    };
    auto check = [&](const Board& before, const Board& after) {
        const auto base = features(before);
        const auto end = features(after);
        check_deltas(before, after, base, end, false);
        check_deltas(before, after, compute_tesauro_inputs(before), compute_tesauro_inputs(after), true);
        ++pairs;
    };
    // Identical bar counts can mask different hit blots in two sibling moves.
    Board first{}, second{};
    first[0] = second[0] = 1;
    first[3] = second[3] = 1;
    first[15] = -1;
    second[18] = -1;
    check(first, second);
    // Moving the front checker changes the opposing side's contact boundary.
    second = first;
    second[3] = 0;
    second[1] = 1;
    check(first, second);
    second = first;
    second[15] = -3;
    second[17] = -2;
    second[0] = 2;
    check(first, second);
    // The previous board can have no numbered positive checker.
    first[3] = 0;
    first[25] = 2;
    second = first;
    second[25] = 1;
    second[22] = 1;
    check(first, second);
    for (int bar = 0; bar <= 3; ++bar) {
        for (unsigned mask = 0; mask < 64; ++mask) {
            Board b{};
            b[25] = bar;
            for (int bit = 0; bit < 6; ++bit)
                if (mask & (1u << bit)) b[24 - bit] = -2;
            features(b);
        }
    }
    for (int back = 19; back <= 24; ++back) {
        for (uint32_t mask = 0; mask < 4096; ++mask) {
            Board b{};
            b[back] = 1;
            for (int i = 0; i < 12; ++i)
                b[back - 12 + i] = mask & (1u << i) ? -2 : 0;
            for (int i = 1; i < back - 12; ++i)
                if (rng() & 1) b[i] = -2;
            features(b);
        }
    }
    std::vector<Board> moves;
    for (int i = 0; i < 8192; ++i) {
        Board b{};
        for (int side : {1, -1}) {
            int n = 1 + rng() % 15;
            int bar = std::min<int>(n, rng() % 4);
            b[side > 0 ? 25 : 0] = bar;
            for (int j = bar; j < n;) {
                int point = 1 + rng() % 24;
                if (b[point] * side >= 0) { b[point] += side; ++j; }
            }
        }
        const int die1 = 1 + rng() % 6;
        const int die2 = 1 + rng() % 6;
        possible_boards_unsorted(b, die1, die2, moves);
        for (const auto& m : moves) {
            check(b, m);
            check(moves[0], m);
        }
        check(b, flip(b));
    }
    std::cout << "boards=" << boards << " pairs=" << pairs << " dense_fallbacks=" << saturated
              << " feature_hash=" << hash << "\n";
}
