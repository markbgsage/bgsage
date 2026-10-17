// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "bgbot/moves.h"

namespace bgsage_python {

struct MoveBoardHash {
    std::size_t operator()(const bgbot::Board& board) const noexcept {
        std::size_t value = 1469598103934665603ULL;
        for (int count : board) {
            value ^= static_cast<unsigned int>(count + 15);
            value *= 1099511628211ULL;
        }
        return value;
    }
};

inline pybind11::tuple possible_moves_all(const std::vector<int>& input) {
    if (input.size() != 26) {
        throw pybind11::value_error("board must contain 26 integers");
    }
    bgbot::Board board;
    int player = 0, opponent = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] < -15 || input[i] > 15 || ((i == 0 || i == 25) && input[i] < 0)) {
            throw pybind11::value_error("invalid checker count");
        }
        board[i] = input[i];
        if (i == 0) opponent += input[i];
        else if (i == 25) player += input[i];
        else { player += std::max(0, input[i]); opponent += std::max(0, -input[i]); }
    }
    if (player > 15 || opponent > 15) {
        throw pybind11::value_error("a side cannot have more than 15 checkers");
    }

    std::vector<bgbot::Board> boards, candidates;
    std::vector<std::uint32_t> masks;
    std::unordered_map<bgbot::Board, std::size_t, MoveBoardHash> indices;
    {
        pybind11::gil_scoped_release release;
        unsigned int roll = 0;
        // Use the existing sorted generator unchanged. Appending only on
        // first sight preserves the exact order of the historical 21-call
        // Python dictionary, including aliases, fans and forced moves.
        for (int first = 1; first <= 6; ++first) {
            for (int second = first; second <= 6; ++second, ++roll) {
                bgbot::possible_boards(board, first, second, candidates);
                for (const auto& candidate : candidates) {
                    const auto inserted = indices.emplace(candidate, boards.size());
                    if (inserted.second) {
                        boards.push_back(candidate);
                        masks.push_back(0);
                    }
                    masks[inserted.first->second] |= std::uint32_t{1} << roll;
                }
            }
        }
    }
    const auto count = static_cast<pybind11::ssize_t>(boards.size());
    pybind11::array_t<std::int8_t> board_array({count, pybind11::ssize_t{26}});
    pybind11::array_t<std::uint32_t> mask_array(count);
    auto* board_data = board_array.mutable_data();
    auto* mask_data = mask_array.mutable_data();
    for (std::size_t i = 0; i < boards.size(); ++i) {
        for (std::size_t p = 0; p < 26; ++p) board_data[i*26+p] = static_cast<std::int8_t>(boards[i][p]);
        mask_data[i] = masks[i];
    }
    return pybind11::make_tuple(std::move(board_array), std::move(mask_array));
}

inline void bind_move_arrays(pybind11::module_& module) {
    module.def("possible_moves_all", &possible_moves_all, pybind11::arg("board"),
        "All distinct legal endpoints and 21-roll bit masks, preserving first-seen order.");
}

}  // namespace bgsage_python
