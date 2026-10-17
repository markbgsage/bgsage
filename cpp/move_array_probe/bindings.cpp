// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
#include "../pybind/move_arrays.h"

PYBIND11_MODULE(_bgsage_move_array_probe, module) {
    bgsage_python::bind_move_arrays(module);
    module.def("possible_moves", [](const std::vector<int>& input, int first, int second) {
        if (input.size() != 26) throw pybind11::value_error("board must contain 26 integers");
        bgbot::Board board;
        std::copy(input.begin(), input.end(), board.begin());
        return bgbot::possible_boards(board, first, second);
    });
}
