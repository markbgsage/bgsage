# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Compact legal move arrays, with compatibility for older engine builds."""
from numbers import Integral

import bgbot_cpp
import numpy as np


ROLLS = tuple((first, second) for first in range(1, 7) for second in range(first, 7))


def _validated_board(board):
    values = tuple(board)
    if len(values) != 26 or any(isinstance(value, bool) or not isinstance(value, Integral)
                                for value in values):
        raise ValueError("board must contain 26 integers")
    values = tuple(map(int, values))
    if (any(abs(value) > 15 for value in values) or min(values[0], values[25]) < 0
            or values[25] + sum(max(0, value) for value in values[1:25]) > 15
            or values[0] + sum(max(0, -value) for value in values[1:25]) > 15):
        raise ValueError("invalid checker count")
    return values


def possible_moves_all(board):
    """Return ``(boards, masks)`` for all 21 rolls in the mover's perspective.

    ``boards`` is a contiguous ``int8`` array of shape ``(N, 26)``; ``masks``
    is contiguous ``uint32`` of shape ``(N,)``. Bit k indicates legality under
    ``ROLLS[k]``. Rows preserve first occurrence while iterating those rolls
    and each roll's existing sorted native move list. A blocked roll includes
    the unchanged board. Each result owns its storage and is independently
    writable. This function performs no evaluation and uses no model weights.

    Older extensions transparently use the equivalent 21-call construction.
    """
    values = _validated_board(board)
    native = getattr(bgbot_cpp, "possible_moves_all", None)
    if native is not None:
        return native(values)
    found = {}
    for k, (first, second) in enumerate(ROLLS):
        for after in bgbot_cpp.possible_moves(values, first, second):
            key = tuple(after)
            found[key] = found.get(key, 0) | (1 << k)
    return (np.asarray(list(found), dtype=np.int8).reshape((-1, 26)),
            np.fromiter(found.values(), dtype=np.uint32, count=len(found)))
