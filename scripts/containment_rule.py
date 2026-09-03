#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""The containment-game rule, Python reference implementation.

A containment game: the escaper E has borne off a bunch of checkers and has
one to three checkers that got hit and must run the whole board home; the
container C arranges its remaining checkers — anchors, blots, a prime,
whatever is left back — to keep hitting them, playing to save a gammon or
occasionally to win. Nothing about C's structure is required: the position
is a containment game because of E's state.

    containment(board)  <=>  for E in (player 2, player 1):
        E_off >= E_OFF_MIN                    (3: the early/late boundary)
        1 <= stragglers <= STRAGGLERS_MAX     (3)
        contact remains (not a pure race)

A straggler is an E checker on the bar or outside E's home board that still
has a C checker AHEAD of it on its way home — i.e. one that can still be
hit. An E checker that is already past every C checker is running free and
does not count, so a fourth free-running checker does not disqualify a
position that is otherwise a three-straggler containment game.

Mirrors ``containment_category`` in cpp/src/neural_net.cpp — the two MUST
agree (tests/test_backgame_category.py checks a sample). Board convention:
player-1 frame, index 25 = P1 bar, index 0 = P2 bar (both positive counts).
"""

from __future__ import annotations

E_OFF_MIN = 3
STRAGGLERS_MAX = 3


def _p2_escaper(board) -> bool:
    """Player 2 (negative) is the escaper of a containment game."""
    on_board = board[0] + sum(-board[i] for i in range(1, 25) if board[i] < 0)
    if 15 - on_board < E_OFF_MIN:
        return False
    # P2 moves upward; a P2 checker at index i still has to pass every P1
    # checker at an index above i (P1's bar, index 25, counts).
    p1_max = max((i for i in range(1, 26) if board[i] > 0), default=-1)
    stragglers = (board[0] if p1_max > 0 else 0) + sum(
        -board[i] for i in range(1, 19) if board[i] < 0 and p1_max > i)
    return 1 <= stragglers <= STRAGGLERS_MAX


def _flip(board):
    # Points reverse and change sign; the two bars swap but stay positive.
    return [board[25]] + [-board[25 - i] for i in range(1, 25)] + [board[0]]


def containment(board) -> bool:
    return _p2_escaper(board) or _p2_escaper(_flip(board))


def escaper_is_p2(board) -> bool | None:
    """Which side is the escaper (None if not a containment game)."""
    if _p2_escaper(board):
        return True
    if _p2_escaper(_flip(board)):
        return False
    return None
