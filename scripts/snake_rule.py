#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""The snake rule, Python reference implementation.

A snake (the Lamford ch. 41 / "Snake" shapes): the holder H has a far-side
prime — a run of >= PRIME_MIN_POINTS consecutive points, each held with >= 2
checkers, entirely on the opponent's half of the board — trapping >= 1
opponent straggler (on the bar or in H's home board) while the opponent's
other checkers are crunched: >= MIN_HOME already in its own home board. It
is a priming / containment structure, not a back game; it reads as one to
Stage 9's plan-pair gate only because H's points sit in the opponent's home
board and H trails in the race.

    snake(board)  <=>  for H in (player 1, player 2):
        longest run of H points (>= 2 checkers) within the far half >= 4
        opponent stragglers (bar + in H's home board) >= 1
        opponent checkers in its own home board >= 10

Mirrors ``snake_category`` in cpp/src/neural_net.cpp and the ``snake``
family filter in scripts/backgame_benchmark.py — all three MUST agree
(tests/test_backgame_category.py checks the folder). Board convention:
player-1 frame, index 25 = P1 bar, index 0 = P2 bar (both positive counts).
"""

from __future__ import annotations

PRIME_MIN_POINTS = 4
MIN_HOME = 10


def _p1_holds_snake(board) -> bool:
    """Player 1 (positive) holds the prime; player 2 (negative) is trapped."""
    run = best = 0
    for i in range(13, 25):
        run = run + 1 if board[i] >= 2 else 0
        best = max(best, run)
    if best < PRIME_MIN_POINTS:
        return False
    straggler = board[0] + sum(-board[i] for i in range(1, 7) if board[i] < 0)
    if straggler < 1:
        return False
    home = sum(-board[i] for i in range(19, 25) if board[i] < 0)
    return home >= MIN_HOME


def _flip(board):
    # Points reverse and change sign; the two bars swap but stay positive.
    return [board[25]] + [-board[25 - i] for i in range(1, 25)] + [board[0]]


def snake(board) -> bool:
    return _p1_holds_snake(board) or _p1_holds_snake(_flip(board))


def holder_is_p1(board) -> bool | None:
    """Which side holds the prime (None if not a snake)."""
    if _p1_holds_snake(board):
        return True
    if _p1_holds_snake(_flip(board)):
        return False
    return None
