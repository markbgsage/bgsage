# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Move notation comes from a legal play of the dice, not from the board diff alone.

Run from the repo root: ``py -3.14 -m pytest bgsage/tests/test_move_notation.py``.
The engine-backed tests skip when ``bgbot_cpp`` is not importable.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if hasattr(os, "add_dll_directory") and os.path.isdir(_CUDA_BIN):
    os.add_dll_directory(_CUDA_BIN)

import pytest

from bgsage.text_export import (
    _legal_single_die_moves,
    _resolve_legal_steps,
    compute_move_notation,
)

STARTING = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]


def board(points):
    """A board from {point: count}: positive = the mover's checkers, negative
    = the opponent's; 25 = the mover's bar, 0 = the opponent's bar."""
    b = [0] * 26
    for p, n in points.items():
        b[p] = n
    return b


def play(before, *steps):
    """Apply from/to steps (mover's perspective), hitting any blot landed on."""
    b = list(before)
    for frm, to in steps:
        b[frm] -= 1
        if to == 0:
            continue
        if b[to] == -1:
            b[to] = 1
            b[0] += 1
        else:
            b[to] += 1
    return b


def test_bear_off_pairing_follows_the_dice():
    # Issue b871348d (2026-09-07): 6-3 with checkers on the 5-, 4-, 2- and
    # 1-points. The 6 can bear the 4-point checker off only once the 3 has
    # emptied the 5-point, so the play is 5/2 4/off; the die-first pairing
    # wrote 5/off 4/2, a two-pip move nobody rolled.
    before = [0, 5, 5, 0, 4, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, -3, -5, 0]
    after = play(before, (5, 2), (4, 0))
    assert compute_move_notation(before, after, 6, 3) == "5/2 4/off"
    assert compute_move_notation(before, after, 3, 6) == "5/2 4/off"
    assert _resolve_legal_steps(before, after, 6, 3) == [(5, 2, False), (4, 0, False)]
    # The runner-up really is 5/off 4/1.
    assert compute_move_notation(before, play(before, (5, 0), (4, 1)), 6, 3) == "5/off 4/1"


def test_familiar_notation_unchanged():
    assert compute_move_notation(STARTING, play(STARTING, (8, 5), (6, 5)), 3, 1) == "8/5 6/5"
    assert compute_move_notation(STARTING, play(STARTING, (13, 7), (8, 7)), 6, 1) == "13/7 8/7"
    # One checker moving twice is a single journey.
    assert compute_move_notation(STARTING, play(STARTING, (13, 8), (8, 5)), 5, 3) == "13/5"
    assert compute_move_notation(STARTING, play(STARTING, (24, 18), (18, 13)), 6, 5) == "24/13"
    # Doubles combine identical journeys.
    assert compute_move_notation(STARTING, play(STARTING, (13, 9), (13, 9), (9, 5), (9, 5)), 4, 4) == "13/5(2)"
    assert compute_move_notation(STARTING, play(STARTING, (13, 11), (13, 11), (6, 4), (6, 4)), 2, 2) == "13/11(2) 6/4(2)"
    # Bearing off with exact and oversize dice.
    home = board({6: 2, 5: 3, 4: 3, 2: 4, 1: 3, 20: -5, 21: -5, 22: -3})
    assert compute_move_notation(home, play(home, (6, 0), (6, 0), (5, 0), (5, 0)), 6, 6) == "6/off(2) 5/off(2)"
    assert compute_move_notation(home, play(home, (6, 0), (5, 4)), 6, 1) == "6/off 5/4"
    # A dance leaves the board unchanged and the notation empty.
    assert compute_move_notation(STARTING, STARTING, 6, 6) == ""


def test_hits_marked_and_never_run_through():
    entering = board({25: 1, 24: 1, 13: 5, 8: 3, 6: 5, 20: -1, 1: -1, 12: -5, 17: -3, 19: -5})
    assert compute_move_notation(entering, play(entering, (25, 20), (6, 4)), 5, 2) == "bar/20* 6/4"
    running = board({24: 2, 13: 5, 8: 3, 6: 5, 18: -1, 1: -1, 12: -5, 17: -3, 19: -5})
    assert compute_move_notation(running, play(running, (24, 18), (18, 13)), 6, 5) == "24/18* 18/13"
    two_blots = board({24: 1, 13: 5, 8: 3, 6: 5, 20: -1, 16: -1, 1: -1, 12: -5, 17: -3, 19: -5})
    assert compute_move_notation(
        two_blots, play(two_blots, (24, 20), (20, 16), (13, 9), (13, 9)), 4, 4
    ) == "24/20* 20/16* 13/9(2)"
    # A journey may end on a hit.
    assert compute_move_notation(two_blots, play(two_blots, (24, 21), (21, 20), (8, 5)), 3, 1) == "24/20* 8/5"


def test_fallback_for_boards_the_roll_cannot_connect():
    after = play(STARTING, (13, 7))
    assert _resolve_legal_steps(STARTING, after, 5, 4) is None
    assert compute_move_notation(STARTING, after, 5, 4) == "13/7"
    # Missing dice are not a legal play either.
    assert _resolve_legal_steps(STARTING, after, 0, 0) is None


# ---------------------------------------------------------------------------
# Against the engine
# ---------------------------------------------------------------------------

bgbot_cpp = pytest.importorskip("bgbot_cpp")


def _random_board(rng, bearoff):
    b = [0] * 26
    for _ in range(rng.randint(4, 15)):
        p = rng.randint(1, 6) if bearoff else rng.randint(1, 25)
        if p <= 24 and b[p] < 0:
            continue
        b[p] += 1
    for _ in range(rng.randint(4, 15)):
        p = rng.randint(19, 24) if bearoff else rng.randint(0, 24)
        if p >= 1 and b[p] > 0:
            continue
        if p == 0:
            b[0] += 1
        else:
            b[p] -= 1
    return b


def test_single_die_generator_matches_engine():
    rng = random.Random(20260914)
    for i in range(3000):
        b = _random_board(rng, i % 2 == 1)
        die = rng.randint(1, 6)
        ours = sorted((f, t, tuple(nb)) for f, t, nb in _legal_single_die_moves(b, die))
        theirs = sorted(
            (m["from"], m["to"], tuple(m["board"])) for m in bgbot_cpp.possible_single_die_moves(b, die)
        )
        assert ours == theirs, (b, die)


def test_every_engine_legal_play_resolves():
    """The engine's legal end boards always resolve into legal steps that add
    up to the board change — the diff pairing is never needed for them."""
    rng = random.Random(7)
    checked = 0
    for i in range(1500):
        b = _random_board(rng, i % 2 == 1)
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        for after in bgbot_cpp.possible_moves(b, d1, d2):
            after = list(after)
            if after == b:
                continue
            steps = _resolve_legal_steps(b, after, d1, d2)
            assert steps is not None, (b, after, d1, d2)
            # Journeys are unordered: a checker entering from the bar can land
            # on a point another checker hit earlier in the turn, so replay the
            # hits first.
            replayed = list(b)
            ordered = [s for s in steps if s[2]] + [s for s in steps if not s[2]]
            for frm, to, hit in ordered:
                replayed[frm] -= 1
                if to == 0:
                    continue
                if hit:
                    assert replayed[to] == -1, (b, after, d1, d2, steps)
                    replayed[to] = 1
                    replayed[0] += 1
                else:
                    replayed[to] += 1
            assert replayed == after, (b, after, d1, d2, steps)
            checked += 1
    assert checked > 5000, checked
