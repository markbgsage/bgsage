# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Exact native-array equivalence, with optional private move-only extension."""
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build/video_move_batch"))
sys.path.append(str(ROOT / "build"))
if os.environ.get("BGSAGE_MOVE_ARRAY_EXTENSION"):
    extension_path = Path(os.environ["BGSAGE_MOVE_ARRAY_EXTENSION"]).resolve()
    if not extension_path.is_relative_to(ROOT):
        raise ValueError("The test extension must stay inside the engine repository")
    extension_spec = importlib.util.spec_from_file_location("bgbot_cpp", extension_path)
    probe = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(probe)
else:
    try:
        import _bgsage_move_array_probe as probe
    except ImportError:
        probe = pytest.importorskip("bgbot_cpp")
spec = importlib.util.spec_from_file_location("move_arrays_under_test", ROOT / "python/bgsage/move_arrays.py")
arrays = importlib.util.module_from_spec(spec)
previous_cpp = sys.modules.get("bgbot_cpp")
try:
    # The move-only test extension is intentionally not an installed engine.
    # No analyzer, model weights or parent app imports are needed by this test.
    sys.modules["bgbot_cpp"] = probe
    spec.loader.exec_module(arrays)
finally:
    if previous_cpp is None:
        sys.modules.pop("bgbot_cpp", None)
    else:
        sys.modules["bgbot_cpp"] = previous_cpp

START = (0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
         5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0)


def flip(board):
    return (board[25],) + tuple(-value for value in board[24:0:-1]) + (board[0],)


def reference(board):
    found = {}
    for k, (first, second) in enumerate(arrays.ROLLS):
        for after in probe.possible_moves(board, first, second):
            key = tuple(after)
            found[key] = found.get(key, 0) | (1 << k)
    return np.asarray(list(found), dtype=np.int8), np.asarray(list(found.values()), dtype=np.uint32)


def positions():
    rng, board = random.Random(78915), START
    yield board
    for _ in range(80):
        first, second = rng.choice(arrays.ROLLS)
        board = flip(rng.choice(probe.possible_moves(board, first, second)))
        if not any(value > 0 for value in board[1:]) or not any(value < 0 for value in board[1:25]):
            board = START
        yield board
        yield flip(board)


def select_implementation(monkeypatch, native):
    if native and not hasattr(probe, "possible_moves_all"):
        pytest.skip("Native array binding requires a rebuilt engine or private probe")
    monkeypatch.setattr(arrays, "bgbot_cpp", probe if native else SimpleNamespace(possible_moves=probe.possible_moves))


@pytest.mark.parametrize("native", [False, True])
def test_every_endpoint_alias_and_order_matches_existing_generator(monkeypatch, native):
    select_implementation(monkeypatch, native)
    for board in positions():
        boards, masks = arrays.possible_moves_all(board)
        expected_boards, expected_masks = reference(board)
        assert np.array_equal(boards, expected_boards)
        assert np.array_equal(masks, expected_masks)
        assert boards.dtype == np.int8 and masks.dtype == np.uint32
        assert boards.flags.c_contiguous and masks.flags.c_contiguous
        assert len({tuple(row) for row in boards}) == len(boards)
        assert np.all((masks > 0) & (masks < (1 << 21)))


@pytest.mark.parametrize("native", [False, True])
def test_bar_fans_partial_doubles_hits_and_bearoff(monkeypatch, native):
    select_implementation(monkeypatch, native)
    fan = [0] * 26
    fan[25], fan[6], fan[12] = 1, 14, -3
    for point in range(19, 25):
        fan[point] = -2
    partial = [0] * 26
    partial[25], partial[21], partial[24] = 1, -2, -2
    bearing = [0] * 26
    bearing[1], bearing[3], bearing[6], bearing[19], bearing[24] = 5, 5, 5, -10, -5
    hitting = list(START)
    hitting[1], hitting[3] = -1, -1
    for board in (fan, partial, bearing, hitting):
        for side in (tuple(board), flip(board)):
            actual = arrays.possible_moves_all(side)
            expected = reference(side)
            assert all(np.array_equal(x, y) for x, y in zip(actual, expected))
    boards, masks = arrays.possible_moves_all(fan)
    assert boards.tolist() == [fan] and masks.tolist() == [(1 << 21) - 1]
    boards, masks = arrays.possible_moves_all(partial)
    two_mask = 1 << arrays.ROLLS.index((2, 2))
    rows = boards[(masks & two_mask) != 0]
    assert len(rows) == 1 and rows[0, 23] == 1 and rows[0, 25] == 0


@pytest.mark.parametrize("native", [False, True])
def test_returned_arrays_own_storage_and_accept_numpy_integers(monkeypatch, native):
    select_implementation(monkeypatch, native)
    a, mask_a = arrays.possible_moves_all(np.asarray(START, dtype=np.int64))
    b, mask_b = arrays.possible_moves_all(START)
    a[:] = 0
    mask_a[:] = 0
    assert np.any(b != 0) and np.all(mask_b != 0)


@pytest.mark.parametrize("native", [False, True])
def test_higher_die_and_maximum_dice_use_are_preserved(monkeypatch, native):
    select_implementation(monkeypatch, native)
    before = [0] * 26
    before[25], before[6], before[22], before[5], before[4], before[1] = 1, 14, -2, -2, -2, -9
    low, high = before.copy(), before.copy()
    low[25], low[24] = 0, 1
    high[25], high[23] = 0, 1
    boards, masks = arrays.possible_moves_all(before)
    roll_mask = 1 << arrays.ROLLS.index((1, 2))
    rows = boards[(masks & roll_mask) != 0].tolist()
    assert high in rows and low not in rows
    partial = list(START)
    partial[24] -= 1
    partial[23] += 1
    boards, masks = arrays.possible_moves_all(START)
    assert partial not in boards[(masks & roll_mask) != 0].tolist()


@pytest.mark.parametrize("native", [False, True])
def test_last_checker_overshoot_retains_every_roll(monkeypatch, native):
    select_implementation(monkeypatch, native)
    before = [0] * 26
    before[1], before[24] = 1, -15
    after = before.copy()
    after[1] = 0
    boards, masks = arrays.possible_moves_all(before)
    assert boards.tolist() == [after]
    assert masks.tolist() == [(1 << 21) - 1]


@pytest.mark.parametrize("invalid", [START[:-1], (False,) + START[1:], (0.,) + START[1:],
                                    (-1,) + START[1:], (16,) + START[1:],
                                    START[:-1] + (1,)])
def test_invalid_boards_rejected_before_native_or_fallback(invalid):
    with pytest.raises(ValueError):
        arrays.possible_moves_all(invalid)


def test_native_calls_are_independent_when_gil_is_released(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    select_implementation(monkeypatch, True)
    boards = list(positions())[:16]
    expected = [reference(board) for board in boards]
    with ThreadPoolExecutor(max_workers=4) as workers:
        actual = list(workers.map(arrays.possible_moves_all, boards))
    assert all(np.array_equal(x, y) for first, second in zip(actual, expected)
               for x, y in zip(first, second))


def test_eight_native_threads_preserve_256_ordered_results_and_owned_storage(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import hashlib

    select_implementation(monkeypatch, True)
    rng, board, boards = random.Random(78915), START, []
    for _ in range(256):
        boards.append(board)
        first, second = rng.choice(arrays.ROLLS)
        board = flip(rng.choice(probe.possible_moves(board, first, second)))
        if not any(value > 0 for value in board[1:]) or not any(value < 0 for value in board[1:25]):
            board = START

    def digest(values):
        rows, masks = values
        return hashlib.sha256(rows.tobytes() + masks.tobytes()).hexdigest()

    expected = [digest(reference(board)) for board in boards]

    def repeated(index):
        previous = None
        for _ in range(3):
            current = arrays.possible_moves_all(boards[index])
            assert digest(current) == expected[index]
            assert all(value.flags.owndata and value.flags.c_contiguous for value in current)
            if previous is not None:
                assert all(not np.shares_memory(old, new) for old in previous for new in current)
                previous[0][:] = 0
                previous[1][:] = 0
                assert digest(current) == expected[index]
            previous = current
        return expected[index]

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert list(workers.map(repeated, range(len(boards)))) == expected
