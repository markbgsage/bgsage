# SPDX-License-Identifier: MPL-2.0
import pytest

from bgsage import STARTING_BOARD, batch_checker_play, possible_moves, possible_single_die_moves


@pytest.mark.parametrize("die", [0, -1, 7, 32])
def test_invalid_die_rejected_by_move_generators(die):
    for first, second in ((die, 1), (1, die), (die, die)):
        with pytest.raises(ValueError, match="between 1 and 6"):
            possible_moves(STARTING_BOARD, first, second)
    with pytest.raises(ValueError, match="between 1 and 6"):
        possible_single_die_moves(STARTING_BOARD, die)


@pytest.mark.parametrize("level", ["1ply", "2ply"])
def test_invalid_dice_rejected_before_worker_dispatch(level):
    good = dict(board=STARTING_BOARD, die1=1, die2=2)
    with pytest.raises(ValueError, match="between 1 and 6"):
        batch_checker_play([good, {**good, "die2": 32}], eval_level=level, n_threads=2)
