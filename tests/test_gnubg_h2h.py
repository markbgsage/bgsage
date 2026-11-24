"""Native GNUbg decisions in the head-to-head simulator."""

import pytest

from bgsage import STARTING_BOARD, possible_moves
from bgsage.gnubg import GnuBgPolicy, _hint_board, resolve_gnubg_cli
from bgsage.h2h import run_head_to_head


@pytest.fixture(scope="module")
def gnubg_cli():
    try:
        return resolve_gnubg_cli()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def test_hint_notation_handles_repeated_paths_and_intermediate_hits():
    before = [0] * 26
    before[13], before[7], before[1] = 2, -1, -1
    after = _hint_board(before, "13/7*/1*(2)")
    assert after[0] == 2
    assert after[13] == after[7] == 0
    assert after[1] == 2

    before[25], before[22], before[23] = 2, -1, -1
    after = _hint_board(before, "bar/23* bar/22*")
    assert after[25] == 0
    assert after[22] == after[23] == 1
    assert after[0] == 2


@pytest.mark.parametrize("action,should_double,should_take", [
    ("No redouble, take (43.8%)", False, True),
    ("Too good to double, pass", False, False),
    ("Redouble, pass", True, False),
    ("Double, take", True, True),
])
def test_cube_verdict_uses_gnubg_recommendation(monkeypatch, action, should_double, should_take):
    monkeypatch.setattr(GnuBgPolicy, "_hint", lambda *args, **kwargs: f"Proper cube action: {action}\n")
    policy = object.__new__(GnuBgPolicy)
    result = policy.cube_action(STARTING_BOARD)
    assert result.should_double is should_double
    assert result.should_take is should_take


def test_native_dice_depth_board_orientation_and_cube(gnubg_cli):
    policy = GnuBgPolicy("3ply", gnubg_cli)
    output = policy._hint(STARTING_BOARD, 1, "centered", (6, 4))
    assert "Cubeful 2-ply" in output
    play = policy.checker_play(STARTING_BOARD, 6, 4).moves[0]
    assert play.board in possible_moves(STARTING_BOARD, 6, 4)

    asymmetric = [0] * 26
    asymmetric[0], asymmetric[8], asymmetric[13], asymmetric[22], asymmetric[25] = 1, 3, -2, -1, 2
    result = policy.cube_action(asymmetric, 4, "player")
    assert result.should_take
    reached_cap = policy.checker_play(STARTING_BOARD, 6, 4, 2, "opponent", max_cube_value=2)
    assert reached_cap.moves[0].board in possible_moves(STARTING_BOARD, 6, 4)


def test_gnubg_level_validation_precedes_executable_lookup():
    with pytest.raises(ValueError, match="GNUbg requires"):
        run_head_to_head(bot_b="gnubg", player_b="1T", games=2)
