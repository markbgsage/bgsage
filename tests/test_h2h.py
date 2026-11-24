"""Verify luck scoring, especially transitions that can change its expectation."""

from math import fsum, sqrt
from pathlib import Path
from statistics import covariance, variance
from types import SimpleNamespace

import pytest

from bgsage import BgBotAnalyzer, GameConfig, simulate_game
from bgsage import h2h


class Reference:
    def __init__(self):
        self.calls = []

    def cube_action(self, board, **kwargs):
        self.calls.append(kwargs)
        scores = [{"die1": lo, "die2": hi, "cubeful_equity": board[0] + (lo + hi) / 10}
                  for lo, hi in sorted(h2h._ROLLS)]
        return SimpleNamespace(details={"nd": scores})


def test_opening_centers_both_openers_and_excludes_doubles(monkeypatch):
    # Deliberately asymmetric tables to check the joint mean and sign.
    monkeypatch.setattr(h2h, "STARTING_BOARD", [1])
    monkeypatch.setattr(h2h, "flip_board", lambda board: [10])
    table = h2h._LuckScorer(Reference(), GameConfig())._opening_table()
    assert len(table) == 30
    assert all(lo != hi for (_, (lo, hi)) in table)
    assert fsum(table.values()) == pytest.approx(0)
    assert table[1, (1, 2)] == pytest.approx(1.3 + 4.5)
    assert table[2, (1, 2)] == pytest.approx(-10.3 + 4.5)


def test_uses_post_take_cube_and_reference_roll_and_skips_cube_pass():
    reference = Reference()
    scorer = h2h._LuckScorer(reference, GameConfig(jacoby=False, max_cube_value=8))
    game = {"turns": [
        {"player": 1, "dice": [6, 1]},
        {"player": 2, "dice": [6, 6], "board": [0], "post_board": [1000],
         "cube_action": "double/take", "cube_value": 2, "cube_owner": "player",
         "checker_cube_value": 4, "checker_cube_owner": "opponent"},
        {"player": 1, "dice": None, "cube_action": "double/pass"},
    ]}
    # Opening roll contributes +0.7. B's 6-6 is +0.5 above the mean in
    # B's cube units, or -2 in A's points.
    assert scorer.score(game) == pytest.approx(-1.3)
    assert len(reference.calls) == 2
    assert reference.calls[-1] == dict(cube_value=4, cube_owner="opponent",
                                      jacoby=False, beaver=False, max_cube_value=8,
                                      incl_2ply_details=True)


def test_custom_start_centers_all_first_rolls_after_cube_decision():
    reference = Reference()
    board = [0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,
             5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2, 0]
    config = GameConfig(start_board=tuple(board), player_on_roll=2)
    scorer = h2h._LuckScorer(reference, config)
    turn = {"player": 2, "board": board, "cube_action": "double/take",
            "checker_cube_value": 2, "checker_cube_owner": "opponent"}
    values = [scorer.score({"turns": [dict(turn, dice=[a, b])]})
              for a in range(1, 7) for b in range(1, 7)]
    assert values[-1] == pytest.approx(-1.0)
    assert fsum(values) == pytest.approx(0)
    assert scorer.score({"turns": [dict(turn, dice=None, cube_action="double/pass")]}) == 0
    assert len(reference.calls) == 36
    assert all(call["cube_value"] == 2 and call["cube_owner"] == "opponent"
               for call in reference.calls)


def test_rejects_duplicate_meant_to_replace_a_roll():
    class IncompleteReference(Reference):
        def cube_action(self, board, **kwargs):
            result = super().cube_action(board, **kwargs)
            result.details["nd"][-1] = result.details["nd"][0]
            return result

    with pytest.raises(ValueError, match="complete.*21-roll"):
        h2h._LuckScorer(IncompleteReference(), GameConfig())._opening_table()


def test_native_dead_cube_reference_bearoff():
    board = [0] * 26
    board[6], board[24] = 1, -1
    native = BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    scorer = h2h._LuckScorer(native, GameConfig(max_cube_value=4))
    values = scorer._table(board, 4, "player")
    assert values[1, 1] == pytest.approx(-1)
    assert values[2, 2] == pytest.approx(1)
    assert fsum(value * (1 if lo == hi else 2) for (lo, hi), value in values.items()) / 36 == pytest.approx(0.5)


@pytest.mark.parametrize("jacoby,cube,owner,multiplier", [
    (True, 1, "centered", 1), (False, 1, "centered", 3), (True, 2, "player", 3),
])
def test_native_terminal_scoring(jacoby, cube, owner, multiplier):
    board = [0] * 26
    board[1], board[2] = 1, -15
    reference = BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    scorer = h2h._LuckScorer(reference, GameConfig(jacoby=jacoby))
    assert scorer._table(board, cube, owner)[1, 2] == pytest.approx(multiplier)


def test_automatic_worker_count_respects_cpu_game_and_platform_limits(monkeypatch):
    monkeypatch.setattr(h2h, "_available_cpus", lambda: 96)
    monkeypatch.setattr(h2h, "sys", SimpleNamespace(platform="linux"))
    assert h2h._worker_count(None, 100) == 96
    assert h2h._worker_count(None, 24) == 24
    assert h2h._worker_count(4, 100) == 4
    monkeypatch.setattr(h2h, "sys", SimpleNamespace(platform="win32"))
    assert h2h._worker_count(None, 100) == 61


def test_explicit_single_thread_applies_to_multipy_and_rollout_prefilter(monkeypatch):
    create_multipy = h2h.bgbot_cpp.create_multipy
    requests = []

    def capture_threads(*args, **kwargs):
        requests.append(kwargs["parallel_threads"])
        return create_multipy(*args, **kwargs)

    monkeypatch.setattr(h2h.bgbot_cpp, "create_multipy", capture_threads)
    BgBotAnalyzer(eval_level="2ply", parallel_threads=1)
    BgBotAnalyzer(eval_level="truncated2", parallel_threads=1)
    assert requests == [1, 1]


def test_run_keeps_simulation_payoffs_and_reports_paired_sample_se(monkeypatch):
    # Some legacy tests prepend their own build path during collection.
    # Spawned workers should use the same native build this process loaded.
    monkeypatch.syspath_prepend(str(Path(h2h.bgbot_cpp.__file__).parent))
    beta = 0.75
    args = dict(player_a="1P", player_b="2ply", games=6, seed=19, beta=beta)
    result = h2h.run_head_to_head(**args, workers=2)
    serial = h2h.run_head_to_head(**args, workers=1)
    assert result["player_a"]["eval_level"] == "1ply"
    assert result["player_b"]["eval_level"] == "2ply"
    records = result["games"]
    players = tuple(BgBotAnalyzer(eval_level=level, parallel_threads=1) for level in ("1ply", "2ply"))
    score_fields = ("seed", "raw_points", "luck_points", "adjusted_points")
    for record, serial_record in zip(records, serial["games"]):
        assert [record[key] for key in score_fields] == [serial_record[key] for key in score_fields]
        replay = simulate_game(seed=record["seed"], players=players)
        sign = 1 if replay["winner"] == 1 else -1
        assert record["raw_points"] == sign * replay["points"]
        assert record["adjusted_points"] == pytest.approx(record["raw_points"] - beta * record["luck_points"])
    raw = [record["raw_points"] for record in records]
    luck = [record["luck_points"] for record in records]
    expected = sqrt((variance(raw) + beta**2 * variance(luck) - 2 * beta * covariance(raw, luck)) / len(raw))
    assert result["summary"]["adjusted"]["se_ppg"] == pytest.approx(expected)
    assert list(h2h._game_seeds(19, 7))[:6] == [record["seed"] for record in records]
