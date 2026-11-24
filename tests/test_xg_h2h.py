"""Regression checks for XG profile pinning and the shared game runner."""

from contextlib import contextmanager

import pytest

from bgsage import xg_h2h
from bgsage.h2h import run_head_to_head


def test_select_xg_profiles_by_name_and_report_staged_depths():
    profiles = {1: "XG 1P:0:-1:-1:-1:-1:-1::0:-1:-1:-1:-1",
                2: "XG 4P:2:3:-1:-1:-1:0::2:3:-1:-1:-1",
                3: "Roller+:1001:-1:-1:-1:-1:-1::1001:-1:-1:-1:-1"}
    selected = xg_h2h._select_profile(profiles, "4ply", "XG 4P")
    assert selected["index"] == 7
    assert selected["stages"] == {"move": (2, 3), "cube": (2, 3)}
    assert xg_h2h._select_profile(profiles, "truncated2", None)["name"] == "Roller+"
    with pytest.raises(ValueError, match="custom profile"):
        xg_h2h._select_profile(profiles, "2ply", "XG 1P")


def test_level_validation_admits_book_and_early_exit_but_rejects_drift():
    profile = {"move": (1001,), "cube": (1001,)}
    xg_h2h._check_level({"level": 998, "level_request": 998}, {"kind": "move"}, profile)
    xg_h2h._check_level({"level": 12, "level_request": 1001}, {"kind": "cube"}, profile)
    with pytest.raises(ValueError, match="returned level"):
        xg_h2h._check_level({"level": 1002, "level_request": 1002}, {"kind": "cube"}, profile)


@pytest.mark.parametrize("xg_seat", ["a", "b"])
def test_xg_uses_standard_game_records_and_fixed_sage_luck(monkeypatch, xg_seat):
    class Batch:
        rounds = 0
        seconds = 0.0

        def analyze(self, requests):
            self.rounds += 1
            assert all(request["max_cube_value"] == 4 for request in requests)
            return [({"board": item["post"]} if item["kind"] == "move"
                     else {"should_double": False, "should_take": True}) for item in requests]

    @contextmanager
    def session(level, name, executable, template, max_cube):
        assert (level, name, max_cube) == ("2ply", "My profile", 4)
        yield Batch(), {"profile": {"name": name}}

    monkeypatch.setattr(xg_h2h, "_session", session)
    args = {"bot_" + xg_seat: "xg", "player_" + xg_seat: "2P"}
    report = run_head_to_head(**args, games=2, seed=11, workers=2, max_cube_value=4,
                              xg_profile="My profile")
    assert report["player_" + xg_seat] == {"bot": "xg", "eval_level": "2ply"}
    assert report["reference"] == "sage_2ply_nd"
    assert report["xg"]["batch_count"] > 0
    assert report["summary"]["estimated_cpu_speedup_at_equal_se"] is None
    for game in report["games"]:
        assert game["adjusted_points"] == pytest.approx(game["raw_points"] - game["luck_points"])


def test_batch_failure_stops_game_streams(monkeypatch):
    class BrokenBatch:
        def analyze(self, requests):
            raise ValueError("invalid XG response")

    @contextmanager
    def session(*args):
        yield BrokenBatch(), {}

    monkeypatch.setattr(xg_h2h, "_session", session)
    with pytest.raises(ValueError, match="invalid XG response"):
        run_head_to_head(bot_b="xg", games=2, workers=2)
