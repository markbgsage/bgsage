# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parallel cubeful rollouts must read complete opening-move cache entries."""

import json
from pathlib import Path

import pytest

from bgsage import BgBotAnalyzer, TrialEvalConfig, WeightConfig


BENCHMARK = Path(__file__).resolve().parents[1] / "data" / "multi_ply_benchmark"


@pytest.mark.parametrize(
    "level,api,position_id,seed,repetitions",
    [
        ("truncated1", "cube_action", "cube-money-anchoring-07", 3972021868, 8),
        ("truncated2", "checker_play", "checker-money-priming-10", 2570830711, 3),
    ],
)
def test_parallel_prefill_matches_serial(level, api, position_id, seed, repetitions):
    # These seeded public calls previously returned different means and standard
    # errors when an early trial wrote a move-1 table alongside its prefill worker.
    positions = json.loads((BENCHMARK / f"{api}.json").read_text(encoding="utf-8"))
    inputs = next(row["inputs"] for row in positions["positions"] if row["id"] == position_id)
    weights = WeightConfig.from_model("stage11")
    serial = BgBotAnalyzer(weights=weights, eval_level=level, parallel_threads=1)
    serial.set_seed(seed)
    expected = getattr(serial, api)(**inputs)

    parallel = BgBotAnalyzer(weights=weights, eval_level=level, parallel_threads=16)
    for _ in range(repetitions):
        parallel.set_seed(seed)
        assert getattr(parallel, api)(**inputs) == expected


def test_partial_opening_stratum_preserves_shared_reply():
    # Trial 37 reuses one opening in the second dice stratum. Move-0 VR is
    # active at this budget, and move 1 reads the reply to the cached opening.
    positions = json.loads((BENCHMARK / "cube_action.json").read_text(encoding="utf-8"))
    inputs = next(row["inputs"] for row in positions["positions"]
                  if row["id"] == "cube-money-attacking-01")
    options = dict(weights=WeightConfig.from_model("stage11"), eval_level="rollout",
                   n_trials=37, truncation_depth=3, decision_ply=1, truncation_ply=2,
                   seed=1720888050, cubeful=True)
    serial = BgBotAnalyzer(parallel_threads=1, **options).cube_action(**inputs)
    # Fixed-seed checkpoint values also catch a repeatable error shared by
    # both schedulers. Leave room for rounding between CPU math kernels.
    assert (serial.probs.win, serial.equity_nd, serial.equity_dt) == pytest.approx(
        (0.58385575, 0.43233198, 0.35523140), abs=1e-5
    )
    assert BgBotAnalyzer(parallel_threads=4, **options).cube_action(**inputs) == serial


@pytest.mark.parametrize("purpose", ["checker", "checker_late", "cube"])
def test_nested_trial_keeps_its_parents_legal_boards(purpose):
    # An inner checker rollout can revisit both the opening prefill and trial
    # generators on its parent's worker while a selected index is still live.
    positions = json.loads((BENCHMARK / "cube_action.json").read_text(encoding="utf-8"))
    inputs = next(row["inputs"] for row in positions["positions"]
                  if row["id"] == "cube-money-attacking-01")
    options = dict(weights=WeightConfig.from_model("stage11"), eval_level="rollout",
                   n_trials=4, truncation_depth=3, decision_ply=1, truncation_ply=1,
                   late_threshold=2, seed=1720888050, cubeful=True)
    options[purpose] = TrialEvalConfig(rollout_trials=2, rollout_depth=1, rollout_ply=1)
    serial = BgBotAnalyzer(parallel_threads=1, **options).cube_action(**inputs)
    assert serial.optimal_action == "No Double"
    if purpose in ("checker", "checker_late"):
        # Late-only nestings once agreed across workers while both indexed a
        # child board. Check the chosen branch and the other branch value.
        assert (serial.equity_nd, serial.equity_dt) == pytest.approx(
            (0.3725155, 0.3370189), abs=1e-5
        )
    assert BgBotAnalyzer(parallel_threads=2, **options).cube_action(**inputs) == serial
