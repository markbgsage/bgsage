# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parallel cubeful rollouts must read complete opening-move cache entries."""

import json
from pathlib import Path

import pytest

from bgsage import BgBotAnalyzer, WeightConfig


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
