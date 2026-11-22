# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Bearoff and NN evaluations must stay distinct when reusing a loaded model."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import weakref

import pytest
import bgbot_cpp
from bgsage.weights import WeightConfig


ROOT = Path(__file__).resolve().parents[1]
BOARD = [
    0, 0, 2, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -3, 0, 0, -4, 0, -1, 0,
]


@pytest.fixture(scope="module")
def model():
    weights = WeightConfig.from_model("stage5")
    strategy = bgbot_cpp.create_strategy(
        weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list
    )
    db = bgbot_cpp.BearoffDB()
    assert db.load(str(ROOT / "data" / "bearoff_1sided.db"))
    return weights, strategy, db


def test_loaded_model_honors_bearoff_database(model):
    weights, raw, db = model
    wrapped = bgbot_cpp.BearoffStrategy(raw, db)

    def evaluate(**kwargs):
        return bgbot_cpp.cube_decision_nply_unified(
            BOARD, 1, bgbot_cpp.CubeOwner.CENTERED, 3,
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list,
            **kwargs,
        )

    expected = evaluate(bearoff_db=db)
    assert expected["equity_dt"] != evaluate()["equity_dt"]
    assert evaluate(loaded_strategy=raw, bearoff_db=db) == expected
    assert evaluate(loaded_strategy=wrapped, bearoff_db=db) == expected
    assert evaluate(loaded_strategy=wrapped) == expected
    outer = bgbot_cpp.BearoffStrategy(wrapped, bgbot_cpp.BearoffDB())
    assert evaluate(loaded_strategy=outer) == expected


def test_warm_cache_distinguishes_nn_and_database(model):
    _, raw, db = model

    def evaluate(database):
        return bgbot_cpp.cubeful_probs_and_equity_nply(
            BOARD, bgbot_cpp.CubeOwner.CENTERED, raw, 2, bearoff_db=database
        )

    with ThreadPoolExecutor(1) as pool:
        expected_db = pool.submit(evaluate, db).result()
    expected_nn = evaluate(None)
    assert expected_db["probs"] != expected_nn["probs"]
    assert evaluate(db) == expected_db
    assert evaluate(None) == expected_nn


def test_loaded_model_distinguishes_automatic_and_explicit_cube_efficiency(model):
    weights, raw, _ = model
    contact = [
        0, 0, 0, 2, 3, 0, 4, -2, 2, 0, 0, 0, -4,
        2, -3, 0, -1, 0, 0, -3, 2, 0, -2, 0, 0, 0,
    ]

    def evaluate(x, threads, strategy=None):
        return bgbot_cpp.cube_decision_nply_unified(
            contact, 1, bgbot_cpp.CubeOwner.CENTERED, 3,
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list,
            cube_x_override=x, n_threads=threads, loaded_strategy=strategy,
        )["equity_nd"]

    # The old integer cache fingerprint rounded both values to zero, even
    # though a negative value requests a board-dependent cube efficiency.
    settings = (-0.00001, 0.00001)
    expected = [evaluate(x, 1) for x in settings]
    assert abs(expected[0] - expected[1]) > 0.005
    # The native pool keeps worker-local caches across public calls.
    for _ in range(4):
        assert [evaluate(x, 4, raw) for x in settings] == expected


def test_nested_native_wrapper_keeps_inner_database_alive(model):
    _, raw, _ = model
    inner_db = bgbot_cpp.BearoffDB()
    reference = weakref.ref(inner_db)
    outer = bgbot_cpp.BearoffStrategy(
        bgbot_cpp.BearoffStrategy(raw, inner_db), bgbot_cpp.BearoffDB()
    )
    del inner_db
    assert reference() is not None
    del outer
    assert reference() is None
