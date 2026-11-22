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


def test_one_ply_cube_uses_analyzer_model_and_bearoff(model):
    weights, raw, db = model
    args = (BOARD, 2, bgbot_cpp.CubeOwner.PLAYER, weights.strategy_type)
    context = dict(away1=3, away2=5)
    expected = bgbot_cpp.evaluate_cube_decision_unified(
        *args, weights.weight_paths_list, weights.hidden_sizes_list, bearoff_db=db, **context)
    # Empty paths ensure a preloaded analyzer never reopens the weights.
    wrapped = bgbot_cpp.BearoffStrategy(raw, db)
    for strategy, database in ((raw, db), (wrapped, db), (wrapped, None)):
        assert bgbot_cpp.evaluate_cube_decision_unified(
            *args, [], [], loaded_strategy=strategy, bearoff_db=database, **context) == expected


def test_checker_batch_matches_individual_opponent_trees(model):
    _, strategy, db = model
    owner = bgbot_cpp.CubeOwner.OPPONENT
    for dice in ((3, 1), (6, 6)):
        moves = bgbot_cpp.possible_moves(BOARD, *dice)
        expected = []
        for endpoint in moves:
            single = bgbot_cpp.cubeful_probs_and_equity_nply(
                bgbot_cpp.flip_board(endpoint), owner, strategy, 3, cube_value=2,
                away1=5, away2=3, bearoff_db=db, root_board=BOARD, max_cube_value=4)
            expected.append((endpoint, single["probs"], single["equity"]))
        for threads in (1, 4):
            assert bgbot_cpp.cubeful_checker_moves_nply(
                BOARD, *dice, owner, strategy, 3, threads, 2, 5, 3, False, True, True, db, 4
            ) == expected


def test_four_ply_checker_subtrees_are_private_to_each_request(model):
    _, strategy, db = model
    owner = bgbot_cpp.CubeOwner.OPPONENT
    contexts = [(0, 0, 0), (5, 3, 4)]

    def evaluate(context, threads):
        away1, away2, cap = context
        return bgbot_cpp.cubeful_checker_moves_nply(
            BOARD, 3, 1, owner, strategy, 4, threads, 2, away1, away2,
            False, True, True, db, cap)

    # A serial batch uses only the thread-local recursion cache. Parallel
    # batches share 2P results among endpoints, including under concurrent scores.
    expected = [evaluate(context, 1) for context in contexts]
    assert expected[0] != expected[1]
    for threads in (4, 16):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(evaluate, context, threads) for context in contexts]
            assert [future.result() for future in futures] == expected


def test_checker_batch_requires_strategy():
    with pytest.raises(TypeError):
        bgbot_cpp.cubeful_checker_moves_nply(
            BOARD, 2, 1, bgbot_cpp.CubeOwner.CENTERED, None, 2, 4, 1,
            0, 0, False, True, True, None, 0)


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
