# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""The cube policy and the cubeless probability policy may share NN work."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from bgsage import BgBotAnalyzer


# The policies choose different continuations on this 1-away/4-away position.
# Substituting the cube search's probabilities shifts gammon wins by 1.52 pp.
MATCH = [1, 2, -1, 3, 3, 2, 3, -1, 2, 0, 0, 0, -3,
         0, 0, 0, 0, -3, 0, -4, 0, -2, 0, 0, 0, 0]
SCORE = dict(cube_value=2, cube_owner="player", away1=1, away2=4)
# This position exercises the specialist root routing and large doubles pools.
SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]


@pytest.mark.parametrize(("board", "options"), [(MATCH, SCORE), (SNAKE, {})])
def test_cube_probabilities_match_independent_cubeless_tree(board, options):
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
    plain = analyzer.cube_action(board, **options)
    # Per-roll details use their own traversal and still run the standalone
    # cubeless search to populate the five headline probabilities.
    independent = analyzer.cube_action(board, incl_2ply_details=True, **options)
    assert plain.probs == independent.probs
    assert plain.cubeless_equity == independent.cubeless_equity
    assert independent.details is not None


def test_concurrent_cube_calls_keep_ordered_batches_local():
    analyzer = BgBotAnalyzer(eval_level="3ply", parallel_threads=2)
    requests = [(MATCH, SCORE), (SNAKE, {}), (MATCH, {}), (SNAKE, SCORE)]

    def evaluate(request):
        board, options = request
        return analyzer.cube_action(board, **options)

    expected = [evaluate(request) for request in requests]
    assert expected[0].probs == expected[2].probs
    assert expected[0].equity_nd != expected[2].equity_nd
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(evaluate, requests * 2)) == expected * 2
