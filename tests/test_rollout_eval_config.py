# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Rollout checker/cube evaluator specification tests."""

from __future__ import annotations

import pytest

import bgbot_cpp
from bgsage import (
    BgBotAnalyzer,
    RolloutConfig,
    TrialEvalConfig,
    rollout_config_from_level,
)


@pytest.mark.parametrize(
    ("name", "ply"),
    [
        ("1P", 1),
        ("2p", 2),
        ("3-ply", 3),
        ("4ply", 4),
    ],
)
def test_named_ply_evaluator_aliases(name, ply):
    config = TrialEvalConfig(name)
    assert config.is_set()
    assert not config.is_rollout()
    assert config.ply == ply


@pytest.mark.parametrize(
    ("name", "trials", "depth", "decision_ply"),
    [
        ("1T", 72, 5, 1),
        ("truncated2", 360, 7, 2),
        ("3-t", 360, 7, 3),
    ],
)
def test_named_truncated_evaluators_have_complete_configs(
    name, trials, depth, decision_ply
):
    evaluator = TrialEvalConfig(name)
    assert evaluator.is_set()
    assert evaluator.is_rollout()
    assert evaluator.rollout_config is not None
    assert evaluator.rollout_config.n_trials == trials
    assert evaluator.rollout_config.truncation_depth == depth
    assert evaluator.rollout_config.decision_ply == decision_ply
    assert evaluator.rollout_config.minimum_rollout_moves == 2


def test_canonical_2t_and_3t_include_standalone_semantics():
    two_t = rollout_config_from_level("2T")
    assert two_t.truncation_ply == 2
    assert two_t.late_ply == 1
    assert two_t.late_threshold == 1
    assert two_t.ultra_late_threshold == 9999
    assert two_t.prefilter_threshold == pytest.approx(0.15)
    assert two_t.nested_cube_1ply_screen
    assert two_t.cube.ply == 2
    assert two_t.cube_late.ply == 2

    three_t = rollout_config_from_level("truncated3")
    assert three_t.truncation_ply == -1
    assert three_t.late_ply == 2
    assert three_t.late_threshold == 2
    assert three_t.ultra_late_threshold == 9999
    assert three_t.prefilter_threshold == pytest.approx(0.15)


def test_complete_rollout_config_is_accepted_directly():
    rollout = RolloutConfig()
    rollout.n_trials = 12
    rollout.truncation_depth = 4
    rollout.decision_ply = 2
    rollout.truncation_ply = 3
    rollout.late_ply = 1
    rollout.late_threshold = 2
    rollout.prefilter_threshold = 0.12
    rollout.minimum_rollout_moves = 3
    rollout.nested_cube_1ply_screen = True
    rollout.cube = TrialEvalConfig("2P")

    evaluator = TrialEvalConfig(rollout=rollout)
    assert evaluator.is_rollout()
    assert evaluator.rollout_config.n_trials == 12
    assert evaluator.rollout_config.truncation_ply == 3
    assert evaluator.rollout_config.minimum_rollout_moves == 3
    assert evaluator.rollout_config.nested_cube_1ply_screen
    assert evaluator.rollout_config.cube.ply == 2


def test_legacy_trial_eval_constructor_is_unchanged():
    evaluator = TrialEvalConfig(
        ply=3, rollout_trials=18, rollout_depth=6, rollout_ply=2
    )
    assert evaluator.ply == 3
    assert evaluator.rollout_trials == 18
    assert evaluator.rollout_depth == 6
    assert evaluator.rollout_ply == 2
    assert evaluator.rollout_config is None
    assert evaluator.is_rollout()


def test_public_rollout_accepts_mixed_string_levels():
    analyzer = BgBotAnalyzer(
        eval_level="rollout",
        n_trials=2,
        truncation_depth=1,
        parallel_threads=2,
        checker="3P",
        cube="2T",
        bearoff_db=False,
    )
    inner = analyzer._analyzer._inner
    config = inner._rollout_strategy.config()
    assert config.checker.ply == 3
    assert config.cube.rollout_config.n_trials == 360
    assert config.cube.rollout_config.decision_ply == 2


def test_public_rollout_accepts_complete_config_object():
    checker = RolloutConfig()
    checker.n_trials = 4
    checker.truncation_depth = 2
    checker.decision_ply = 1
    checker.ultra_late_threshold = 7

    analyzer = BgBotAnalyzer(
        eval_level="rollout",
        n_trials=2,
        truncation_depth=1,
        checker=checker,
        cube=TrialEvalConfig("1P"),
        bearoff_db=False,
    )
    inner = analyzer._analyzer._inner
    stored = inner._rollout_strategy.config().checker.rollout_config
    assert stored.n_trials == 4
    assert stored.truncation_depth == 2
    assert stored.ultra_late_threshold == 7


@pytest.mark.parametrize("value", ["5P", "4T", "rollout", "banana"])
def test_invalid_named_evaluator_is_rejected(value):
    with pytest.raises(ValueError, match="unknown rollout evaluator level"):
        TrialEvalConfig(value)


def test_invalid_public_evaluator_type_is_rejected():
    with pytest.raises(TypeError, match="checker must be"):
        BgBotAnalyzer(
            eval_level="rollout",
            n_trials=1,
            truncation_depth=1,
            checker=object(),
            bearoff_db=False,
        )
