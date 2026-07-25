# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""High-level analysis interface for the Open Sage bot engine.

This module provides :class:`BgBotAnalyzer`, the main entry point for
checker play and cube action analysis at any evaluation level (1-ply
through N-ply and Monte Carlo rollout).

Typical usage::

    from bgsage import BgBotAnalyzer

    analyzer = BgBotAnalyzer()                       # 1-ply, cubeful
    result = analyzer.checker_play(STARTING_BOARD, 3, 1)
    for m in result.moves:
        print(f"{m.equity:+.3f}  {m.probs.win:.1%}")
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import bgbot_cpp

from .luck import roll_luck as _roll_luck
from .types import (
    CheckerPlayResult,
    CubeActionResult,
    LuckResult,
    MoveAnalysis,
    PostMoveAnalysis,
    Probabilities,
)
from .weights import WeightConfig, WeightConfigPair, bearoff_db_path, default_weights

# ---------------------------------------------------------------------------
# Cube owner mapping
# ---------------------------------------------------------------------------

OWNER_MAP: dict[str, Any] = {
    "centered": bgbot_cpp.CubeOwner.CENTERED,
    "player": bgbot_cpp.CubeOwner.PLAYER,
    "opponent": bgbot_cpp.CubeOwner.OPPONENT,
}

_FLIP_OWNER: dict[Any, Any] = {
    bgbot_cpp.CubeOwner.CENTERED: bgbot_cpp.CubeOwner.CENTERED,
    bgbot_cpp.CubeOwner.PLAYER: bgbot_cpp.CubeOwner.OPPONENT,
    bgbot_cpp.CubeOwner.OPPONENT: bgbot_cpp.CubeOwner.PLAYER,
}


def resolve_owner(cube_owner: str | Any) -> Any:
    """Convert a string cube owner to a ``bgbot_cpp.CubeOwner`` enum value."""
    if isinstance(cube_owner, str):
        return OWNER_MAP[cube_owner.lower()]
    return cube_owner


class RolloutCancelled(Exception):
    """Raised when a rollout is cancelled via cancel()."""
    pass


def _default_parallel_threads() -> int:
    env_threads = os.getenv("BGBOT_MULTIPLY_THREADS", "")
    if not env_threads:
        return max(2, os.cpu_count() or 2)
    try:
        parsed = int(env_threads)
    except ValueError:
        return 0
    return max(parsed, 0)


# ---------------------------------------------------------------------------
# Internal cubeless analyzers
# ---------------------------------------------------------------------------


class _CubelessBase:
    """Shared infrastructure for cubeless analyzers."""

    def __init__(self, weights: WeightConfig | WeightConfigPair):
        self._weights = weights
        self._strategy_1ply = bgbot_cpp.create_strategy(
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list)
        self._bearoff_db = None  # Set by BgBotAnalyzer after construction

    def _score_candidates(
        self,
        candidates: list,
        board: list[int],
        cube_owner: str | None = None,
        cube_value: int = 1,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        strategy=None,
    ) -> list[tuple[float, float, list[int], list[float]]]:
        if strategy is None:
            strategy = self._strategy_1ply
        owner = resolve_owner(cube_owner) if cube_owner else None
        is_match = away1 > 0 or away2 > 0
        scored = []
        for b in candidates:
            bl = list(b)
            r = strategy.evaluate_board(bl, board)
            cl_eq = r["equity"]
            probs = list(r["probs"])
            if owner is not None:
                race = bgbot_cpp.is_race(bl)
                pp, op = bgbot_cpp.pip_counts(bl)
                x = bgbot_cpp.cube_efficiency(probs, race, pp, op)
                if is_match:
                    cf_eq = bgbot_cpp.cl2cf(probs, cube_value, owner, x,
                                            away1, away2, is_crawford,
                                            jacoby=jacoby)
                else:
                    jacoby_active = (
                        jacoby and owner == bgbot_cpp.CubeOwner.CENTERED
                    )
                    cf_eq = bgbot_cpp.cl2cf_money(probs, owner, x,
                                                  jacoby_active=jacoby_active)
            else:
                cf_eq = cl_eq
            scored.append((cf_eq, cl_eq, bl, probs))
        scored.sort(key=lambda item: -item[0])
        return scored

    @staticmethod
    def _filter_candidates(
        scored_1ply: list,
        threshold: float,
        max_moves: int,
    ) -> tuple[list, set]:
        best_eq = scored_1ply[0][0]
        survivors = [
            item
            for item in scored_1ply
            if (best_eq - item[0]) < threshold
        ][:max_moves]
        survivor_set = {tuple(item[2]) for item in survivors}
        return survivors, survivor_set

    @staticmethod
    def _force_include(
        scored_pool: list,
        survivors: list,
        survivor_set: set,
        force_boards,
    ) -> None:
        """Force caller-specified post-move boards into the deep-eval survivor
        set so they are always evaluated at the full (N-ply / rollout) level,
        even when the move filter would otherwise drop them.

        Used by the app's expert re-evaluation to guarantee the move a player
        actually made is scored at the expert level — i.e. apples-to-apples with
        the best move, instead of carrying only its cheap 1-ply filter equity.

        ``scored_pool`` is the list of ``(cf_eq, cl_eq, board, probs)`` tuples
        from which survivors are drawn (the 1-ply scored list, or the 2-ply
        rescored list for the rollout prefilter path). Boards not present in the
        pool (already a survivor, or not a legal candidate) are skipped. Mutates
        ``survivors`` / ``survivor_set`` in place.
        """
        if not force_boards:
            return
        force_set = {tuple(b) for b in force_boards}
        for item in scored_pool:
            tb = tuple(item[2])
            if tb in force_set and tb not in survivor_set:
                survivors.append(item)
                survivor_set.add(tb)

    @staticmethod
    def _promote_second_best(results: list, board: list[int], evaluate_fn) -> None:
        results.sort(key=lambda x: -x["equity"])
        while len(results) >= 2 and results[1].get("is_1ply_only"):
            r = results[1]
            equity, probs, eval_level, extra = evaluate_fn(r["board"], board)
            r["equity"] = equity
            r["probs"] = probs
            r["eval_level"] = eval_level
            r.pop("is_1ply_only", None)
            r.update(extra)
            results.sort(key=lambda x: -x["equity"])

    @staticmethod
    def _finalize_results(results: list[dict]) -> list[dict]:
        results.sort(key=lambda x: -x["equity"])
        if results:
            best = results[0]["equity"]
            for r in results:
                r["equity_diff"] = r["equity"] - best
        return results

    @staticmethod
    def _format_cube_result(r: dict, eval_level: str = "1-ply") -> dict:
        return {
            "probs": list(r["probs"]),
            "cubeless_equity": r.get("cubeless_equity", 0),
            "equity_nd": r["equity_nd"],
            "equity_dt": r["equity_dt"],
            "equity_dp": r["equity_dp"],
            "should_double": bool(r["should_double"]),
            "should_take": bool(r["should_take"]),
            "optimal_equity": r["optimal_equity"],
            "is_beaver": bool(r.get("is_beaver", False)),
            "cubeless_se": r.get("cubeless_se", None),
            "equity_nd_se": r.get("equity_nd_se", None),
            "equity_dt_se": r.get("equity_dt_se", None),
            "eval_level": eval_level,
        }


class _OnePlyAnalyzer(_CubelessBase):

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        force_boards=None,
    ) -> list[dict]:
        # ``force_boards`` is a no-op at 1-ply: every candidate is already
        # evaluated at the full level, so there's nothing to force into a
        # filtered deep-eval set. Accepted for call-signature symmetry.
        del force_boards
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []
        results = []
        for b in candidates:
            bl = list(b)
            r = self._strategy_1ply.evaluate_board(bl, board)
            results.append({
                "board": bl,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "eval_level": "1-ply",
            })
        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        if incl_2ply_details:
            raise ValueError("incl_2ply_details requires at least 2-ply evaluation")
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.evaluate_cube_decision_unified(
            board, cube_value, owner,
            self._weights.strategy_type,
            self._weights.weight_paths_list,
            self._weights.hidden_sizes_list,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver,
            bearoff_db=self._bearoff_db,
        )
        return self._format_cube_result(r, eval_level="1-ply")


class _MultiPlyAnalyzer(_CubelessBase):

    FILTER_MAX_MOVES = 5
    FILTER_THRESHOLD = 0.08

    def __init__(self, weights, n_plies, parallel_evaluate=True, parallel_threads=0):
        super().__init__(weights)
        self._n_plies = n_plies
        self._parallel_evaluate = parallel_evaluate
        requested_threads = parallel_threads
        if parallel_threads <= 0 and n_plies > 1:
            requested_threads = _default_parallel_threads()
        elif n_plies > 1:
            requested_threads = max(2, parallel_threads)
        self._parallel_threads = requested_threads
        self._strategy_nply = bgbot_cpp.create_multipy(
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list,
            n_plies=n_plies,
            parallel_evaluate=parallel_evaluate,
            parallel_threads=self._parallel_threads,
        )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        force_boards=None,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby,
        )
        survivors, survivor_set = self._filter_candidates(
            scored_1ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )
        # Force-include caller-specified boards (e.g. the move actually played
        # in an expert re-eval) so they're evaluated at N-ply even when the
        # filter dropped them.
        self._force_include(scored_1ply, survivors, survivor_set, force_boards)

        results = []
        for feq, cleq, b, p0 in survivors:
            r = self._strategy_nply.evaluate_board(b, board)
            results.append({
                "board": b,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "eval_level": f"{self._n_plies}-ply",
            })

        for feq, cleq, b, p in scored_1ply:
            if tuple(b) not in survivor_set:
                results.append({
                    "board": b,
                    "equity": cleq,
                    "probs": p,
                    "is_1ply_only": True,
                    "eval_level": "1-ply",
                })

        n_plies = self._n_plies
        strategy = self._strategy_nply

        def _nply_eval(b, board_ref):
            r = strategy.evaluate_board(b, board_ref)
            return r["equity"], list(r["probs"]), f"{n_plies}-ply", {}

        self._promote_second_best(results, board, _nply_eval)
        self._strategy_nply.clear_cache()
        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.cube_decision_nply_unified(
            board, cube_value, owner, self._n_plies,
            self._weights.strategy_type,
            self._weights.weight_paths_list,
            self._weights.hidden_sizes_list,
            n_threads=self._parallel_threads,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver,
            bearoff_db=self._bearoff_db,
            incl_2ply_details=incl_2ply_details,
        )
        result = self._format_cube_result(r, eval_level=f"{self._n_plies}-ply")

        # Pass through 2-ply details if present
        if "details" in r:
            result["details"] = r["details"]

        # The C++ binding computes N-ply cubeless probs with parallelism and
        # bearoff DB. No need to recompute in Python.

        # Always clear the shared position cache after cube analysis.
        # cube_decision_nply() fills the cache internally; without clearing,
        # subsequent analyses (especially on bearoff positions where the
        # non-bearoff branch above is skipped) can return stale cached values.
        self._strategy_nply.clear_cache()

        return result


class _RolloutAnalyzer(_CubelessBase):

    FILTER_MAX_MOVES = 5
    FILTER_THRESHOLD = 0.08

    def __init__(
        self, weights, n_trials=1296, truncation_depth=0,
        decision_ply=1, truncation_ply=-1, n_threads=0, seed=42,
        late_ply=-1, late_threshold=20,
        parallelize_trials=True,
        checker=None, checker_late=None,
        cube=None, cube_late=None,
        ultra_late_threshold=9999,
        cubeful_trial_moves=True,
        cubeful_late_threshold=0,
        prefilter_threshold=0.0,
        target_se=0.0,
        max_batches=50,
        filter_max_moves=None,
        filter_threshold=None,
    ):
        super().__init__(weights)
        # Candidate-selection filter for checker_play_analytics: how many of
        # the top scored moves get rolled out (the rest keep their filter-stage
        # equity). Instance attrs shadow the TINY class defaults when a wider
        # search interval is requested.
        if filter_max_moves is not None:
            self.FILTER_MAX_MOVES = int(filter_max_moves)
        if filter_threshold is not None:
            self.FILTER_THRESHOLD = float(filter_threshold)
        # Two-stage candidate filter for checker_play_analytics: when
        # prefilter_threshold > 0, stage 1 is a loose 1-ply cull at that
        # threshold (no max_moves cap), stage 2 is the standard TINY filter
        # at 2-ply on stage 1's survivors. Survivors of stage 2 are rolled
        # out. When 0, the legacy single-stage 1-ply TINY filter is used.
        self._prefilter_threshold = float(prefilter_threshold)
        self._strategy_2ply = None
        if self._prefilter_threshold > 0:
            self._strategy_2ply = bgbot_cpp.create_multipy(
                weights.strategy_type,
                weights.weight_paths_list,
                weights.hidden_sizes_list,
                n_plies=2,
                parallel_evaluate=True,
                parallel_threads=max(2, _default_parallel_threads()),
            )
        requested_threads = n_threads
        if n_threads <= 0:
            requested_threads = _default_parallel_threads()
        elif n_threads < 1:
            requested_threads = 1
        self._parallel_threads = max(1, requested_threads)
        self._parallelize_trials = bool(parallelize_trials)
        self._rollout_config = {
            "n_trials": n_trials,
            "truncation_depth": truncation_depth,
            "decision_ply": decision_ply,
            "truncation_ply": truncation_ply,
            "n_threads": self._parallel_threads,
            "seed": seed,
            "late_ply": late_ply,
            "late_threshold": late_threshold,
        }
        self._cancel_event = threading.Event()

        # Convert None to default TrialEvalConfig
        _empty = bgbot_cpp.TrialEvalConfig()
        checker_cfg = checker if checker is not None else _empty
        checker_late_cfg = checker_late if checker_late is not None else _empty
        cube_cfg = cube if cube is not None else _empty
        cube_late_cfg = cube_late if cube_late is not None else _empty

        self._rollout_strategy = bgbot_cpp.create_rollout(
            weights.strategy_type, weights.weight_paths_list, weights.hidden_sizes_list,
            n_trials=n_trials,
            truncation_depth=truncation_depth,
            decision_ply=decision_ply,
            truncation_ply=truncation_ply,
            n_threads=self._parallel_threads,
            seed=seed,
            late_ply=late_ply,
            late_threshold=late_threshold,
            parallelize_trials=parallelize_trials,
            checker=checker_cfg,
            checker_late=checker_late_cfg,
            cube=cube_cfg,
            cube_late=cube_late_cfg,
            ultra_late_threshold=ultra_late_threshold,
            cubeful_trial_moves=cubeful_trial_moves,
            cubeful_late_threshold=cubeful_late_threshold,
            target_se=target_se,
            max_batches=max_batches,
        )

    def set_seed(self, seed):
        """Reseed the rollout RNG (for independent seeded batches; keeps the
        SharedPosCache warm). Clears the strategy's cached stratified dice."""
        self._rollout_strategy.set_seed(int(seed))

    def cancel(self):
        """Request cancellation of in-progress rollout."""
        self._cancel_event.set()
        self._rollout_strategy.cancel()

    def reset_cancel(self):
        """Clear cancellation flag for reuse."""
        self._cancel_event.clear()
        self._rollout_strategy.reset_cancel()

    def _check_cancel(self):
        """Raise RolloutCancelled if cancellation was requested."""
        if self._cancel_event.is_set():
            raise RolloutCancelled()

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        force_boards=None,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby,
        )

        # Stage 1: 1-ply filter. If prefilter_threshold > 0, this is a loose
        # cull (no max_moves cap) that just drops obvious garbage; stage 2 at
        # 2-ply then applies the TINY filter. Otherwise it's a single-stage
        # TINY filter at 1-ply.
        if self._prefilter_threshold > 0:
            stage1_survivors, _ = self._filter_candidates(
                scored_1ply, self._prefilter_threshold, max_moves=len(scored_1ply)
            )
        else:
            stage1_survivors = scored_1ply

        # Stage 2: 2-ply rescore + TINY filter (only when prefilter is on and
        # more than one candidate survived stage 1).
        scored_2ply = None
        scored_2ply_set: set = set()
        if self._prefilter_threshold > 0 and len(stage1_survivors) > 1:
            stage1_boards = [item[2] for item in stage1_survivors]
            scored_2ply = self._score_candidates(
                stage1_boards, board, cube_owner,
                cube_value=cube_value, away1=away1, away2=away2,
                is_crawford=is_crawford, jacoby=jacoby,
                strategy=self._strategy_2ply,
            )
            scored_2ply_set = {tuple(item[2]) for item in scored_2ply}
            survivors, survivor_set = self._filter_candidates(
                scored_2ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
            )
            self._strategy_2ply.clear_cache()
        else:
            survivors, survivor_set = self._filter_candidates(
                stage1_survivors, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
            )

        # Ensure at least 2 candidates get rolled out so the cubeful sort
        # always has at least two rollout-quality entries to compare.
        fallback_pool = scored_2ply if scored_2ply is not None else scored_1ply
        if len(survivors) < 2 and len(fallback_pool) >= 2:
            for item in fallback_pool:
                if tuple(item[2]) not in survivor_set:
                    survivors.append(item)
                    survivor_set.add(tuple(item[2]))
                    if len(survivors) >= 2:
                        break
        # Force-include caller-specified boards (e.g. the move actually played
        # in an expert re-eval) so they're rolled out even when the filter
        # dropped them. Pulls from the same pool survivors are drawn from so the
        # forced entry is rolled out (not left at its 2-ply / 1-ply fallback).
        self._force_include(fallback_pool, survivors, survivor_set, force_boards)
        self._check_cancel()

        n_trials = self._rollout_config["n_trials"]
        results = []
        total_moves = len(survivors)
        # When cube info is present, use cubeful_evaluate_board: the C++
        # function returns rollout-level cubeful equity that already
        # incorporates the opponent's optimal cube action at the start of
        # their turn (it delegates to cubeful_cube_decision on the flipped
        # opp position and collapses ND/DT/DP into opp's optimal). Cubeless
        # probs/equity come from the same trials, already inverted to SP's
        # perspective at the post-move board `b`. No perspective fiddling
        # needed at the python layer.
        owner = resolve_owner(cube_owner) if cube_owner else None
        use_cubeful_rollout = owner is not None
        for i, (feq, cleq, b, p0) in enumerate(survivors):
            self._check_cancel()

            # Trial-level progress callback: maps trial progress within
            # the current move to overall progress across all moves.
            def _trial_progress(completed_trials, total_trials, _move_idx=i):
                if progress_callback:
                    overall = _move_idx * n_trials + completed_trials
                    overall_total = total_moves * n_trials
                    progress_callback(overall, overall_total, results)

            try:
                if use_cubeful_rollout:
                    r = self._rollout_strategy.cubeful_evaluate_board(
                        b, board,
                        cube_value=cube_value, owner=owner,
                        away1=away1, away2=away2, is_crawford=is_crawford,
                        jacoby=jacoby, beaver=beaver,
                        progress=_trial_progress,
                    )
                else:
                    r = self._rollout_strategy.evaluate_board(
                        b, board, _trial_progress)
            except bgbot_cpp.RolloutCancelled:
                raise RolloutCancelled()

            probs = list(r["probs"])
            # Override with exact bearoff DB probs when available.
            if self._bearoff_db is not None and self._bearoff_db.is_bearoff(b):
                probs = self._bearoff_db.lookup_probs(b, post_move=True)

            entry = {
                "board": b,
                "equity": r["equity"],          # cubeless equity
                "probs": probs,
                "std_error": r.get("std_error", 0),
                "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
                "eval_level": "Rollout",
            }
            if use_cubeful_rollout:
                # cubeful_evaluate_board returns rollout-level cubeful that
                # already accounts for opp's optimal cube action — store it
                # so _CubefulAnalyzer uses it directly instead of recomputing.
                entry["rollout_cubeful_equity"] = r["cubeful_equity"]
                entry["rollout_cubeful_se"] = r["cubeful_se"]
            results.append(entry)

        # Non-rolled-out moves: those that survived stage 1 (1-ply prefilter)
        # but failed stage 2 (2-ply TINY) get their 2-ply equity; those that
        # failed stage 1 keep their 1-ply equity. When prefilter is off, all
        # non-survivors get their 1-ply equity (legacy single-stage path).
        if scored_2ply is not None:
            for feq, cleq, b, p in scored_2ply:
                if tuple(b) not in survivor_set:
                    results.append({
                        "board": b,
                        "equity": cleq,
                        "probs": p,
                        "eval_level": "2-ply",
                    })
        for feq, cleq, b, p in scored_1ply:
            t = tuple(b)
            if t in survivor_set or t in scored_2ply_set:
                continue
            results.append({
                "board": b,
                "equity": cleq,
                "probs": p,
                "eval_level": "1-ply",
            })

        return self._finalize_results(results)

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        self._check_cancel()
        owner = resolve_owner(cube_owner)

        # Wire trial-level progress for cube rollout
        def _cube_trial_progress(completed_trials, total_trials):
            if progress_callback:
                progress_callback(completed_trials, total_trials, [])

        try:
            r = self._rollout_strategy.cube_decision(
                board, cube_value, owner,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver,
                progress=_cube_trial_progress if progress_callback else None,
            )
        except bgbot_cpp.RolloutCancelled:
            raise RolloutCancelled()
        result = self._format_cube_result(r, eval_level="Rollout")

        # Override cubeless probs with exact bearoff DB values when available.
        # The rollout's cubeful equities (ND/DT/DP) are still valuable, but
        # the cubeless probs from Monte Carlo are noisy — the DB is exact.
        if self._bearoff_db is not None and self._bearoff_db.is_bearoff(board):
            pre_roll = self._bearoff_db.lookup_probs(board, post_move=False)
            result["probs"] = pre_roll
            result["cubeless_equity"] = (
                2.0 * pre_roll[0] - 1.0
                + pre_roll[1] - pre_roll[3]
                + pre_roll[2] - pre_roll[4]
            )

        return result


class _CubefulAnalyzer:
    """Cubeful wrapper around any cubeless analyzer."""

    def __init__(self, inner: _CubelessBase):
        self._inner = inner
        self._weights = inner._weights
        if isinstance(inner, _MultiPlyAnalyzer):
            self._cubeful_ply = inner._n_plies
        elif isinstance(inner, _RolloutAnalyzer):
            # Use the rollout's decision_ply for cubeful equity per-move.
            # This gives N-ply cubeful evaluation matching the rollout's strength,
            # rather than falling back to crude 1-ply Janowski.
            dp = inner._rollout_config["decision_ply"]
            self._cubeful_ply = max(dp, 1)
        else:
            self._cubeful_ply = 1

    def set_seed(self, seed):
        """Delegate reseeding to the wrapped (rollout) analyzer if supported."""
        if hasattr(self._inner, "set_seed"):
            self._inner.set_seed(seed)

    def _cubeful_equity(
        self, post_move_board, probs, owner,
        cube_value=1, away1=0, away2=0, is_crawford=False, jacoby=True,
        beaver=True,
    ) -> float:
        is_match = away1 > 0 or away2 > 0
        if self._cubeful_ply == 1:
            race = bgbot_cpp.is_race(post_move_board)
            pp, op = bgbot_cpp.pip_counts(post_move_board)
            x = bgbot_cpp.cube_efficiency(probs, race, pp, op)
            if is_match:
                return bgbot_cpp.cl2cf(probs, cube_value, owner, x,
                                       away1, away2, is_crawford,
                                       jacoby=jacoby)
            else:
                jacoby_active = (
                    jacoby and owner == bgbot_cpp.CubeOwner.CENTERED
                )
                return bgbot_cpp.cl2cf_money(probs, owner, x,
                                             jacoby_active=jacoby_active)
        else:
            opp_pre_roll = bgbot_cpp.flip_board(post_move_board)
            opp_owner = _FLIP_OWNER[owner]
            db = getattr(self._inner, '_bearoff_db', None)
            n_threads = getattr(self._inner, '_parallel_threads', 1)
            if is_match:
                opp_eq = bgbot_cpp.cubeful_equity_nply(
                    opp_pre_roll, opp_owner,
                    self._inner._strategy_1ply, self._cubeful_ply,
                    n_threads=n_threads,
                    cube_value=cube_value,
                    away1=away2, away2=away1, is_crawford=is_crawford,
                    jacoby=jacoby, beaver=beaver,
                    bearoff_db=db,
                )
            else:
                opp_eq = bgbot_cpp.cubeful_equity_nply(
                    opp_pre_roll, opp_owner,
                    self._inner._strategy_1ply, self._cubeful_ply,
                    n_threads=n_threads,
                    jacoby=jacoby, beaver=beaver,
                    bearoff_db=db,
                )
            return -opp_eq

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        force_boards=None,
    ) -> list[dict]:
        owner = resolve_owner(cube_owner)
        inner = self._inner
        is_rollout = isinstance(inner, _RolloutAnalyzer)

        results = inner.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
            force_boards=force_boards,
        )
        if not results:
            return results

        if is_rollout:
            # Rollout path: survivors carry rollout-level cubeful equity from
            # cube_decision (set in _RolloutAnalyzer.checker_play_analytics).
            # Non-survivors fall back to N-ply cubeful_equity_nply at the
            # rollout's decision_ply — faster than running cube_decision per
            # candidate, but at a lower ply than the rollout itself, so the
            # cubeful values may disagree with the rollout-level cube action
            # (e.g. 3-ply says D/T where 3T says D/P).
            for m in results:
                cubeless_eq = m["equity"]
                if "rollout_cubeful_equity" in m:
                    cf_eq = m["rollout_cubeful_equity"]
                else:
                    cf_eq = self._cubeful_equity(
                        m["board"], m["probs"], owner,
                        cube_value=cube_value, away1=away1, away2=away2,
                        is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
                    )
                m["cubeless_equity"] = cubeless_eq
                m["equity"] = cf_eq

            results.sort(key=lambda x: -x["equity"])

            # Promote non-rollout entries that rank above any rollout entry,
            # so the top of the displayed list is at rollout-level cubeful and
            # the symmetry checker_play.best.cubeful == -cube_action.optimal
            # holds (required for parity with the multi-ply path that uses
            # cubeful_equity_nply uniformly across survivors and non-survivors).
            # Pathological positions where many candidates flip cube action
            # between decision_ply and rollout-level can trigger many extra
            # rollouts here — that's the cost of getting the displayed best
            # right at the rollout level.
            inner_ro = inner  # _RolloutAnalyzer
            # The trial loop's denominator (survivors x n_trials) was fixed
            # before any promotion, so it cannot cover the extra rollouts below
            # — and promotions are discovered one at a time, so no honest
            # percentage exists here. Keep reporting past that total instead:
            # ``completed > total`` is the caller's signal that the planned
            # trials are done and we are finalizing the best move. Without this
            # the caller sees a legitimate 100% and then silence for however
            # long the promotions take.
            n_trials_ro = inner_ro._rollout_config["n_trials"]
            trial_total = n_trials_ro * sum(
                1 for r in results if r.get("eval_level") == "Rollout"
            )
            promoted = 0
            while (results and "rollout_cubeful_equity" not in results[0]):
                promoted += 1
                if progress_callback:
                    progress_callback(trial_total + promoted, trial_total, results)
                m = results[0]
                try:
                    pr = inner_ro._rollout_strategy.cubeful_evaluate_board(
                        m["board"], m["board"],
                        cube_value=cube_value, owner=owner,
                        away1=away1, away2=away2, is_crawford=is_crawford,
                        jacoby=jacoby, beaver=beaver,
                    )
                except bgbot_cpp.RolloutCancelled:
                    raise RolloutCancelled()
                probs = list(pr["probs"])
                if inner_ro._bearoff_db is not None and \
                        inner_ro._bearoff_db.is_bearoff(m["board"]):
                    probs = inner_ro._bearoff_db.lookup_probs(
                        m["board"], post_move=True)
                m["probs"] = probs
                m["prob_std_errors"] = list(pr.get("prob_std_errors", [0] * 5))
                m["cubeless_equity"] = pr["equity"]
                m["std_error"] = pr.get("std_error") or 0
                m["equity"] = pr["cubeful_equity"]
                m["rollout_cubeful_equity"] = pr["cubeful_equity"]
                m["rollout_cubeful_se"] = pr["cubeful_se"]
                m["eval_level"] = "Rollout"
                m.pop("is_1ply_only", None)
                results.sort(key=lambda x: -x["equity"])

            if results:
                best = results[0]["equity"]
                for r in results:
                    r["equity_diff"] = r["equity"] - best
            return results

        # Non-rollout path: N-ply cubeful equity + promotion loop
        workers = getattr(inner, "_parallel_threads", 0)
        if not isinstance(workers, int) or workers <= 1:
            workers = max(2, os.cpu_count() or 2)
        workers = max(1, workers)

        # When the inner analyzer is multi-ply with N > 1, route each candidate
        # through cubeful_probs_and_equity_nply so the per-move probs come from
        # the CUBE-AWARE tree (opponent move selection respects cube state /
        # match equity). Otherwise the displayed probs are from MultiPlyStrategy's
        # cubeless tree, where the opponent picks gammon-greedy moves in match
        # play. The fix mirrors the post_move_analytics path.
        use_cube_aware_probs = (
            isinstance(inner, _MultiPlyAnalyzer)
            and self._cubeful_ply > 1
            and owner is not None
        )

        if use_cube_aware_probs:
            total_threads = getattr(inner, "_parallel_threads", 1) or 1
            db = getattr(inner, "_bearoff_db", None)
            opp_owner = _FLIP_OWNER[owner]
            strat_1ply = inner._strategy_1ply
            cubeful_ply = self._cubeful_ply

            # Parallelize ACROSS candidates with n_threads=1 inside each call.
            # The cubeful tree at 3-ply on a single candidate is small (~10ms
            # serial), so outer parallelism overlaps far more effectively than
            # inner parallelism would. Avoid oversubscription by giving each
            # candidate a single thread.
            def _convert_move(m: dict) -> tuple[list, float, float]:
                opp_pre_roll = bgbot_cpp.flip_board(m["board"])
                r = bgbot_cpp.cubeful_probs_and_equity_nply(
                    opp_pre_roll, opp_owner, strat_1ply, cubeful_ply,
                    n_threads=1,
                    cube_value=cube_value,
                    away1=away2, away2=away1, is_crawford=is_crawford,
                    jacoby=jacoby, beaver=beaver,
                    bearoff_db=db,
                )
                opp_probs = r["probs"]
                # invert opp's POV probs → player's POV
                probs = [1.0 - opp_probs[0],
                         opp_probs[3], opp_probs[4],
                         opp_probs[1], opp_probs[2]]
                cl_eq = (2.0 * probs[0] - 1.0
                         + probs[1] - probs[3]
                         + probs[2] - probs[4])
                # cubeful_probs_and_equity_nply returns the opponent's equity at
                # the flipped board; negate to get the current player's equity.
                cf_eq = -r["equity"]
                return probs, cl_eq, cf_eq

            with ThreadPoolExecutor(max_workers=total_threads) as pool:
                converted = list(pool.map(_convert_move, results))

            for m, (probs, cl_eq, cf_eq) in zip(results, converted):
                m["probs"] = probs
                m["cubeless_equity"] = cl_eq
                m["equity"] = cf_eq
        else:
            def _convert_move_legacy(m: dict) -> tuple[float, float]:
                cubeless_eq = m["equity"]
                cf_eq = self._cubeful_equity(
                    m["board"], m["probs"], owner,
                    cube_value=cube_value, away1=away1, away2=away2,
                    is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
                )
                return cubeless_eq, cf_eq

            with ThreadPoolExecutor(max_workers=workers) as pool:
                converted = list(pool.map(_convert_move_legacy, results))

            for m, (cubeless_eq, cf_eq) in zip(results, converted):
                m["cubeless_equity"] = m["equity"]
                m["equity"] = cf_eq

        results.sort(key=lambda x: -x["equity"])

        def _nply_eval(b, board_ref):
            if isinstance(inner, _MultiPlyAnalyzer):
                r = inner._strategy_nply.evaluate_board(b, board_ref)
                eval_level = f"{inner._n_plies}-ply"
                extra = {}
            else:
                return None
            probs = list(r["probs"])
            cf_eq = self._cubeful_equity(
                b, probs, owner,
                cube_value=cube_value, away1=away1, away2=away2,
                is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
            )
            extra["cubeless_equity"] = r["equity"]
            return cf_eq, probs, eval_level, extra

        while len(results) >= 2 and results[1].get("is_1ply_only"):
            r = results[1]
            ret = _nply_eval(r["board"], board)
            if ret is None:
                break
            cf_eq, probs, eval_level, extra = ret
            r["equity"] = cf_eq
            r["probs"] = probs
            r["eval_level"] = eval_level
            r.pop("is_1ply_only", None)
            r.update(extra)
            results.sort(key=lambda x: -x["equity"])

        if results:
            best = results[0]["equity"]
            for r in results:
                r["equity_diff"] = r["equity"] - best

        return results

    def cube_action_analytics(
        self, board, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        return self._inner.cube_action_analytics(
            board, cube_value, cube_owner,
            progress_callback=progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver,
            incl_2ply_details=incl_2ply_details,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dict_to_move_analysis(d: dict, include_game_plans: bool = False) -> MoveAnalysis:
    """Convert an internal result dict to a :class:`MoveAnalysis`."""
    player_gp = None
    opponent_gp = None
    if include_game_plans:
        b = d["board"]
        gp = bgbot_cpp.classify_game_plan(b)
        player_gp = gp.name.lower() if hasattr(gp, "name") else str(gp).lower()
        flipped = bgbot_cpp.flip_board(b)
        gp2 = bgbot_cpp.classify_game_plan(flipped)
        opponent_gp = gp2.name.lower() if hasattr(gp2, "name") else str(gp2).lower()

    return MoveAnalysis(
        board=d["board"],
        equity=d["equity"],
        cubeless_equity=d.get("cubeless_equity", d["equity"]),
        probs=Probabilities.from_list(d["probs"]),
        equity_diff=d.get("equity_diff", 0.0),
        eval_level=d["eval_level"],
        player_game_plan=player_gp,
        opponent_game_plan=opponent_gp,
        std_error=d.get("std_error"),
        prob_std_errors=d.get("prob_std_errors"),
    )


def _optimal_action(should_double: bool, should_take: bool,
                    is_beaver: bool = False) -> str:
    if not should_double:
        return "No Double"
    if is_beaver:
        return "Double/Beaver"
    if should_take:
        return "Double/Take"
    return "Double/Pass"


class BgBotAnalyzer:
    """High-level interface to the Open Sage bot engine.

    Thread-safe: multiple threads can call methods concurrently.

    Args:
        weights: Weight file configuration (defaults to production model).
        eval_level: ``'1ply'``, ``'2ply'``, ``'3ply'``, ``'4ply'``,
            ``'truncated1'``, ``'truncated2'``, ``'truncated3'``, or ``'rollout'``.
        cubeful: If True, compute cubeful equities via Janowski.
        filter_max_moves: Candidate-selection filter width for
            ``eval_level='rollout'`` checker play — how many of the top
            scored moves are rolled out (default 5; the rest keep their
            filter-stage equity). Ignored by other eval levels.
        filter_threshold: Equity window for the same filter — candidates
            more than this far below the best are dropped even within the
            ``filter_max_moves`` cap (default 0.08).
        parallel_threads: Thread count (0 = auto-detect).
        n_trials: Rollout trial count.
        truncation_depth: Rollout truncation (0 = play to completion).
        decision_ply: Ply depth for move selection during rollout trials.
        truncation_ply: Ply depth for evaluation at the truncation point
            (-1 = same as ``decision_ply``). Only used by ``eval_level='rollout'``;
            the named ``truncated1/2/3`` levels fix their own truncation ply.
        late_ply: Ply for move selection after ``late_threshold`` half-moves
            (-1 = same as ``decision_ply``).
        late_threshold: Half-move index where decision ply switches to ``late_ply``.
        seed: RNG seed for rollout.
        checker: TrialEvalConfig for checker play during rollout trials.
        checker_late: TrialEvalConfig for late-game checker play.
        cube: TrialEvalConfig for cube decisions during rollout trials.
        cube_late: TrialEvalConfig for late-game cube decisions.
        prefilter_threshold: Two-stage checker_play filter for rollout levels.
            When > 0, stage 1 culls candidates with > prefilter_threshold
            1-ply equity error from best (no count cap), then stage 2 applies
            the TINY filter at 2-ply on the survivors before rollout. Defaults
            to 0.15 for truncated2, truncated3, and any non-truncated rollout
            (eval_level="rollout" with truncation_depth=0); 0.0 (legacy
            single-stage 1-ply TINY) for truncated1 and user-configured
            truncated rollouts. Pass an explicit value to override.
    """

    def __init__(
        self,
        weights: WeightConfig | WeightConfigPair | None = None,
        eval_level: str = "1ply",
        cubeful: bool = True,
        *,
        filter_max_moves: int = 5,
        filter_threshold: float = 0.08,
        parallel_threads: int = 0,
        n_trials: int = 1296,
        truncation_depth: int = 0,
        decision_ply: int = 1,
        truncation_ply: int = -1,
        late_ply: int = -1,
        late_threshold: int = 20,
        seed: int = 42,
        bearoff_db: bool | str = True,
        checker=None,
        checker_late=None,
        cube=None,
        cube_late=None,
        ultra_late_threshold: int = 9999,
        cubeful_trial_moves: bool = True,
        cubeful_late_threshold: int = 0,
        prefilter_threshold: float | None = None,
        target_se: float = 0.0,
        max_batches: int = 50,
    ):
        if weights is None:
            weights = default_weights()
        self._weights = weights
        self._eval_level = eval_level
        # Two-stage checker_play filter: 1-ply loose cull (prefilter_threshold)
        # then 2-ply TINY. Defaults to 0.15 for truncated2, truncated3, and
        # any non-truncated rollout (eval_level="rollout" with
        # truncation_depth=0); 0.0 (disabled, legacy single-stage 1-ply
        # TINY) elsewhere. Rationale: the rollout is expensive enough at all
        # those levels that pre-rollout filtering is worth the small 2-ply
        # cost; 1T and user-configured truncated rollouts are cheap enough
        # that the extra 2-ply scoring isn't worth it. User-supplied value
        # overrides the per-level default.
        if prefilter_threshold is None:
            is_full_rollout = (
                eval_level == "rollout" and truncation_depth == 0
            )
            if is_full_rollout:
                # Keep the loose stage-1 cull comfortably wider than the
                # final candidate filter so a widened filter_threshold
                # (wide/gigantic search interval) is never strangled by the
                # prefilter. The default filter (0.08) yields the historical
                # 0.15.
                prefilter_threshold = max(0.15, filter_threshold + 0.05)
            elif eval_level in ("truncated2", "truncated3"):
                prefilter_threshold = 0.15
            else:
                prefilter_threshold = 0.0
        self._prefilter_threshold = float(prefilter_threshold)
        # Flag-gated cube-aware trial moves (see ROLLOUT.md). Default
        # False keeps existing rollout behavior. When True, all rollout-based
        # eval levels select trial moves by cubeful equity against the trial's
        # cube state instead of cubeless equity.
        self._cubeful_trial_moves = bool(cubeful_trial_moves)
        # Per-move drop-to-cubeless threshold for full-game rollouts where
        # ultra_late_threshold=9999 keeps cubeful active for ~50 half-moves.
        # 0 = inherit from ultra_late_threshold (no separate drop).
        self._cubeful_late_threshold = int(cubeful_late_threshold)
        # Load bearoff database
        self._bearoff_db = None
        if bearoff_db:
            db_path = bearoff_db if isinstance(bearoff_db, str) else bearoff_db_path()
            if db_path:
                self._bearoff_db = bgbot_cpp.BearoffDB()
                if not self._bearoff_db.load(db_path):
                    self._bearoff_db = None

        if eval_level == "1ply":
            inner: _CubelessBase = _OnePlyAnalyzer(weights)
        elif eval_level in ("2ply", "3ply", "4ply"):
            n_plies = int(eval_level[0])
            inner = _MultiPlyAnalyzer(
                weights, n_plies=n_plies,
                parallel_threads=parallel_threads,
            )
        elif eval_level == "truncated1":
            # 1T: 72 trials (2x36) -- a multiple of 36, so the first roll is
            # exactly stratified. XG Roller's 42 is not, which 2x over-weights
            # 6 ordered first rolls and biases the result (benchmark PR 2.23 ->
            # 0.50 going 42 -> 72). Otherwise XG-Roller-style: trunc-5, 1-ply.
            inner = _RolloutAnalyzer(
                weights,
                n_trials=72,
                truncation_depth=5,
                decision_ply=1,
                n_threads=parallel_threads,
                seed=seed,
                ultra_late_threshold=2,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
            )
        elif eval_level == "truncated2":
            # 2T: trunc-7, 360 trials. Checker 2-ply on the first ply then
            # 1-ply (late_threshold=1); cube 2-ply throughout; 2-ply truncation
            # eval; no ultra-late 1-ply drop (ultra_late=9999). Beats XG Roller+
            # (benchmark PR 0.89 -> 0.36). Was: late_threshold=2, ultra_late=2,
            # 1-ply late cube + 1-ply truncation eval (the ply-drop economies
            # that cost ~0.5 PR).
            inner = _RolloutAnalyzer(
                weights,
                n_trials=360,
                truncation_depth=7,
                decision_ply=2,
                truncation_ply=2,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=1,
                late_threshold=1,
                cube=bgbot_cpp.TrialEvalConfig(ply=2),
                cube_late=bgbot_cpp.TrialEvalConfig(ply=2),
                ultra_late_threshold=9999,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
            )
        elif eval_level == "truncated3":
            # 3T: trunc-7, 3-ply early then 2-ply late (ultra_late=9999 disables
            # the 1-ply drop). Closer to XG Roller++. (Was trunc-5 / 1-ply late.)
            inner = _RolloutAnalyzer(
                weights,
                n_trials=360,
                truncation_depth=7,
                decision_ply=3,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=2,
                late_threshold=2,
                ultra_late_threshold=9999,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
            )
        elif eval_level == "rollout":
            inner = _RolloutAnalyzer(
                weights,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
                truncation_ply=truncation_ply,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=late_ply,
                late_threshold=late_threshold,
                checker=checker,
                checker_late=checker_late,
                cube=cube,
                cube_late=cube_late,
                ultra_late_threshold=ultra_late_threshold,
                cubeful_trial_moves=self._cubeful_trial_moves,
                cubeful_late_threshold=self._cubeful_late_threshold,
                prefilter_threshold=self._prefilter_threshold,
                target_se=target_se,
                max_batches=max_batches,
                filter_max_moves=filter_max_moves,
                filter_threshold=filter_threshold,
            )
        else:
            raise ValueError(f"Unknown eval_level: {eval_level!r}")

        # Set bearoff DB on inner analyzer and its C++ strategies
        if self._bearoff_db is not None:
            inner._bearoff_db = self._bearoff_db
            # Wrap the 1-ply strategy in BearoffStrategy so 1-ply scoring used
            # for filter ranking returns exact DB probs at bearoff positions.
            # Without this, the 1-ply NN can mis-rank candidates whose post-move
            # boards are bearoff (e.g. a position the NN says has P(GL)=0.01 but
            # the DB knows is exactly 0.139), causing the filter to drop near-
            # equivalent moves before N-ply evaluation.
            inner._strategy_1ply = bgbot_cpp.BearoffStrategy(
                inner._strategy_1ply, self._bearoff_db)
            if isinstance(inner, _MultiPlyAnalyzer):
                bgbot_cpp.multipy_set_bearoff_db(inner._strategy_nply, self._bearoff_db)
            elif isinstance(inner, _RolloutAnalyzer):
                bgbot_cpp.rollout_set_bearoff_db(inner._rollout_strategy, self._bearoff_db)
                if inner._strategy_2ply is not None:
                    bgbot_cpp.multipy_set_bearoff_db(inner._strategy_2ply, self._bearoff_db)

        if cubeful:
            self._analyzer = _CubefulAnalyzer(inner)
        else:
            self._analyzer = inner

    def cancel(self):
        """Request cancellation of an in-progress rollout.

        Thread-safe. Only effective when the analyzer uses rollout evaluation.
        After calling cancel(), the next or in-progress checker_play() or
        cube_action() call will raise :class:`RolloutCancelled`.
        Call :meth:`reset_cancel` before reusing the analyzer.
        """
        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner
        if isinstance(inner, _RolloutAnalyzer):
            inner.cancel()

    def reset_cancel(self):
        """Clear cancellation flag so the analyzer can be reused."""
        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner
        if isinstance(inner, _RolloutAnalyzer):
            inner.reset_cancel()

    def epc(self, board: list[int], player: int = 0) -> float | None:
        """Return Effective Pip Count for a player in a bearoff position.

        EPC = mean_rolls × (49/6), where mean_rolls is the expected number
        of rolls to bear off all checkers (including the upcoming roll).

        Args:
            board: 26-element board array.
            player: 0 = player on roll, 1 = opponent.

        Returns:
            EPC as a float, or None if the position is not in the bearoff DB.
        """
        if self._bearoff_db is None or not self._bearoff_db.is_bearoff(board):
            return None
        return self._bearoff_db.lookup_epc(board, player)

    def set_seed(self, seed: int) -> None:
        """Reseed the underlying rollout RNG, if this analyzer is rollout-based.

        Used to run independent seeded batches of the same position while
        keeping the rollout SharedPosCache warm (e.g. lockstep target-SE checker
        rollouts). No-op for non-rollout eval levels.
        """
        inner = getattr(self, "_analyzer", None)
        if inner is not None and hasattr(inner, "set_seed"):
            inner.set_seed(seed)

    def checker_play(
        self,
        board: list[int],
        die1: int,
        die2: int,
        cube_value: int = 1,
        cube_owner: str = "centered",
        include_game_plans: bool = False,
        progress_callback: Any | None = None,
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        force_boards: list[list[int]] | None = None,
    ) -> CheckerPlayResult:
        """Analyze all legal moves for a checker play decision.

        Returns moves sorted by equity (best first). Each move includes
        post-move probabilities and equity difference from the best move.

        Args:
            board: 26-element board array.
            die1, die2: Dice values (1-6).
            cube_value: Current cube value.
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            include_game_plans: If True, populate ``player_game_plan`` and
                ``opponent_game_plan`` on each :class:`MoveAnalysis`.
            progress_callback: Optional ``callback(completed, total, partial)``
                for rollout progress.
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            beaver: If True, opponent can beaver after being doubled
                (money games only). Auto-disabled for match play.
            force_boards: Optional list of post-move boards (mover's
                perspective) to always evaluate at the full level, bypassing
                the move filter. Boards that aren't legal candidates are
                ignored. Used by the app's expert re-evaluation so the move a
                player actually made is always scored at the expert level.
        """
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
            beaver=beaver, force_boards=force_boards,
        )
        moves = [_dict_to_move_analysis(d, include_game_plans) for d in raw]
        eval_level = moves[0].eval_level if moves else self._eval_level
        return CheckerPlayResult(
            moves=moves, board=board, die1=die1, die2=die2,
            eval_level=eval_level,
        )

    def post_move_analytics(
        self,
        board: list[int],
        cube_owner: str = "centered",
        cube_value: int = 1,
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        progress_callback: Any | None = None,
    ) -> PostMoveAnalysis:
        """Evaluate a post-move position (right before the opponent's turn).

        Returns cubeless probabilities, cubeless equity, and cubeful equity
        from the perspective of the player who just moved.

        Args:
            board: 26-element post-move board array (player who moved's perspective).
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            cube_value: Current cube value.
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            progress_callback: Optional ``callback(completed, total)`` called
                from inside the rollout's worker threads.  Only meaningful for
                rollout eval levels — N-ply and 1-ply paths return synchronously
                and don't emit progress.
        """
        if away1 > 0 or away2 > 0:
            jacoby = False

        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner

        # Evaluate the post-move board (NN outputs from mover's perspective).
        # When the inner analyzer is a rollout AND cube-aware trial moves is
        # enabled, route through cubeful_evaluate_board so the trial loop
        # receives the cube state and selects moves by cubeful equity. The
        # cubeless probs and cubeful equity both come from the same trials.
        # Otherwise keep the cubeless-rollout + post-hoc Janowski path so
        # existing behavior is byte-identical when the flag is off.
        owner = resolve_owner(cube_owner)
        use_cubeful_rollout = (
            isinstance(inner, _RolloutAnalyzer)
            and self._cubeful_trial_moves
        )

        if isinstance(inner, _MultiPlyAnalyzer):
            n_plies = inner._n_plies
            inner._strategy_nply.clear_cache()
            # Route through cubeful_probs_nply (N-ply CUBE-AWARE tree) when
            # cube info is meaningful so the returned probs reflect match-aware
            # interior picks. Falls back to cubeless evaluate_board when no
            # cube info is present (no-op for 1-ply where there's nothing to
            # change).
            if owner is not None and n_plies > 1:
                # Mirror the flip/perspective dance in _cubeful_equity: flip
                # the board to opp's POV, flip the cube owner, swap away
                # scores; cubeful_probs_nply returns probs in opp's POV at the
                # flipped pre-roll position; invert to get back to ours.
                opp_pre_roll = bgbot_cpp.flip_board(board)
                opp_owner = _FLIP_OWNER[owner]
                opp_probs = bgbot_cpp.cubeful_probs_nply(
                    opp_pre_roll, opp_owner,
                    inner._strategy_1ply, n_plies,
                    n_threads=getattr(inner, '_parallel_threads', 1),
                    cube_value=cube_value,
                    away1=away2, away2=away1, is_crawford=is_crawford,
                    jacoby=jacoby,
                    bearoff_db=getattr(inner, '_bearoff_db', None),
                )
                # invert_probs from bgsage.board: [W, gW, bW, gL, bL] in opp's
                # POV → [1-W, gL, bL, gW, bW] in our POV
                probs = [1.0 - opp_probs[0],
                         opp_probs[3], opp_probs[4],
                         opp_probs[1], opp_probs[2]]
                cl_eq = (2.0 * probs[0] - 1.0
                         + probs[1] - probs[3]
                         + probs[2] - probs[4])
                r = {"probs": probs, "equity": cl_eq}
            else:
                r = inner._strategy_nply.evaluate_board(board, board)
            eval_level = f"{n_plies}-ply"
        elif isinstance(inner, _RolloutAnalyzer):
            if use_cubeful_rollout:
                r = inner._rollout_strategy.cubeful_evaluate_board(
                    board, board,
                    cube_value=cube_value, owner=owner,
                    away1=away1, away2=away2, is_crawford=is_crawford,
                    jacoby=jacoby,
                    progress=progress_callback,
                )
            else:
                r = inner._rollout_strategy.evaluate_board(
                    board, board,
                    progress=progress_callback,
                )
            eval_level = "Rollout"
        else:
            r = inner._strategy_1ply.evaluate_board(board, board)
            eval_level = "1-ply"

        probs_list = list(r["probs"])
        cl_eq = r["equity"]

        if use_cubeful_rollout:
            # cubeful_evaluate_board returns cubeful_equity in basis-cube
            # units already (computed from the rollout's own trial paths,
            # including the opponent's optimal cube action at the start of
            # their turn). No post-hoc Janowski needed.
            cf_eq = float(r["cubeful_equity"])
        else:
            # Cubeful equity via post-hoc Janowski on cubeless probs.
            race = bgbot_cpp.is_race(board)
            pp, op = bgbot_cpp.pip_counts(board)
            x = bgbot_cpp.cube_efficiency(probs_list, race, pp, op)
            is_match = away1 > 0 or away2 > 0
            if is_match:
                cf_eq = bgbot_cpp.cl2cf(probs_list, cube_value, owner, x,
                                        away1, away2, is_crawford,
                                        jacoby=jacoby)
            else:
                jacoby_active = (
                    jacoby and owner == bgbot_cpp.CubeOwner.CENTERED
                )
                cf_eq = bgbot_cpp.cl2cf_money(probs_list, owner, x,
                                              jacoby_active=jacoby_active)

        return PostMoveAnalysis(
            probs=Probabilities.from_list(probs_list),
            cubeless_equity=cl_eq,
            cubeful_equity=cf_eq,
            eval_level=eval_level,
            cubeless_se=(float(r["std_error"])
                         if isinstance(r, dict) and r.get("std_error") is not None
                         else None),
            cubeful_se=(float(r["cubeful_se"])
                        if isinstance(r, dict) and r.get("cubeful_se") is not None
                        else None),
        )

    def cube_action(
        self,
        board: list[int],
        cube_value: int = 1,
        cube_owner: str = "centered",
        *,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        incl_2ply_details: bool = False,
        progress_callback=None,
    ) -> CubeActionResult:
        """Analyze the cube decision for a pre-roll position.

        Returns cubeful equities for No Double, Double/Take, Double/Pass,
        with the optimal action and pre-roll cubeless probabilities.

        Args:
            board: 26-element board array.
            cube_value: Current cube value.
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
            away1: Points player needs to win (0 = money game).
            away2: Points opponent needs to win (0 = money game).
            is_crawford: True if this is the Crawford game.
            jacoby: If True, gammons/backgammons don't count when cube is
                centered (money games only). Auto-disabled for match play.
            beaver: If True, opponent can beaver (redouble while retaining
                ownership) after being doubled. Money games only.
                Auto-disabled for match play.
            incl_2ply_details: If True, include per-roll details for the
                first two turns under both ND and DT scenarios. Requires
                2-ply or higher evaluation. At 2-ply only the player-roll
                level is captured (per-roll equities at 1-ply, no
                opponent_rolls); the headline equities match the plain
                2-ply call.
        """
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.cube_action_analytics(
            board, cube_value, cube_owner,
            progress_callback=progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver,
            incl_2ply_details=incl_2ply_details,
        )
        probs = Probabilities.from_list(raw["probs"])
        is_beaver = raw.get("is_beaver", False)
        return CubeActionResult(
            probs=probs,
            cubeless_equity=raw["cubeless_equity"],
            equity_nd=raw["equity_nd"],
            equity_dt=raw["equity_dt"],
            equity_dp=raw["equity_dp"],
            should_double=raw["should_double"],
            should_take=raw["should_take"],
            optimal_equity=raw["optimal_equity"],
            optimal_action=_optimal_action(
                raw["should_double"], raw["should_take"], is_beaver),
            eval_level=raw["eval_level"],
            is_beaver=is_beaver,
            cubeless_se=raw.get("cubeless_se"),
            equity_nd_se=raw.get("equity_nd_se"),
            equity_dt_se=raw.get("equity_dt_se"),
            details=raw.get("details"),
        )

    def roll_luck(
        self,
        board: list[int],
        die1: int,
        die2: int,
        *,
        cube_value: int = 1,
        cube_owner: str = "centered",
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
        beaver: bool = True,
        is_opening_roll: bool = False,
    ) -> LuckResult | None:
        """Compute how lucky ``(die1, die2)`` was from ``board``.

        Convenience wrapper: runs the cube analysis this analyzer's eval level
        implies (with per-roll details) and computes luck from it. Callers that
        already hold a ``cube_action(incl_2ply_details=True)`` result should call
        :func:`bgsage.roll_luck` on it directly to avoid a second evaluation.

        ``is_opening_roll=True`` excludes doubles (an opening roll has 15 rolls,
        not 21). Returns ``None`` on a degenerate position. See
        :func:`bgsage.luck.roll_luck` for the meaning of the result.
        """
        cube = self.cube_action(
            board, cube_value, cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
            jacoby=jacoby, beaver=beaver, incl_2ply_details=True,
        )
        return _roll_luck(cube, die1, die2, is_opening_roll=is_opening_roll)


def create_analyzer(
    level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    cubeful: bool = True,
    **kwargs: Any,
) -> BgBotAnalyzer:
    """Convenience factory for creating analyzers."""
    return BgBotAnalyzer(weights=weights, eval_level=level, cubeful=cubeful, **kwargs)
