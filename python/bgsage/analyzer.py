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

from .types import (
    CheckerPlayResult,
    CubeActionResult,
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
        self._is_pair = isinstance(weights, WeightConfigPair)
        if self._is_pair:
            paths, hiddens = weights.weight_args
            self._strategy_1ply = bgbot_cpp.GamePlanPairStrategy(paths, hiddens)
        else:
            self._strategy_1ply = bgbot_cpp.GamePlanStrategy(*weights.weight_args)
        self._bearoff_db = None  # Set by BgBotAnalyzer after construction

    def _score_candidates_1ply(
        self,
        candidates: list,
        board: list[int],
        cube_owner: str | None = None,
        cube_value: int = 1,
        away1: int = 0,
        away2: int = 0,
        is_crawford: bool = False,
        jacoby: bool = True,
    ) -> list[tuple[float, float, list[int], list[float]]]:
        owner = resolve_owner(cube_owner) if cube_owner else None
        is_match = away1 > 0 or away2 > 0
        scored = []
        for b in candidates:
            bl = list(b)
            r = self._strategy_1ply.evaluate_board(bl, board)
            cl_eq = r["equity"]
            probs = list(r["probs"])
            if owner is not None:
                race = bgbot_cpp.is_race(bl)
                x = bgbot_cpp.cube_efficiency(bl, race)
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
            "eval_level": eval_level,
        }


class _OnePlyAnalyzer(_CubelessBase):

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True,
    ) -> list[dict]:
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
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        if incl_2ply_details:
            raise ValueError("incl_2ply_details requires at least 3-ply evaluation")
        owner = resolve_owner(cube_owner)
        if self._is_pair:
            paths, hiddens = self._weights.weight_args
            r = bgbot_cpp.evaluate_cube_decision_pair(
                board, cube_value, owner, paths, hiddens,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver,
                bearoff_db=self._bearoff_db,
            )
        else:
            r = bgbot_cpp.evaluate_cube_decision(
                board, cube_value, owner, *self._weights.weight_args,
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
        if self._is_pair:
            paths, hiddens = weights.weight_args
            self._strategy_nply = bgbot_cpp.create_multipy_pair(
                paths, hiddens,
                n_plies=n_plies,
                parallel_evaluate=parallel_evaluate,
                parallel_threads=self._parallel_threads,
            )
        else:
            self._strategy_nply = bgbot_cpp.create_multipy_5nn(
                *weights.weight_args,
                n_plies=n_plies,
                parallel_evaluate=parallel_evaluate,
                parallel_threads=self._parallel_threads,
            )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates_1ply(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby,
        )
        survivors, survivor_set = self._filter_candidates(
            scored_1ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )

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
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        owner = resolve_owner(cube_owner)
        if self._is_pair:
            paths, hiddens = self._weights.weight_args
            r = bgbot_cpp.cube_decision_nply_pair(
                board, cube_value, owner, self._n_plies, paths, hiddens,
                n_threads=self._parallel_threads,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver,
                bearoff_db=self._bearoff_db,
                incl_2ply_details=incl_2ply_details,
            )
        else:
            r = bgbot_cpp.cube_decision_nply(
                board, cube_value, owner, self._n_plies, *self._weights.weight_args,
                n_threads=self._parallel_threads,
                away1=away1, away2=away2, is_crawford=is_crawford,
                jacoby=jacoby, beaver=beaver,
                bearoff_db=self._bearoff_db,
                incl_2ply_details=incl_2ply_details,
            )
        result = self._format_cube_result(r, eval_level=f"{self._n_plies}-ply")

        # Pass through player_rolls if present
        if "player_rolls" in r:
            result["player_rolls"] = r["player_rolls"]

        # The C++ binding already computes N-ply cubeless probs (with bearoff DB
        # when applicable). Only override if the binding didn't use bearoff DB,
        # to get N-ply probs instead of 1-ply probs for the cubeless display.
        if self._bearoff_db is None or not self._bearoff_db.is_bearoff(board):
            flipped = bgbot_cpp.flip_board(board)
            nply_eval = self._strategy_nply.evaluate_board(flipped, board)
            nply_probs = list(nply_eval["probs"])
            nply_pre_roll = [
                1.0 - nply_probs[0],
                nply_probs[3],
                nply_probs[4],
                nply_probs[1],
                nply_probs[2],
            ]
            result["probs"] = nply_pre_roll
            result["cubeless_equity"] = (
                2.0 * nply_pre_roll[0] - 1.0
                + nply_pre_roll[1] - nply_pre_roll[3]
                + nply_pre_roll[2] - nply_pre_roll[4]
            )
            self._strategy_nply.clear_cache()

        return result


class _RolloutAnalyzer(_CubelessBase):

    FILTER_MAX_MOVES = 5
    FILTER_THRESHOLD = 0.08

    def __init__(
        self, weights, n_trials=1296, truncation_depth=0,
        decision_ply=1, n_threads=0, seed=42,
        late_ply=-1, late_threshold=20,
        parallelize_trials=True,
        checker=None, checker_late=None,
        cube=None, cube_late=None,
        ultra_late_threshold=2,
    ):
        super().__init__(weights)
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

        if self._is_pair:
            paths, hiddens = weights.weight_args
            self._rollout_strategy = bgbot_cpp.create_rollout_pair(
                paths, hiddens,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
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
            )
            self._strategy_3ply = bgbot_cpp.create_multipy_pair(
                paths, hiddens, n_plies=3,
            )
        else:
            self._rollout_strategy = bgbot_cpp.create_rollout_5nn(
                *weights.weight_args,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
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
            )
            # Create a 3-ply strategy for pre-filtering (accurate move count)
            self._strategy_3ply = bgbot_cpp.create_multipy_5nn(
                *weights.weight_args, n_plies=3,
            )

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
        away1=0, away2=0, is_crawford=False, jacoby=True,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_1ply = self._score_candidates_1ply(
            candidates, board, cube_owner,
            cube_value=cube_value, away1=away1, away2=away2,
            is_crawford=is_crawford, jacoby=jacoby,
        )
        survivors_1ply, survivor_1ply_set = self._filter_candidates(
            scored_1ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )
        # Ensure at least 2 candidates go into the 3-ply rescore so the
        # cubeful promotion loop won't need surprise extra rollouts.
        if len(survivors_1ply) < 2 and len(scored_1ply) >= 2:
            for item in scored_1ply:
                if tuple(item[2]) not in survivor_1ply_set:
                    survivors_1ply.append(item)
                    survivor_1ply_set.add(tuple(item[2]))
                    if len(survivors_1ply) >= 2:
                        break

        # 3-ply rescore: re-evaluate 1-ply survivors at 3-ply, then re-filter
        # to get accurate move count before starting the rollout.
        self._check_cancel()
        scored_3ply = []
        for feq, cleq, b, p in survivors_1ply:
            r = self._strategy_3ply.evaluate_board(b, board)
            scored_3ply.append((r["equity"], cleq, b, list(r["probs"])))
        scored_3ply.sort(key=lambda x: -x[0])
        survivors, survivor_set = self._filter_candidates(
            scored_3ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )
        # Maintain the min-2 guarantee after 3-ply filtering too.
        if len(survivors) < 2 and len(scored_3ply) >= 2:
            for item in scored_3ply:
                if tuple(item[2]) not in survivor_set:
                    survivors.append(item)
                    survivor_set.add(tuple(item[2]))
                    if len(survivors) >= 2:
                        break

        n_trials = self._rollout_config["n_trials"]
        results = []
        total_moves = len(survivors)
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
                r = self._rollout_strategy.evaluate_board(b, board, _trial_progress)
            except bgbot_cpp.RolloutCancelled:
                raise RolloutCancelled()
            results.append({
                "board": b,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "std_error": r.get("std_error", 0),
                "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
                "eval_level": "Rollout",
            })

        # Non-rolled-out moves: use 3-ply results where available, 1-ply otherwise.
        scored_3ply_map = {tuple(b): (eq, p) for eq, _, b, p in scored_3ply}
        for feq, cleq, b, p in scored_1ply:
            bkey = tuple(b)
            if bkey not in survivor_set:
                eq3, p3 = scored_3ply_map.get(bkey, (None, None))
                if eq3 is not None:
                    results.append({
                        "board": b,
                        "equity": eq3,
                        "probs": p3,
                        "eval_level": "3-ply",
                    })
                else:
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
        return self._format_cube_result(r, eval_level="Rollout")


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

    def _cubeful_equity(
        self, post_move_board, probs, owner,
        cube_value=1, away1=0, away2=0, is_crawford=False, jacoby=True,
        beaver=True,
    ) -> float:
        is_match = away1 > 0 or away2 > 0
        if self._cubeful_ply == 1:
            race = bgbot_cpp.is_race(post_move_board)
            x = bgbot_cpp.cube_efficiency(post_move_board, race)
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
            if is_match:
                opp_eq = bgbot_cpp.cubeful_equity_nply(
                    opp_pre_roll, opp_owner,
                    self._inner._strategy_1ply, self._cubeful_ply,
                    cube_value=cube_value,
                    away1=away2, away2=away1, is_crawford=is_crawford,
                    jacoby=jacoby, beaver=beaver,
                    bearoff_db=db,
                )
            else:
                opp_eq = bgbot_cpp.cubeful_equity_nply(
                    opp_pre_roll, opp_owner,
                    self._inner._strategy_1ply, self._cubeful_ply,
                    jacoby=jacoby, beaver=beaver,
                    bearoff_db=db,
                )
            return -opp_eq

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
    ) -> list[dict]:
        owner = resolve_owner(cube_owner)
        inner = self._inner
        is_rollout = isinstance(inner, _RolloutAnalyzer)

        results = inner.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
        )
        if not results:
            return results

        if is_rollout:
            # Rollout path: all heavy computation is done. Just apply 1-ply
            # Janowski cubeful equity to each move's probs (instant) and sort.
            for m in results:
                cubeless_eq = m["equity"]
                cf_eq = self._cubeful_equity(
                    m["board"], m["probs"], owner,
                    cube_value=cube_value, away1=away1, away2=away2,
                    is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
                )
                m["cubeless_equity"] = cubeless_eq
                m["equity"] = cf_eq

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

        def _convert_move(m: dict) -> tuple[float, float]:
            cubeless_eq = m["equity"]
            cf_eq = self._cubeful_equity(
                m["board"], m["probs"], owner,
                cube_value=cube_value, away1=away1, away2=away2,
                is_crawford=is_crawford, jacoby=jacoby, beaver=beaver,
            )
            return cubeless_eq, cf_eq

        with ThreadPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert_move, results))

        for m, (cubeless_eq, cf_eq) in zip(results, converted):
            cubeless_eq = m["equity"]
            m["cubeless_equity"] = cubeless_eq
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
        away1=0, away2=0, is_crawford=False, jacoby=True, beaver=True,
        incl_2ply_details=False,
    ) -> dict:
        return self._inner.cube_action_analytics(
            board, cube_value, cube_owner,
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
        filter_max_moves: Max moves to carry through N-ply/rollout.
        filter_threshold: Equity threshold for move filter.
        parallel_threads: Thread count (0 = auto-detect).
        n_trials: Rollout trial count.
        truncation_depth: Rollout truncation (0 = play to completion).
        decision_ply: Ply depth for move selection during rollout trials.
        late_ply: Ply for move selection after ``late_threshold`` half-moves
            (-1 = same as ``decision_ply``).
        late_threshold: Half-move index where decision ply switches to ``late_ply``.
        seed: RNG seed for rollout.
        checker: TrialEvalConfig for checker play during rollout trials.
        checker_late: TrialEvalConfig for late-game checker play.
        cube: TrialEvalConfig for cube decisions during rollout trials.
        cube_late: TrialEvalConfig for late-game cube decisions.
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
        late_ply: int = -1,
        late_threshold: int = 20,
        seed: int = 42,
        bearoff_db: bool | str = True,
        checker=None,
        checker_late=None,
        cube=None,
        cube_late=None,
        ultra_late_threshold: int = 2,
    ):
        if weights is None:
            weights = default_weights()
        self._weights = weights
        self._eval_level = eval_level

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
            inner = _RolloutAnalyzer(
                weights,
                n_trials=42,
                truncation_depth=5,
                decision_ply=1,
                n_threads=parallel_threads,
                seed=seed,
            )
        elif eval_level == "truncated2":
            inner = _RolloutAnalyzer(
                weights,
                n_trials=360,
                truncation_depth=7,
                decision_ply=2,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=1,
                late_threshold=2,
            )
        elif eval_level == "truncated3":
            inner = _RolloutAnalyzer(
                weights,
                n_trials=360,
                truncation_depth=5,
                decision_ply=3,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=2,
                late_threshold=2,
            )
        elif eval_level == "rollout":
            inner = _RolloutAnalyzer(
                weights,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
                n_threads=parallel_threads,
                seed=seed,
                late_ply=late_ply,
                late_threshold=late_threshold,
                checker=checker,
                checker_late=checker_late,
                cube=cube,
                cube_late=cube_late,
                ultra_late_threshold=ultra_late_threshold,
            )
        else:
            raise ValueError(f"Unknown eval_level: {eval_level!r}")

        # Set bearoff DB on inner analyzer and its C++ strategies
        if self._bearoff_db is not None:
            inner._bearoff_db = self._bearoff_db
            if isinstance(inner, _MultiPlyAnalyzer):
                bgbot_cpp.multipy_set_bearoff_db(inner._strategy_nply, self._bearoff_db)
            elif isinstance(inner, _RolloutAnalyzer):
                bgbot_cpp.rollout_set_bearoff_db(inner._rollout_strategy, self._bearoff_db)

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
        """
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback,
            away1=away1, away2=away2, is_crawford=is_crawford, jacoby=jacoby,
            beaver=beaver,
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
        """
        if away1 > 0 or away2 > 0:
            jacoby = False

        inner = self._analyzer
        if isinstance(inner, _CubefulAnalyzer):
            inner = inner._inner

        # Evaluate the post-move board (NN outputs from mover's perspective)
        if isinstance(inner, _MultiPlyAnalyzer):
            r = inner._strategy_nply.evaluate_board(board, board)
            eval_level = f"{inner._n_plies}-ply"
            inner._strategy_nply.clear_cache()
        elif isinstance(inner, _RolloutAnalyzer):
            r = inner._rollout_strategy.evaluate_board(board, board)
            eval_level = "Rollout"
        else:
            r = inner._strategy_1ply.evaluate_board(board, board)
            eval_level = "1-ply"

        probs_list = list(r["probs"])
        cl_eq = r["equity"]

        # Cubeful equity via Janowski
        owner = resolve_owner(cube_owner)
        race = bgbot_cpp.is_race(board)
        x = bgbot_cpp.cube_efficiency(board, race)
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
                first two turns under the No Double scenario. Requires
                3-ply or higher evaluation.
        """
        if away1 > 0 or away2 > 0:
            jacoby = False
            beaver = False
        raw = self._analyzer.cube_action_analytics(
            board, cube_value, cube_owner,
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
            player_rolls=raw.get("player_rolls"),
        )


def create_analyzer(
    level: str = "1ply",
    weights: WeightConfig | WeightConfigPair | None = None,
    cubeful: bool = True,
    **kwargs: Any,
) -> BgBotAnalyzer:
    """Convenience factory for creating analyzers."""
    return BgBotAnalyzer(weights=weights, eval_level=level, cubeful=cubeful, **kwargs)
