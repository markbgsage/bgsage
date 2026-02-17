"""High-level analysis interface for the bgsage backgammon engine.

This module provides :class:`BgBotAnalyzer`, the main entry point for
checker play and cube action analysis at any evaluation level (0-ply
through N-ply and Monte Carlo rollout).

Typical usage::

    from bgsage import BgBotAnalyzer

    analyzer = BgBotAnalyzer()                       # 0-ply, cubeful
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
from .weights import WeightConfig

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

    def __init__(self, weights: WeightConfig):
        self._weights = weights
        self._strategy_0ply = bgbot_cpp.GamePlanStrategy(*weights.weight_args)

    def _score_candidates_0ply(
        self,
        candidates: list,
        board: list[int],
        cube_owner: str | None = None,
    ) -> list[tuple[float, float, list[int], list[float]]]:
        owner = resolve_owner(cube_owner) if cube_owner else None
        scored = []
        for b in candidates:
            bl = list(b)
            r = self._strategy_0ply.evaluate_board(bl, board)
            cl_eq = r["equity"]
            probs = list(r["probs"])
            if owner is not None:
                race = bgbot_cpp.is_race(bl)
                x = bgbot_cpp.cube_efficiency(bl, race)
                cf_eq = bgbot_cpp.cl2cf_money(probs, owner, x)
            else:
                cf_eq = cl_eq
            scored.append((cf_eq, cl_eq, bl, probs))
        scored.sort(key=lambda item: -item[0])
        return scored

    @staticmethod
    def _filter_candidates(
        scored_0ply: list,
        threshold: float,
        max_moves: int,
    ) -> tuple[list, set]:
        best_eq = scored_0ply[0][0]
        survivors = [
            item
            for item in scored_0ply
            if (best_eq - item[0]) < threshold
        ][:max_moves]
        survivor_set = {tuple(item[2]) for item in survivors}
        return survivors, survivor_set

    @staticmethod
    def _promote_second_best(results: list, board: list[int], evaluate_fn) -> None:
        results.sort(key=lambda x: -x["equity"])
        while len(results) >= 2 and results[1].get("is_0ply_only"):
            r = results[1]
            equity, probs, eval_level, extra = evaluate_fn(r["board"], board)
            r["equity"] = equity
            r["probs"] = probs
            r["eval_level"] = eval_level
            r.pop("is_0ply_only", None)
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
    def _format_cube_result(r: dict, eval_level: str = "0-ply") -> dict:
        return {
            "probs": list(r["probs"]),
            "cubeless_equity": r.get("cubeless_equity", 0),
            "equity_nd": r["equity_nd"],
            "equity_dt": r["equity_dt"],
            "equity_dp": r["equity_dp"],
            "should_double": bool(r["should_double"]),
            "should_take": bool(r["should_take"]),
            "optimal_equity": r["optimal_equity"],
            "cubeless_se": r.get("cubeless_se", None),
            "eval_level": eval_level,
        }


class _ZeroPlyAnalyzer(_CubelessBase):

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []
        results = []
        for b in candidates:
            bl = list(b)
            r = self._strategy_0ply.evaluate_board(bl, board)
            results.append({
                "board": bl,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "eval_level": "0-ply",
            })
        return self._finalize_results(results)

    def cube_action_analytics(self, board, cube_value=1, cube_owner="centered") -> dict:
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.evaluate_cube_decision(
            board, cube_value, owner, *self._weights.weight_args
        )
        return self._format_cube_result(r, eval_level="0-ply")


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
        self._strategy_nply = bgbot_cpp.create_multipy_5nn(
            *weights.weight_args,
            n_plies=n_plies,
            parallel_evaluate=parallel_evaluate,
            parallel_threads=self._parallel_threads,
        )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_0ply = self._score_candidates_0ply(candidates, board, cube_owner)
        survivors, survivor_set = self._filter_candidates(
            scored_0ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
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

        for feq, cleq, b, p in scored_0ply:
            if tuple(b) not in survivor_set:
                results.append({
                    "board": b,
                    "equity": cleq,
                    "probs": p,
                    "is_0ply_only": True,
                    "eval_level": "0-ply",
                })

        n_plies = self._n_plies
        strategy = self._strategy_nply

        def _nply_eval(b, board_ref):
            r = strategy.evaluate_board(b, board_ref)
            return r["equity"], list(r["probs"]), f"{n_plies}-ply", {}

        self._promote_second_best(results, board, _nply_eval)
        self._strategy_nply.clear_cache()
        return self._finalize_results(results)

    def cube_action_analytics(self, board, cube_value=1, cube_owner="centered") -> dict:
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.cube_decision_nply(
            board, cube_value, owner, self._n_plies, *self._weights.weight_args,
            n_threads=self._parallel_threads,
        )
        result = self._format_cube_result(r, eval_level=f"{self._n_plies}-ply")

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
        decision_ply=0, vr_ply=0, n_threads=0, seed=42,
    ):
        super().__init__(weights)
        self._rollout_config = {
            "n_trials": n_trials,
            "truncation_depth": truncation_depth,
            "decision_ply": decision_ply,
            "vr_ply": vr_ply,
            "n_threads": n_threads,
            "seed": seed,
        }
        self._rollout_strategy = bgbot_cpp.create_rollout_5nn(
            *weights.weight_args,
            n_trials=n_trials,
            truncation_depth=truncation_depth,
            decision_ply=decision_ply,
            vr_ply=vr_ply,
            n_threads=n_threads,
            seed=seed,
        )

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
    ) -> list[dict]:
        candidates = bgbot_cpp.possible_moves(board, die1, die2)
        if not candidates:
            return []

        scored_0ply = self._score_candidates_0ply(candidates, board, cube_owner)
        survivors, survivor_set = self._filter_candidates(
            scored_0ply, self.FILTER_THRESHOLD, self.FILTER_MAX_MOVES
        )

        results = []
        total = len(survivors)
        for i, (feq, cleq, b, p0) in enumerate(survivors):
            r = self._rollout_strategy.evaluate_board(b, board)
            results.append({
                "board": b,
                "equity": r["equity"],
                "probs": list(r["probs"]),
                "std_error": r.get("std_error", 0),
                "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
                "eval_level": "Rollout",
            })
            if progress_callback:
                progress_callback(i + 1, total, results)

        for feq, cleq, b, p in scored_0ply:
            if tuple(b) not in survivor_set:
                results.append({
                    "board": b,
                    "equity": cleq,
                    "probs": p,
                    "is_0ply_only": True,
                    "eval_level": "0-ply",
                })

        rollout_strategy = self._rollout_strategy

        def _rollout_eval(b, board_ref):
            r = rollout_strategy.evaluate_board(b, board_ref)
            extra = {
                "std_error": r.get("std_error", 0),
                "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
            }
            return r["equity"], list(r["probs"]), "Rollout", extra

        self._promote_second_best(results, board, _rollout_eval)
        return self._finalize_results(results)

    def cube_action_analytics(self, board, cube_value=1, cube_owner="centered") -> dict:
        owner = resolve_owner(cube_owner)
        r = bgbot_cpp.cube_decision_rollout(
            board, cube_value, owner, *self._weights.weight_args,
            **self._rollout_config,
        )
        return self._format_cube_result(r, eval_level="Rollout")


class _CubefulAnalyzer:
    """Cubeful wrapper around any cubeless analyzer."""

    def __init__(self, inner: _CubelessBase):
        self._inner = inner
        self._weights = inner._weights
        if isinstance(inner, _MultiPlyAnalyzer):
            self._cubeful_ply = inner._n_plies
        else:
            self._cubeful_ply = 0

    def _cubeful_equity(self, post_move_board, probs, owner) -> float:
        if self._cubeful_ply == 0:
            race = bgbot_cpp.is_race(post_move_board)
            x = bgbot_cpp.cube_efficiency(post_move_board, race)
            return bgbot_cpp.cl2cf_money(probs, owner, x)
        else:
            opp_pre_roll = bgbot_cpp.flip_board(post_move_board)
            opp_owner = _FLIP_OWNER[owner]
            opp_eq = bgbot_cpp.cubeful_equity_nply(
                opp_pre_roll, opp_owner,
                self._inner._strategy_0ply, self._cubeful_ply,
            )
            return -opp_eq

    def checker_play_analytics(
        self, board, die1, die2, cube_value=1, cube_owner="centered",
        progress_callback=None,
    ) -> list[dict]:
        owner = resolve_owner(cube_owner)
        results = self._inner.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback
        )
        if not results:
            return results

        workers = getattr(self._inner, "_parallel_threads", 0)
        if not isinstance(workers, int) or workers <= 1:
            workers = max(2, os.cpu_count() or 2)
        workers = max(1, workers)

        def _convert_move(m: dict) -> tuple[float, float]:
            cubeless_eq = m["equity"]
            cf_eq = self._cubeful_equity(m["board"], m["probs"], owner)
            return cubeless_eq, cf_eq

        with ThreadPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert_move, results))

        for m, (cubeless_eq, cf_eq) in zip(results, converted):
            cubeless_eq = m["equity"]
            m["cubeless_equity"] = cubeless_eq
            m["equity"] = cf_eq

        results.sort(key=lambda x: -x["equity"])

        inner = self._inner

        def _cubeful_eval(b, board_ref):
            if isinstance(inner, _MultiPlyAnalyzer):
                r = inner._strategy_nply.evaluate_board(b, board_ref)
                eval_level = f"{inner._n_plies}-ply"
                extra = {}
            elif isinstance(inner, _RolloutAnalyzer):
                r = inner._rollout_strategy.evaluate_board(b, board_ref)
                eval_level = "Rollout"
                extra = {
                    "std_error": r.get("std_error", 0),
                    "prob_std_errors": list(r.get("prob_std_errors", [0] * 5)),
                }
            else:
                return None
            probs = list(r["probs"])
            cf_eq = self._cubeful_equity(b, probs, owner)
            extra["cubeless_equity"] = r["equity"]
            return cf_eq, probs, eval_level, extra

        while len(results) >= 2 and results[1].get("is_0ply_only"):
            r = results[1]
            ret = _cubeful_eval(r["board"], board)
            if ret is None:
                break
            cf_eq, probs, eval_level, extra = ret
            r["equity"] = cf_eq
            r["probs"] = probs
            r["eval_level"] = eval_level
            r.pop("is_0ply_only", None)
            r.update(extra)
            results.sort(key=lambda x: -x["equity"])

        if results:
            best = results[0]["equity"]
            for r in results:
                r["equity_diff"] = r["equity"] - best

        return results

    def cube_action_analytics(self, board, cube_value=1, cube_owner="centered") -> dict:
        return self._inner.cube_action_analytics(board, cube_value, cube_owner)


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


def _optimal_action(should_double: bool, should_take: bool) -> str:
    if not should_double:
        return "No Double"
    if should_take:
        return "Double/Take"
    return "Double/Pass"


class BgBotAnalyzer:
    """High-level interface to the backgammon bot engine.

    Thread-safe: multiple threads can call methods concurrently.

    Args:
        weights: Weight file configuration (defaults to bundled Stage 5 models).
        eval_level: ``'0ply'``, ``'1ply'``, ``'2ply'``, ``'3ply'``, or ``'rollout'``.
        cubeful: If True, compute cubeful equities via Janowski.
        filter_max_moves: Max moves to carry through N-ply/rollout.
        filter_threshold: Equity threshold for move filter.
        parallel_threads: Thread count (0 = auto-detect).
        n_trials: Rollout trial count.
        truncation_depth: Rollout truncation (0 = play to completion).
        decision_ply: Ply depth for move selection during rollout trials.
        vr_ply: Ply depth for variance reduction (-1 = disable).
        seed: RNG seed for rollout.
    """

    def __init__(
        self,
        weights: WeightConfig | None = None,
        eval_level: str = "0ply",
        cubeful: bool = True,
        *,
        filter_max_moves: int = 5,
        filter_threshold: float = 0.08,
        parallel_threads: int = 0,
        n_trials: int = 1296,
        truncation_depth: int = 0,
        decision_ply: int = 0,
        vr_ply: int = 0,
        seed: int = 42,
    ):
        if weights is None:
            weights = WeightConfig.default()
        self._weights = weights
        self._eval_level = eval_level

        if eval_level == "0ply":
            inner: _CubelessBase = _ZeroPlyAnalyzer(weights)
        elif eval_level in ("1ply", "2ply", "3ply"):
            n_plies = int(eval_level[0])
            inner = _MultiPlyAnalyzer(
                weights, n_plies=n_plies,
                parallel_threads=parallel_threads,
            )
        elif eval_level == "rollout":
            inner = _RolloutAnalyzer(
                weights,
                n_trials=n_trials,
                truncation_depth=truncation_depth,
                decision_ply=decision_ply,
                vr_ply=vr_ply,
                n_threads=parallel_threads,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown eval_level: {eval_level!r}")

        if cubeful:
            self._analyzer = _CubefulAnalyzer(inner)
        else:
            self._analyzer = inner

    def checker_play(
        self,
        board: list[int],
        die1: int,
        die2: int,
        cube_value: int = 1,
        cube_owner: str = "centered",
        include_game_plans: bool = False,
        progress_callback: Any | None = None,
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
        """
        raw = self._analyzer.checker_play_analytics(
            board, die1, die2, cube_value, cube_owner, progress_callback
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
    ) -> PostMoveAnalysis:
        """Evaluate a post-move position (right before the opponent's turn).

        Returns cubeless probabilities, cubeless equity, and cubeful equity
        from the perspective of the player who just moved.

        Args:
            board: 26-element post-move board array (player who moved's perspective).
            cube_owner: ``'centered'``, ``'player'``, or ``'opponent'``.
        """
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
            r = inner._strategy_0ply.evaluate_board(board, board)
            eval_level = "0-ply"

        probs_list = list(r["probs"])
        cl_eq = r["equity"]

        # Cubeful equity via Janowski
        owner = resolve_owner(cube_owner)
        race = bgbot_cpp.is_race(board)
        x = bgbot_cpp.cube_efficiency(board, race)
        cf_eq = bgbot_cpp.cl2cf_money(probs_list, owner, x)

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
    ) -> CubeActionResult:
        """Analyze the cube decision for a pre-roll position.

        Returns cubeful equities for No Double, Double/Take, Double/Pass,
        with the optimal action and pre-roll cubeless probabilities.
        """
        raw = self._analyzer.cube_action_analytics(board, cube_value, cube_owner)
        probs = Probabilities.from_list(raw["probs"])
        return CubeActionResult(
            probs=probs,
            cubeless_equity=raw["cubeless_equity"],
            equity_nd=raw["equity_nd"],
            equity_dt=raw["equity_dt"],
            equity_dp=raw["equity_dp"],
            should_double=raw["should_double"],
            should_take=raw["should_take"],
            optimal_equity=raw["optimal_equity"],
            optimal_action=_optimal_action(raw["should_double"], raw["should_take"]),
            eval_level=raw["eval_level"],
            cubeless_se=raw.get("cubeless_se"),
        )


def create_analyzer(
    level: str = "0ply",
    weights: WeightConfig | None = None,
    cubeful: bool = True,
    **kwargs: Any,
) -> BgBotAnalyzer:
    """Convenience factory for creating analyzers."""
    return BgBotAnalyzer(weights=weights, eval_level=level, cubeful=cubeful, **kwargs)
