"""Batch position evaluation for the bgsage backgammon engine.

Evaluates lists of positions in parallel, returning cubeless probabilities,
cubeless equity, cubeful equity, and (for pre-roll) cube decisions.

Parallelization happens in C++ across positions.  Individual multi-ply
evaluations run serially so there is no double-parallelization.

Typical usage::

    from bgsage.batch import batch_evaluate, batch_post_move_evaluate

    # Pre-roll positions (with cube decisions)
    positions = [
        {"board": STARTING_BOARD, "cube_value": 1, "cube_owner": "centered"},
        {"board": other_board,    "cube_value": 2, "cube_owner": "player"},
    ]
    results = batch_evaluate(positions, eval_level="2ply")
    for r in results:
        print(f"Eq: {r.cubeless_equity:+.3f}  CF: {r.cubeful_equity:+.3f}")

    # Post-move positions (after a player moves, before opponent rolls)
    post_positions = [
        {"board": post_move_board1, "cube_owner": "centered"},
        {"board": post_move_board2, "cube_owner": "player"},
    ]
    results = batch_post_move_evaluate(post_positions, eval_level="0ply")
    for r in results:
        print(f"CL={r.cubeless_equity:+.3f}  CF={r.cubeful_equity:+.3f}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import bgbot_cpp

from .analyzer import resolve_owner
from .types import CheckerPlayResult, CubeActionResult, MoveAnalysis, PostMoveAnalysis, Probabilities
from .weights import WeightConfig


@dataclass
class PositionEval:
    """Result of evaluating a single pre-roll position."""

    probs: Probabilities
    cubeless_equity: float
    cubeful_equity: float
    equity_nd: float
    equity_dt: float
    equity_dp: float
    should_double: bool
    should_take: bool
    optimal_action: str


def batch_evaluate(
    positions: list[dict],
    eval_level: str = "0ply",
    weights: WeightConfig | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
) -> list[PositionEval]:
    """Evaluate a batch of pre-roll positions in parallel.

    Each position dict must contain:
        board: list[int]  — 26-element board from mover's perspective
        cube_value: int   — current cube value (1, 2, 4, ...)
        cube_owner: str   — "centered", "player", or "opponent"

    Args:
        positions: List of position dicts.
        eval_level: ``"0ply"``, ``"1ply"``, ``"2ply"``, or ``"3ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.

    Returns:
        List of :class:`PositionEval`, one per input position.
    """
    if weights is None:
        weights = WeightConfig.default()

    # Build (board, cube_value, CubeOwner) tuples for C++
    cpp_positions = []
    for p in positions:
        board = list(p["board"])
        cube_value = p.get("cube_value", 1)
        cube_owner = resolve_owner(p.get("cube_owner", "centered"))
        cpp_positions.append((board, cube_value, cube_owner))

    if eval_level == "0ply":
        strategy = bgbot_cpp.GamePlanStrategy(*weights.weight_args)
        raw_results = bgbot_cpp.batch_evaluate_positions(
            cpp_positions, strategy, n_threads,
        )
    elif eval_level in ("1ply", "2ply", "3ply"):
        n_plies = int(eval_level[0])
        # parallel_evaluate=False so each N-ply eval is serial;
        # parallelism is across positions in the C++ batch function.
        strategy = bgbot_cpp.create_multipy_5nn(
            *weights.weight_args,
            n_plies=n_plies,
            filter_max_moves=filter_max_moves,
            filter_threshold=filter_threshold,
            parallel_evaluate=False,
            parallel_threads=0,
        )
        raw_results = bgbot_cpp.batch_evaluate_positions(
            cpp_positions, strategy, n_threads,
        )
    else:
        raise ValueError(
            f"Unsupported eval_level for batch: {eval_level!r}. "
            f"Use '0ply', '1ply', '2ply', or '3ply'."
        )

    return [
        PositionEval(
            probs=Probabilities.from_list(list(r["probs"])),
            cubeless_equity=r["cubeless_equity"],
            cubeful_equity=r["cubeful_equity"],
            equity_nd=r["equity_nd"],
            equity_dt=r["equity_dt"],
            equity_dp=r["equity_dp"],
            should_double=bool(r["should_double"]),
            should_take=bool(r["should_take"]),
            optimal_action=r["optimal_action"],
        )
        for r in raw_results
    ]


def batch_post_move_evaluate(
    positions: list[dict],
    eval_level: str = "0ply",
    weights: WeightConfig | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
) -> list[PostMoveAnalysis]:
    """Evaluate a batch of post-move positions in parallel.

    "Post-move" means the board is from the perspective of the player who
    just moved, right before the opponent rolls.  The NN is evaluated
    directly (no flip/invert), producing post-move probabilities.

    Each position dict must contain:
        board: list[int]   — 26-element post-move board (mover's perspective)
        cube_owner: str    — "centered", "player", or "opponent"

    Args:
        positions: List of position dicts.
        eval_level: ``"0ply"``, ``"1ply"``, ``"2ply"``, or ``"3ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.

    Returns:
        List of :class:`~bgsage.types.PostMoveAnalysis`, one per input position.
    """
    if weights is None:
        weights = WeightConfig.default()

    # Build (board, CubeOwner) tuples for C++
    cpp_positions = []
    for p in positions:
        board = list(p["board"])
        cube_owner = resolve_owner(p.get("cube_owner", "centered"))
        cpp_positions.append((board, cube_owner))

    if eval_level == "0ply":
        strategy = bgbot_cpp.GamePlanStrategy(*weights.weight_args)
        raw_results = bgbot_cpp.batch_evaluate_post_move(
            cpp_positions, strategy, n_threads,
        )
        level_label = "0-ply"
    elif eval_level in ("1ply", "2ply", "3ply"):
        n_plies = int(eval_level[0])
        strategy = bgbot_cpp.create_multipy_5nn(
            *weights.weight_args,
            n_plies=n_plies,
            filter_max_moves=filter_max_moves,
            filter_threshold=filter_threshold,
            parallel_evaluate=False,
            parallel_threads=0,
        )
        raw_results = bgbot_cpp.batch_evaluate_post_move(
            cpp_positions, strategy, n_threads,
        )
        level_label = f"{n_plies}-ply"
    else:
        raise ValueError(
            f"Unsupported eval_level for batch: {eval_level!r}. "
            f"Use '0ply', '1ply', '2ply', or '3ply'."
        )

    return [
        PostMoveAnalysis(
            probs=Probabilities.from_list(list(r["probs"])),
            cubeless_equity=r["cubeless_equity"],
            cubeful_equity=r["cubeful_equity"],
            eval_level=level_label,
        )
        for r in raw_results
    ]


def batch_cube_action(
    positions: list[dict],
    eval_level: str = "0ply",
    weights: WeightConfig | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
) -> list[CubeActionResult]:
    """Evaluate cube decisions for a batch of pre-roll positions in parallel.

    This is a convenience wrapper around :func:`batch_evaluate` that returns
    :class:`~bgsage.types.CubeActionResult` objects (the same type returned by
    ``BgBotAnalyzer.cube_action()``).

    Cube decisions use Janowski interpolation on the N-ply cubeless probs
    (same approach as ``batch_evaluate``), **not** the expensive recursive
    N-ply cubeful search from ``cube_decision_nply``.  This is much faster
    and adequate for most use cases.

    Each position dict must contain:
        board: list[int]  — 26-element board from mover's perspective
        cube_value: int   — current cube value (1, 2, 4, ...)
        cube_owner: str   — "centered", "player", or "opponent"

    Args:
        positions: List of position dicts.
        eval_level: ``"0ply"``, ``"1ply"``, ``"2ply"``, or ``"3ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.

    Returns:
        List of :class:`~bgsage.types.CubeActionResult`, one per input position.
    """
    evals = batch_evaluate(
        positions,
        eval_level=eval_level,
        weights=weights,
        n_threads=n_threads,
        filter_max_moves=filter_max_moves,
        filter_threshold=filter_threshold,
    )

    n_plies = 0 if eval_level == "0ply" else int(eval_level[0])
    level_label = "0-ply" if n_plies == 0 else f"{n_plies}-ply"

    return [
        CubeActionResult(
            probs=e.probs,
            cubeless_equity=e.cubeless_equity,
            equity_nd=e.equity_nd,
            equity_dt=e.equity_dt,
            equity_dp=e.equity_dp,
            should_double=e.should_double,
            should_take=e.should_take,
            optimal_equity=(
                e.equity_nd if not e.should_double
                else e.equity_dt if e.should_take
                else e.equity_dp
            ),
            optimal_action=e.optimal_action,
            eval_level=level_label,
        )
        for e in evals
    ]


def batch_checker_play(
    positions: list[dict],
    eval_level: str = "0ply",
    weights: WeightConfig | None = None,
    *,
    n_threads: int = 0,
    filter_max_moves: int = 5,
    filter_threshold: float = 0.08,
) -> list[CheckerPlayResult]:
    """Evaluate checker play for a batch of positions in parallel.

    For each input position (board + dice + cube), generates all legal moves,
    scores them, applies filtering, and optionally re-scores survivors at
    N-ply.  Returns the full ranked move list for each position.

    Each position dict must contain:
        board: list[int]    — 26-element pre-move board (mover's perspective)
        die1: int           — first die value (1-6)
        die2: int           — second die value (1-6)
        cube_value: int     — current cube value (1, 2, 4, ...)
        cube_owner: str     — "centered", "player", or "opponent"

    Args:
        positions: List of position dicts.
        eval_level: ``"0ply"``, ``"1ply"``, ``"2ply"``, or ``"3ply"``.
        weights: Weight configuration (defaults to production model).
        n_threads: Number of threads for position-level parallelism
            (0 = auto-detect all cores).
        filter_max_moves: Max moves for N-ply move filter.
        filter_threshold: Equity threshold for N-ply move filter.

    Returns:
        List of :class:`~bgsage.types.CheckerPlayResult`, one per input
        position.  Each contains a ``moves`` list sorted best-first by
        cubeful equity, with survivors evaluated at the requested ply and
        the rest at 0-ply.
    """
    if weights is None:
        weights = WeightConfig.default()

    # Build input dicts with CubeOwner enum for C++
    cpp_inputs = []
    for p in positions:
        cpp_inputs.append({
            "board": list(p["board"]),
            "die1": p["die1"],
            "die2": p["die2"],
            "cube_value": p.get("cube_value", 1),
            "cube_owner": resolve_owner(p.get("cube_owner", "centered")),
        })

    strategy_0ply = bgbot_cpp.GamePlanStrategy(*weights.weight_args)

    if eval_level == "0ply":
        raw_results = bgbot_cpp.batch_checker_play(
            cpp_inputs, strategy_0ply,
            filter_max_moves, filter_threshold, n_threads,
        )
        level_label = "0-ply"
    elif eval_level in ("1ply", "2ply", "3ply"):
        n_plies = int(eval_level[0])
        strategy_nply = bgbot_cpp.create_multipy_5nn(
            *weights.weight_args,
            n_plies=n_plies,
            filter_max_moves=filter_max_moves,
            filter_threshold=filter_threshold,
            parallel_evaluate=False,
            parallel_threads=0,
        )
        raw_results = bgbot_cpp.batch_checker_play(
            cpp_inputs, strategy_0ply, strategy_nply,
            filter_max_moves, filter_threshold, n_threads,
        )
        level_label = f"{n_plies}-ply"
    else:
        raise ValueError(
            f"Unsupported eval_level: {eval_level!r}. "
            f"Use '0ply', '1ply', '2ply', or '3ply'."
        )

    results = []
    for i, raw in enumerate(raw_results):
        p = positions[i]
        moves = [
            MoveAnalysis(
                board=list(m["board"]),
                equity=m["equity"],
                cubeless_equity=m["cubeless_equity"],
                probs=Probabilities.from_list(list(m["probs"])),
                equity_diff=m["equity_diff"],
                eval_level=m["eval_level"],
            )
            for m in raw["moves"]
        ]
        results.append(CheckerPlayResult(
            moves=moves,
            board=list(p["board"]),
            die1=p["die1"],
            die2=p["die2"],
            eval_level=level_label,
        ))
    return results
