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
from .types import PostMoveAnalysis, Probabilities
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
