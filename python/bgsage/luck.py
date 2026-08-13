# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Per-roll Luck: how lucky a dice roll was, in equity units.

Luck measures how much the roll that actually happened helped (or hurt) the
player on roll, versus an average roll from the same position::

    luck = (equity of the best play with the actual roll)
         - (weight-averaged equity over every possible roll)

Both equities are cubeful and from the roller's perspective. Positive luck means
a lucky roll (its best play beats the position's roll-average); negative means
unlucky. Averaged over many rolls, luck tends to zero, so summed luck over a game
separates dice fortune from decision quality.

The per-roll equities are the ND (No Double) per-roll cubeful equities already
produced by :meth:`bgsage.BgBotAnalyzer.cube_action` with
``incl_2ply_details=True``. Luck is therefore a *pure function over analytics the
engine has already computed* — :func:`luck_from_equities` and :func:`roll_luck`
run no additional neural-network evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import CubeActionResult, LuckResult, RollEquity


def _luck_ply_from_eval_level(eval_level: str) -> tuple[int, str]:
    """Accuracy of a cube analysis's per-roll equities, from its eval level.

    ``cube_action(incl_2ply_details=True)`` evaluates each player roll at
    ``(N-1)``-ply when the cube analysis itself runs at ``N``-ply, so an N-ply
    cube analysis yields ``(N-1)``-ply luck. Falls back to 1-ply for eval levels
    with no leading ply count (e.g. ``"Rollout"``).
    """
    digits = ""
    for ch in eval_level.strip():
        if ch.isdigit():
            digits += ch
        else:
            break
    n = int(digits) if digits else 0
    per_roll_ply = max(1, n - 1) if n else 1
    return per_roll_ply, f"{per_roll_ply}-ply"


def luck_from_equities(
    per_roll: Sequence[RollEquity],
    die1: int,
    die2: int,
    *,
    ply: int,
    level_label: str,
    is_opening_roll: bool = False,
) -> LuckResult | None:
    """Compute luck from pre-computed per-roll equities (pure, no evaluation).

    ``per_roll`` holds one :class:`~bgsage.types.RollEquity` per possible roll
    (21 rolls, or the 15 non-doubles for an opening roll). ``die1``/``die2`` are
    the roll that actually happened; die order does not matter.

    Returns ``None`` when luck cannot be computed — the actual roll is absent
    from ``per_roll`` or the total weight is zero.
    """
    rolls = [r for r in per_roll if not (is_opening_roll and r.die1 == r.die2)]

    lo, hi = min(die1, die2), max(die1, die2)
    actual_equity: float | None = None
    for r in rolls:
        if min(r.die1, r.die2) == lo and max(r.die1, r.die2) == hi:
            actual_equity = r.equity
            break

    total_weight = sum(r.weight for r in rolls)
    if actual_equity is None or total_weight <= 0:
        return None

    average_equity = sum(r.equity * r.weight for r in rolls) / total_weight
    return LuckResult(
        luck=actual_equity - average_equity,
        actual_equity=actual_equity,
        average_equity=average_equity,
        ply=ply,
        level_label=level_label,
        per_roll=list(rolls),
    )


def roll_luck(
    cube: CubeActionResult,
    die1: int,
    die2: int,
    *,
    ply: int | None = None,
    level_label: str | None = None,
    is_opening_roll: bool = False,
) -> LuckResult | None:
    """Compute luck from a cube analysis — the bot analytics for a position.

    ``cube`` must have been produced by
    :meth:`bgsage.BgBotAnalyzer.cube_action` with ``incl_2ply_details=True``; its
    ``details["nd"]`` per-roll cubeful equities are the input. ``die1``/``die2``
    are the roll that actually happened.

    ``ply`` / ``level_label`` describe the accuracy of those per-roll equities;
    when omitted they are derived from ``cube.eval_level`` (an N-ply cube analysis
    gives ``(N-1)``-ply per-roll equities). Returns ``None`` if ``cube`` carries
    no per-roll details or the actual roll is missing from them.
    """
    nd = (cube.details or {}).get("nd") or []
    if not nd:
        return None

    per_roll = [
        RollEquity(
            die1=entry["die1"],
            die2=entry["die2"],
            equity=entry["cubeful_equity"],
            weight=1 if entry["die1"] == entry["die2"] else 2,
        )
        for entry in nd
    ]

    if ply is None or level_label is None:
        derived_ply, derived_label = _luck_ply_from_eval_level(cube.eval_level)
        ply = derived_ply if ply is None else ply
        level_label = derived_label if level_label is None else level_label

    return luck_from_equities(
        per_roll, die1, die2,
        ply=ply, level_label=level_label, is_opening_roll=is_opening_roll,
    )
