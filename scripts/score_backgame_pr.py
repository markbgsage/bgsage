#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score the engine against a back game benchmark category's rollout reference.

Reads ``<category> rollout.jsonl`` -- the rollout-grade reference produced for
every decision in ``<category> benchmark.txt`` -- plays each decision at one or
more evaluation levels, and reports a Performance Rating: the average equity
error per decision x 500 (the XG convention).

The scoring formulas are imported from ``benchmark_money`` rather than restated,
so this benchmark and the money/match ones can never disagree about what an
error is:

* **Checker**: the bot's chosen play against the reference's best.
  ``error = max(0, best_equity - chosen_equity)``.
* **Cube**: up to two sub-decisions per position. The doubler's error is
  ``max(0, max(ND, min(DT,DP)) - actual)``; the receiver's is
  ``max(0, actual - min(DT,DP))``.

Which cube sub-decisions count is decided HERE, from the rollout reference,
rather than trusted from the 2T/3P screen that generated the decision list --
the rollout is the better evidence for whether a double was ever a live
question. A position whose doubler decision is trivial (an obvious no-double,
too-good or hopeless spot) contributes only its take decision, or nothing.

One caveat the numbers do not show on their own. The reference carries EVERY
legal move, but only the handful that survived the move filter carry
rollout-grade equities; the rest carry the 1-ply or 2-ply value the filter
scored them at. When a bot picks one of those, its error mixes precision
scales -- and it is NOT a small effect for a bot whose taste differs from
the reference player's: measured 2026-09-02, filter-graded picks were 3.3%
of stage11's decisions but 24% of its error mass (43% in "21 backgame"),
and re-rolling eleven of them flipped eight in stage11's favour (+0.15
equity over-charged per such pick). So the report prints, beside the full
PR, the PR over rollout-graded picks only ("PR(RO)"), the share of checker
picks that were filter-graded, and the share of the error mass they carry.
Read the two PRs as a bracket: the truth lies between them until the
filter-graded candidates are rolled out (scripts/rollout_backgame_candidates
in the parent repo completes a reference that way).

Usage::

    py -3.14 scripts/score_backgame_pr.py --category "21 backgame"
    py -3.14 scripts/score_backgame_pr.py --category "21 backgame" \\
        --level 1ply 2ply 3ply
    py -3.14 scripts/score_backgame_pr.py --category "21 backgame" \\
        --level 3ply --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT / "build", _SCRIPT_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if (_PROJECT_ROOT / "build").is_dir():
        os.add_dll_directory(str(_PROJECT_ROOT / "build"))

from backgame_benchmark import benchmark_file  # noqa: E402
from benchmark_money import (  # noqa: E402
    BLUNDER_THRESHOLD, PR_MULTIPLIER, TRIVIAL_SPREAD, _is_trivial_cube,
)

DEFAULT_LEVELS = ("1ply", "2ply", "3ply")


def rollout_file(category: str) -> Path:
    return benchmark_file(category).parent / f"{category} rollout.jsonl"


def load_reference(category: str, limit: int | None) -> list[dict]:
    path = rollout_file(category)
    if not path.exists():
        raise SystemExit(
            f"No rollout reference for {category!r}: {path} does not exist.\n"
            f"Build it with scripts/rollout_backgame_benchmark.py (parent repo).")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:limit] if limit else rows


def cube_subdecisions(entry: dict) -> tuple[bool, bool]:
    """Which of a cube position's two sub-decisions count, per the rollout."""
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    has_double = not _is_trivial_cube(nd, dt, dp)
    has_take = bool(entry.get("should_double_ref")) and (
        abs(dt - dp) >= TRIVIAL_SPREAD or bool(entry.get("is_beaver")))
    return has_double, has_take


def score_level(rows: list[dict], level: str, threads: int,
                model: str | None = None) -> dict:
    """Play every decision at ``level`` and total the equity errors."""
    from bgsage import BgBotAnalyzer
    from bgsage.weights import WeightConfigPair

    weights = WeightConfigPair.from_model(model) if model else None
    analyzer = BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                             parallel_threads=threads)

    sums = {"checker": 0.0, "cube": 0.0}
    counts = {"checker": 0, "cube": 0}
    blunders = {"checker": 0, "cube": 0}
    rollout_grade_picks = mismatches = 0
    # Checker error carried by rollout-graded picks vs filter-graded picks:
    # the second is scored against a 1-/2-ply value and over-charges any bot
    # whose picks fall outside the reference player's filter set.
    ro_sum = filt_sum = 0.0
    ro_n = 0
    started = time.perf_counter()

    for i, entry in enumerate(rows, start=1):
        if entry["kind"] == "checker":
            die1, die2 = entry["dice"]
            result = analyzer.checker_play(
                entry["board"], die1, die2,
                cube_value=entry["cube_value"], cube_owner=entry["cube_owner"],
                jacoby=True, beaver=True)
            if not result.moves:
                continue
            chosen = tuple(result.moves[0].board)
            ref = {tuple(m["board"]): m for m in entry["moves"]}
            picked = ref.get(chosen)
            if picked is None:
                # Should not happen: the reference carries every legal move.
                mismatches += 1
                continue
            is_ro = picked["eval_level"] == "Rollout"
            rollout_grade_picks += is_ro
            error = max(0.0, entry["moves"][0]["equity"] - picked["equity"])
            sums["checker"] += error
            counts["checker"] += 1
            blunders["checker"] += error > BLUNDER_THRESHOLD
            if is_ro:
                ro_sum += error
                ro_n += 1
            else:
                filt_sum += error
        else:
            has_double, has_take = cube_subdecisions(entry)
            if not (has_double or has_take):
                continue
            nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
            action = analyzer.cube_action(
                entry["board"], cube_value=entry["cube_value"],
                cube_owner=entry["cube_owner"], jacoby=True, beaver=True)
            if has_double:
                optimal = max(nd, min(dt, dp))
                actual = min(dt, dp) if action.should_double else nd
                error = max(0.0, optimal - actual)
                sums["cube"] += error
                counts["cube"] += 1
                blunders["cube"] += error > BLUNDER_THRESHOLD
            if has_take:
                optimal = min(dt, dp)
                actual = dt if action.should_take else dp
                error = max(0.0, actual - optimal)
                sums["cube"] += error
                counts["cube"] += 1
                blunders["cube"] += error > BLUNDER_THRESHOLD

        if i % 100 == 0:
            print(f"    {level}: {i}/{len(rows)} decisions "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    n_total = counts["checker"] + counts["cube"]
    err_total = sums["checker"] + sums["cube"]

    def _pr(total: float, n: int) -> float:
        return (total / n * PR_MULTIPLIER) if n else 0.0

    return {
        "level": level,
        "pr": _pr(err_total, n_total),
        # Cube sub-decisions are always reference-grade, so they count here.
        "pr_rollout_graded": _pr(ro_sum + sums["cube"], ro_n + counts["cube"]),
        "checker_pr": _pr(sums["checker"], counts["checker"]),
        "cube_pr": _pr(sums["cube"], counts["cube"]),
        "n": n_total,
        "n_checker": counts["checker"],
        "n_cube": counts["cube"],
        "blunders": blunders["checker"] + blunders["cube"],
        "mean_error": err_total / n_total if n_total else 0.0,
        "rollout_grade_pct": (100 * rollout_grade_picks / counts["checker"]
                              if counts["checker"] else 0.0),
        "filter_graded_mass_pct": 100 * filt_sum / err_total if err_total else 0.0,
        "mismatches": mismatches,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True, help='e.g. "21 backgame"')
    parser.add_argument("--level", nargs="+", default=list(DEFAULT_LEVELS),
                        help=f"Levels to score (default: {' '.join(DEFAULT_LEVELS)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N reference decisions")
    parser.add_argument("--model", default=None,
                        help="Registry model to score (default: the production "
                             "model). e.g. stage11")
    parser.add_argument("--threads", type=int, default=0,
                        help="Engine threads (0 = every CPU)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Also write the results here, for aggregating "
                             "many categories scored in parallel")
    args = parser.parse_args()

    if args.model:
        print(f"Scoring model: {args.model}")
    rows = load_reference(args.category, args.limit)
    n_checker = sum(1 for r in rows if r["kind"] == "checker")
    live_cube = sum(1 for r in rows if r["kind"] == "cube" and any(cube_subdecisions(r)))
    print(f"{args.category}: {len(rows)} reference decisions "
          f"({n_checker} checker, {len(rows) - n_checker} cube; "
          f"{live_cube} cube positions with a live sub-decision)")
    print(f"Reference: {rows[0]['n_trials']} paths/position, "
          f"blunder > {BLUNDER_THRESHOLD}, PR = mean error x {PR_MULTIPLIER}\n")

    results = []
    for level in args.level:
        print(f"  scoring {level}...", flush=True)
        results.append(score_level(rows, level, args.threads, args.model))

    print(f"\n{'level':>8} {'PR':>8} {'PR(RO)':>8} {'checker':>8} {'cube':>8} "
          f"{'decisions':>10} {'blunders':>9} {'mean err':>9} "
          f"{'RO-grade':>9} {'filt mass':>10} {'time':>7}")
    print("-" * 106)
    for r in results:
        print(f"{r['level']:>8} {r['pr']:8.2f} {r['pr_rollout_graded']:8.2f} "
              f"{r['checker_pr']:8.2f} {r['cube_pr']:8.2f} {r['n']:10d} "
              f"{r['blunders']:9d} {r['mean_error']:9.5f} "
              f"{r['rollout_grade_pct']:8.1f}% {r['filter_graded_mass_pct']:9.1f}% "
              f"{r['seconds']:6.0f}s")
    if any(r["mismatches"] for r in results):
        print("\nWARNING: some chosen plays were not in the reference move list "
              "- the reference is supposed to carry every legal move.")
    print("\nRO-grade = share of checker picks whose reference equity is "
          "rollout-grade rather than a filter-level estimate; PR(RO) scores "
          "only those picks (plus cube decisions), and 'filt mass' is the "
          "share of the error mass carried by filter-graded picks. The full "
          "PR and PR(RO) bracket the truth until those candidates are rolled out.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"category": args.category, "n_reference": len(rows),
             "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
