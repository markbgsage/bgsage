#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score XG (eXtreme Gammon) against the money benchmark via Batch-Analyze results.

Companion to ``benchmark_money.benchmark_pr``, but the engine being scored is XG.
Rather than call XG position-by-position (it has no API), we read the ``.xg`` files
XG's *Batch Analyze* produced for the Sage-vs-Sage transcripts in
``data/money_benchmark/xg/``, pick out XG's **best** decision for each position --
its #1 ranked move, or its recommended cube action -- and score *that* decision
against the benchmark's SAVED reference analytics, exactly as ``benchmark_pr`` scores
a bot. XG's own equities/errors are ignored; only its chosen decision is used.

This scores XG using batch results: drop a folder of ``seed_<N>.txt`` transcripts into
XG, run Batch Analyze with "Save Games after analyze", and the resulting ``seed_<N>.xg``
files carry XG's picks for every position in those games.

Matching: each ``.xg`` game is one captured seed game, and within a single game a
position (board + dice) is unique, so we match each XG turn to that seed's
``build/stage1/seed_<N>.json`` decision by ``(kind, board, dice)``, then pull the
refined reference (and its precision tier) from ``benchmark.json`` via that decision's
``key``. Boards are reconciled to the mover's perspective: ``xg_compare`` gives
``board_before`` in player-1 perspective (flip on bot turns) while move-list boards are
already mover's-perspective. Decisions whose reference lacks the precision their
closeness requires (e.g. a not-yet-computed rollout) are skipped, same as benchmark_pr.

Usage:
  python scripts/benchmark_pr_xg.py            # score every game that has a .xg file
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# benchmark_money sets up the bgsage import path (and DLL dirs) at import time and
# supplies the scoring/aggregate/report helpers we reuse. Its bgsage imports are
# deferred, so importing it here does NOT load the engine -- and this script never
# needs it (pure parse + match + score).
import benchmark_money as bm
from bgsage.board import flip_board
from bgsage.xg_compare import parse_xg_game

log = logging.getLogger("benchmark_pr_xg")

_XG_DIR = bm._XG_DIR
_STAGE1_DIR = bm._STAGE1_DIR
DEFAULT_DATASET = bm.DEFAULT_DATASET


def _seeds_with_xg(xg_dir: Path) -> list[int]:
    """Seeds (game numbers) that have a ``seed_<N>.xg`` file."""
    seeds = []
    for p in sorted(xg_dir.glob("seed_*.xg")):
        try:
            seeds.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(seeds))


def _seed_lookup(seed: int) -> dict:
    """``{(kind, board_tuple, dice_key): decision}`` for one captured game.

    ``dice_key`` is the sorted dice tuple for checker plays, ``None`` for cube
    decisions. Unique within a single game (positions don't recur mid-game).
    """
    data = json.loads((_STAGE1_DIR / f"seed_{seed}.json").read_text(encoding="utf-8"))
    lookup = {}
    for d in data["decisions"]:
        dice_key = tuple(sorted(d["dice"])) if d["kind"] == "checker" else None
        lookup[(d["kind"], tuple(d["board"]), dice_key)] = d
    return lookup


def _norm_bars(board) -> tuple:
    """Normalise XG's signed-bar convention to bgsage's (bars are non-negative counts).

    XG signs the bars by owner, so the opponent's bar comes out negative; bgsage stores
    both bars (index 0 = opponent, index 25 = mover) as positive counts. Points (1-24)
    already share the signed-by-owner convention, so only the two bar cells need fixing.
    Without this, any board with a checker on the (opponent's) bar -- i.e. just been hit
    -- fails to match the benchmark's legal post-move boards.
    """
    b = list(board)
    b[0] = abs(b[0])
    b[25] = abs(b[25])
    return tuple(b)


def _to_mover(board_before, player) -> tuple:
    """XG ``board_before`` (player-1 perspective) -> mover's perspective, bgsage bars.

    Flip on bot turns (bgsage ``flip_board``), then normalise the bars. Flipping first
    moves a wrong-signed bar to the other index, which the bar-normalisation then fixes.
    """
    b = list(board_before) if player == "user" else list(flip_board(board_before))
    return _norm_bars(b)


def _xg_turn_decisions(turns):
    """Yield ``(kind, mover_board, dice_key, choice)`` for each XG decision in a game.

    ``board_before`` is player-1 perspective -> convert to mover's perspective (flip on
    bot turns) + normalise bars. ``checker_analysis`` boards are already mover's-
    perspective; only their bars need normalising.
    """
    for t in turns:
        bb = t.get("board_before")
        if bb is None:
            continue
        mb = _to_mover(bb, t["player"])

        ca = t.get("cube_analysis")
        if ca is not None:
            yield ("cube", mb, None,
                   {"should_double": ca["should_double"], "should_take": ca["should_take"]})

        cha = t.get("checker_analysis")
        dice = t.get("dice")
        if cha and dice:
            yield ("checker", mb, tuple(sorted(dice)), {"best_board": _norm_bars(cha[0]["board"])})


def _score_xg_checker(refined: dict, xg_best_board: tuple):
    """XG's #1 move scored against the saved reference. None if it isn't a legal ref move."""
    ref_by_board = {tuple(m["board"]): m["equity"] for m in refined["moves"]}
    chosen_eq = ref_by_board.get(tuple(xg_best_board))
    if chosen_eq is None:
        return None
    best_eq = refined["moves"][0]["equity"]
    return bm._scored("checker", "checker", refined.get("game_plan"),
                      max(0.0, best_eq - chosen_eq))


def _score_xg_cube(refined: dict, should_double: bool, should_take: bool) -> list:
    """XG's cube recommendation scored against the saved reference (doubler + receiver).

    Identical formulas to ``benchmark_money._score_cube`` -- only the chosen action
    comes from XG instead of a bot.
    """
    nd, dt, dp = refined["equity_nd"], refined["equity_dt"], refined["equity_dp"]
    plan = refined.get("game_plan")
    out = []
    if refined.get("has_double"):
        optimal = max(nd, min(dt, dp))
        actual = min(dt, dp) if should_double else nd
        out.append(bm._scored("cube", "double", plan, max(0.0, optimal - actual)))
    if refined.get("has_take"):
        optimal = min(dt, dp)
        actual = dt if should_take else dp
        out.append(bm._scored("cube", "take", plan, max(0.0, actual - optimal)))
    return out


def benchmark_pr(xg_dir: Path | str = _XG_DIR,
                 dataset_path: Path | str = DEFAULT_DATASET,
                 progress: bool = True, max_seed: int | None = None) -> dict:
    """Score XG against the benchmark for every game that has a ``.xg`` file.

    Returns the same result dict as ``benchmark_money.benchmark_pr`` (total/checker/cube
    PR + per-game-plan breakdown + blunders), plus ``skipped`` / ``mismatches`` /
    ``unmatched`` / ``n_games`` coverage counts.
    """
    xg_dir = Path(xg_dir)
    refined_by_key = {d["key"]: d for d in bm._read_dataset(dataset_path)["decisions"]}

    seeds = _seeds_with_xg(xg_dir)
    if max_seed is not None:
        seeds = [s for s in seeds if s <= max_seed]
    if not seeds:
        raise SystemExit(f"No .xg files found in {xg_dir}"
                         + (f" with seed <= {max_seed}" if max_seed is not None else ""))
    log.info("Found .xg files for %d games: seeds %d-%d", len(seeds), seeds[0], seeds[-1])

    seen: set = set()
    records: list = []
    skipped = {"rollout": 0, "3t": 0}
    unmatched = 0
    mismatches = 0
    n_xg_decisions = 0

    for seed in seeds:
        stage1_path = _STAGE1_DIR / f"seed_{seed}.json"
        if not stage1_path.exists():
            log.warning("seed %d has a .xg file but no stage1 capture -- skipping", seed)
            continue
        lookup = _seed_lookup(seed)
        turns = parse_xg_game((xg_dir / f"seed_{seed}.xg").read_bytes())

        for kind, mb, dk, choice in _xg_turn_decisions(turns):
            n_xg_decisions += 1
            sd = lookup.get((kind, mb, dk))
            if sd is None:
                unmatched += 1
                continue
            key = sd["key"]
            if key in seen:
                continue  # already scored (e.g. a shared opening from an earlier seed)
            refined = refined_by_key.get(key)
            if refined is None:
                unmatched += 1
                continue
            seen.add(key)
            miss = bm._missing_tier(refined)
            if miss is not None:
                skipped[miss] += 1
                continue
            if kind == "checker":
                s = _score_xg_checker(refined, choice["best_board"])
                if s is None:
                    mismatches += 1
                    continue
                records.append({"key": key, "scored": [s]})
            else:
                records.append({"key": key, "scored": _score_xg_cube(
                    refined, choice["should_double"], choice["should_take"])})

    result = bm._aggregate(records)
    n_skipped = skipped["rollout"] + skipped["3t"]
    result["skipped"] = n_skipped
    result["skipped_missing_rollout"] = skipped["rollout"]
    result["skipped_missing_3t"] = skipped["3t"]
    result["mismatches"] = mismatches
    result["unmatched"] = unmatched
    result["n_games"] = len(seeds)

    if progress:
        log.info("XG: %d games, %d XG decisions parsed, %d unique positions scored, "
                 "%d skipped (missing precision), %d unmatched, %d mismatches",
                 len(seeds), n_xg_decisions, result["n_decisions"], n_skipped,
                 unmatched, mismatches)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score XG against the money benchmark via .xg files")
    parser.add_argument("--xg-dir", type=Path, default=_XG_DIR,
                        help="Folder of seed_<N>.xg files (default: data/money_benchmark/xg)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Benchmark dataset JSON")
    parser.add_argument("--max-seed", type=int, default=None,
                        help="Only score the first N games (.xg files with seed <= MAX_SEED)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    result = benchmark_pr(xg_dir=args.xg_dir, dataset_path=args.dataset, max_seed=args.max_seed)
    print(f"\nScored: XG (eXtreme Gammon) over {result['n_games']} games")
    bm._print_report(result)
    if result.get("unmatched"):
        print(f"Outside benchmark: {result['unmatched']} XG decisions are positions the "
              f"dataset does not capture (forced moves, trivial checker spreads, trivial "
              f"cube actions) -- correctly not scored")


if __name__ == "__main__":
    main()
