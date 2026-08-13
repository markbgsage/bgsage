#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score XG (eXtreme Gammon) against the MATCH benchmark via Batch-Analyze results.

The match-play analog of ``benchmark_pr_xg.py``. Rather than call XG
position-by-position (it has no API), we read the ``.xg`` files XG's *Batch
Analyze* produced for the Sage-vs-Sage match transcripts in
``data/match_benchmark/{L}pt/xg/``, pick out XG's **best** decision for each
position (its #1 ranked move, or its recommended cube action), and score *that*
decision against the benchmark's SAVED reference analytics -- exactly as
``benchmark_match.benchmark_pr`` scores a bot. XG's own equities/errors are
ignored; only its chosen decision is used.

Workflow: drop the folder of ``match_seed_<N>.txt`` transcripts (written by
``benchmark_match build --stages pass1``) into XG, run Batch Analyze with
"Save Games after analyze", and the resulting ``match_seed_<N>.xg`` files carry
XG's picks for every position. Then run this script.

Matching: each ``.xg`` file is one captured match (multiple games). XG turns are
matched to ``build/stage1/match_seed_<N>.json`` decisions by
``(game_number, kind, board, dice)`` -- a position is unique within a single
game, and game_number disambiguates the same board recurring across games of a
match at different scores. Boards are reconciled to the mover's perspective
(``board_before`` is player-1 perspective; flip on bot turns) and bars are
normalised to bgsage's convention. Decisions whose reference lacks the precision
their closeness requires (e.g. a not-yet-computed rollout) are skipped, same as
``benchmark_pr``.

Usage:
  python scripts/benchmark_pr_xg_match.py --match-length 5
  python scripts/benchmark_pr_xg_match.py --match-length 5 --max-seed 50
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# benchmark_money supplies the scoring/aggregate/report helpers; benchmark_match
# supplies the path layout. Both defer their bgsage imports, but we DO need the
# engine's parser here, imported below.
import benchmark_money as bm        # noqa: E402
import benchmark_match as bmm       # noqa: E402
from bgsage.board import flip_board          # noqa: E402
from bgsage.xg_compare import parse_xg_match  # noqa: E402

log = logging.getLogger("benchmark_pr_xg_match")


def _seeds_with_xg(xg_dir: Path) -> list[int]:
    """Seeds (match numbers) that have a ``match_seed_<N>.xg`` file."""
    seeds = []
    for p in sorted(xg_dir.glob("match_seed_*.xg")):
        try:
            seeds.append(int(p.stem.split("_")[-1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(seeds))


def _match_lookup(stage1_path: Path) -> dict:
    """``{(game_number, kind, board_tuple, dice_key): decision}`` for one match.

    ``dice_key`` is the sorted dice tuple for checker plays, ``None`` for cube
    decisions. ``game_number`` disambiguates a board recurring across games.
    """
    data = json.loads(stage1_path.read_text(encoding="utf-8"))
    lookup = {}
    for d in data["decisions"]:
        dice_key = tuple(sorted(d["dice"])) if d["kind"] == "checker" else None
        lookup[(d["game_number"], d["kind"], tuple(d["board"]), dice_key)] = d
    return lookup


def _norm_bars(board) -> tuple:
    """Normalise XG's signed-bar convention to bgsage's (bars are non-negative)."""
    b = list(board)
    b[0] = abs(b[0])
    b[25] = abs(b[25])
    return tuple(b)


def _to_mover(board_before, player) -> tuple:
    """XG ``board_before`` (player-1 perspective) -> mover's perspective, bgsage bars."""
    b = list(board_before) if player == "user" else list(flip_board(board_before))
    return _norm_bars(b)


def _xg_turn_decisions(turns):
    """Yield ``(kind, mover_board, dice_key, choice)`` for each XG decision in a game."""
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
    """XG's #1 move scored against the saved reference. None if not a legal ref move."""
    ref_by_board = {tuple(m["board"]): m["equity"] for m in refined["moves"]}
    chosen_eq = ref_by_board.get(tuple(xg_best_board))
    if chosen_eq is None:
        return None
    best_eq = refined["moves"][0]["equity"]
    return bm._scored("checker", "checker", refined.get("game_plan"),
                      max(0.0, best_eq - chosen_eq))


def _score_xg_cube(refined: dict, should_double: bool, should_take: bool) -> list:
    """XG's cube recommendation scored against the saved reference (doubler + receiver)."""
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


def benchmark_pr(match_length: int, dataset_path: Path | str | None = None,
                 xg_dir: Path | str | None = None,
                 progress: bool = True, max_seed: int | None = None) -> dict:
    """Score XG against the match benchmark for every match that has a ``.xg`` file."""
    paths = bmm._Paths(match_length)
    xg_dir = Path(xg_dir) if xg_dir is not None else paths.xg
    dataset_path = Path(dataset_path) if dataset_path is not None else paths.dataset
    refined_by_key = {d["key"]: d for d in bm._read_dataset(dataset_path)["decisions"]}

    seeds = _seeds_with_xg(xg_dir)
    if max_seed is not None:
        seeds = [s for s in seeds if s <= max_seed]
    if not seeds:
        raise SystemExit(f"No match_seed_*.xg files found in {xg_dir}"
                         + (f" with seed <= {max_seed}" if max_seed is not None else ""))
    log.info("Found .xg files for %d matches: seeds %d-%d", len(seeds), seeds[0], seeds[-1])

    seen: set = set()
    records: list = []
    skipped = {"rollout": 0, "3t": 0}
    unmatched = 0
    mismatches = 0
    n_xg_decisions = 0
    n_games_total = 0

    for seed in seeds:
        stage1_path = paths.stage1 / f"match_seed_{seed}.json"
        if not stage1_path.exists():
            log.warning("seed %d has a .xg file but no stage1 capture -- skipping", seed)
            continue
        lookup = _match_lookup(stage1_path)
        match_data = parse_xg_match((xg_dir / f"match_seed_{seed}.xg").read_bytes())

        for game in match_data["games"]:
            n_games_total += 1
            game_number = game["game_number"]
            for kind, mb, dk, choice in _xg_turn_decisions(game["turns"]):
                n_xg_decisions += 1
                sd = lookup.get((game_number, kind, mb, dk))
                if sd is None:
                    unmatched += 1
                    continue
                key = sd["key"]
                if key in seen:
                    continue  # already scored (e.g. a shared opening from an earlier match)
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
    result["n_matches"] = len(seeds)
    result["n_games"] = n_games_total

    if progress:
        log.info("XG: %d matches (%d games), %d XG decisions parsed, %d unique positions scored, "
                 "%d skipped (missing precision), %d unmatched, %d mismatches",
                 len(seeds), n_games_total, n_xg_decisions, result["n_decisions"], n_skipped,
                 unmatched, mismatches)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score XG against the match benchmark via .xg files")
    parser.add_argument("--match-length", type=int, required=True, help="Match length in points")
    parser.add_argument("--xg-dir", type=Path, default=None,
                        help="Folder of match_seed_<N>.xg files (default: data/match_benchmark/{L}pt/xg)")
    parser.add_argument("--dataset", type=Path, default=None, help="Benchmark dataset JSON")
    parser.add_argument("--max-seed", type=int, default=None,
                        help="Only score the first N matches (.xg files with seed <= MAX_SEED)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    result = benchmark_pr(match_length=args.match_length, dataset_path=args.dataset,
                          xg_dir=args.xg_dir, max_seed=args.max_seed)
    print(f"\nScored: XG (eXtreme Gammon) over {result['n_matches']} matches "
          f"({result['n_games']} games), {args.match_length}-point")
    bm._print_report(result)
    if result.get("unmatched"):
        print(f"Outside benchmark: {result['unmatched']} XG decisions are positions the "
              f"dataset does not capture (forced moves, trivial checker spreads, trivial "
              f"cube actions) -- correctly not scored")


if __name__ == "__main__":
    main()
