#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Harvest XG analysis/rollout results from .xg files into the XG results cache.

Two modes:

``--set rollout``   Read the staged files in ``data/<benchmark>/xg_batch/rollout/``
                    after (or during) an XG Batch Rollout run. Decisions whose
                    rollout has completed (a ``TRolloutContext`` exists in the
                    file's temp.xgr stream) are appended to
                    ``data/<benchmark>/xg_results/rollout.jsonl``. Safe to run
                    mid-batch: pending decisions are just reported, and already
                    -cached keys are skipped.

``--set rollerpp``  Read the EXISTING Roller++ Batch-Analyze files (money:
                    ``xg/seed_<N>_pp.xg``; match: ``xg_snapshots/roller_pp/``)
                    and cache XG's stored evaluations for 3T-tier benchmark
                    decisions — no new XG run needed. Per-move / per-cube
                    evaluation levels are preserved (XG only escalates close
                    decisions to its Roller++ mini-rollout; the rest carry the
                    ply level XG's move filter used).

The cache is keyed by the benchmark decision ``key`` and lives entirely under
``xg_results/`` — separate from the Sage reference data and Sage score caches.

Usage:
  python scripts/xg_harvest_results.py --benchmark money --set rollerpp
  python scripts/xg_harvest_results.py --benchmark money --set rollout
  python scripts/xg_harvest_results.py --benchmark match --set rollout --match-length 5
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import xg_batch_common as xbc
from bgsage import xg_file


def _ctx_ok(ctx) -> bool:
    """A rollout context holds usable (possibly early-stopped) data."""
    return ctx["rolled"] > 0 and not ctx["user_interrupted"]


def _ctx_fully_rolled(ctx) -> bool:
    return _ctx_ok(ctx) and ctx["rolled"] >= max(ctx["min_roll"], 1)


def _harvest_file(path, paths, targets_by_seed, seed, batch_set, stats, new_records):
    arch = xg_file.XgArchive.load(path)
    tempxg = arch.get("temp.xg")
    xgr = arch.get("temp.xgr")
    index = xbc.GameFileIndex(tempxg)
    source = {"file": path.name, "seed": seed}

    for d in targets_by_seed.get(seed, []):
        off = xbc.find_decision_record(index, paths, d)
        if off is None:
            stats["not_found"] += 1
            continue
        if d["kind"] == "checker":
            rec = xg_file.parse_move_record(tempxg, off)
            if batch_set == "rollout":
                # The mark request persists in the file (timedelay bits; XG sets
                # the done bits rather than clearing them). Only cache once every
                # requested move has a usable rollout context AND the decision is
                # finished: either XG's done-bits cover the request, or every
                # context reached its minimum game count. (A move XG eliminated
                # early via the multiple-rollout 1% rule has rolled < min_roll but
                # its done-bit set — that is complete, not pending.) Mid-batch
                # partials and user-interrupted contexts stay "pending" and are
                # picked up by a later harvest.
                requested = rec["timedelay"] | rec["timedelay_done"]
                if requested:
                    req_idx = [i for i in range(rec["n_moves"]) if requested >> i & 1]
                else:  # marks cleared in XG: fall back to whatever has contexts
                    req_idx = [i for i in range(rec["n_moves"])
                               if rec["rollout_indices"][i] >= 0]
                if not req_idx or not xgr or any(
                        rec["rollout_indices"][i] < 0 for i in req_idx):
                    stats["pending"] += 1
                    continue
                ctxs = [xg_file.parse_rollout_context(xgr, rec["rollout_indices"][i])
                        for i in req_idx]
                done_covers = requested != 0 and \
                    (rec["timedelay_done"] & requested) == requested
                if any(not _ctx_ok(c) for c in ctxs) or not (
                        done_covers or all(_ctx_fully_rolled(c) for c in ctxs)):
                    stats["pending"] += 1
                    continue
                stats["rolled_moves"] += len(req_idx)
                for ctx in ctxs:
                    stats["duration"] += ctx["duration"]
                    stats["trials"] += ctx["rolled"]
            entry = xbc.checker_cache_record(d, rec, xgr, f"xg_{batch_set}", source)
            new_records.append(entry)
            stats["checker"] += 1
            if batch_set == "rollerpp":
                top = entry["moves"][0]["eval_level"] if entry["moves"] else "none"
                stats[f"top_level:{top}"] += 1
        else:
            rec = xg_file.parse_cube_record(tempxg, off)
            if batch_set == "rollout":
                if rec["rollout_index"] < 0 or not xgr:
                    stats["pending"] += 1
                    continue
                ctx = xg_file.parse_rollout_context(xgr, rec["rollout_index"])
                # Both lines (ND and D/T) must be rolled — equity_dt comes from
                # the D/T line. rolled2 == 0 with a finished ND line means the
                # preset's "Roll for both Double and NoDouble" was off.
                if ctx["rolled"] > 0 and ctx["rolled2"] == 0 and not ctx["roll_both"]:
                    stats["nd_only"] += 1
                done = rec["timedelay_done"]
                complete = _ctx_ok(ctx) and ctx["rolled2"] > 0 and (
                    done or (_ctx_fully_rolled(ctx)
                             and ctx["rolled2"] >= max(ctx["min_roll"], 1)))
                if not complete:
                    stats["pending"] += 1
                    continue
                stats["duration"] += ctx["duration"]
                stats["trials"] += ctx["rolled"] + ctx["rolled2"]
            entry = xbc.cube_cache_record(d, rec, xgr, f"xg_{batch_set}", source)
            new_records.append(entry)
            stats["cube"] += 1
            if batch_set == "rollerpp":
                stats[f"cube_level:{entry['eval_level']}"] += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--set", dest="batch_set", choices=["rollout", "rollerpp"],
                    default="rollout")
    ap.add_argument("--src", type=Path, default=None,
                    help="Override the source directory of .xg files")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="Benchmark tiers to harvest (default: rollout->rollout, "
                         "rollerpp->3T)")
    args = ap.parse_args(argv)

    paths = xbc.paths_for(args.benchmark, args.match_length)
    tiers = set(args.tiers) if args.tiers else (
        {"rollout"} if args.batch_set == "rollout" else {"3T"})
    decisions = xbc.load_decisions(paths, tiers)
    cache_path = paths.cache_file(args.batch_set)
    cache = xbc.load_cache(cache_path)
    targets = [d for d in decisions if d["key"] not in cache]
    print(f"{paths.name} / {args.batch_set}: {len(decisions)} target decisions "
          f"(tiers {sorted(tiers)}), {len(decisions) - len(targets)} already cached, "
          f"{len(targets)} to harvest")
    if not targets:
        return

    targets_by_seed = xbc.group_by_seed(targets)

    if args.src is not None:
        src_dir = args.src
        pattern = paths.staged_name if args.batch_set == "rollout" else paths.base_pattern
    elif args.batch_set == "rollout":
        src_dir, pattern = paths.batch_dir / "rollout", paths.staged_name
    else:
        src_dir, pattern = paths.xg_base_dir, paths.base_pattern

    stats = Counter()
    new_records = []
    for seed in sorted(targets_by_seed):
        path = src_dir / pattern.format(seed=seed)
        if not path.exists():
            stats["missing_file"] += 1
            stats["missing_file_decisions"] += len(targets_by_seed[seed])
            continue
        try:
            _harvest_file(path, paths, targets_by_seed, seed, args.batch_set,
                          stats, new_records)
        except Exception as e:  # noqa: BLE001 - report and continue with other files
            print(f"  ERROR parsing {path.name}: {e}")
            stats["parse_errors"] += 1

    if new_records:
        xbc.append_cache(cache_path, new_records)

    print(f"\nHarvested {stats['checker']} checker + {stats['cube']} cube decisions "
          f"-> {cache_path}")
    if args.batch_set == "rollout":
        print(f"  pending (marked but not rolled yet): {stats['pending']}")
        if stats["nd_only"]:
            print(f"  WARNING: {stats['nd_only']} cube rollouts have only the No-Double "
                  f"line — the XG preset's \"Roll for both Double and NoDouble\" appears "
                  f"to be OFF. Fix the preset and re-roll those files.")
        if stats["rolled_moves"]:
            print(f"  checker rollout lines: {stats['rolled_moves']}")
        if stats["duration"]:
            n = stats["checker"] + stats["cube"]
            print(f"  XG compute harvested: {stats['duration'] / 3600:.2f} h total, "
                  f"{stats['duration'] / max(n, 1):.1f} s per decision, "
                  f"{stats['trials']} trials")
            remaining = len(targets) - stats["checker"] - stats["cube"]
            if n and remaining > 0:
                proj = stats["duration"] / n * remaining
                print(f"  projected XG time for the {remaining} remaining decisions: "
                      f"{proj / 3600:.1f} h")
    else:
        levels = {k.split(":", 1)[1]: v for k, v in stats.items()
                  if k.startswith("top_level:")}
        cube_levels = {k.split(":", 1)[1]: v for k, v in stats.items()
                       if k.startswith("cube_level:")}
        if levels:
            print(f"  XG #1-move eval level distribution: {dict(sorted(levels.items()))}")
        if cube_levels:
            print(f"  cube eval level distribution: {dict(sorted(cube_levels.items()))}")
    if stats["not_found"]:
        print(f"  {stats['not_found']} decisions not located in their game file")
    if stats["missing_file"]:
        print(f"  {stats['missing_file']} source files missing "
              f"({stats['missing_file_decisions']} decisions)")
    if stats["parse_errors"]:
        print(f"  {stats['parse_errors']} files failed to parse")


if __name__ == "__main__":
    sys.exit(main())
