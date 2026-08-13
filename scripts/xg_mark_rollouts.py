#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Stage .xg files with batch-rollout marks for XG's Batch Rollout feature.

For every benchmark decision whose Sage reference is a full rollout
(``tier == "rollout"``), this script locates the decision inside the
Roller++-batch-analyzed ``.xg`` game file it came from, sets XG's
"marked for later rollout" fields on it, and writes the marked copy to
``data/<benchmark>/xg_batch/rollout/``. You then queue that folder ONCE in XG
(Batch Rollout > Select game to rolls > "+" > multi-select all files > pick the
rollout preset > Start) and let it run; XG writes results back into the staged
files, which ``xg_harvest_results.py`` then parses into the XG results cache.

Which moves get rolled (checker decisions): the best ``--max-sage-moves`` of
Sage's own rollout candidates that XG stored, plus XG's top ``--xg-top`` moves.
Cube decisions are marked whole (XG rolls ND and D/T lines together when the
preset has "Roll for both Double and NoDouble" on).

Idempotent: decisions already present in the results cache are not re-marked,
and existing staged files are not overwritten without ``--force`` (they may
contain completed rollouts you haven't harvested yet).

Usage:
  python scripts/xg_mark_rollouts.py --benchmark money --seeds 1     # single-file test
  python scripts/xg_mark_rollouts.py --benchmark money               # everything
  python scripts/xg_mark_rollouts.py --benchmark match --match-length 5
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import xg_batch_common as xbc
from bgsage import xg_file

BATCH_SET = "rollout"

XG_INSTRUCTIONS = """\
Next steps in XG (one time for this folder):
  1. Open XG. Menu "Batch Rollout" > "Select games to rolls" (Ctrl+Alt+T).
  2. Click the "+" icon, browse to:
       {folder}
     and multi-select all the staged .xg files.
  3. Pick / configure the rollout preset (Options > Rollouts, or "change
     setting" in the dialog). Recommended, to mirror the Sage reference:
       - Minimum games: 1296, Maximum games: 20736
       - "Roll until error is less than": 0.005  (95% CI ~ 2 x Sage's 0.0025 SE gate)
       - No truncation ("Truncate after N moves" OFF)
       - Variance reduction: ON
       - Checker play: 3-ply for first moves AND next moves
       - Cube decision: 3-ply for first moves AND next moves
       - "Roll for both Double and NoDouble": ON
       - Seed: default
     For already-rolled decisions choose "Do nothing".
  4. Press Start and leave it running. Progress is saved into the files as it
     goes, so you can stop/restart XG and re-run the remaining files any time.
  5. When done (or partially done): python scripts/xg_harvest_results.py \
--benchmark {benchmark} --set rollout
"""

# Sage evaluation-level choice caches (per-decision equity error vs the Sage
# reference; the chosen move is recovered from the error). Money benchmark names.
SAGE_SCORE_CACHES = {"Sage 1T": "sage_1t_clean72", "Sage 2T": "sage_2t_clean"}
XG_PICK_LEVELS = ("roller", "rollerplus", "rollerpp")


def _extra_level_boards(d, xg_maps, stage2, sage_caches):
    """The 6 evaluation levels' own chosen moves for checker decision ``d``.

    XG Roller/Roller+/Roller++ from their batch-analyze files; Sage 3T from
    stage2_3t (argmax); Sage 1T/2T recovered from their score-cache errors.
    Returned as a set of board tuples so plan_checker_flags can flag them.
    """
    key = d["key"]
    extra = set()
    for m in xg_maps.values():
        p = m.get(key)
        if p and "checker" in p:
            extra.add(p["checker"])
    s2 = stage2.get(key)
    if s2 and s2.get("moves"):
        extra.add(tuple(max(s2["moves"], key=lambda mv: mv["equity"])["board"]))
    for sc in sage_caches:
        r = sc.get(key)
        if not r:
            continue
        ce = [x["error"] for x in r["scored"] if x["bucket"] == "checker"]
        if ce:
            pk = xbc.recover_checker_pick(d, ce[0])
            if pk:
                extra.add(pk)
    return extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Only stage these seeds (default: all with target decisions)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stage at most N files (first N seeds with targets)")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="Flag moves within this Sage-equity gap of the best move")
    ap.add_argument("--min-moves", type=int, default=2,
                    help="Always flag at least this many moves per checker decision")
    ap.add_argument("--max-moves", type=int, default=4,
                    help="Flag at most this many moves per checker decision")
    ap.add_argument("--xg-top", type=int, default=0,
                    help="Also flag XG's top N stored moves (default 0 = off)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite staged files that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Locate decisions and plan flags but write nothing")
    args = ap.parse_args(argv)

    paths = xbc.paths_for(args.benchmark, args.match_length)
    decisions = xbc.load_decisions(paths, tiers={"rollout"})
    cache = xbc.load_cache(paths.cache_file(BATCH_SET))
    targets = [d for d in decisions if d["key"] not in cache]
    print(f"{paths.name}: {len(decisions)} rollout-tier decisions, "
          f"{len(decisions) - len(targets)} already cached, {len(targets)} to mark")
    if not targets:
        return

    by_seed = xbc.group_by_seed(targets)
    seeds = sorted(by_seed)
    if args.seeds is not None:
        seeds = [s for s in seeds if s in set(args.seeds)]
    if args.limit is not None:
        seeds = seeds[:args.limit]

    # Caches for each evaluation level's own pick (flagged so nothing is a mismatch).
    stage2 = xbc.load_cache(paths.dataset.parent / "build" / "stage2_3t.jsonl")
    scores_dir = paths.dataset.parent / "scores"
    sage_caches = [xbc.load_cache(scores_dir / f"{name}.jsonl")
                   for name in SAGE_SCORE_CACHES.values()]
    missing_sage = [lbl for lbl, name in SAGE_SCORE_CACHES.items()
                    if not (scores_dir / f"{name}.jsonl").exists()]
    if not stage2:
        print("  WARNING: stage2_3t.jsonl not found -- Sage 3T picks won't be flagged")
    if missing_sage:
        print(f"  WARNING: score caches missing for {missing_sage} -- those picks "
              f"won't be flagged")

    out_dir = paths.batch_dir / BATCH_SET
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    totals = {"files": 0, "skipped_existing": 0, "missing_base": 0,
              "checker_marked": 0, "cube_marked": 0, "flagged_moves": 0,
              "not_found": 0, "sage_best_missing": 0,
              "level_pick_lines": 0, "level_pick_missing": 0}
    manifest_lines = []

    for seed in seeds:
        base = paths.base_file(seed)
        out = paths.staged_file(BATCH_SET, seed)
        if not base.exists():
            print(f"  seed {seed}: base file missing ({base.name}) -- "
                  f"{len(by_seed[seed])} decisions cannot be marked")
            totals["missing_base"] += 1
            continue
        if out.exists() and not args.force:
            if not args.dry_run:
                print(f"  seed {seed}: staged file exists, skipping (harvest it or --force)")
            totals["skipped_existing"] += 1
            continue

        arch = xg_file.XgArchive.load(base)
        tempxg = bytearray(arch.get("temp.xg"))
        index = xbc.GameFileIndex(tempxg)

        # each XG level's own pick for this seed's decisions (Roller++ = base file)
        xg_maps = {lvl: xbc.xg_level_picks(paths, lvl, seed, by_seed[seed])
                   for lvl in XG_PICK_LEVELS}

        marked = []
        n_move_marks = n_cube_marks = n_move_lines = 0
        for d in by_seed[seed]:
            off = xbc.find_decision_record(index, paths, d)
            if off is None:
                totals["not_found"] += 1
                continue
            if d["kind"] == "checker":
                rec = xg_file.parse_move_record(tempxg, off)
                extra = _extra_level_boards(d, xg_maps, stage2, sage_caches)
                plan = xbc.plan_checker_flags(d, rec, threshold=args.threshold,
                                              min_moves=args.min_moves,
                                              max_moves=args.max_moves,
                                              xg_top=args.xg_top, extra_boards=extra)
                if not plan.indices:
                    totals["not_found"] += 1
                    continue
                xg_file.set_move_timedelay(tempxg, off, plan.bits)
                n_move_marks += 1
                n_move_lines += len(plan.indices)
                totals["flagged_moves"] += len(plan.indices)
                totals["level_pick_lines"] += plan.extra_present
                totals["level_pick_missing"] += plan.extra_missing
                if plan.sage_best_missing:
                    totals["sage_best_missing"] += 1
                marked.append({"key": d["key"], "kind": "checker", "offset": off,
                               "move_indices": plan.indices,
                               "sage_missing": plan.sage_missing,
                               "sage_best_missing": plan.sage_best_missing})
            else:
                xg_file.set_cube_timedelay(tempxg, off, marked=True)
                n_cube_marks += 1
                marked.append({"key": d["key"], "kind": "cube", "offset": off})

        if not marked:
            continue
        if index.header_offset is None:
            print(f"  seed {seed}: no match header record -- skipping file")
            continue
        if args.dry_run:
            totals["files"] += 1
            totals["checker_marked"] += n_move_marks
            totals["cube_marked"] += n_cube_marks
            continue
        # XG's TotTimeDelayMove counts flagged move-LINES (individual candidate
        # moves), not records; TotTimeDelayCube counts flagged cube records. XG
        # recomputes these on save, but we match its convention so the file we
        # hand it is self-consistent.
        xg_file.set_header_timedelay_totals(tempxg, index.header_offset,
                                            n_move_lines, n_cube_marks)
        arch.set("temp.xg", bytes(tempxg))
        arch.set("temp.xgi", xg_file.rebuild_xgi(tempxg))
        arch.save(out)

        # sanity: re-load and confirm the marks round-tripped
        check = xg_file.XgArchive.load(out)
        check_data = check.get("temp.xg")
        hdr = xg_file.parse_header(check_data, index.header_offset)
        assert hdr["tot_timedelay_move"] == n_move_lines, "header totals did not round-trip"
        for m in marked:
            if m["kind"] == "checker":
                rec = xg_file.parse_move_record(check_data, m["offset"])
                want = sum(1 << i for i in m["move_indices"])
                assert rec["timedelay"] == want, f"move marks did not round-trip at {m['offset']}"
            else:
                rec = xg_file.parse_cube_record(check_data, m["offset"])
                assert rec["timedelay"], f"cube mark did not round-trip at {m['offset']}"

        totals["files"] += 1
        totals["checker_marked"] += n_move_marks
        totals["cube_marked"] += n_cube_marks
        manifest_lines.append(json.dumps({
            "file": out.name, "seed": seed, "staged_at":
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "marked": marked}, separators=(",", ":")))
        print(f"  seed {seed}: marked {n_move_marks} checker + {n_cube_marks} cube "
              f"decisions -> {out.name}")

    if manifest_lines:
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(manifest_lines) + "\n")

    verb = "Would stage" if args.dry_run else "Staged"
    print(f"\n{verb} {totals['files']} files in {out_dir}")
    print(f"  decisions marked: {totals['checker_marked']} checker "
          f"({totals['flagged_moves']} move rollout lines), {totals['cube_marked']} cube")
    print(f"  level-pick moves flagged (Sage 1T/2T/3T + XG Roller/+/++): "
          f"{totals['level_pick_lines']} added, {totals['level_pick_missing']} not in "
          f"XG's stored moves (can't be rolled)")
    if totals["skipped_existing"]:
        print(f"  {totals['skipped_existing']} files already staged (not overwritten)")
    if totals["missing_base"]:
        print(f"  {totals['missing_base']} seeds missing their base Roller++ .xg file")
    if totals["not_found"]:
        print(f"  {totals['not_found']} decisions could not be located/flagged in their game file")
    if totals["sage_best_missing"]:
        print(f"  WARNING: {totals['sage_best_missing']} decisions where Sage's #1 rollout "
              f"move is not among XG's stored moves (XG cannot roll it; tracked in manifest)")
    if totals["files"] and not args.dry_run:
        print("\n" + XG_INSTRUCTIONS.format(folder=out_dir, benchmark=args.benchmark))


if __name__ == "__main__":
    sys.exit(main())
