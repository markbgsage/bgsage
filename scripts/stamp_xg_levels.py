#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Stamp per-XG-eval-level copies of the pristine native .xg folder for Batch Analyze.

XG's Batch Analyze writes its analysis INTO each ``.xg`` (one analysis per record --
re-analyzing overwrites it) and skips already-analyzed positions, so testing XG at
several eval levels needs a SEPARATE folder copy per level. This copies the pristine
master ``xg_native/`` to ``xg_native_<tag>/`` for each requested level, leaving the
master untouched.

Workflow per level:
  1. run this to stamp the folders,
  2. in XG: File -> Batch Analyze, point at ``xg_native_<tag>/``, pick the matching
     level (see the table below), "Save Games after analyze" ON,
  3. ``python scripts/benchmark_pr_xg_pasko.py --xg-dir <that folder>`` to harvest the PR.

The analyzed folder is the durable cache -- re-harvesting is free; never analyze the
master, and never analyze one folder at two levels.

Level tags (tag -> XG Batch-Analyze menu selection -> Sage equivalent):

    xg1ply        1-ply                 Sage 1P
    xg2ply        2-ply                 Sage 2P
    xg3ply        3-ply                 Sage 3P
    xg4ply        4-ply                 Sage 4P
    xgroller      XG Roller             Sage 1T
    xgrollerplus  XG Roller+            Sage 2T
    xgrollerpp    XG Roller++           Sage 3T

Usage::

    python scripts/stamp_xg_levels.py                       # the DEFAULT_LEVELS set
    python scripts/stamp_xg_levels.py --levels xg3ply xgrollerplus
    python scripts/stamp_xg_levels.py --list               # print the mapping, stamp nothing
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import benchmark_pasko as bp  # noqa: E402  (pasko data dir; no engine load)

#: tag -> (XG Batch-Analyze menu label, Sage equivalent) -- also the documented mapping.
LEVELS: dict[str, tuple[str, str]] = {
    "xg1ply":       ("1-ply",      "Sage 1P"),
    "xg2ply":       ("2-ply",      "Sage 2P"),
    "xg3ply":       ("3-ply",      "Sage 3P"),
    "xg4ply":       ("4-ply",      "Sage 4P"),
    "xgroller":     ("XG Roller",  "Sage 1T"),
    "xgrollerplus": ("XG Roller+", "Sage 2T"),
    "xgrollerpp":   ("XG Roller++", "Sage 3T"),
}

#: Stamped by default: the three plies that mirror the Sage 1P/2P/3P table, plus the
#: three XG Roller truncated-rollout levels (Roller / Roller+ / Roller++ = Sage 1T/2T/3T).
DEFAULT_LEVELS = ["xg1ply", "xg2ply", "xg3ply", "xgroller", "xgrollerplus", "xgrollerpp"]

#: The pristine native-.xg master that every per-level copy is stamped from.
MASTER = bp._DATA_DIR / "xg_native"


def _print_table(tags):
    w = max(len(t) for t in LEVELS)
    print(f"{'folder tag':<{w}}  {'XG Batch-Analyze level':<22}  Sage equiv")
    print(f"{'-'*w}  {'-'*22}  {'-'*10}")
    for t in tags:
        xg, sage = LEVELS[t]
        print(f"{t:<{w}}  {xg:<22}  {sage}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--levels", nargs="*", default=None,
                        help=f"Level tags to stamp (default: {' '.join(DEFAULT_LEVELS)})")
    parser.add_argument("--master", type=Path, default=MASTER,
                        help=f"Pristine native .xg folder to copy (default: {MASTER})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing per-level folder (default: skip it)")
    parser.add_argument("--sync", action="store_true",
                        help="Add only NEW master games to each existing level folder "
                             "(never overwrites already-analyzed .xg). Use after extending "
                             "the game set; missing folders are stamped fresh.")
    parser.add_argument("--list", action="store_true", help="Print the level mapping and exit")
    args = parser.parse_args(argv)

    tags = args.levels or DEFAULT_LEVELS
    bad = [t for t in tags if t not in LEVELS]
    if bad:
        raise SystemExit(f"Unknown level tag(s): {bad}. Known: {list(LEVELS)}")

    if args.list:
        _print_table(tags)
        return

    if not args.master.is_dir():
        raise SystemExit(f"Master folder not found: {args.master}\n"
                         "Run: python scripts/export_pasko_benchmark_xg.py")
    n_master = len(list(args.master.glob("seed_*.xg")))
    if n_master == 0:
        raise SystemExit(f"No seed_*.xg in {args.master} -- export the games first.")

    if args.sync:
        print(f"Master: {args.master}  ({n_master} games)\n")
        print(f"{'folder':<34}  {'XG level':<22}  status")
        print(f"{'-'*34}  {'-'*22}  ------")
        for tag in tags:
            dest = args.master.parent / f"{args.master.name}_{tag}"
            xg = LEVELS[tag][0]
            if not dest.is_dir():
                shutil.copytree(args.master, dest)
                print(f"{dest.name:<34}  {xg:<22}  created ({n_master} games)")
                continue
            added = 0
            for src in sorted(args.master.glob("seed_*.xg")):
                d = dest / src.name
                if not d.exists():
                    shutil.copy2(src, d)
                    added += 1
            total = len(list(dest.glob("seed_*.xg")))
            print(f"{dest.name:<34}  {xg:<22}  +{added} new (now {total}); "
                  f"existing analysis untouched")
        print("\nRe-run XG Batch Analyze on each folder (it skips already-analyzed games), "
              "then harvest with benchmark_pr_xg_pasko.py --xg-dir <folder>.")
        return

    print(f"Master: {args.master}  ({n_master} games)\n")
    print(f"{'folder':<34}  {'XG level to pick':<22}  {'Sage equiv':<10}  status")
    print(f"{'-'*34}  {'-'*22}  {'-'*10}  ------")
    for tag in tags:
        dest = args.master.parent / f"{args.master.name}_{tag}"
        xg, sage = LEVELS[tag]
        if dest.exists() and not args.force:
            status = "exists (skipped)"
        else:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(args.master, dest)
            status = f"stamped {len(list(dest.glob('seed_*.xg')))} games"
        print(f"{dest.name:<34}  {xg:<22}  {sage:<10}  {status}")

    print("\nNext, per folder: XG -> Batch Analyze at the level shown, 'Save Games after "
          "analyze' ON, then:\n  python scripts/benchmark_pr_xg_pasko.py --xg-dir <folder>")


if __name__ == "__main__":
    main()
