#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Sage-vs-XG rollout benchmark report — the two standing analyses in one run.

This is the single entry point for the Sage-vs-XG comparison. It (optionally)
harvests the latest XG rollout results, then prints two reports with consistent
formatting:

  1. EVAL-LEVEL PR TABLE (benchmark_pr_xg_levels)
       Performance Rating of six evaluators — Sage 1T/2T/3T and XG
       Roller/Roller+/Roller++ — against XG's full rollout as the reference, over
       the benchmark's rollout-tier positions. Lower PR = the level's chosen
       decisions are closer to the full rollout.

  2. DISPUTE REPORT (xg_dispute_analysis)
       On positions where Sage 3T and XG Roller++ pick DIFFERENT decisions and we
       have both a Sage and an XG full rollout, which bot is closer to each
       rollout — measured separately against the Sage rollout and the XG rollout.

Methodology (so results are reproducible and checkable by others):
  - Reference = full rollouts (1296+ games, roll-until-CI-0.005, 3-ply checker &
    cube, VR on). XG rollouts come from XG's Batch Rollout (parsed from the .xg
    files into data/<bench>/xg_results/rollout.jsonl); the Sage rollout is the
    benchmark reference in data/<bench>/benchmark.json.
  - Each evaluator's *chosen* decision is taken from cached data, not recomputed:
    Sage 3T from build/stage2_3t.jsonl; Sage 1T/2T recovered from their
    scores/*.jsonl errors; XG Roller/Roller+/Roller++ from the seed_<N>_*.xg
    batch-analyze files.
  - Every evaluator's own pick is force-rolled by XG (the flag policy), so
    mismatches are rare; a pick XG couldn't roll (outside its stored move list on
    high-mobility positions) is charged the biggest error among the rolled moves.
  - Only positions XG has actually rolled so far are included, so the numbers
    firm up as more of the batch completes. The header prints the current sample.

Usage:
  python scripts/xg_benchmark_report.py                 # harvest + both reports (money)
  python scripts/xg_benchmark_report.py --no-harvest    # use cached data as-is
  python scripts/xg_benchmark_report.py --benchmark match --match-length 5
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import benchmark_pr_xg_levels as levels
import xg_dispute_analysis as dispute
import xg_harvest_results


_RULE = "=" * 78


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--no-harvest", action="store_true",
                    help="Skip harvesting new XG rollout results; use cached data as-is")
    args = ap.parse_args(argv)

    if not args.no_harvest:
        print(_RULE)
        print("Harvesting latest XG rollout results ...")
        print(_RULE)
        xg_harvest_results.main(["--benchmark", args.benchmark,
                                 "--match-length", str(args.match_length),
                                 "--set", "rollout"])

    print("\n" + _RULE)
    print("REPORT 1 — Evaluation-level PR vs XG full rollout")
    print(_RULE)
    levels.print_report(levels.score(args.benchmark, args.match_length), args.benchmark)

    print("\n" + _RULE)
    print("REPORT 2 — Sage 3T vs XG Roller++ on disputed positions")
    print(_RULE)
    dispute.print_report(dispute.score(args.benchmark, args.match_length), args.benchmark)


if __name__ == "__main__":
    sys.exit(main())
