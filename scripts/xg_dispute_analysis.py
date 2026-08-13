#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Dispute analysis: Sage 3T vs XG Roller++ on positions where they disagree.

For every rollout-tier position we have BOTH a Sage full rollout (the benchmark
reference) and an XG full rollout (xg_results/rollout.jsonl), find the ones where
Sage 3T and XG Roller++ make DIFFERENT decisions, and ask: which bot is closer to
each rollout? Reported separately for the Sage rollout and the XG rollout.

Per rollout benchmark, over the disputed positions:
  - fraction where Sage 3T's decision matches the rollout's best,
  - fraction where XG Roller++'s decision matches the rollout's best,
  - fraction where neither does (rollout prefers a third option),
  - average equity error of each bot vs that rollout.

Decisions: checker = the chosen move; cube = the action (ND / Double-Take /
Double-Pass). Sage 3T pick from build/stage2_3t.jsonl; XG Roller++ pick from the
seed_<N>_pp.xg batch-analyze file. Cube actions from each side's nd/dt/dp.

Coverage: a bot's pick can only be scored against a rollout if that rollout
actually rolled it. The XG rollout rolled both bots' picks (forced via the flag
policy); the Sage rollout only rolled Sage's own filter survivors, so XG
Roller++'s pick is sometimes absent (Sage's filter pruned it). To keep the two
comparisons cleanly comparable, both run on the SAME common set: disputed
positions where both bots' picks were rolled by BOTH rollouts. Disputes failing
that (a pick -- usually XG's -- missing from a rolled set) are excluded from both
and counted.

Usage:
  python scripts/xg_dispute_analysis.py --benchmark money
"""

import argparse
import json
import sys
from collections import defaultdict

import xg_batch_common as xbc
from benchmark_pr_xg_reference import _load_jsonl, _cube_action


def _rolled_map(moves, level_tag):
    """{board_tuple: equity} for moves actually rolled (eval_level == level_tag)."""
    return {tuple(m["board"]): m["equity"] for m in moves
            if m.get("eval_level") == level_tag}


def _cube_label(nd, dt, dp):
    """Discrete cube action: 'ND', 'D/T', or 'D/P'."""
    if min(dt, dp) > nd:
        return "D/T" if dt <= dp else "D/P"
    return "ND"


def _cube_error(nd, dt, dp, has_double, has_take, sd, st):
    """Standard doubler+receiver equity error of action (sd, st) vs rollout nd/dt/dp."""
    err = 0.0
    if has_double:
        err += max(0.0, max(nd, min(dt, dp)) - (min(dt, dp) if sd else nd))
    if has_take:
        err += max(0.0, (dt if st else dp) - min(dt, dp))
    return err


class Tally:
    def __init__(self):
        self.n = 0
        self.sage_correct = 0
        self.xg_correct = 0
        self.neither = 0
        self.sage_err = 0.0
        self.xg_err = 0.0

    def add(self, sage_ok, xg_ok, sage_err, xg_err):
        self.n += 1
        self.sage_correct += sage_ok
        self.xg_correct += xg_ok
        self.neither += (not sage_ok and not xg_ok)
        self.sage_err += sage_err
        self.xg_err += xg_err

    def report(self, label):
        if not self.n:
            print(f"  {label}: (no covered disputed positions)")
            return
        n = self.n
        print(f"  {label}: n={n}")
        print(f"    Sage 3T correct:  {self.sage_correct/n:5.1%}   "
              f"XG Roller++ correct: {self.xg_correct/n:5.1%}   "
              f"neither: {self.neither/n:5.1%}")
        print(f"    avg error:  Sage 3T {self.sage_err/n:.4f}   "
              f"XG Roller++ {self.xg_err/n:.4f}   "
              f"(PR: {self.sage_err/n*500:.2f} vs {self.xg_err/n*500:.2f})")


def score(benchmark: str, match_length: int = 5) -> dict:
    paths = xbc.paths_for(benchmark, match_length)
    dataset = json.loads(paths.dataset.read_text(encoding="utf-8"))
    by_key = {d["key"]: d for d in dataset["decisions"]}
    xro = _load_jsonl(paths.cache_file("rollout"))
    stage2 = _load_jsonl(paths.dataset.parent / "build" / "stage2_3t.jsonl")

    decs = [by_key[k] for k in xro if by_key.get(k) and by_key[k]["tier"] == "rollout"]
    by_seed = defaultdict(list)
    for d in decs:
        by_seed[d["seed"]].append(d)
    xgpp = {}
    for s in by_seed:
        xgpp.update(xbc.xg_level_picks(paths, "rollerpp", s, by_seed[s]))

    chk = {"sage": Tally(), "xg": Tally(), "disputed": 0,
           "excluded": 0, "no_pick": 0}
    cube = {"sage": Tally(), "xg": Tally(), "disputed": 0}

    for d in decs:
        key = d["key"]
        r = xro[key]
        s2 = stage2.get(key)
        xp = xgpp.get(key)
        if not s2 or not xp:
            continue

        if d["kind"] == "checker":
            if not s2.get("moves") or "checker" not in xp:
                chk["no_pick"] += 1
                continue
            sage_pick = tuple(max(s2["moves"], key=lambda m: m["equity"])["board"])
            xg_pick = xp["checker"]
            if sage_pick == xg_pick:
                continue  # not disputed
            chk["disputed"] += 1

            # Common set: both bots' picks must be rolled by BOTH rollouts, so the
            # XG-rollout and Sage-rollout comparisons run on the identical set of
            # positions (same n) and are directly comparable. The XG rollout rolled
            # both picks by design; the Sage rollout only rolled its own filter
            # survivors, so this drops disputes where a pick (usually XG's) wasn't
            # a Sage rollout candidate.
            xg_rolled = _rolled_map(r["moves"], "rollout")
            sage_rolled = _rolled_map(d["moves"], "Rollout")
            if not (sage_pick in xg_rolled and xg_pick in xg_rolled
                    and sage_pick in sage_rolled and xg_pick in sage_rolled):
                chk["excluded"] += 1
                continue
            for tag, rolled in (("xg", xg_rolled), ("sage", sage_rolled)):
                best_b = max(rolled, key=rolled.get)
                best = rolled[best_b]
                chk[tag].add(sage_pick == best_b, xg_pick == best_b,
                             best - rolled[sage_pick], best - rolled[xg_pick])

        else:  # cube
            if "cube" not in xp:
                continue
            s_sd, s_st = _cube_action(s2["equity_nd"], s2["equity_dt"], s2["equity_dp"])
            x_sd, x_st = xp["cube"]
            s_label = _cube_label(s2["equity_nd"], s2["equity_dt"], s2["equity_dp"])
            # XG Roller++'s label from its own nd/dt/dp isn't in xp; derive from action
            x_label = ("ND" if not x_sd else ("D/T" if x_st else "D/P"))
            if s_label == x_label:
                continue
            cube["disputed"] += 1
            hd, ht = d.get("has_double"), d.get("has_take")
            for tag, nd, dt, dp in (("xg", r["equity_nd"], r["equity_dt"], r["equity_dp"]),
                                    ("sage", d["equity_nd"], d["equity_dt"], d["equity_dp"])):
                opt = _cube_label(nd, dt, dp)
                cube[tag].add(s_label == opt, x_label == opt,
                              _cube_error(nd, dt, dp, hd, ht, s_sd, s_st),
                              _cube_error(nd, dt, dp, hd, ht, x_sd, x_st))

    return {"n_positions": len(decs), "checker": chk, "cube": cube}


def print_report(data: dict, benchmark: str) -> None:
    """Print the dispute report. Shared by main() and xg_benchmark_report."""
    chk, cube = data["checker"], data["cube"]
    print(f"Dispute analysis: Sage 3T vs XG Roller++ ({benchmark})")
    print(f"  {data['n_positions']} positions with both a Sage and an XG full rollout.")
    print("  On positions where the two bots pick DIFFERENT decisions, which bot is")
    print("  closer to each rollout? (Sage rollout vs XG rollout scored separately.)\n")
    scored = chk["xg"].n
    print(f"CHECKER: {chk['disputed']} disputed positions "
          f"({scored} scored on the common set where both picks were rolled by both "
          f"rollouts; {chk['excluded']} excluded)")
    chk["xg"].report("vs XG rollout")
    chk["sage"].report("vs Sage rollout")
    print()
    print(f"CUBE: {cube['disputed']} disputed positions (bots pick different cube actions)")
    cube["xg"].report("vs XG rollout")
    cube["sage"].report("vs Sage rollout")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    args = ap.parse_args(argv)
    print_report(score(args.benchmark, args.match_length), args.benchmark)


if __name__ == "__main__":
    sys.exit(main())
