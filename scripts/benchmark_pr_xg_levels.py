#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Multi-level PR against the XG full-rollout reference (rollout-tier only).

Scores six evaluation levels -- Sage 1T/2T/3T and XG Roller/Roller+/Roller++ --
against XG's full rollout on the rollout-tier positions we have XG rollout data
for. Everything comes from existing caches; nothing is re-computed:

  Sage 1T pick  <- recovered from scores/sage_1t_clean72.jsonl (error -> move)
  Sage 2T pick  <- recovered from scores/sage_2t_clean.jsonl
  Sage 3T pick  <- build/stage2_3t.jsonl (argmax equity)
  XG Roller++   <- xg/seed_<N>_pp.xg     (batch-analyze #1 move / cube action)
  XG Roller+    <- xg/seed_<N>_p.xg
  XG Roller     <- xg/seed_<N>_roller.xg
  reference     <- xg_results/rollout.jsonl (XG full rollout)

Pick recovery: the Sage score caches store each level's equity error vs the Sage
benchmark reference (best_eq - chosen_eq). The chosen move is the benchmark move
whose equity equals best_eq - error; the cube action is the reference-optimal
action, flipped when the error is non-zero. Ambiguous checker ties (two moves at
the same equity) are left unscored for that level.

Mismatches: a level's pick may not be among the moves XG rolled (the
within-0.02-of-Sage's-best set). Such an off-book pick is charged the biggest
error among the rolled moves (best - worst rolled) -- a conservative penalty
that avoids flattering weaker levels by dropping their bad picks. The mismatch
column counts how many picks got this treatment per level.

Usage:
  python scripts/benchmark_pr_xg_levels.py --benchmark money
"""

import argparse
import json
import sys
from collections import defaultdict

import benchmark_money as bm
import xg_batch_common as xbc
from bgsage import xg_file
from benchmark_pr_xg_reference import _load_jsonl, _xg_checker_reference, _cube_action

SAGE_LEVELS = [
    ("Sage 1T", "sage_1t_clean72"),
    ("Sage 2T", "sage_2t_clean"),
    ("Sage 3T", "stage2_3t"),          # special-cased: direct choices, not error recovery
]
XG_LEVELS = [
    ("XG Roller", "roller"),
    ("XG Roller+", "rollerplus"),
    ("XG Roller++", "rollerpp"),
]


def _recover_cube_action(decision: dict, scored: list):
    """Recover (should_double, should_take) from the cube errors + Sage reference."""
    nd, dt, dp = decision["equity_nd"], decision["equity_dt"], decision["equity_dp"]
    errs = {s["sub"]: s["error"] for s in scored if s["bucket"] == "cube"}
    opt_double = min(dt, dp) > nd
    opt_take = dt <= dp
    sd = (opt_double if errs.get("double", 0.0) == 0.0 else not opt_double)
    st = (opt_take if errs.get("take", 0.0) == 0.0 else not opt_take)
    return sd, st


def score(benchmark: str, match_length: int = 5) -> dict:
    """A pick XG didn't roll (a mismatch) is charged the biggest error among the
    moves XG DID roll (best - worst rolled) rather than dropped, so off-book
    picks are penalised instead of flattering weaker levels. This is a
    conservative lower bound: the pick sat outside Sage's within-0.02 candidate
    set, so its true error is probably larger than the worst rolled move's."""
    paths = xbc.paths_for(benchmark, match_length)
    dataset = json.loads(paths.dataset.read_text(encoding="utf-8"))
    by_key = {d["key"]: d for d in dataset["decisions"]}
    xg_rollout = _load_jsonl(paths.cache_file("rollout"))
    processed = {by_key[k]["seed"] for k in xg_rollout if k in by_key}

    # rollout-tier decisions we can reference
    decisions = [by_key[k] for k in xg_rollout
                 if by_key.get(k) and by_key[k]["tier"] == bm.TIER_ROLLOUT]
    by_seed = defaultdict(list)
    for d in decisions:
        by_seed[d["seed"]].append(d)

    stage2 = _load_jsonl(paths.dataset.parent / "build" / "stage2_3t.jsonl")
    scores_dir = paths.dataset.parent / "scores"
    sage_scores = {name: _load_jsonl(scores_dir / f"{name}.jsonl")
                   for _, name in SAGE_LEVELS if name != "stage2_3t"}

    # XG per-level file picks
    xg_picks = {label: {} for label, _ in XG_LEVELS}
    for label, lvl in XG_LEVELS:
        for seed in sorted(by_seed):
            xg_picks[label].update(xbc.xg_level_picks(paths, lvl, seed, by_seed[seed]))

    players = [lbl for lbl, _ in SAGE_LEVELS] + [lbl for lbl, _ in XG_LEVELS]
    scored = {p: defaultdict(list) for p in players}   # key -> [scored items]
    mismatch = {p: 0 for p in players}
    ambiguous = {p: 0 for p in players}

    for d in decisions:
        key = d["key"]
        if d["kind"] == "checker":
            ref_moves = _xg_checker_reference(xg_rollout[key])
            if not ref_moves:
                continue
            ref_by = {tuple(m["board"]): m["equity"] for m in ref_moves}
            best_eq = ref_moves[0]["equity"]
            # biggest error among the moves XG actually rolled (best - worst)
            worst_avail_err = max(0.0, best_eq - min(ref_by.values()))

            picks = {}
            # Sage levels
            s2 = stage2.get(key)
            if s2 and s2.get("moves"):
                picks["Sage 3T"] = tuple(max(s2["moves"], key=lambda m: m["equity"])["board"])
            for label, name in [("Sage 1T", "sage_1t_clean72"), ("Sage 2T", "sage_2t_clean")]:
                sc = sage_scores[name].get(key)
                if not sc:
                    continue
                cerr = [s["error"] for s in sc["scored"] if s["bucket"] == "checker"]
                if not cerr:
                    continue
                pk = xbc.recover_checker_pick(d, cerr[0])
                if pk is None:
                    ambiguous[label] += 1
                else:
                    picks[label] = pk
            # XG levels
            for label, _ in XG_LEVELS:
                p = xg_picks[label].get(key)
                if p and "checker" in p:
                    picks[label] = p["checker"]

            for p in players:
                pk = picks.get(p)
                if pk is None:
                    continue
                if pk not in ref_by:
                    # off-book pick: penalise with the worst rolled move's error
                    mismatch[p] += 1
                    scored[p][key].append(bm._scored("checker", "checker",
                                                     d.get("game_plan"), worst_avail_err))
                    continue
                scored[p][key].append(bm._scored("checker", "checker",
                                                 d.get("game_plan"), max(0.0, best_eq - ref_by[pk])))
        else:  # cube
            nd, dt, dp = (xg_rollout[key]["equity_nd"], xg_rollout[key]["equity_dt"],
                          xg_rollout[key]["equity_dp"])
            actions = {}
            s2 = stage2.get(key)
            if s2:
                actions["Sage 3T"] = _cube_action(s2["equity_nd"], s2["equity_dt"], s2["equity_dp"])
            for label, name in [("Sage 1T", "sage_1t_clean72"), ("Sage 2T", "sage_2t_clean")]:
                sc = sage_scores[name].get(key)
                if sc:
                    actions[label] = _recover_cube_action(d, sc["scored"])
            for label, _ in XG_LEVELS:
                p = xg_picks[label].get(key)
                if p and "cube" in p:
                    actions[label] = p["cube"]

            for p in players:
                act = actions.get(p)
                if act is None:
                    continue
                sd, st = act
                items = []
                if d.get("has_double"):
                    opt = max(nd, min(dt, dp)); a = min(dt, dp) if sd else nd
                    items.append(bm._scored("cube", "double", d.get("game_plan"), max(0.0, opt - a)))
                if d.get("has_take"):
                    opt = min(dt, dp); a = dt if st else dp
                    items.append(bm._scored("cube", "take", d.get("game_plan"), max(0.0, a - opt)))
                if items:
                    scored[p][key].extend(items)

    def agg(p):
        return bm._aggregate([{"scored": [s]} for its in scored[p].values() for s in its])

    return {
        "processed_seeds": sorted(processed),
        "n_decisions": len(decisions),
        "players": players,
        "results": {p: {"pr": agg(p), "mismatch": mismatch[p], "ambiguous": ambiguous[p]}
                    for p in players},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = score(args.benchmark, args.match_length)
    if args.json:
        print(json.dumps(res, indent=2)); return
    print_report(res, args.benchmark)


def print_report(res: dict, benchmark: str) -> None:
    """Print the eval-level PR table. Shared by main() and xg_benchmark_report."""
    print(f"PR vs XG full rollout ({benchmark}) -- rollout-tier positions only")
    print(f"  {len(res['processed_seeds'])} processed seeds, {res['n_decisions']} decisions")
    print("  lower PR = closer to the full rollout; a level's own pick is always rolled,")
    print("  and the few unrollable picks are charged the biggest rolled-move error.\n")
    hdr = f"{'level':<13}{'total PR':>9}{'checker':>9}{'cube':>8}{'n':>7}{'mism':>6}"
    print(hdr); print("-" * len(hdr))
    for p in res["players"]:
        r = res["results"][p]["pr"]
        print(f"{p:<13}{r['total_pr']:>9.2f}{r['checker_pr']:>9.2f}{r['cube_pr']:>8.2f}"
              f"{r['n_decisions']:>7}{res['results'][p]['mismatch']:>6}")


if __name__ == "__main__":
    sys.exit(main())
