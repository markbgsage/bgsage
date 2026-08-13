#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Benchmark PR scored against XG references instead of Sage references.

The normal benchmark scores a bot against Sage's own tiered reference (Sage 3T
for 3T-tier decisions, Sage full rollout for rollout-tier decisions). This
script swaps the reference to XG's data:

    3T-tier decision      -> XG Roller++ result (xg_results/rollerpp.jsonl)
    rollout-tier decision -> XG full rollout   (xg_results/rollout.jsonl)

and scores two players against it:

    Sage 3T      -- Sage's truncated-3 pick (argmax equity in build/stage2_3t.jsonl)
    XG Roller++  -- XG's Roller++ pick (argmax equity in xg_results/rollerpp.jsonl)

Only decisions from games that have XG *rollout* data (the seeds present in
rollout.jsonl) are considered; 3P-tier decisions are excluded (no XG reference).

Error formulas are the benchmark's own (bm._scored / bm._aggregate), so the PR
numbers are directly comparable to the Sage-referenced benchmark PRs.

Caveats surfaced in the report:
  * On 3T-tier decisions XG Roller++ IS the reference, so it scores ~0 there by
    construction. The only head-to-head where neither player is the reference is
    the rollout tier. The report breaks PR down by tier for exactly this reason.
  * For rollout-tier decisions XG only rolled the moves within 0.02 of Sage's
    best, so a pick outside that set (a mismatch) is charged the biggest error
    among the rolled moves (best - worst rolled) -- a conservative penalty, not
    dropped, so weaker levels aren't flattered.

Usage:
  python scripts/benchmark_pr_xg_reference.py --benchmark money
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import benchmark_money as bm
import xg_batch_common as xbc


def _load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def _checker_pick(record: dict, level_filter: str | None = None) -> tuple | None:
    """argmax-equity move board from a cache/stage record's move list."""
    moves = record.get("moves") or []
    if level_filter:
        moves = [m for m in moves if m.get("eval_level") == level_filter] or moves
    if not moves:
        return None
    return tuple(max(moves, key=lambda m: m["equity"])["board"])


def _xg_checker_reference(record: dict) -> list[dict] | None:
    """Reference move list (board+equity, best first) from an XG cache record.

    Rollout records: only the rolled moves carry an XG rollout equity, so the
    reference is restricted to those. Roller++ records: all XG-stored moves.
    """
    rolled = [m for m in record["moves"] if m.get("eval_level") == "rollout"]
    moves = rolled if rolled else record["moves"]
    if not moves:
        return None
    return sorted(({"board": list(m["board"]), "equity": m["equity"]} for m in moves),
                  key=lambda m: m["equity"], reverse=True)


def _cube_action(nd: float, dt: float, dp: float) -> tuple[bool, bool]:
    return (min(dt, dp) > nd, dt <= dp)


def score(benchmark: str, match_length: int = 5) -> dict:
    paths = xbc.paths_for(benchmark, match_length)
    dataset = json.loads(paths.dataset.read_text(encoding="utf-8"))
    by_key = {d["key"]: d for d in dataset["decisions"]}

    stage2 = _load_jsonl(paths.dataset.parent / "build" / "stage2_3t.jsonl")
    xg_rollout = _load_jsonl(paths.cache_file("rollout"))
    xg_rollerpp = _load_jsonl(paths.cache_file("rollerpp"))

    processed_seeds = {by_key[k]["seed"] for k in xg_rollout if k in by_key}

    # per-bot, per-tier, per-decision scored items; plus mismatch counts.
    # keyed by decision so a fair (common-set) comparison can be aggregated.
    scored = {b: {"3T": defaultdict(list), "rollout": defaultdict(list)}
              for b in ("sage3t", "xgpp")}
    mismatch = {b: defaultdict(int) for b in ("sage3t", "xgpp")}
    n_considered = 0

    for d in dataset["decisions"]:
        if d.get("seed") not in processed_seeds:
            continue
        tier = d.get("tier")
        if tier == bm.TIER_ROLLOUT and d["key"] in xg_rollout:
            ref_rec, ref_tier = xg_rollout[d["key"]], "rollout"
        elif tier == bm.TIER_3T and d["key"] in xg_rollerpp:
            ref_rec, ref_tier = xg_rollerpp[d["key"]], "3T"
        else:
            continue  # 3P, or a rollout-tier decision not yet rolled
        n_considered += 1

        s2 = stage2.get(d["key"])
        xgpp_rec = xg_rollerpp.get(d["key"])
        if s2 is None or xgpp_rec is None:
            continue

        if d["kind"] == "checker":
            ref_moves = _xg_checker_reference(ref_rec)
            if not ref_moves:
                continue
            ref_by_board = {tuple(m["board"]): m["equity"] for m in ref_moves}
            best_eq = ref_moves[0]["equity"]
            # off-book picks are charged the biggest error among the rolled moves
            worst_avail_err = max(0.0, best_eq - min(ref_by_board.values()))
            picks = {"sage3t": _checker_pick(s2), "xgpp": _checker_pick(xgpp_rec)}
            for bot, pick in picks.items():
                if pick is None:
                    continue  # pick unknown (no stored moves) -> not scoreable
                err = (worst_avail_err if pick not in ref_by_board
                       else max(0.0, best_eq - ref_by_board[pick]))
                if pick not in ref_by_board:
                    mismatch[bot][ref_tier] += 1
                scored[bot][ref_tier][d["key"]].append(
                    bm._scored("checker", "checker", d.get("game_plan"), err))
        else:  # cube
            nd, dt, dp = ref_rec["equity_nd"], ref_rec["equity_dt"], ref_rec["equity_dp"]
            actions = {
                "sage3t": _cube_action(s2["equity_nd"], s2["equity_dt"], s2["equity_dp"]),
                "xgpp": (xgpp_rec["should_double"], xgpp_rec["should_take"]),
            }
            for bot, (sd, stk) in actions.items():
                items = []
                if d.get("has_double"):
                    opt = max(nd, min(dt, dp))
                    act = min(dt, dp) if sd else nd
                    items.append(bm._scored("cube", "double", d.get("game_plan"),
                                            max(0.0, opt - act)))
                if d.get("has_take"):
                    opt = min(dt, dp)
                    act = dt if stk else dp
                    items.append(bm._scored("cube", "take", d.get("game_plan"),
                                            max(0.0, act - opt)))
                if items:
                    scored[bot][ref_tier][d["key"]].extend(items)

    def agg(items):
        return bm._aggregate([{"scored": [s]} for s in items])

    def flat(bot, tiers, keys=None):
        items = []
        for tier in tiers:
            for k, its in scored[bot][tier].items():
                if keys is None or k in keys:
                    items += its
        return items

    # common (both-scoreable) decision set per tier, for a fair head-to-head
    common = {tier: set(scored["sage3t"][tier]) & set(scored["xgpp"][tier])
              for tier in ("3T", "rollout")}

    out = {"processed_seeds": sorted(processed_seeds), "n_considered": n_considered,
           "bots": {}, "common_rollout": {}}
    for bot in ("sage3t", "xgpp"):
        by_tier = {}
        for tier in ("3T", "rollout"):
            by_tier[tier] = {"pr": agg(flat(bot, [tier])),
                             "mismatch": mismatch[bot][tier]}
        out["bots"][bot] = {"combined": agg(flat(bot, ["3T", "rollout"])),
                            "by_tier": by_tier}
        out["common_rollout"][bot] = agg(flat(bot, ["rollout"], common["rollout"]))
    out["common_rollout"]["n_keys"] = len(common["rollout"])
    return out


def _fmt(a: dict) -> str:
    return (f"PR={a['total_pr']:.2f} (checker={a['checker_pr']:.2f} "
            f"cube={a['cube_pr']:.2f}) n={a['n_decisions']} "
            f"[{a['n_checker']}c+{a['n_cube']}q] blunders={a['blunders']['total']}")


NAMES = {"sage3t": "Sage 3T", "xgpp": "XG Roller++"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="dump raw result JSON")
    args = ap.parse_args(argv)

    res = score(args.benchmark, args.match_length)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    seeds = res["processed_seeds"]
    print(f"\nXG-referenced benchmark PR ({args.benchmark})")
    print(f"  reference: 3T-tier -> XG Roller++, rollout-tier -> XG full rollout")
    print(f"  {len(seeds)} processed seeds: {seeds}")
    print(f"  {res['n_considered']} scoreable decisions (3T + rollout tiers)\n")
    for bot in ("sage3t", "xgpp"):
        b = res["bots"][bot]
        print(f"{NAMES[bot]}:")
        print(f"  combined : {_fmt(b['combined'])}")
        for tier in ("3T", "rollout"):
            t = b["by_tier"][tier]
            note = "  <- XG Roller++ IS the reference here (PR~0 by construction)" \
                if (bot == "xgpp" and tier == "3T") else ""
            print(f"  {tier:>7} : {_fmt(t['pr'])} mismatch={t['mismatch']}{note}")
        print()
    print("Head-to-head on the rollout tier (the only tier where neither player is "
          "the reference),")
    print(f"restricted to the {res['common_rollout']['n_keys']} decisions BOTH players "
          "can be scored on:")
    for bot in ("sage3t", "xgpp"):
        print(f"  {NAMES[bot]:>12}: {_fmt(res['common_rollout'][bot])}")


if __name__ == "__main__":
    sys.exit(main())
