#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Score every XG batch-analysis level against the money or match benchmark,
and record XG's picks in the same form the Sage score caches use.

XG has no API, so its decisions come from the ``.xg`` files its Batch Analyze
wrote for the benchmark transcripts, one set per XG level:

    money   data/money_benchmark/xg/seed_<N>{_3p,,_roller,_p,_pp}.xg
            (3-ply, 4-ply, Roller, Roller+, Roller++ -- the plain file is the
            4-ply batch; XG re-analyses in place, so 3-ply was kept as _3p)
    match   data/match_benchmark/<L>pt/xg_snapshots/{ply3,ply4,roller,roller_p,roller_pp}/match_seed_<N>.xg

For each level and decision XG's #1 ranked move (or its cube action) is scored
against the benchmark's saved reference with the benchmark's own formulas, and
the run writes, per level, the resume cache ``scores/xg_<level>.jsonl``
({key, scored}) plus ``scores/xg_<level>.picks.jsonl`` ({key, kind, pick}) --
the same layout the Sage caches use -- so the XG-reference rescoring and the
disputes report read XG's and Sage's picks the same way.

    python scripts/benchmark_pr_xg_levels_all.py --benchmark money
    python scripts/benchmark_pr_xg_levels_all.py --benchmark match --match-length 5
    python scripts/benchmark_pr_xg_levels_all.py --benchmark money --level rollerpp ply3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import benchmark_money as bm  # noqa: E402
import xg_batch_common as xbc  # noqa: E402

XG_LEVELS = [("ply3", "XG 3-ply"), ("ply4", "XG 4-ply"), ("roller", "XG Roller"),
             ("rollerplus", "XG Roller+"), ("rollerpp", "XG Roller++")]


def score_level(paths: xbc.BenchmarkPaths, level: str, decisions: list[dict]) -> tuple[list[dict], dict]:
    """(records, counts) for one XG level over the scoreable decisions."""
    by_seed = xbc.group_by_seed(decisions)
    records, counts = [], {"scored": 0, "no_file": 0, "unmatched": 0, "mismatch": 0}
    for seed, decs in sorted(by_seed.items()):
        if not paths.xg_level_file(level, seed).exists():
            counts["no_file"] += len(decs)
            continue
        picks = xbc.xg_level_picks(paths, level, seed, decs)
        for d in decs:
            p = picks.get(d["key"])
            if p is None:
                counts["unmatched"] += 1
                continue
            plan = d.get("game_plan")
            if d["kind"] == "checker":
                board = list(p["checker"])
                ref = {tuple(m["board"]): m["equity"] for m in d["moves"]}
                eq = ref.get(tuple(board))
                if eq is None:
                    counts["mismatch"] += 1
                    records.append({"key": d["key"], "kind": "checker", "scored": None, "pick": board})
                    continue
                err = max(0.0, d["moves"][0]["equity"] - eq)
                records.append({"key": d["key"], "kind": "checker", "pick": board,
                                "scored": [bm._scored("checker", "checker", plan, err)]})
            else:
                sd, st = p["cube"]
                nd, dt, dp = d["equity_nd"], d["equity_dt"], d["equity_dp"]
                scored = []
                if d.get("has_double"):
                    optimal = max(nd, min(dt, dp)); actual = min(dt, dp) if sd else nd
                    scored.append(bm._scored("cube", "double", plan, max(0.0, optimal - actual)))
                if d.get("has_take"):
                    optimal = min(dt, dp); actual = dt if st else dp
                    scored.append(bm._scored("cube", "take", plan, max(0.0, actual - optimal)))
                records.append({"key": d["key"], "kind": "cube", "scored": scored,
                                "pick": {"should_double": bool(sd), "should_take": bool(st)}})
            counts["scored"] += 1
    return records, counts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["money", "match"], default="money")
    ap.add_argument("--match-length", type=int, default=5)
    ap.add_argument("--level", nargs="+", default=[l for l, _ in XG_LEVELS])
    ap.add_argument("--json", type=Path, default=None, help="also write the per-level breakdowns here")
    args = ap.parse_args(argv)

    paths = xbc.paths_for(args.benchmark, args.match_length)
    data = json.loads(paths.dataset.read_text(encoding="utf-8")) if paths.dataset.exists() else None
    if data is None:
        import gzip
        data = json.loads(gzip.decompress(Path(str(paths.dataset) + ".gz").read_bytes()).decode("utf-8"))
    decisions = [d for d in data["decisions"] if bm._missing_tier(d) is None]
    scores_dir = paths.dataset.parent / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.benchmark}: {len(decisions)} scoreable decisions")
    out = {}
    names = dict(XG_LEVELS)
    for level in args.level:
        records, counts = score_level(paths, level, decisions)
        agg = bm._aggregate(r for r in records if r["scored"] is not None)
        agg.update(level=level, name=names.get(level, level), counts=counts)
        out[level] = agg
        (scores_dir / f"xg_{level}.jsonl").write_text(
            "".join(json.dumps({"key": r["key"], "scored": r["scored"]}, separators=(",", ":")) + "\n"
                    for r in records if r["scored"] is not None), encoding="utf-8")
        (scores_dir / f"xg_{level}.picks.jsonl").write_text(
            "".join(json.dumps({"key": r["key"], "kind": r["kind"], "pick": r["pick"]}, separators=(",", ":")) + "\n"
                    for r in records), encoding="utf-8")
        plans = " ".join(f"{p[:4]} {agg['by_game_plan'][p]['pr']:.2f}" for p in bm.GAME_PLANS)
        print(f"  {names.get(level, level):13s} PR {agg['total_pr']:.2f}  checker {agg['checker_pr']:.2f}  "
              f"cube {agg['cube_pr']:.2f}  n {agg['n_decisions']}  | {plans}  | {counts}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
