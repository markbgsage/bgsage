#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Every evaluator's PR against XG's OWN tiered reference on the money benchmark.

The benchmark grades against Sage's tiered reference (3-ply, 3T or a full
rollout per decision, by closeness). This mirrors it with XG's analysis as the
truth at every tier, so the ranking can be judged by XG's numbers as well:

    3P-tier decision      -> XG 3-ply   (the seed_<N>_3p.xg batch: XG's move list
                                         with equities / its cube equities)
    3T-tier decision      -> XG Roller++ (xg_results/rollerpp.jsonl)
    rollout-tier decision -> XG rollout  (xg_results/rollout.jsonl)

and scores every evaluator over the same decisions -- Sage at each level from
its recorded picks (``scores/sage_<level>_<suffix>.picks.jsonl``, written by
the parent repo's distributed scorer or any scorer that records picks) and XG
at each batch level from ``scores/xg_<level>.picks.jsonl``
(``benchmark_pr_xg_levels_all.py``). Error formulas are the benchmark's own, so
the PRs are directly comparable with the Sage-referenced table.

A pick outside the moves XG evaluated at that tier (XG rolls / re-evaluates
only the moves within 0.02 of Sage's best) is charged the biggest error among
the moves it did evaluate -- a conservative penalty rather than a dropped
decision, so weaker levels are not flattered. The mismatch count is reported.

    python scripts/benchmark_pr_xg_reference_all.py --sage-suffix stage11
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import benchmark_money as bm  # noqa: E402
import xg_batch_common as xbc  # noqa: E402
from bgsage import xg_file  # noqa: E402

SAGE_LEVELS = [("truncated3", "Sage 3T"), ("truncated2", "Sage 2T"), ("truncated1", "Sage 1T"),
               ("4ply", "Sage 4P"), ("3ply", "Sage 3P"), ("2ply", "Sage 2P"), ("1ply", "Sage 1P")]
XG_LEVELS = [("rollerpp", "XG Roller++"), ("rollerplus", "XG Roller+"), ("roller", "XG Roller"),
             ("ply4", "XG 4-ply"), ("ply3", "XG 3-ply")]
ROW_ORDER = ["Sage 3T", "XG Roller++", "Sage 2T", "XG Roller+", "Sage 1T", "XG Roller",
             "Sage 4P", "XG 4-ply", "Sage 3P", "XG 3-ply", "Sage 2P", "Sage 1P"]


def _load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); out[r["key"]] = r
    return out


def _xg_3ply_reference(paths: xbc.BenchmarkPaths, decisions: list[dict]) -> dict[str, dict]:
    """{key: {'moves': [{board, equity}], ...} | {'equity_nd','equity_dt','equity_dp'}}
    from XG's 3-ply batch files, for the 3P-tier decisions."""
    out: dict[str, dict] = {}
    for seed, decs in sorted(xbc.group_by_seed(decisions).items()):
        path = paths.xg_level_file("ply3", seed)
        if not path.exists():
            continue
        tx = xg_file.XgArchive.load(path).get("temp.xg")
        index = xbc.GameFileIndex(tx)
        for d in decs:
            off = xbc.find_decision_record(index, paths, d)
            if off is None:
                continue
            if d["kind"] == "checker":
                rec = xg_file.parse_move_record(tx, off)
                moves = [{"board": list(m["board"]), "equity": float(m["eval"][6])} for m in rec["moves"]]
                if moves:
                    out[d["key"]] = {"moves": sorted(moves, key=lambda m: -m["equity"])}
            else:
                rec = xg_file.parse_cube_record(tx, off)
                if rec.get("flag_double", -100) != -100 or rec.get("equity_nd"):
                    out[d["key"]] = {"equity_nd": rec["equity_nd"], "equity_dt": rec["equity_dt"],
                                     "equity_dp": rec["equity_dp"]}
    return out


def _ref_moves(record: dict, tier: str) -> list[dict] | None:
    moves = record.get("moves") or []
    if tier == "rollout":
        rolled = [m for m in moves if m.get("eval_level") == "rollout"]
        moves = rolled if rolled else moves
    if not moves:
        return None
    return sorted(({"board": list(m["board"]), "equity": m["equity"]} for m in moves), key=lambda m: -m["equity"])


def score(sage_suffix: str, match_length: int = 5, tiers: tuple = ("3P", "3T", "rollout")) -> dict:
    paths = xbc.paths_for("money", match_length)
    data = json.loads(paths.dataset.read_text(encoding="utf-8"))
    decisions = [d for d in data["decisions"] if bm._missing_tier(d) is None]
    tier_name = {bm.TIER_ROLLOUT: "rollout", bm.TIER_3T: "3T", bm.TIER_3P: "3P"}
    decisions = [d for d in decisions if tier_name.get(d.get("tier")) in tiers]
    by_tier = defaultdict(list)
    for d in decisions:
        by_tier[d.get("tier")].append(d)
    refs = {"rollout": _load_jsonl(paths.cache_file("rollout")),
            "3T": _load_jsonl(paths.cache_file("rollerpp")),
            "3P": _xg_3ply_reference(paths, by_tier.get(bm.TIER_3P, []))}
    scores_dir = paths.dataset.parent / "scores"
    picks = {}
    for level, name in SAGE_LEVELS:
        picks[name] = _load_jsonl(scores_dir / f"sage_{level}_{sage_suffix}.picks.jsonl")
    for level, name in XG_LEVELS:
        picks[name] = _load_jsonl(scores_dir / f"xg_{level}.picks.jsonl")
    evaluators = [n for n in ROW_ORDER if picks.get(n)]

    records = {n: [] for n in evaluators}
    mismatch = {n: 0 for n in evaluators}
    n_by_tier = defaultdict(int)
    for d in decisions:
        tier = {bm.TIER_ROLLOUT: "rollout", bm.TIER_3T: "3T", bm.TIER_3P: "3P"}.get(d.get("tier"))
        ref = refs.get(tier, {}).get(d["key"]) if tier else None
        if ref is None:
            continue
        n_by_tier[tier] += 1
        plan = d.get("game_plan")
        if d["kind"] == "checker":
            moves = _ref_moves(ref, tier)
            if not moves:
                n_by_tier[tier] -= 1
                continue
            by_board = {tuple(m["board"]): m["equity"] for m in moves}
            best = moves[0]["equity"]
            worst_err = max(0.0, best - min(by_board.values()))
            for n in evaluators:
                p = picks[n].get(d["key"])
                if not p or p.get("kind") != "checker" or p.get("pick") is None:
                    continue
                b = tuple(p["pick"])
                if b in by_board:
                    err = max(0.0, best - by_board[b])
                else:
                    err = worst_err; mismatch[n] += 1
                records[n].append({"key": d["key"], "scored": [bm._scored("checker", "checker", plan, err)]})
        else:
            nd, dt, dp = ref["equity_nd"], ref["equity_dt"], ref["equity_dp"]
            for n in evaluators:
                p = picks[n].get(d["key"])
                if not p or p.get("kind") != "cube" or p.get("pick") is None:
                    continue
                sd, st = p["pick"]["should_double"], p["pick"]["should_take"]
                scored = []
                if d.get("has_double"):
                    optimal = max(nd, min(dt, dp)); actual = min(dt, dp) if sd else nd
                    scored.append(bm._scored("cube", "double", plan, max(0.0, optimal - actual)))
                if d.get("has_take"):
                    optimal = min(dt, dp); actual = dt if st else dp
                    scored.append(bm._scored("cube", "take", plan, max(0.0, actual - optimal)))
                records[n].append({"key": d["key"], "scored": scored})
    out = {"n_by_tier": dict(n_by_tier), "rows": {}}
    for n in evaluators:
        agg = bm._aggregate(records[n]); agg["mismatches"] = mismatch[n]
        out["rows"][n] = agg
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sage-suffix", default="stage11", help="Sage picks label suffix (sage_<level>_<suffix>.picks.jsonl)")
    ap.add_argument("--tiers", nargs="+", default=["3P", "3T", "rollout"], help="which reference tiers to include")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = score(args.sage_suffix, tiers=tuple(args.tiers))
    print(f"money benchmark scored against XG's tiered reference: {res['n_by_tier']}")
    print(f"{'Bot':13s} {'PR':>6s} {'Chk':>6s} {'Cube':>6s} " + " ".join(f"{p[:5]:>6s}" for p in bm.GAME_PLANS) + "   n   mism")
    for n, a in res["rows"].items():
        plans = " ".join(f"{a['by_game_plan'][p]['pr']:6.2f}" for p in bm.GAME_PLANS)
        print(f"{n:13s} {a['total_pr']:6.2f} {a['checker_pr']:6.2f} {a['cube_pr']:6.2f} {plans} {a['n_decisions']:6d} {a['mismatches']:5d}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
