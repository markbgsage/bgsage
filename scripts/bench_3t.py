# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Mark Higgins
"""Cube-action rollout evaluation-level benchmark.

Reads the cube-action reference positions from ``refpos.txt`` (one level up
from the bgsage repo root). By default it evaluates production 3T; use
``--full-rollout --checker-eval 3P --cube-eval 1T`` to benchmark a full
outer rollout with independent named checker/cube evaluators. Always uses
16 threads.

Saves results to JSON and compares a later run against a saved baseline,
flagging any value that moved more than the "material" band:

    * equity (ND / DT)       : 0.01
    * probability (W/GW/...)  : 0.005

Usage:
    python scripts/bench_3t.py --save baseline.json
    python scripts/bench_3t.py --compare baseline.json
    python scripts/bench_3t.py --repeat 3 --compare baseline.json
    python scripts/bench_3t.py --full-rollout --checker-eval 3P --cube-eval 1T

Set ``BGSAGE_BUILD_DIR`` to benchmark a non-default extension build.

Runs correctly regardless of the current working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Path / DLL bootstrap (works from any CWD) ────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent          # bgsage repo root
_BUILD_DIR = Path(os.getenv("BGSAGE_BUILD_DIR", _REPO_ROOT / "build")).resolve()
_PY_PKG = _REPO_ROOT / "python"
_REFPOS_DEFAULT = _REPO_ROOT.parent / "refpos.txt"           # parent (bgbot) folder

if hasattr(os, "add_dll_directory"):
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PY_PKG))

import bgbot_cpp                             # noqa: E402
from bgsage import BgBotAnalyzer            # noqa: E402
from bgsage.weights import PRODUCTION_MODEL  # noqa: E402

N_THREADS = 16
EQUITY_BAND = 0.01
PROB_BAND = 0.005
_OWNER_MAP = {0: "centered", 1: "player", 2: "opponent"}

# Canonical production 3T config (the "truncated3" branch in analyzer.py).
# Driven via eval_level="rollout" so individual knobs can be overridden from
# the CLI; verified bit-identical to eval_level="truncated3" for cube_action.
TRUNCATED3 = dict(
    n_trials=360,
    truncation_depth=7,
    decision_ply=3,
    late_ply=2,
    late_threshold=2,
    ultra_late_threshold=9999,
    cubeful_late_threshold=0,
)

FULL_ROLLOUT = dict(
    n_trials=1296,
    truncation_depth=0,
    decision_ply=1,
    truncation_ply=-1,
    late_ply=-1,
    late_threshold=20,
    ultra_late_threshold=9999,
    cubeful_late_threshold=0,
)


def build_analyzer(args) -> BgBotAnalyzer:
    if args.named_level:
        return BgBotAnalyzer(
            eval_level=args.named_level, parallel_threads=N_THREADS)
    if (args.truncated3 and not args.checker_eval and not args.cube_eval
            and not args.full_rollout):
        return BgBotAnalyzer(eval_level="truncated3", parallel_threads=N_THREADS)
    cfg = dict(FULL_ROLLOUT if args.full_rollout else TRUNCATED3)
    if args.n_trials is not None:        cfg["n_trials"] = args.n_trials
    if args.trunc_depth is not None:     cfg["truncation_depth"] = args.trunc_depth
    if args.decision_ply is not None:    cfg["decision_ply"] = args.decision_ply
    if args.late_ply is not None:        cfg["late_ply"] = args.late_ply
    if args.late_threshold is not None:  cfg["late_threshold"] = args.late_threshold
    if args.ultra_late is not None:      cfg["ultra_late_threshold"] = args.ultra_late
    if args.cubeful_late is not None:    cfg["cubeful_late_threshold"] = args.cubeful_late
    if args.checker_ply is not None:
        cfg["checker"] = bgbot_cpp.TrialEvalConfig(ply=args.checker_ply)
    if args.cube_ply is not None:
        cfg["cube"] = bgbot_cpp.TrialEvalConfig(ply=args.cube_ply)
    if args.checker_eval is not None:
        cfg["checker"] = args.checker_eval
    if args.cube_eval is not None:
        cfg["cube"] = args.cube_eval
    return BgBotAnalyzer(eval_level="rollout", parallel_threads=N_THREADS, **cfg)


def parse_refpos(path: Path) -> list[dict]:
    positions: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            checkers = list(map(int, parts[0].split(",")))
            meta = list(map(int, parts[1].split(",")))
            ref = {}
            if len(parts) >= 3:
                v = list(map(float, parts[2].split(",")))
                # v = [ND, DT, win, gw, bw, loss, gl, bl]
                ref = {"nd": v[0], "dt": v[1], "win": v[2], "gw": v[3],
                       "bw": v[4], "gl": v[6], "bl": v[7]}
            positions.append({
                "board": checkers,
                "cube_value": meta[0],
                "cube_owner": _OWNER_MAP[meta[1]],
                "away1": meta[2] if meta[2] > 0 else 0,
                "away2": meta[3] if meta[3] > 0 else 0,
                "is_crawford": bool(meta[4]) if len(meta) > 4 else False,
                "jacoby": bool(meta[5]) if len(meta) > 5 else True,
                "beaver": bool(meta[6]) if len(meta) > 6 else True,
                "ref": ref,
            })
    return positions


def evaluate(analyzer: BgBotAnalyzer, positions: list[dict]) -> dict:
    results = []
    total_time = 0.0
    for pos in positions:
        kw = {
            "cube_value": pos["cube_value"], "cube_owner": pos["cube_owner"],
            "away1": pos["away1"], "away2": pos["away2"],
            "is_crawford": pos["is_crawford"],
            "jacoby": pos["jacoby"], "beaver": pos["beaver"],
        }
        t0 = time.perf_counter()
        r = analyzer.cube_action(pos["board"], **kw)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        p = r.probs
        results.append({
            "action": r.optimal_action,
            "nd": float(r.equity_nd), "dt": float(r.equity_dt),
            "win": float(p.win), "gw": float(p.gammon_win), "bw": float(p.backgammon_win),
            "gl": float(p.gammon_loss), "bl": float(p.backgammon_loss),
            "nd_se": float(r.equity_nd_se) if r.equity_nd_se is not None else None,
            "dt_se": float(r.equity_dt_se) if r.equity_dt_se is not None else None,
            "cl_se": float(r.cubeless_se) if r.cubeless_se is not None else None,
            "time": elapsed,
        })
    return {"results": results, "total_time": total_time}


def warm_up(analyzer: BgBotAnalyzer, board: list[int]) -> None:
    """Touch the selected model and hot code paths without running a rollout."""
    inner = analyzer._analyzer
    if hasattr(inner, "_inner"):
        inner = inner._inner
    strategy = inner._strategy_1ply
    for _ in range(8):
        strategy.evaluate_board(board, board)


def _fmt(x) -> str:
    return f"{x:.4f}" if x is not None else "  n/a "


def print_results(positions: list[dict], data: dict) -> None:
    for i, (pos, r) in enumerate(zip(positions, data["results"])):
        print(f"\nPos {i+1:2d}: {pos['cube_owner']} cv={pos['cube_value']}"
              f" away=({pos['away1']},{pos['away2']}) craw={int(pos['is_crawford'])}"
              f" jac={int(pos['jacoby'])} bvr={int(pos['beaver'])}")
        print(f"  Action: {r['action']}")
        print(f"  ND={r['nd']:+.4f} (SE={_fmt(r['nd_se'])})"
              f"  DT={r['dt']:+.4f} (SE={_fmt(r['dt_se'])})  CL_SE={_fmt(r['cl_se'])}")
        print(f"  Probs: W={r['win']:.4f} GW={r['gw']:.4f} BW={r['bw']:.4f}"
              f" GL={r['gl']:.4f} BL={r['bl']:.4f}")
        print(f"  Time: {r['time']:.3f}s")
        ref = pos.get("ref", {})
        if ref:
            print(f"  Ref(RO): ND={ref['nd']:+.4f} DT={ref['dt']:+.4f}"
                  f"  W={ref['win']:.4f} GW={ref['gw']:.4f} BW={ref['bw']:.4f}"
                  f" GL={ref['gl']:.4f} BL={ref['bl']:.4f}")


_EQUITY_KEYS = ("nd", "dt")
_PROB_KEYS = ("win", "gw", "bw", "gl", "bl")


def compare(baseline: dict, current: dict) -> None:
    base, cur = baseline["results"], current["results"]
    n = min(len(base), len(cur))
    n_values = n_outside = 0
    worst_eq = worst_prob = 0.0
    flagged: list[str] = []
    for i in range(n):
        b, c = base[i], cur[i]
        for k in _EQUITY_KEYS:
            d = abs(c[k] - b[k]); n_values += 1; worst_eq = max(worst_eq, d)
            if d > EQUITY_BAND:
                n_outside += 1
                flagged.append(f"  Pos {i+1:2d} {k.upper():3s}: "
                               f"{b[k]:+.4f} -> {c[k]:+.4f}  (d={d:.4f} > {EQUITY_BAND})")
        for k in _PROB_KEYS:
            d = abs(c[k] - b[k]); n_values += 1; worst_prob = max(worst_prob, d)
            if d > PROB_BAND:
                n_outside += 1
                flagged.append(f"  Pos {i+1:2d} {k.upper():3s}: "
                               f"{b[k]:.4f} -> {c[k]:.4f}  (d={d:.4f} > {PROB_BAND})")
    bt, ct = baseline["total_time"], current["total_time"]
    speedup = bt / ct if ct > 0 else float("inf")
    print("\n" + "=" * 90)
    print("COMPARISON vs baseline")
    print("=" * 90)
    print(f"  Baseline total: {bt:.3f}s    Current total: {ct:.3f}s    "
          f"Speedup: {speedup:.3f}x  ({bt - ct:+.3f}s)")
    print(f"  Values within band: {n_values - n_outside}/{n_values}  "
          f"({100.0 * (n_values - n_outside) / n_values:.1f}%)")
    print(f"  Worst equity delta: {worst_eq:.4f} (band {EQUITY_BAND})    "
          f"Worst prob delta: {worst_prob:.4f} (band {PROB_BAND})")
    if flagged:
        print(f"  {n_outside} value(s) outside band:")
        for line in flagged:
            print(line)
    else:
        print("  All values within band. OK")
    print("\n  Per-position time (baseline -> current):")
    for i in range(n):
        print(f"    Pos {i+1:2d}: {base[i]['time']:7.3f}s -> {cur[i]['time']:7.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refpos", default=str(_REFPOS_DEFAULT))
    parser.add_argument("--save", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the set N times; report the fastest run")
    parser.add_argument("--limit", type=int, default=0,
                        help="benchmark only the first N positions (0 = all)")
    parser.add_argument("--label", default="")
    parser.add_argument("--truncated3", action="store_true",
                        help="use eval_level='truncated3' directly (canonical)")
    parser.add_argument(
        "--named-level",
        choices=("truncated1", "truncated2", "truncated3"),
        help="benchmark a canonical standalone truncated level",
    )
    parser.add_argument("--full-rollout", action="store_true",
                        help="use a full 1296-trial, play-to-completion outer rollout")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--trunc-depth", type=int, default=None)
    parser.add_argument("--decision-ply", type=int, default=None)
    parser.add_argument("--late-ply", type=int, default=None)
    parser.add_argument("--late-threshold", type=int, default=None)
    parser.add_argument("--ultra-late", type=int, default=None)
    parser.add_argument("--cubeful-late", type=int, default=None)
    parser.add_argument("--checker-ply", type=int, default=None)
    parser.add_argument("--cube-ply", type=int, default=None)
    parser.add_argument("--checker-eval", default=None,
                        help="checker evaluator shorthand, e.g. 3P or 2T")
    parser.add_argument("--cube-eval", default=None,
                        help="cube evaluator shorthand, e.g. 3P or 2T")
    args = parser.parse_args()

    refpos_path = Path(args.refpos)
    if not refpos_path.exists():
        print(f"ERROR: refpos.txt not found at {refpos_path}")
        sys.exit(1)
    positions = parse_refpos(refpos_path)
    if args.limit > 0:
        positions = positions[:args.limit]
    analyzer = build_analyzer(args)
    if args.named_level:
        mode = args.named_level
    elif args.truncated3 and not args.full_rollout:
        mode = "truncated3"
    elif args.full_rollout:
        mode = "full rollout"
    else:
        mode = "rollout(config-driven 3T)"
    if args.checker_eval or args.cube_eval:
        mode += (
            f", checker={args.checker_eval or 'inherited'},"
            f" cube={args.cube_eval or 'inherited'}"
        )

    warm_up(analyzer, positions[0]["board"])

    print(f"Cube-Action Rollout Benchmark   {args.label}")
    print(f"Model: {PRODUCTION_MODEL}   Threads: {N_THREADS}   "
          f"Positions: {len(positions)}   Mode: {mode}")
    print(f"Extension: {Path(bgbot_cpp.__file__).resolve()}")
    print(f"refpos: {refpos_path}")
    print("=" * 90)

    data = None
    for run in range(args.repeat):
        d = evaluate(analyzer, positions)
        if args.repeat > 1:
            print(f"--- run {run+1}/{args.repeat}: total {d['total_time']:.3f}s ---")
        if data is None or d["total_time"] < data["total_time"]:
            data = d

    print_results(positions, data)
    print("\n" + "=" * 90)
    print(f"Total wall-clock time: {data['total_time']:.3f}s   "
          f"(avg {data['total_time']/len(positions):.3f}s/pos)")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved results to {args.save}")
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        compare(baseline, data)


if __name__ == "__main__":
    main()
