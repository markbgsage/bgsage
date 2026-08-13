# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""3T post-move evaluation benchmark on the refpos positions.

Runs BgBotAnalyzer.post_move_analytics on each refpos board (treated as a
post-move position from the mover's perspective, with the refpos cube/match
state) at the production 3T level, 16 threads. Reports cubeful equity,
cubeless equity, probabilities and wall-clock time; save/compare against a
baseline with the material band (equity max(SE-floor, 0.01), probs 0.005).

Usage:
    python scripts/bench_3t_postmove.py --save logs/postmove_baseline.json
    python scripts/bench_3t_postmove.py --compare logs/postmove_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "build"
_PY_PKG = _REPO_ROOT / "python"
_REFPOS_DEFAULT = _REPO_ROOT.parent / "refpos.txt"

if hasattr(os, "add_dll_directory"):
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PY_PKG))

from bgsage import BgBotAnalyzer              # noqa: E402
from bgsage.weights import PRODUCTION_MODEL   # noqa: E402

N_THREADS = 16
EQUITY_BAND = 0.01
PROB_BAND = 0.005
_OWNER_MAP = {0: "centered", 1: "player", 2: "opponent"}


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
            positions.append({
                "board": checkers,
                "cube_value": meta[0],
                "cube_owner": _OWNER_MAP[meta[1]],
                "away1": meta[2] if meta[2] > 0 else 0,
                "away2": meta[3] if meta[3] > 0 else 0,
                "is_crawford": bool(meta[4]) if len(meta) > 4 else False,
                "jacoby": bool(meta[5]) if len(meta) > 5 else True,
            })
    return positions


def evaluate(analyzer: BgBotAnalyzer, positions: list[dict]) -> dict:
    results = []
    total_time = 0.0
    for pos in positions:
        t0 = time.perf_counter()
        r = analyzer.post_move_analytics(
            pos["board"],
            cube_owner=pos["cube_owner"], cube_value=pos["cube_value"],
            away1=pos["away1"], away2=pos["away2"],
            is_crawford=pos["is_crawford"], jacoby=pos["jacoby"],
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        p = r.probs
        results.append({
            "cubeful": float(r.cubeful_equity),
            "cubeless": float(r.cubeless_equity),
            "cubeful_se": (float(r.cubeful_se)
                           if getattr(r, "cubeful_se", None) is not None else None),
            "cubeless_se": (float(r.cubeless_se)
                            if getattr(r, "cubeless_se", None) is not None else None),
            "probs": [float(p.win), float(p.gammon_win),
                      float(p.backgammon_win), float(p.gammon_loss),
                      float(p.backgammon_loss)],
            "eval_level": r.eval_level,
            "time": elapsed,
        })
    return {"results": results, "total_time": total_time}


def print_results(data: dict) -> None:
    for i, r in enumerate(data["results"]):
        print(f"Pos {i+1:2d}: CF={r['cubeful']:+.4f}  CL={r['cubeless']:+.4f}  "
              f"W={r['probs'][0]:.4f} GW={r['probs'][1]:.4f} "
              f"BW={r['probs'][2]:.4f} GL={r['probs'][3]:.4f} "
              f"BL={r['probs'][4]:.4f}  [{r['eval_level']}]  "
              f"time={r['time']:.3f}s")


def compare(baseline: dict, current: dict) -> None:
    n_vals = n_outside = 0
    worst_eq = worst_prob = 0.0
    flagged: list[str] = []
    for i, (b, c) in enumerate(zip(baseline["results"], current["results"])):
        for key in ("cubeful", "cubeless"):
            d = abs(c[key] - b[key])
            n_vals += 1
            worst_eq = max(worst_eq, d)
            se = b.get(f"{key}_se") or c.get(f"{key}_se")
            band = max(EQUITY_BAND, se) if se else EQUITY_BAND
            if d > band:
                n_outside += 1
                flagged.append(f"  Pos {i+1:2d} {key}: {b[key]:+.4f} -> "
                               f"{c[key]:+.4f}  (d={d:.4f} > band {band:.4f})")
        for k in range(5):
            dp = abs(c["probs"][k] - b["probs"][k])
            n_vals += 1
            worst_prob = max(worst_prob, dp)
            if dp > PROB_BAND:
                n_outside += 1
                flagged.append(f"  Pos {i+1:2d} prob[{k}]: {b['probs'][k]:.4f}"
                               f" -> {c['probs'][k]:.4f}  (d={dp:.4f} > {PROB_BAND})")
    bt, ct = baseline["total_time"], current["total_time"]
    print("\n" + "=" * 90)
    print("COMPARISON vs baseline (post-move eval)")
    print("=" * 90)
    print(f"  Baseline total: {bt:.3f}s    Current total: {ct:.3f}s    "
          f"Speedup: {bt / ct:.3f}x  ({bt - ct:+.3f}s)")
    print(f"  Values within band: {n_vals - n_outside}/{n_vals}")
    print(f"  Worst equity delta: {worst_eq:.4f} (band {EQUITY_BAND})    "
          f"Worst prob delta: {worst_prob:.4f} (band {PROB_BAND})")
    if flagged:
        print(f"  {len(flagged)} value(s) outside band:")
        for line in flagged:
            print(line)
    else:
        print("  All values within band. OK")
    print("\n  Per-position time (baseline -> current):")
    for i, (b, c) in enumerate(zip(baseline["results"], current["results"])):
        print(f"    Pos {i+1:2d}: {b['time']:7.3f}s -> {c['time']:7.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refpos", default=str(_REFPOS_DEFAULT))
    parser.add_argument("--save", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    positions = parse_refpos(Path(args.refpos))
    analyzer = BgBotAnalyzer(eval_level="truncated3",
                             parallel_threads=N_THREADS)

    print(f"3T Post-Move Eval Benchmark   {args.label}")
    print(f"Model: {PRODUCTION_MODEL}   Threads: {N_THREADS}   "
          f"Positions: {len(positions)}")
    print("=" * 90)

    data = None
    for run in range(args.repeat):
        d = evaluate(analyzer, positions)
        if args.repeat > 1:
            print(f"--- run {run+1}/{args.repeat}: total {d['total_time']:.3f}s ---")
        if data is None or d["total_time"] < data["total_time"]:
            data = d

    print_results(data)
    print(f"\nTotal wall-clock time: {data['total_time']:.3f}s   "
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
