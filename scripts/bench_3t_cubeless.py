# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""3T cubeless rollout benchmark on the refpos positions.

Runs a pure cubeless truncated rollout (the post-move cubeless probabilities
calculation: RolloutStrategy::rollout_position via the evaluate_board binding)
on each refpos board, treated as a post-move position from the mover's
perspective, at the production 3T trial config, 16 threads, fixed seed.
Reports cubeless equity, probabilities and standard errors; save/compare
against a baseline with the material band (equity max(SE, 0.01), probs
max(SE, 0.005)).

Usage:
    python scripts/bench_3t_cubeless.py --save logs/cubeless_baseline.json
    python scripts/bench_3t_cubeless.py --compare logs/cubeless_baseline.json
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

import bgbot_cpp                                              # noqa: E402
from bgsage.weights import (WeightConfig, PRODUCTION_MODEL,   # noqa: E402
                            bearoff_db_path)

N_THREADS = 16
EQUITY_BAND = 0.01
PROB_BAND = 0.005

# Production 3T trial config (matches analyzer's truncated3), cubeless.
CONFIG = dict(
    n_trials=360,
    truncation_depth=7,
    decision_ply=3,
    late_ply=2,
    late_threshold=2,
    ultra_late_threshold=9999,
    n_threads=N_THREADS,
    seed=42,
    enable_vr=True,
)


def parse_refpos(path: Path) -> list[list[int]]:
    boards: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            boards.append(list(map(int, parts[0].split(","))))
    return boards


def make_strategy():
    w = WeightConfig.default()
    strat = bgbot_cpp.create_rollout(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list, **CONFIG)
    db = bgbot_cpp.BearoffDB()
    if db.load(bearoff_db_path()):
        bgbot_cpp.rollout_set_bearoff_db(strat, db)
        return strat, db   # keep db alive
    print("WARNING: bearoff DB not loaded")
    return strat, None


def evaluate(strat, boards: list[list[int]]) -> dict:
    results = []
    total_time = 0.0
    for b in boards:
        t0 = time.perf_counter()
        r = strat.evaluate_board(b, b)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        results.append({
            "equity": float(r["equity"]),
            "se": float(r["std_error"]),
            "probs": [float(x) for x in r["probs"]],
            "prob_ses": [float(x) for x in r["prob_std_errors"]],
            "time": elapsed,
        })
    return {"results": results, "total_time": total_time}


def print_results(data: dict) -> None:
    for i, r in enumerate(data["results"]):
        print(f"Pos {i+1:2d}: CL={r['equity']:+.4f} (SE={r['se']:.4f})  "
              f"W={r['probs'][0]:.4f} GW={r['probs'][1]:.4f} "
              f"BW={r['probs'][2]:.4f} GL={r['probs'][3]:.4f} "
              f"BL={r['probs'][4]:.4f}  time={r['time']:.3f}s")


def compare(baseline: dict, current: dict) -> None:
    n_vals = n_outside = 0
    worst_eq = worst_prob = 0.0
    flagged: list[str] = []
    for i, (b, c) in enumerate(zip(baseline["results"], current["results"])):
        d = abs(c["equity"] - b["equity"])
        n_vals += 1
        worst_eq = max(worst_eq, d)
        band = max(EQUITY_BAND, b["se"])
        if d > band:
            n_outside += 1
            flagged.append(f"  Pos {i+1:2d} equity: {b['equity']:+.4f} -> "
                           f"{c['equity']:+.4f}  (d={d:.4f} > band {band:.4f})")
        for k in range(5):
            dp = abs(c["probs"][k] - b["probs"][k])
            n_vals += 1
            worst_prob = max(worst_prob, dp)
            pband = max(PROB_BAND, b["prob_ses"][k])
            if dp > pband:
                n_outside += 1
                flagged.append(f"  Pos {i+1:2d} prob[{k}]: {b['probs'][k]:.4f}"
                               f" -> {c['probs'][k]:.4f}  (d={dp:.4f} > {pband:.4f})")
    bt, ct = baseline["total_time"], current["total_time"]
    print("\n" + "=" * 90)
    print("COMPARISON vs baseline (cubeless rollout)")
    print("=" * 90)
    print(f"  Baseline total: {bt:.3f}s    Current total: {ct:.3f}s    "
          f"Speedup: {bt / ct:.3f}x  ({bt - ct:+.3f}s)")
    print(f"  Values within band: {n_vals - n_outside}/{n_vals}")
    print(f"  Worst equity delta: {worst_eq:.4f} (band floor {EQUITY_BAND})    "
          f"Worst prob delta: {worst_prob:.4f} (band floor {PROB_BAND})")
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

    boards = parse_refpos(Path(args.refpos))
    strat, _db = make_strategy()

    print(f"3T Cubeless Rollout Benchmark   {args.label}")
    print(f"Model: {PRODUCTION_MODEL}   Threads: {N_THREADS}   "
          f"Positions: {len(boards)}   Trials: {CONFIG['n_trials']}")
    print("=" * 90)

    data = None
    for run in range(args.repeat):
        d = evaluate(strat, boards)
        if args.repeat > 1:
            print(f"--- run {run+1}/{args.repeat}: total {d['total_time']:.3f}s ---")
        if data is None or d["total_time"] < data["total_time"]:
            data = d

    print_results(data)
    print(f"\nTotal wall-clock time: {data['total_time']:.3f}s   "
          f"(avg {data['total_time']/len(boards):.3f}s/pos)")

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
