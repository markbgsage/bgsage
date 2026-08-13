# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Standalone N-ply cubeless analytics benchmark.

Two parts:
1. Post-move cubeless probabilities (MultiPlyStrategy::evaluate_probs) on the
   refpos boards at 2/3/4-ply, with wall-clock timing. Deterministic — the
   compare bands are flat (0.01 equity / 0.005 probability).
2. Benchmark ER (mean millipip error of best_move_index picks vs the GNUbg
   rollout reference) on the contact and race benchmarks at 2-ply, plus a
   deterministic contact subsample at 3-ply. This gates the cubeless N-ply
   move-selection path the same way MODEL_BENCHMARKS.md scores models.

Usage:
    python scripts/bench_nply_cubeless.py --save logs/nply_cubeless_baseline.json
    python scripts/bench_nply_cubeless.py --compare logs/nply_cubeless_baseline.json
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
from bgsage.data import load_benchmark_file                   # noqa: E402
from bgsage.weights import (WeightConfig, PRODUCTION_MODEL,   # noqa: E402
                            bearoff_db_path)

N_THREADS = 16
EQUITY_BAND = 0.01
PROB_BAND = 0.005
PLIES = [2, 3, 4]
ER_BENCHMARKS = [
    # (name, ply, max_scenarios) — step-subsampled deterministically
    ("contact", 2, 0),
    ("race", 2, 0),
    ("contact", 3, 500),
]


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


def make_multipy(w, db, n_plies):
    mp = bgbot_cpp.create_multipy(
        w.strategy_type, w.weight_paths_list, w.hidden_sizes_list,
        n_plies=n_plies, parallel_evaluate=True, parallel_threads=N_THREADS)
    if db is not None:
        bgbot_cpp.multipy_set_bearoff_db(mp, db)
    return mp


def load_scenarios(bm_name, max_scenarios):
    bm_path = str(_REPO_ROOT / "data" / f"{bm_name}.bm")
    full = load_benchmark_file(bm_path)
    n_total = full.size()
    if max_scenarios > 0 and max_scenarios < n_total:
        step = max(1, n_total // max_scenarios)
        sub = load_benchmark_file(bm_path, step=step)
        return sub, sub.size(), n_total
    return full, n_total, n_total


def evaluate(w, db, boards) -> dict:
    out = {"postmove": {}, "er": []}
    for plies in PLIES:
        mp = make_multipy(w, db, plies)
        results = []
        total = 0.0
        for b in boards:
            t0 = time.perf_counter()
            r = mp.evaluate_board(b, b)
            el = time.perf_counter() - t0
            total += el
            results.append({
                "equity": float(r["equity"]),
                "probs": [float(x) for x in r["probs"]],
                "time": el,
            })
        out["postmove"][str(plies)] = {"results": results, "total_time": total}
        mp.clear_cache()

    for bm_name, ply, max_sc in ER_BENCHMARKS:
        scenarios, n, n_total = load_scenarios(bm_name, max_sc)
        mp = make_multipy(w, db, ply)
        t0 = time.perf_counter()
        result = bgbot_cpp.score_benchmarks_multipy(scenarios, mp, N_THREADS)
        el = time.perf_counter() - t0
        out["er"].append({
            "name": bm_name, "ply": ply, "n": n, "n_total": n_total,
            "score": float(result.score()), "time": el,
        })
        mp.clear_cache()
    return out


def print_results(data: dict) -> None:
    for plies in PLIES:
        d = data["postmove"][str(plies)]
        print(f"--- post-move cubeless probs, {plies}-ply "
              f"(total {d['total_time']:.3f}s) ---")
        for i, r in enumerate(d["results"]):
            print(f"  Pos {i+1:2d}: CL={r['equity']:+.4f}  "
                  f"W={r['probs'][0]:.4f} GW={r['probs'][1]:.4f} "
                  f"BW={r['probs'][2]:.4f} GL={r['probs'][3]:.4f} "
                  f"BL={r['probs'][4]:.4f}  time={r['time']:.3f}s")
    print("--- benchmark ER (best_move_index picks) ---")
    for e in data["er"]:
        cnt = f"{e['n']}" if e["n"] == e["n_total"] else f"{e['n']}/{e['n_total']}"
        print(f"  {e['name']:8s} {e['ply']}-ply: ER={e['score']:7.2f}  "
              f"({cnt} scenarios, {e['time']:.1f}s)")


def compare(baseline: dict, current: dict) -> None:
    print("\n" + "=" * 90)
    print("COMPARISON vs baseline (standalone N-ply cubeless)")
    print("=" * 90)
    n_vals = n_outside = 0
    worst_eq = worst_prob = 0.0
    flagged: list[str] = []
    for plies in PLIES:
        b_res = baseline["postmove"][str(plies)]["results"]
        c_res = current["postmove"][str(plies)]["results"]
        for i, (b, c) in enumerate(zip(b_res, c_res)):
            d = abs(c["equity"] - b["equity"])
            n_vals += 1
            worst_eq = max(worst_eq, d)
            if d > EQUITY_BAND:
                n_outside += 1
                flagged.append(f"  {plies}-ply Pos {i+1:2d} equity: "
                               f"{b['equity']:+.4f} -> {c['equity']:+.4f} (d={d:.4f})")
            for k in range(5):
                dp = abs(c["probs"][k] - b["probs"][k])
                n_vals += 1
                worst_prob = max(worst_prob, dp)
                if dp > PROB_BAND:
                    n_outside += 1
                    flagged.append(f"  {plies}-ply Pos {i+1:2d} prob[{k}]: "
                                   f"{b['probs'][k]:.4f} -> {c['probs'][k]:.4f} (d={dp:.4f})")
        bt = baseline["postmove"][str(plies)]["total_time"]
        ct = current["postmove"][str(plies)]["total_time"]
        print(f"  {plies}-ply post-move: {bt:.3f}s -> {ct:.3f}s  "
              f"(speedup {bt / ct:.3f}x)")
    print(f"  Values within band: {n_vals - n_outside}/{n_vals}    "
          f"Worst equity delta: {worst_eq:.4f} (band {EQUITY_BAND})    "
          f"Worst prob delta: {worst_prob:.4f} (band {PROB_BAND})")
    if flagged:
        print(f"  {len(flagged)} value(s) outside band:")
        for line in flagged:
            print(line)
    print("\n  Benchmark ER (baseline -> current; lower is better):")
    for b, c in zip(baseline["er"], current["er"]):
        print(f"    {b['name']:8s} {b['ply']}-ply: {b['score']:7.2f} -> "
              f"{c['score']:7.2f}  (delta {c['score'] - b['score']:+.2f})   "
              f"time {b['time']:.1f}s -> {c['time']:.1f}s "
              f"(speedup {b['time'] / c['time']:.3f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refpos", default=str(_REFPOS_DEFAULT))
    parser.add_argument("--save", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    boards = parse_refpos(Path(args.refpos))
    w = WeightConfig.default()
    db = bgbot_cpp.BearoffDB()
    if not db.load(bearoff_db_path()):
        sys.exit("FATAL: bearoff DB failed to load — results not comparable")

    print(f"Standalone N-ply Cubeless Benchmark   {args.label}")
    print(f"Model: {PRODUCTION_MODEL}   Threads: {N_THREADS}   "
          f"Positions: {len(boards)}")
    print("=" * 90)

    data = evaluate(w, db, boards)
    print_results(data)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved results to {args.save}")
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        compare(baseline, data)


if __name__ == "__main__":
    main()
