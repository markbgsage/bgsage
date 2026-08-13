# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""3T checker-play benchmark on the refpos positions.

Each refpos position gets a fixed random dice roll (generated once with a
fixed seed, re-rolled until at least two legal moves exist, then persisted to
scripts/bench_checker_dice.json so every future run uses the same dice).
Runs BgBotAnalyzer.checker_play at the production 3T level
(eval_level="truncated3"), 16 threads, and reports per-move equities, SEs and
wall-clock time.

Save/compare semantics mirror bench_3t.py: a later run is compared against a
saved baseline move-by-move (matched by post-move board), flagging equity
moves outside max(rollout SE, 0.01) and probability moves outside 0.005.

Usage:
    python scripts/bench_3t_checker.py --save logs/checker_baseline.json
    python scripts/bench_3t_checker.py --compare logs/checker_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "build"
_PY_PKG = _REPO_ROOT / "python"
_REFPOS_DEFAULT = _REPO_ROOT.parent / "refpos.txt"
_DICE_FILE = _REPO_ROOT / "scripts" / "bench_checker_dice.json"
_DICE_SEED = 20260610

if hasattr(os, "add_dll_directory"):
    _cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda):
        os.add_dll_directory(_cuda)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

sys.path.insert(0, str(_BUILD_DIR))
sys.path.insert(0, str(_PY_PKG))

import bgbot_cpp                              # noqa: E402
from bgsage import BgBotAnalyzer              # noqa: E402
from bgsage.weights import PRODUCTION_MODEL   # noqa: E402

N_THREADS = 16
EQUITY_BAND_FLOOR = 0.01
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
                "beaver": bool(meta[6]) if len(meta) > 6 else True,
            })
    return positions


def load_or_generate_dice(positions: list[dict]) -> list[tuple[int, int]]:
    """One fixed dice roll per position, persisted across runs.

    Generated once with a fixed seed; rolls without at least two legal moves
    are re-rolled. The result is saved to bench_checker_dice.json so the
    benchmark inputs never change between runs.
    """
    if _DICE_FILE.exists():
        with open(_DICE_FILE) as f:
            data = json.load(f)
        dice = [tuple(d) for d in data["dice"]]
        if len(dice) == len(positions):
            return dice
        print(f"WARNING: dice file has {len(dice)} entries for "
              f"{len(positions)} positions — regenerating")

    rng = random.Random(_DICE_SEED)
    dice = []
    for pos in positions:
        while True:
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            moves = bgbot_cpp.possible_moves(pos["board"], d1, d2)
            if len(moves) >= 2:
                dice.append((d1, d2))
                break
    with open(_DICE_FILE, "w") as f:
        json.dump({"seed": _DICE_SEED, "dice": [list(d) for d in dice]}, f,
                  indent=2)
    print(f"Generated and saved dice to {_DICE_FILE}")
    return dice


def evaluate(analyzer: BgBotAnalyzer, positions: list[dict],
             dice: list[tuple[int, int]]) -> dict:
    results = []
    total_time = 0.0
    for pos, (d1, d2) in zip(positions, dice):
        t0 = time.perf_counter()
        r = analyzer.checker_play(
            pos["board"], d1, d2,
            cube_value=pos["cube_value"], cube_owner=pos["cube_owner"],
            away1=pos["away1"], away2=pos["away2"],
            is_crawford=pos["is_crawford"],
            jacoby=pos["jacoby"], beaver=pos["beaver"],
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        results.append({
            "die1": d1, "die2": d2, "time": elapsed,
            "moves": [{
                "board": list(m.board),
                "equity": float(m.equity),
                "cubeless_equity": float(m.cubeless_equity),
                "eval_level": m.eval_level,
                "se": float(m.std_error) if m.std_error is not None else None,
                "probs": [float(m.probs.win), float(m.probs.gammon_win),
                          float(m.probs.backgammon_win),
                          float(m.probs.gammon_loss),
                          float(m.probs.backgammon_loss)],
            } for m in r.moves],
        })
    return {"results": results, "total_time": total_time}


def print_results(data: dict) -> None:
    for i, r in enumerate(data["results"]):
        best = r["moves"][0]
        n_ro = sum(1 for m in r["moves"] if m["eval_level"] == "Rollout")
        se = best["se"]
        se_s = f"{se:.4f}" if se is not None else "  n/a "
        print(f"Pos {i+1:2d}: dice {r['die1']}{r['die2']}  "
              f"{len(r['moves']):2d} moves ({n_ro} rolled out)  "
              f"best eq={best['equity']:+.4f} (SE={se_s})  "
              f"time={r['time']:.3f}s")


def compare(baseline: dict, current: dict) -> None:
    n_vals = n_outside = 0
    worst_eq = worst_prob = 0.0
    best_flips = 0
    flagged: list[str] = []
    for i, (b, c) in enumerate(zip(baseline["results"], current["results"])):
        cmap = {tuple(m["board"]): m for m in c["moves"]}
        if tuple(b["moves"][0]["board"]) != tuple(c["moves"][0]["board"]):
            best_flips += 1
            flagged.append(f"  Pos {i+1:2d}: best move changed "
                           f"({b['moves'][0]['equity']:+.4f} vs "
                           f"{c['moves'][0]['equity']:+.4f} for new best)")
        for bm in b["moves"]:
            cm = cmap.get(tuple(bm["board"]))
            if cm is None:
                flagged.append(f"  Pos {i+1:2d}: move missing in current run!")
                n_outside += 1
                continue
            band = EQUITY_BAND_FLOOR
            if bm["se"] is not None:
                band = max(band, bm["se"])
            d = abs(cm["equity"] - bm["equity"])
            n_vals += 1
            worst_eq = max(worst_eq, d)
            if d > band:
                n_outside += 1
                flagged.append(
                    f"  Pos {i+1:2d} [{bm['eval_level']}] eq: "
                    f"{bm['equity']:+.4f} -> {cm['equity']:+.4f} "
                    f"(d={d:.4f} > band {band:.4f})")
            for k in range(5):
                dp = abs(cm["probs"][k] - bm["probs"][k])
                n_vals += 1
                worst_prob = max(worst_prob, dp)
                if dp > PROB_BAND:
                    n_outside += 1
                    flagged.append(
                        f"  Pos {i+1:2d} [{bm['eval_level']}] prob[{k}]: "
                        f"{bm['probs'][k]:.4f} -> {cm['probs'][k]:.4f} "
                        f"(d={dp:.4f} > {PROB_BAND})")
    bt, ct = baseline["total_time"], current["total_time"]
    print("\n" + "=" * 90)
    print("COMPARISON vs baseline (checker play)")
    print("=" * 90)
    print(f"  Baseline total: {bt:.3f}s    Current total: {ct:.3f}s    "
          f"Speedup: {bt / ct:.3f}x  ({bt - ct:+.3f}s)")
    print(f"  Values within band: {n_vals - n_outside}/{n_vals}")
    print(f"  Worst equity delta: {worst_eq:.4f}    "
          f"Worst prob delta: {worst_prob:.4f} (band {PROB_BAND})")
    print(f"  Best-move changes: {best_flips}")
    if flagged:
        print(f"  {len(flagged)} item(s) flagged:")
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
    dice = load_or_generate_dice(positions)

    analyzer = BgBotAnalyzer(eval_level="truncated3",
                             parallel_threads=N_THREADS)

    print(f"3T Checker-Play Benchmark   {args.label}")
    print(f"Model: {PRODUCTION_MODEL}   Threads: {N_THREADS}   "
          f"Positions: {len(positions)}")
    print("=" * 90)

    data = None
    for run in range(args.repeat):
        d = evaluate(analyzer, positions, dice)
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
