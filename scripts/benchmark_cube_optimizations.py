"""Cube analytics optimization benchmark.

Evaluates the 11 reference positions in refpos.txt at 3-ply, 4-ply, 2T, 3T
using n_threads=16, captures cubeful ND/DT equities, 5 cubeless probs,
standard errors (rollout only), and wall-clock time.

Writes a JSON baseline file that can be compared against after each
optimization attempt.

Usage:
    python bgsage/scripts/benchmark_cube_optimizations.py --out baseline.json
    python bgsage/scripts/benchmark_cube_optimizations.py --out current.json
    python bgsage/scripts/benchmark_cube_optimizations.py --compare baseline.json current.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Setup paths and CUDA DLL
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CUDA_DLL = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.isdir(_CUDA_DLL):
    os.add_dll_directory(_CUDA_DLL)
sys.path.insert(0, str(_REPO_ROOT / "build"))
sys.path.insert(0, str(_REPO_ROOT / "bgsage" / "python"))

import bgbot_cpp  # noqa: E402
from bgsage import BgBotAnalyzer  # noqa: E402


REFPOS_PATH = _REPO_ROOT / "refpos.txt"

LEVELS = ["3ply", "4ply", "truncated2", "truncated3"]
LEVEL_DISPLAY = {"3ply": "3-ply", "4ply": "4-ply",
                 "truncated2": "2T", "truncated3": "3T"}

N_THREADS = 16
N_TIMING_REPS = 3


def parse_refpos(path: Path) -> list[dict]:
    """Parse refpos.txt into a list of position dicts."""
    positions = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            checkers_str = parts[0]
            meta_str = parts[1]
            board = [int(x) for x in checkers_str.split(",")]
            meta = [int(x) if "." not in x else float(x)
                    for x in meta_str.split(",")]
            # meta: [cube_value, cube_owner, away1, away2, is_crawford, jacoby, beaver]
            cube_value = int(meta[0])
            cube_owner_id = int(meta[1])
            away1 = int(meta[2])
            away2 = int(meta[3])
            is_crawford = bool(meta[4])
            jacoby = bool(meta[5])
            beaver = bool(meta[6])
            owner_map = {0: "centered", 1: "player", 2: "opponent"}
            # For money game, away1/away2 should be 0 (not -1)
            is_money = (away1 < 0 or away2 < 0)
            positions.append({
                "line": lineno,
                "index": len(positions),
                "board": board,
                "cube_value": cube_value,
                "cube_owner": owner_map[cube_owner_id],
                "away1": 0 if is_money else away1,
                "away2": 0 if is_money else away2,
                "is_crawford": is_crawford,
                "jacoby": jacoby if is_money else False,
                "beaver": beaver if is_money else False,
                "is_money": is_money,
            })
    return positions


def make_analyzer(level: str) -> BgBotAnalyzer:
    """Create an analyzer for the given level with n_threads=16."""
    return BgBotAnalyzer(
        eval_level=level,
        parallel_threads=N_THREADS,
    )


def run_cube_action(analyzer: BgBotAnalyzer, pos: dict) -> dict:
    """Run cube_action and capture probs, equities, and (for rollout) SEs.

    For rollout levels we also access the raw rollout result to capture
    prob_std_errors, which BgBotAnalyzer.cube_action does not expose.
    """
    result = analyzer.cube_action(
        pos["board"],
        cube_value=pos["cube_value"],
        cube_owner=pos["cube_owner"],
        away1=pos["away1"],
        away2=pos["away2"],
        is_crawford=pos["is_crawford"],
        jacoby=pos["jacoby"],
        beaver=pos["beaver"],
    )
    out = {
        "equity_nd": result.equity_nd,
        "equity_dt": result.equity_dt,
        "equity_dp": result.equity_dp,
        "cubeless_equity": result.cubeless_equity,
        "probs": [result.probs.win, result.probs.gammon_win,
                  result.probs.backgammon_win, result.probs.gammon_loss,
                  result.probs.backgammon_loss],
        "equity_nd_se": result.equity_nd_se,
        "equity_dt_se": result.equity_dt_se,
        "cubeless_se": result.cubeless_se,
        "optimal_action": result.optimal_action,
    }
    return out


def run_cube_action_raw_rollout(analyzer: BgBotAnalyzer, pos: dict) -> dict:
    """For rollout levels, call the raw C++ rollout to also capture prob SEs."""
    inner = analyzer._analyzer  # _CubefulAnalyzer wrapping _RolloutAnalyzer
    rollout = inner._inner._rollout_strategy  # C++ RolloutStrategy
    owner_map = {
        "centered": bgbot_cpp.CubeOwner.CENTERED,
        "player": bgbot_cpp.CubeOwner.PLAYER,
        "opponent": bgbot_cpp.CubeOwner.OPPONENT,
    }
    r = rollout.cube_decision(
        pos["board"], pos["cube_value"], owner_map[pos["cube_owner"]],
        away1=pos["away1"], away2=pos["away2"], is_crawford=pos["is_crawford"],
        jacoby=pos["jacoby"], beaver=pos["beaver"],
    )
    return {
        "equity_nd": r["equity_nd"],
        "equity_dt": r["equity_dt"],
        "equity_dp": r.get("equity_dp", 1.0),
        "cubeless_equity": r.get("cubeless_equity", 0.0),
        "probs": list(r["probs"]),
        "equity_nd_se": r.get("equity_nd_se"),
        "equity_dt_se": r.get("equity_dt_se"),
        "cubeless_se": r.get("cubeless_se"),
        "prob_std_errors": list(r.get("prob_std_errors", [])) or None,
    }


def is_rollout_level(level: str) -> bool:
    return level.startswith("truncated") or level == "rollout"


def benchmark_level(level: str, positions: list[dict],
                    reps: int = N_TIMING_REPS, verbose: bool = True) -> dict:
    """Run all positions at a given level, reps times, record timing + values."""
    if verbose:
        print(f"\n=== Level: {LEVEL_DISPLAY[level]} ===", flush=True)
    analyzer = make_analyzer(level)

    # Warm-up: run first position once, not timed
    _ = run_cube_action(analyzer, positions[0])

    per_position = []
    for pos in positions:
        times_ns = []
        values = None
        for rep in range(reps):
            t0 = time.perf_counter_ns()
            if is_rollout_level(level):
                v = run_cube_action_raw_rollout(analyzer, pos)
            else:
                v = run_cube_action(analyzer, pos)
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)
            if values is None:
                values = v  # values are deterministic (seed=42), record once
        min_ms = min(times_ns) / 1e6
        median_ms = sorted(times_ns)[len(times_ns) // 2] / 1e6
        per_position.append({
            "index": pos["index"],
            "line": pos["line"],
            "min_ms": min_ms,
            "median_ms": median_ms,
            "values": values,
        })
        if verbose:
            nd = values["equity_nd"]
            dt = values["equity_dt"]
            print(f"  pos {pos['index']:2d} (line {pos['line']:2d}): "
                  f"min={min_ms:8.1f} ms  median={median_ms:8.1f} ms  "
                  f"ND={nd:+.4f}  DT={dt:+.4f}", flush=True)

    total_min = sum(p["min_ms"] for p in per_position)
    total_median = sum(p["median_ms"] for p in per_position)
    if verbose:
        print(f"  total: min={total_min:.1f} ms  median={total_median:.1f} ms",
              flush=True)

    return {
        "level": level,
        "level_display": LEVEL_DISPLAY[level],
        "total_min_ms": total_min,
        "total_median_ms": total_median,
        "positions": per_position,
    }


def run_full_benchmark(out_path: Path, levels: list[str]) -> dict:
    positions = parse_refpos(REFPOS_PATH)
    print(f"Loaded {len(positions)} positions from {REFPOS_PATH}", flush=True)
    print(f"Levels: {[LEVEL_DISPLAY[l] for l in levels]}", flush=True)
    print(f"n_threads={N_THREADS}, timing reps={N_TIMING_REPS}", flush=True)

    t0 = time.perf_counter()
    results = {}
    for level in levels:
        results[level] = benchmark_level(level, positions)
    elapsed = time.perf_counter() - t0

    out = {
        "n_threads": N_THREADS,
        "reps": N_TIMING_REPS,
        "n_positions": len(positions),
        "levels": levels,
        "elapsed_s": elapsed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}", flush=True)
    print(f"Total benchmark time: {elapsed:.1f} s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Tolerance bounds
TOL_EQUITY = 0.01      # ND / DT cubeful equity
TOL_PROB = 0.005       # cubeless probs
SE_REL_TOL = 0.10      # standard errors allowed to grow by at most 10%


def compare(baseline_path: Path, current_path: Path) -> int:
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    any_violation = False
    print(f"\nComparing {current_path.name} vs baseline {baseline_path.name}")
    print("=" * 90)

    for level in baseline["levels"]:
        if level not in current["results"]:
            continue
        b_lvl = baseline["results"][level]
        c_lvl = current["results"][level]
        b_total = b_lvl["total_min_ms"]
        c_total = c_lvl["total_min_ms"]
        speedup = b_total / c_total if c_total > 0 else 0.0
        delta_pct = (c_total - b_total) / b_total * 100.0 if b_total > 0 else 0.0

        print(f"\n[{LEVEL_DISPLAY[level]}] "
              f"total min-time: {b_total:.0f} -> {c_total:.0f} ms "
              f"({delta_pct:+.1f}%, {speedup:.2f}x)")

        max_nd = 0.0
        max_dt = 0.0
        max_prob = 0.0
        max_se_rel = 0.0
        violations = []

        for bp, cp in zip(b_lvl["positions"], c_lvl["positions"]):
            bv = bp["values"]
            cv = cp["values"]
            d_nd = abs(bv["equity_nd"] - cv["equity_nd"])
            d_dt = abs(bv["equity_dt"] - cv["equity_dt"])
            d_probs = [abs(a - b) for a, b in zip(bv["probs"], cv["probs"])]
            d_prob = max(d_probs)
            max_nd = max(max_nd, d_nd)
            max_dt = max(max_dt, d_dt)
            max_prob = max(max_prob, d_prob)

            # SE comparison (only if both have SEs)
            se_rel = 0.0
            for key in ("equity_nd_se", "equity_dt_se", "cubeless_se"):
                bse = bv.get(key)
                cse = cv.get(key)
                if bse is not None and cse is not None and bse > 1e-6:
                    rel = (cse - bse) / bse
                    se_rel = max(se_rel, rel)
            max_se_rel = max(max_se_rel, se_rel)

            msgs = []
            if d_nd > TOL_EQUITY:
                msgs.append(f"ND d={d_nd:.4f}")
            if d_dt > TOL_EQUITY:
                msgs.append(f"DT d={d_dt:.4f}")
            if d_prob > TOL_PROB:
                msgs.append(f"probs d={d_prob:.4f}")
            if se_rel > SE_REL_TOL:
                msgs.append(f"SE rel d={se_rel:+.1%}")
            if msgs:
                violations.append(f"    pos {bp['index']:2d}: " + ", ".join(msgs))

        print(f"  max d:  ND={max_nd:.4f}  DT={max_dt:.4f}  "
              f"probs={max_prob:.4f}  SE rel={max_se_rel:+.1%}")
        tol_ok = (max_nd <= TOL_EQUITY and max_dt <= TOL_EQUITY
                  and max_prob <= TOL_PROB and max_se_rel <= SE_REL_TOL)
        status = "OK" if tol_ok else "VIOLATION"
        print(f"  tolerance: {status}")
        if violations:
            any_violation = True
            for v in violations:
                print(v)

    print("\n" + "=" * 90)
    if any_violation:
        print("RESULT: Tolerance violations detected — change must be reverted.")
        return 1
    print("RESULT: All levels within tolerance.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="Write benchmark results JSON")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"),
                    type=Path, help="Compare two benchmark result files")
    ap.add_argument("--levels", nargs="+", default=LEVELS,
                    choices=LEVELS,
                    help="Subset of levels to benchmark")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    if args.out is None:
        ap.error("--out is required when not using --compare")

    run_full_benchmark(args.out, args.levels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
