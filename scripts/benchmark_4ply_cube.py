"""4-ply cube action benchmark for 11 reference positions.

Evaluates each position at 4-ply with 16 threads, prints outputs and wall-clock time.
On first run, saves baseline values to a JSON file. On subsequent runs, compares
against the baseline and checks accuracy thresholds.

Usage:
    python bgsage/scripts/benchmark_4ply_cube.py           # run benchmark
    python bgsage/scripts/benchmark_4ply_cube.py --save     # force save new baseline
"""

from __future__ import annotations

import json
import os
import sys
import time

# Add CUDA DLLs and build directory before importing bgbot_cpp
if sys.platform == "win32":
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
    if os.path.isdir(build_dir):
        os.add_dll_directory(os.path.abspath(build_dir))

sys.path.insert(0, "build")
sys.path.insert(0, "bgsage/python")

from bgsage import BgBotAnalyzer
from bgsage.weights import PRODUCTION_MODEL

N_THREADS = 16
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "benchmark_4ply_baseline.json")

# Accuracy thresholds: optimizations must not change values by more than this
EQ_THRESHOLD = 0.01     # cubeful equity (ND or DT)
PROB_THRESHOLD = 0.005  # any cubeless probability

# 11 reference positions from refpos.txt
POSITIONS = [
    {
        "board": [0,0,-2,0,2,3,3,3,0,0,0,0,-4,2,0,0,-2,-3,0,-4,1,0,0,0,1,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [0,-2,-3,-1,2,3,3,3,2,2,0,0,-3,0,0,0,0,0,0,-3,-3,0,0,0,0,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [0,0,0,0,0,-3,3,3,2,2,0,0,-4,3,0,0,0,-3,0,-3,2,0,0,-2,0,0],
        "cube_value": 2, "cube_owner": "player",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [1,2,2,2,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-6,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [1,2,-1,2,0,0,3,0,1,0,0,0,-4,5,0,0,0,-4,0,-5,0,0,0,0,2,0],
        "cube_value": 1, "cube_owner": "centered",
        "away1": 5, "away2": 4,
    },
    {
        "board": [0,0,0,1,2,-2,3,2,3,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0],
        "cube_value": 2, "cube_owner": "player",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [0,0,0,0,2,2,2,3,2,0,0,0,0,0,0,0,0,-1,0,-3,-3,-3,2,2,-5,0],
        "cube_value": 2, "cube_owner": "player",
        "jacoby": True, "beaver": True,
    },
    {
        "board": [0,3,1,2,2,2,3,1,1,0,0,-2,0,0,0,0,0,-2,0,-1,0,-2,-2,-3,-3,0],
        "cube_value": 1, "cube_owner": "centered",
        "away1": 3, "away2": 2,
    },
    {
        "board": [0,0,-1,2,2,2,3,0,1,0,1,0,-4,3,0,0,0,-4,0,-4,-1,0,0,-1,0,1],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": False, "beaver": True,
    },
    {
        "board": [0,-2,0,0,0,0,5,0,3,0,0,0,-4,5,0,0,0,-4,0,-4,0,-1,0,0,1,1],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": False,
    },
    {
        "board": [0,3,2,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,-3,-2,-1,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
]


def run_benchmark():
    analyzer = BgBotAnalyzer(eval_level="4ply", parallel_threads=N_THREADS)
    results = []

    for pos in POSITIONS:
        kw = {
            "cube_value": pos["cube_value"],
            "cube_owner": pos["cube_owner"],
            "away1": pos.get("away1", 0),
            "away2": pos.get("away2", 0),
            "is_crawford": pos.get("is_crawford", False),
            "jacoby": pos.get("jacoby", False),
            "beaver": pos.get("beaver", False),
        }
        t0 = time.perf_counter()
        r = analyzer.cube_action(pos["board"], **kw)
        elapsed = time.perf_counter() - t0

        results.append({
            "nd": r.equity_nd,
            "dt": r.equity_dt,
            "probs": [r.probs.win, r.probs.gammon_win, r.probs.backgammon_win,
                      r.probs.gammon_loss, r.probs.backgammon_loss],
            "action": r.optimal_action,
            "time": elapsed,
        })

    return results


def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


def save_baseline(results):
    data = []
    for r in results:
        data.append({
            "nd": round(r["nd"], 6),
            "dt": round(r["dt"], 6),
            "probs": [round(p, 6) for p in r["probs"]],
        })
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Baseline saved to {BASELINE_FILE}")


def main():
    force_save = "--save" in sys.argv

    print(f"4-Ply Cube Action Benchmark ({len(POSITIONS)} positions)")
    print(f"Model: {PRODUCTION_MODEL}, Threads: {N_THREADS}")
    print("=" * 80)

    t0_total = time.perf_counter()
    results = run_benchmark()
    total_time = sum(r["time"] for r in results)

    baseline = load_baseline()
    has_baseline = baseline is not None and not force_save

    max_nd_diff = 0.0
    max_dt_diff = 0.0
    max_prob_diff = 0.0

    for i, r in enumerate(results):
        print(f"\n  Pos {i+1:2d} [{r['time']:6.2f}s]  {r['action']}")
        print(f"    ND={r['nd']:+.4f}  DT={r['dt']:+.4f}")
        p = r["probs"]
        print(f"    Probs: W={p[0]:.4f} GW={p[1]:.4f} BW={p[2]:.4f}"
              f" GL={p[3]:.4f} BL={p[4]:.4f}")

        if has_baseline:
            b = baseline[i]
            nd_diff = abs(r["nd"] - b["nd"])
            dt_diff = abs(r["dt"] - b["dt"])
            prob_diffs = [abs(r["probs"][k] - b["probs"][k]) for k in range(5)]
            max_p = max(prob_diffs)
            max_nd_diff = max(max_nd_diff, nd_diff)
            max_dt_diff = max(max_dt_diff, dt_diff)
            max_prob_diff = max(max_prob_diff, max_p)
            if nd_diff > 0.001 or dt_diff > 0.001 or max_p > 0.001:
                print(f"    vs baseline: dND={nd_diff:.4f} dDT={dt_diff:.4f} dProb={max_p:.4f}")

    print(f"\n{'=' * 80}")
    print(f"Total time:  {total_time:.2f}s")

    if has_baseline:
        print(f"Max diffs vs baseline:  ND={max_nd_diff:.4f}  DT={max_dt_diff:.4f}  Prob={max_prob_diff:.4f}")
        eq_ok = max_nd_diff <= EQ_THRESHOLD and max_dt_diff <= EQ_THRESHOLD
        prob_ok = max_prob_diff <= PROB_THRESHOLD
        if eq_ok and prob_ok:
            print(f"PASS: All values within thresholds (eq<={EQ_THRESHOLD}, prob<={PROB_THRESHOLD})")
        else:
            print("FAIL: Values exceed thresholds!")
            if not eq_ok:
                print(f"  Equity: ND={max_nd_diff:.4f} DT={max_dt_diff:.4f} (limit {EQ_THRESHOLD})")
            if not prob_ok:
                print(f"  Prob: {max_prob_diff:.4f} (limit {PROB_THRESHOLD})")
    else:
        save_baseline(results)


if __name__ == "__main__":
    main()
