"""Cube analysis profiling benchmark.

Evaluates 8 fixed positions at a given evaluation level and prints results
with timing.  Use for regression testing: values should not change materially
across code changes, while wall-clock time should decrease with optimisations.

Usage:
    python bgsage/scripts/profile_cube_benchmark.py 3ply
    python bgsage/scripts/profile_cube_benchmark.py 2T

Supported levels: 1ply, 2ply, 3ply, 4ply, 1T, 2T, 3T
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "build")
sys.path.insert(0, "bgsage/python")

from bgsage import BgBotAnalyzer
from bgsage.weights import PRODUCTION_MODEL

# ── Positions ──────────────────────────────────────────────────────────────

POSITIONS = [
    {
        "label": "Bearoff race, centered, money",
        "board": [0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,-1,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "label": "Bearoff race, centered, 5pt match 0-0",
        "board": [0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,-1,0],
        "cube_value": 1, "cube_owner": "centered",
        "away1": 5, "away2": 5,
    },
    {
        "label": "Complex contact, player cube=2, money",
        "board": [0,0,0,0,0,-3,4,2,3,0,0,0,-4,4,-1,-1,2,-2,0,-4,0,0,0,0,0,0],
        "cube_value": 2, "cube_owner": "player",
        "jacoby": True, "beaver": True,
    },
    {
        "label": "Complex contact, player cube=4, 7pt match 0-2",
        "board": [0,0,0,2,2,-2,4,0,2,0,0,0,-4,3,0,0,0,-2,0,-3,0,2,-2,-2,0,0],
        "cube_value": 4, "cube_owner": "player",
        "away1": 7, "away2": 5,
    },
    {
        "label": "Mixed contact, centered, money",
        "board": [0,4,1,0,0,3,3,0,1,2,0,0,0,0,1,-2,0,-3,0,-6,-2,0,-1,-1,0,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "label": "Late race, centered, money",
        "board": [1,2,2,2,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,-3,-4,0],
        "cube_value": 1, "cube_owner": "centered",
        "jacoby": True, "beaver": True,
    },
    {
        "label": "Contact, player cube=2, 4pt match 0-1",
        "board": [0,0,2,2,2,3,3,0,1,0,0,0,-4,0,0,0,0,-3,0,-3,1,-2,-2,-1,0,0],
        "cube_value": 2, "cube_owner": "player",
        "away1": 4, "away2": 3,
    },
    {
        "label": "Contact, player cube=2, money",
        "board": [0,0,0,0,0,0,4,1,3,0,0,0,-4,3,-2,-1,0,0,0,-3,-2,-3,2,0,2,0],
        "cube_value": 2, "cube_owner": "player",
        "jacoby": True, "beaver": True,
    },
]

# ── Truncated rollout presets (XG Roller equivalents) ──────────────────────

TRUNCATED_PRESETS = {
    "1T": {"n_trials": 42, "truncation_depth": 5, "decision_ply": 1,
            "late_ply": -1, "late_threshold": 20},
    "2T": {"n_trials": 360, "truncation_depth": 7, "decision_ply": 2,
            "late_ply": 1, "late_threshold": 2},
    "3T": {"n_trials": 360, "truncation_depth": 5, "decision_ply": 3,
            "late_ply": 2, "late_threshold": 2},
}

N_THREADS = 16


def make_analyzer(level: str) -> BgBotAnalyzer:
    if level.endswith("T"):
        preset = TRUNCATED_PRESETS[level]
        return BgBotAnalyzer(
            eval_level="rollout",
            parallel_threads=N_THREADS,
            **preset,
        )
    else:
        return BgBotAnalyzer(
            eval_level=level,
            parallel_threads=N_THREADS,
        )


def main():
    parser = argparse.ArgumentParser(description="Cube analysis profiling benchmark")
    parser.add_argument("level", choices=["1ply", "2ply", "3ply", "4ply", "1T", "2T", "3T"],
                        help="Evaluation level")
    args = parser.parse_args()
    level = args.level
    is_rollout = level.endswith("T")

    analyzer = make_analyzer(level)

    print(f"Cube Analysis Profiling Benchmark -- Level: {level}")
    print(f"Model: {PRODUCTION_MODEL}, Threads: {N_THREADS}")
    print("=" * 70)

    total_time = 0.0
    for i, pos in enumerate(POSITIONS):
        kw = {
            "cube_value": pos["cube_value"],
            "cube_owner": pos["cube_owner"],
            "away1": pos.get("away1", 0),
            "away2": pos.get("away2", 0),
            "is_crawford": pos.get("is_crawford", False),
            "jacoby": pos.get("jacoby", True),
            "beaver": pos.get("beaver", True),
        }
        t0 = time.perf_counter()
        r = analyzer.cube_action(pos["board"], **kw)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        p = r.probs
        print(f"\n  Position {i+1}: {pos['label']}")
        print(f"    Action: {r.optimal_action}")
        print(f"    ND={r.equity_nd:+.4f}  DT={r.equity_dt:+.4f}", end="")
        if is_rollout:
            parts = []
            if r.equity_nd_se is not None:
                parts.append(f"ND SE={r.equity_nd_se:.4f}")
            if r.equity_dt_se is not None:
                parts.append(f"DT SE={r.equity_dt_se:.4f}")
            if r.cubeless_se is not None:
                parts.append(f"CL SE={r.cubeless_se:.4f}")
            if parts:
                print(f"  ({', '.join(parts)})", end="")
        print()
        print(f"    Probs: W={p.win:.4f} GW={p.gammon_win:.4f} BW={p.backgammon_win:.4f}"
              f" GL={p.gammon_loss:.4f} BL={p.backgammon_loss:.4f}")
        print(f"    Time: {elapsed:.3f}s")

    print(f"\n{'=' * 70}")
    print(f"Total time: {total_time:.3f}s")


if __name__ == "__main__":
    main()
