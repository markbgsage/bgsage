"""3T (XG Roller++) cube action benchmark using reference positions.

Reads 11 positions from ../refpos.txt, evaluates each at the 3T truncated
rollout level (360 trials, truncation_depth=5, decision_ply=3, late_ply=2,
late_threshold=2), prints cubeful equities, cubeless probabilities, standard
errors, and wall-clock time.  Caps CPU usage at 16 threads.

Usage:
    python bgsage/scripts/benchmark_3t_cube.py
"""

from __future__ import annotations

import os
import sys
import time

# DLL setup for Windows (CUDA + build dir)
if hasattr(os, "add_dll_directory"):
    cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    for d in ("build", "build_msvc"):
        if os.path.isdir(d):
            os.add_dll_directory(os.path.abspath(d))

sys.path.insert(0, "build")
sys.path.insert(0, "bgsage/python")

from bgsage import BgBotAnalyzer
from bgsage.weights import PRODUCTION_MODEL

N_THREADS = 16

# ── Parse refpos.txt ─────────────────────────────────────────────────────────

def parse_refpos(path: str) -> list[dict]:
    positions = []
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
            # meta: cube_value, cube_owner(0=centered,1=player,2=opponent),
            #       away1, away2(-1 for money), is_crawford, jacoby, beaver
            cube_value = meta[0]
            owner_map = {0: "centered", 1: "player", 2: "opponent"}
            cube_owner = owner_map[meta[1]]
            away1 = meta[2] if meta[2] > 0 else 0
            away2 = meta[3] if meta[3] > 0 else 0
            is_crawford = bool(meta[4]) if len(meta) > 4 else False
            jacoby = bool(meta[5]) if len(meta) > 5 else True
            beaver = bool(meta[6]) if len(meta) > 6 else True

            ref = {}
            if len(parts) >= 3:
                vals = list(map(float, parts[2].split(",")))
                ref = {
                    "nd": vals[0], "dt": vals[1],
                    "probs": vals[2:8],
                }

            positions.append({
                "board": checkers,
                "cube_value": cube_value,
                "cube_owner": cube_owner,
                "away1": away1,
                "away2": away2,
                "is_crawford": is_crawford,
                "jacoby": jacoby,
                "beaver": beaver,
                "ref": ref,
            })
    return positions


def main():
    # Find refpos.txt one level up from bgsage/
    refpos_path = os.path.join(os.path.dirname(__file__), "..", "..", "refpos.txt")
    refpos_path = os.path.normpath(refpos_path)
    if not os.path.exists(refpos_path):
        print(f"ERROR: refpos.txt not found at {refpos_path}")
        sys.exit(1)

    positions = parse_refpos(refpos_path)
    print(f"Loaded {len(positions)} positions from {refpos_path}")

    # Parse optional --truncation-ply override
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--truncation-ply", type=int, default=None,
                        help="Override truncation ply (default: same as decision_ply)")
    args, _ = parser.parse_known_args()

    extra_kw = {}
    if args.truncation_ply is not None:
        extra_kw["truncation_ply"] = args.truncation_ply
        print(f"Truncation ply override: {args.truncation_ply}")

    analyzer = BgBotAnalyzer(
        eval_level="rollout",
        n_trials=360,
        truncation_depth=5,
        decision_ply=3,
        late_ply=2,
        late_threshold=2,
        parallel_threads=N_THREADS,
        **extra_kw,
    )

    print(f"\n3T Cube Action Benchmark")
    print(f"Model: {PRODUCTION_MODEL}, Threads: {N_THREADS}")
    print(f"Config: 360 trials, trunc_depth=5, decision_ply=3, late_ply=2, late_threshold=2")
    print("=" * 90)

    total_time = 0.0
    for i, pos in enumerate(positions):
        kw = {
            "cube_value": pos["cube_value"],
            "cube_owner": pos["cube_owner"],
            "away1": pos["away1"],
            "away2": pos["away2"],
            "is_crawford": pos["is_crawford"],
            "jacoby": pos["jacoby"],
            "beaver": pos["beaver"],
        }

        t0 = time.perf_counter()
        r = analyzer.cube_action(pos["board"], **kw)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        p = r.probs
        nd_se = r.equity_nd_se if r.equity_nd_se is not None else 0
        dt_se = r.equity_dt_se if r.equity_dt_se is not None else 0
        cl_se = r.cubeless_se if r.cubeless_se is not None else 0

        print(f"\nPos {i+1:2d}: {pos['cube_owner']} cv={pos['cube_value']}"
              f" away=({pos['away1']},{pos['away2']})"
              f" jacoby={pos['jacoby']} beaver={pos['beaver']}")
        print(f"  Board: {pos['board']}")
        print(f"  Action: {r.optimal_action}")
        print(f"  ND={r.equity_nd:+.4f} (SE={nd_se:.4f})"
              f"  DT={r.equity_dt:+.4f} (SE={dt_se:.4f})"
              f"  CL_SE={cl_se:.4f}")
        print(f"  Probs: W={p.win:.4f} GW={p.gammon_win:.4f}"
              f" BW={p.backgammon_win:.4f}"
              f" GL={p.gammon_loss:.4f} BL={p.backgammon_loss:.4f}")
        print(f"  Time: {elapsed:.3f}s")

        ref = pos.get("ref", {})
        if ref:
            rp = ref["probs"]
            print(f"  Ref: ND={ref['nd']:+.4f} DT={ref['dt']:+.4f}"
                  f"  W={rp[0]:.4f} GW={rp[1]:.4f} BW={rp[2]:.4f}"
                  f" GL={rp[3]:.4f} BL={rp[4]:.4f}")

    print(f"\n{'=' * 90}")
    print(f"Total wall-clock time: {total_time:.3f}s")
    print(f"Average per position: {total_time/len(positions):.3f}s")


if __name__ == "__main__":
    main()
