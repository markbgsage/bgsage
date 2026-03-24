"""Diagnose whether VR introduces bias in full rollouts.

Runs the same position with VR on and VR off using the same seed,
then compares cubeless and cubeful equity estimates.
Also runs with thinned VR disabled (VR at every move) as a control.
"""

from __future__ import annotations

import os
import sys
import time


def _setup_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    root_dir = os.path.dirname(project_dir)
    build_dirs = [
        os.path.join(project_dir, "build_msvc"),
        os.path.join(root_dir, "build_msvc"),
        os.path.join(project_dir, "build"),
        os.path.join(root_dir, "build"),
    ]
    search_paths = [os.path.join(project_dir, "python"), *build_dirs]
    for path in reversed(search_paths):
        if os.path.isdir(path):
            sys.path.insert(0, path)
    if hasattr(os, "add_dll_directory"):
        cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
        if os.path.isdir(cuda_x64):
            os.add_dll_directory(cuda_x64)
        for build_dir in build_dirs:
            if os.path.isdir(build_dir):
                os.add_dll_directory(build_dir)


POSITION = [2, 2, 0, 0, 0, 0, 4, 1, 2, 0, 1, 0, -4, 3, 0, 0, 0, -3, 0, -4, -2, 0, 1, 0, 1, 0]


def run_cube_rollout(weights, n_trials, n_threads, seed, enable_vr):
    import bgbot_cpp
    rollout = bgbot_cpp.create_rollout_5nn(
        *weights.weight_args,
        n_trials=n_trials,
        truncation_depth=0,
        decision_ply=1,
        n_threads=n_threads,
        seed=seed,
        enable_vr=enable_vr,
    )
    start = time.perf_counter()
    result = rollout.cube_decision(
        POSITION,
        cube_value=1,
        owner=bgbot_cpp.CubeOwner.CENTERED,
        jacoby=True,
        beaver=True,
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    _setup_paths()
    from bgsage.weights import WeightConfig
    weights = WeightConfig.default()
    weights.validate()

    N_TRIALS = 1296
    N_THREADS = 16

    print(f"Position: {POSITION}")
    print(f"Full rollout: {N_TRIALS} trials, {N_THREADS} threads, 1-ply decisions")
    print()

    # Run multiple seeds to assess consistency
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000]

    vr_nd = []
    vr_dt = []
    vr_cl = []
    novr_nd = []
    novr_dt = []
    novr_cl = []

    print(f"{'Seed':>6}  {'VR ND':>8} {'noVR ND':>8} {'diff':>7}"
          f"  {'VR CL':>8} {'noVR CL':>8} {'diff':>7}"
          f"  {'VR SE':>7} {'noVR SE':>7}")
    print("-" * 95)

    for seed in seeds:
        r_vr, t_vr = run_cube_rollout(weights, N_TRIALS, N_THREADS, seed, True)
        r_novr, t_novr = run_cube_rollout(weights, N_TRIALS, N_THREADS, seed, False)

        vr_nd.append(r_vr["equity_nd"])
        vr_dt.append(r_vr["equity_dt"])
        vr_cl.append(r_vr["cubeless_equity"])
        novr_nd.append(r_novr["equity_nd"])
        novr_dt.append(r_novr["equity_dt"])
        novr_cl.append(r_novr["cubeless_equity"])

        d_nd = r_vr["equity_nd"] - r_novr["equity_nd"]
        d_cl = r_vr["cubeless_equity"] - r_novr["cubeless_equity"]

        print(f"{seed:>6}  {r_vr['equity_nd']:>+8.4f} {r_novr['equity_nd']:>+8.4f} {d_nd:>+7.4f}"
              f"  {r_vr['cubeless_equity']:>+8.4f} {r_novr['cubeless_equity']:>+8.4f} {d_cl:>+7.4f}"
              f"  {r_vr['cubeless_se']:>7.4f} {r_novr['cubeless_se']:>7.4f}")

    print()
    import statistics
    print("Summary across seeds:")
    print(f"  VR ND mean:   {statistics.mean(vr_nd):>+8.4f}  stdev: {statistics.stdev(vr_nd):.4f}")
    print(f"  noVR ND mean: {statistics.mean(novr_nd):>+8.4f}  stdev: {statistics.stdev(novr_nd):.4f}")
    print(f"  VR CL mean:   {statistics.mean(vr_cl):>+8.4f}  stdev: {statistics.stdev(vr_cl):.4f}")
    print(f"  noVR CL mean: {statistics.mean(novr_cl):>+8.4f}  stdev: {statistics.stdev(novr_cl):.4f}")
    print()
    nd_diffs = [a - b for a, b in zip(vr_nd, novr_nd)]
    cl_diffs = [a - b for a, b in zip(vr_cl, novr_cl)]
    print(f"  Mean VR-noVR diff (ND): {statistics.mean(nd_diffs):>+8.4f}  stdev: {statistics.stdev(nd_diffs):.4f}")
    print(f"  Mean VR-noVR diff (CL): {statistics.mean(cl_diffs):>+8.4f}  stdev: {statistics.stdev(cl_diffs):.4f}")

    # Check significance: t-test on the paired differences
    n = len(nd_diffs)
    nd_mean_diff = statistics.mean(nd_diffs)
    nd_se_diff = statistics.stdev(nd_diffs) / (n ** 0.5)
    cl_mean_diff = statistics.mean(cl_diffs)
    cl_se_diff = statistics.stdev(cl_diffs) / (n ** 0.5)
    print()
    print(f"  ND bias t-stat: {nd_mean_diff / nd_se_diff if nd_se_diff > 0 else 0:.2f}"
          f"  (mean diff / SE = {nd_mean_diff:.4f} / {nd_se_diff:.4f})")
    print(f"  CL bias t-stat: {cl_mean_diff / cl_se_diff if cl_se_diff > 0 else 0:.2f}"
          f"  (mean diff / SE = {cl_mean_diff:.4f} / {cl_se_diff:.4f})")


if __name__ == "__main__":
    main()
