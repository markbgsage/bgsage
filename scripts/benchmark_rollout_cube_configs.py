"""Benchmark rollout cube action with different checker/cube evaluation configs.

Compares wall-clock time, cubeful equities, cubeless probabilities, and standard
errors across rollout configurations with separate checker/cube strengths.

Usage:
    python bgsage/scripts/benchmark_rollout_cube_configs.py
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
N_TRIALS = 1296
N_THREADS = 16
SEED = 42


def run_config(name, weights, extra_kwargs):
    import bgbot_cpp

    rollout = bgbot_cpp.create_rollout_5nn(
        *weights.weight_args,
        n_trials=N_TRIALS,
        truncation_depth=0,  # full rollout (play to completion)
        decision_ply=1,      # base default (overridden by checker/cube configs)
        n_threads=N_THREADS,
        seed=SEED,
        **extra_kwargs,
    )

    # Warm up (clear any one-time init)
    rollout.clear_internal_caches()

    start = time.perf_counter()
    result = rollout.cube_decision(
        POSITION,
        cube_value=1,
        owner=bgbot_cpp.CubeOwner.CENTERED,
        jacoby=True,
        beaver=True,
    )
    elapsed = time.perf_counter() - start

    return {
        "name": name,
        "time": elapsed,
        "equity_nd": result["equity_nd"],
        "equity_dt": result["equity_dt"],
        "equity_dp": result["equity_dp"],
        "nd_se": result["equity_nd_se"],
        "dt_se": result["equity_dt_se"],
        "should_double": result["should_double"],
        "should_take": result["should_take"],
        "probs": result["probs"],
        "prob_se": result["prob_std_errors"],
        "cubeless_eq": result["cubeless_equity"],
        "cubeless_se": result["cubeless_se"],
    }


def main():
    _setup_paths()

    import bgbot_cpp
    from bgsage.weights import WeightConfig

    weights = WeightConfig.default()
    weights.validate()

    # Define configurations
    configs = [
        ("Fast, no VR", {
            "enable_vr": False,
        }),
        ("Fast, VR", {
            "enable_vr": True,
        }),
        ("Medium", {
            "enable_vr": True,
            "checker": bgbot_cpp.TrialEvalConfig(ply=2),
            "cube": bgbot_cpp.TrialEvalConfig(ply=3),
        }),
        ("Default", {
            "enable_vr": True,
            "checker": bgbot_cpp.TrialEvalConfig(ply=3),
            "cube": bgbot_cpp.TrialEvalConfig(
                rollout_trials=42, rollout_depth=5, rollout_ply=1),
        }),
        ("Slow", {
            "enable_vr": True,
            "checker": bgbot_cpp.TrialEvalConfig(ply=3),
            "cube": bgbot_cpp.TrialEvalConfig(
                rollout_trials=360, rollout_depth=7, rollout_ply=2),
        }),
    ]

    print(f"Position: {POSITION}")
    print(f"Cube: centered at 1, money game (Jacoby on, Beaver on)")
    print(f"Rollout: {N_TRIALS} trials, {N_THREADS} threads, full play-out")
    print()

    results = []
    for name, kwargs in configs:
        print(f"Running: {name}...", end=" ", flush=True)
        r = run_config(name, weights, kwargs)
        results.append(r)
        print(f"{r['time']:.1f}s")

    # Print results table
    print()
    print("=" * 100)
    print(f"{'Config':<16} {'Time':>6} {'ND':>8} {'DT':>8} {'DP':>6}"
          f"  {'ND SE':>7} {'DT SE':>7} {'CL SE':>7}"
          f"  {'D?':>3} {'T?':>3}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<16} {r['time']:>5.1f}s"
              f" {r['equity_nd']:>+8.4f} {r['equity_dt']:>+8.4f} {r['equity_dp']:>+6.3f}"
              f"  {r['nd_se']:>7.4f} {r['dt_se']:>7.4f} {r['cubeless_se']:>7.4f}"
              f"  {'Y' if r['should_double'] else 'N':>3}"
              f" {'Y' if r['should_take'] else 'N':>3}")

    print()
    print("Cubeless probabilities:")
    print(f"{'Config':<16} {'P(win)':>7} {'P(gw)':>7} {'P(bw)':>7}"
          f" {'P(gl)':>7} {'P(bl)':>7} {'CL Eq':>8}")
    print("-" * 75)
    for r in results:
        p = r["probs"]
        print(f"{r['name']:<16}"
              f" {p[0]:>7.4f} {p[1]:>7.4f} {p[2]:>7.4f}"
              f" {p[3]:>7.4f} {p[4]:>7.4f} {r['cubeless_eq']:>+8.4f}")

    print()
    print("Probability standard errors:")
    print(f"{'Config':<16} {'SE(win)':>7} {'SE(gw)':>7} {'SE(bw)':>7}"
          f" {'SE(gl)':>7} {'SE(bl)':>7}")
    print("-" * 60)
    for r in results:
        se = r["prob_se"]
        print(f"{r['name']:<16}"
              f" {se[0]:>7.4f} {se[1]:>7.4f} {se[2]:>7.4f}"
              f" {se[3]:>7.4f} {se[4]:>7.4f}")


if __name__ == "__main__":
    main()
