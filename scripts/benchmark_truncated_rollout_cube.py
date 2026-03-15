"""Benchmark truncated cubeful rollout for a fixed cube-action position.

This targets the XG Roller++ Cube-equivalent configuration from CLAUDE.md:
  n_trials=360, truncation_depth=7, decision_ply=3, late_ply=2, late_threshold=2

It reports cubeful ND/DT/DP equities, cubeless probabilities, standard errors,
and wall-clock timing for the cube-action rollout call.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any


POSITION = [
    0, 0, 0, 2, 2, -2, 3, 2, 2, 0, 0, 0, -3,
    4, 0, 0, 0, -3, 0, -3, -2, -2, 0, 0, 0, 0,
]

ROLLOUT_CONFIG = {
    "n_trials": 360,
    "truncation_depth": 7,
    "decision_ply": 3,
    "late_ply": 2,
    "late_threshold": 2,
    "filter_max_moves": 5,
    "filter_threshold": 0.08,
    "enable_vr": True,
}

PROB_LABELS = (
    "win",
    "gammon_win",
    "backgammon_win",
    "gammon_loss",
    "backgammon_loss",
)


def _setup_paths() -> None:
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


def _get_weights():
    from bgsage.weights import WeightConfig

    weights = WeightConfig.default()
    weights.validate()
    return weights


def _create_rollout_strategy(weights, config: dict[str, Any], n_threads: int, seed: int):
    import bgbot_cpp

    kwargs = dict(config)
    kwargs["n_threads"] = n_threads
    kwargs["seed"] = seed
    return bgbot_cpp.create_rollout_5nn(
        weights.purerace,
        weights.racing,
        weights.attacking,
        weights.priming,
        weights.anchoring,
        weights.n_hidden_purerace,
        weights.n_hidden_racing,
        weights.n_hidden_attacking,
        weights.n_hidden_priming,
        weights.n_hidden_anchoring,
        **kwargs,
    )


def _run_once(weights, config: dict[str, Any], n_threads: int, seed: int,
              profile: bool, standalone: bool, rollout_strategy=None) -> dict[str, Any]:
    import bgbot_cpp

    start = time.perf_counter()
    if standalone:
        kwargs = dict(config)
        kwargs["n_threads"] = n_threads
        kwargs["seed"] = seed
        if profile:
            kwargs["profile"] = True

        try:
            result = bgbot_cpp.cube_decision_rollout(
                POSITION,
                cube_value=1,
                owner=bgbot_cpp.CubeOwner.CENTERED,
                purerace_weights=weights.purerace,
                racing_weights=weights.racing,
                attacking_weights=weights.attacking,
                priming_weights=weights.priming,
                anchoring_weights=weights.anchoring,
                n_hidden_purerace=weights.n_hidden_purerace,
                n_hidden_racing=weights.n_hidden_racing,
                n_hidden_attacking=weights.n_hidden_attacking,
                n_hidden_priming=weights.n_hidden_priming,
                n_hidden_anchoring=weights.n_hidden_anchoring,
                jacoby=True,
                beaver=True,
                **kwargs,
            )
        except TypeError:
            kwargs.pop("profile", None)
            result = bgbot_cpp.cube_decision_rollout(
                POSITION,
                cube_value=1,
                owner=bgbot_cpp.CubeOwner.CENTERED,
                purerace_weights=weights.purerace,
                racing_weights=weights.racing,
                attacking_weights=weights.attacking,
                priming_weights=weights.priming,
                anchoring_weights=weights.anchoring,
                n_hidden_purerace=weights.n_hidden_purerace,
                n_hidden_racing=weights.n_hidden_racing,
                n_hidden_attacking=weights.n_hidden_attacking,
                n_hidden_priming=weights.n_hidden_priming,
                n_hidden_anchoring=weights.n_hidden_anchoring,
                jacoby=True,
                beaver=True,
                **kwargs,
            )
    else:
        result = rollout_strategy.cube_decision(
            POSITION,
            cube_value=1,
            owner=bgbot_cpp.CubeOwner.CENTERED,
            jacoby=True,
            beaver=True,
        )

    elapsed = time.perf_counter() - start
    result["elapsed_seconds"] = elapsed
    return result


def _format_summary(results: list[dict[str, Any]], args: argparse.Namespace,
                    config: dict[str, Any], mode: str) -> dict[str, Any]:
    timings = [r["elapsed_seconds"] for r in results]
    last = results[-1]

    summary: dict[str, Any] = {
        "mode": mode,
        "position": POSITION,
        "cube_value": 1,
        "cube_owner": "centered",
        "jacoby": True,
        "beaver": True,
        "threads": args.threads,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "config": dict(config),
        "timing": {
            "last_seconds": last["elapsed_seconds"],
            "mean_seconds": statistics.fmean(timings),
            "min_seconds": min(timings),
            "max_seconds": max(timings),
        },
        "cubeful": {
            "equity_nd": last["equity_nd"],
            "equity_nd_se": last["equity_nd_se"],
            "equity_dt": last["equity_dt"],
            "equity_dt_se": last["equity_dt_se"],
            "equity_dp": last["equity_dp"],
            "should_double": last["should_double"],
            "should_take": last["should_take"],
            "optimal_equity": last["optimal_equity"],
            "is_beaver": last.get("is_beaver", False),
        },
        "cubeless": {
            "equity": last["cubeless_equity"],
            "equity_se": last.get("cubeless_se"),
            "probs": dict(zip(PROB_LABELS, last["probs"])),
        },
    }

    prob_ses = last.get("prob_std_errors")
    if prob_ses is not None:
        summary["cubeless"]["prob_std_errors"] = dict(zip(PROB_LABELS, prob_ses))

    if "profile" in last:
        summary["profile"] = last["profile"]

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print("Truncated Rollout Cube Benchmark")
    print("=" * 80)
    print(f"Position: {summary['position']}")
    print("Cube: centered 1, money game, Jacoby on, Beaver on")
    print(f"Mode: {summary['mode']}")
    cfg = summary["config"]
    print(
        "Config: "
        f"trials={cfg['n_trials']}, trunc_depth={cfg['truncation_depth']}, "
        f"decision_ply={cfg['decision_ply']}, late_ply={cfg['late_ply']}, "
        f"late_threshold={cfg['late_threshold']}, vr={cfg['enable_vr']}"
    )
    print(
        f"Execution: repeats={summary['repeats']}, warmups={summary['warmups']}, "
        f"threads={summary['threads']}, seed={summary['seed']}"
    )
    print()

    timing = summary["timing"]
    print("Timing")
    print(
        f"  last={timing['last_seconds']:.6f}s  mean={timing['mean_seconds']:.6f}s  "
        f"min={timing['min_seconds']:.6f}s  max={timing['max_seconds']:.6f}s"
    )
    print()

    cubeful = summary["cubeful"]
    print("Cubeful")
    print(f"  ND: {cubeful['equity_nd']:+.6f}  SE={cubeful['equity_nd_se']:.6f}")
    print(f"  DT: {cubeful['equity_dt']:+.6f}  SE={cubeful['equity_dt_se']:.6f}")
    print(f"  DP: {cubeful['equity_dp']:+.6f}")
    print(
        f"  Decision: {'Double' if cubeful['should_double'] else 'No Double'} / "
        f"{'Take' if cubeful['should_take'] else 'Pass'}"
    )
    if cubeful["is_beaver"]:
        print("  Beaver: yes")
    print(f"  Optimal equity: {cubeful['optimal_equity']:+.6f}")
    print()

    cubeless = summary["cubeless"]
    print("Cubeless")
    print(f"  Equity: {cubeless['equity']:+.6f}  SE={cubeless['equity_se']:.6f}")
    prob_ses = cubeless.get("prob_std_errors")
    for label in PROB_LABELS:
        value = cubeless["probs"][label]
        if prob_ses is None:
            print(f"  {label:16s} {value:.6f}  SE=n/a")
        else:
            print(f"  {label:16s} {value:.6f}  SE={prob_ses[label]:.6f}")

    profile = summary.get("profile")
    if profile:
        print()
        print("Profile")
        for key, value in profile.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=0, help="Rollout worker threads (0=auto)")
    parser.add_argument("--seed", type=int, default=42, help="Rollout RNG seed")
    parser.add_argument("--warmups", type=int, default=1, help="Untimed warmup runs")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs")
    parser.add_argument("--n-trials", type=int, default=ROLLOUT_CONFIG["n_trials"], help="Number of rollout trials")
    parser.add_argument("--truncation-depth", type=int, default=ROLLOUT_CONFIG["truncation_depth"], help="Half-move truncation depth")
    parser.add_argument("--decision-ply", type=int, default=ROLLOUT_CONFIG["decision_ply"], help="Decision ply before late threshold")
    parser.add_argument("--late-ply", type=int, default=ROLLOUT_CONFIG["late_ply"], help="Decision ply after late threshold")
    parser.add_argument("--late-threshold", type=int, default=ROLLOUT_CONFIG["late_threshold"], help="Half-move index that switches to late ply")
    parser.add_argument("--filter-max-moves", type=int, default=ROLLOUT_CONFIG["filter_max_moves"], help="Move filter survivor cap")
    parser.add_argument("--filter-threshold", type=float, default=ROLLOUT_CONFIG["filter_threshold"], help="Move filter equity threshold")
    parser.add_argument("--disable-vr", action="store_true", help="Disable variance reduction")
    parser.add_argument("--standalone", action="store_true", help="Benchmark the one-shot binding that rebuilds rollout state each call")
    parser.add_argument("--profile", action="store_true", help="Request internal C++ profile data if supported")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    args = parser.parse_args()

    _setup_paths()
    weights = _get_weights()

    config = {
        "n_trials": args.n_trials,
        "truncation_depth": args.truncation_depth,
        "decision_ply": args.decision_ply,
        "late_ply": args.late_ply,
        "late_threshold": args.late_threshold,
        "filter_max_moves": args.filter_max_moves,
        "filter_threshold": args.filter_threshold,
        "enable_vr": not args.disable_vr,
    }

    rollout_strategy = None
    mode = "standalone" if args.standalone else "reused_strategy"
    if not args.standalone:
        rollout_strategy = _create_rollout_strategy(
            weights, config, args.threads, args.seed
        )

    for _ in range(args.warmups):
        _run_once(
            weights,
            config,
            args.threads,
            args.seed,
            profile=args.profile,
            standalone=args.standalone,
            rollout_strategy=rollout_strategy,
        )

    results = [
        _run_once(
            weights,
            config,
            args.threads,
            args.seed,
            profile=args.profile,
            standalone=args.standalone,
            rollout_strategy=rollout_strategy,
        )
        for _ in range(args.repeats)
    ]
    summary = _format_summary(results, args, config, mode)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)


if __name__ == "__main__":
    main()
