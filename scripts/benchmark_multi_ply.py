# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Time every fixed checker/cube/post-move case and retain full API results.

Run from bgsage, for example:
    python scripts/benchmark_multi_ply.py --levels 2ply 3ply --threads 2

Each run creates metadata.json, results.jsonl and summary.json in a new directory.
Timings include the public API call only; model loading and output I/O are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

from build_multi_ply_benchmark import validate_set


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multi_ply_benchmark"
APIS = ("checker_play", "cube_action", "post_move_analytics")
TRUNCATED_LEVELS = {"1T": "truncated1", "2T": "truncated2", "3T": "truncated3"}
LEVELS = ("1ply", "2ply", "3ply", "4ply", *TRUNCATED_LEVELS.values())
CACHE_POLICY = (
    "One analyzer per level; API order then complete position passes per repeat. "
    "No explicit cache reset or warmup. Normal engine cache behavior is retained, "
    "including possible reuse between positions/APIs/repeats; these are not cold-cache timings. "
    "First calls may include lazy thread-pool or kernel initialization."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inside_repo(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside the bgsage repository: {path}")
    return path


def file_identity(path: Path) -> dict:
    path = inside_repo(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_engine(extension: Path | None, python_dir: Path = ROOT / "python"):
    if extension is None:
        names = {"bgbot_cpp" + suffix for suffix in EXTENSION_SUFFIXES}
        candidates = [p for p in (ROOT / "build").rglob("bgbot_cpp*") if p.name in names]
        direct = [p for p in candidates if p.parent == ROOT / "build"]
        candidates = direct or candidates
        if len(candidates) != 1:
            raise ValueError("Select one compatible in-repository bgbot_cpp build with --extension; "
                             f"found {len(candidates)} candidates")
        extension = candidates[0]
    extension = inside_repo(extension)
    spec = importlib.util.spec_from_file_location("bgbot_cpp", extension)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load engine extension: {extension}")
    cpp = importlib.util.module_from_spec(spec)
    sys.modules["bgbot_cpp"] = cpp
    spec.loader.exec_module(cpp)
    python_dir = inside_repo(python_dir)
    sys.path.insert(0, str(python_dir))
    import bgsage
    if not Path(bgsage.__file__).resolve().is_relative_to(python_dir):
        raise ValueError(f"Benchmark did not load the selected bgsage package from {python_dir}")
    return bgsage, extension


def timing_stats(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(values), "total_seconds": sum(values),
        "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values),
        "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_seconds": ordered[0], "max_seconds": ordered[-1],
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def value_metrics(api: str, result: dict) -> dict:
    if api == "checker_play":
        moves = result["moves"]
        return {"legal_moves": len(moves), "best_equity": moves[0]["equity"],
                "best_cubeless_equity": moves[0]["cubeless_equity"]}
    if api == "cube_action":
        return {key: result[key] for key in
                ("equity_nd", "equity_dt", "equity_dp", "optimal_equity", "cubeless_equity")}
    return {key: result[key] for key in ("cubeless_equity", "cubeful_equity")}


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        for mode, plan in ((None, None), (record["mode"], None),
                           (None, record["source_game_plan"]),
                           (record["mode"], record["source_game_plan"])):
            groups[(record["api"], record["level"], mode, plan)].append(record)
    summaries = []
    for (api, level, mode, plan), rows in groups.items():
        metrics = defaultdict(list)
        for row in rows:
            for key, value in row["value_metrics"].items():
                metrics[key].append(value)
        summaries.append({
            "api": api, "level": level, "mode": mode, "source_game_plan": plan,
            "timing": timing_stats([row["elapsed_seconds"] for row in rows]),
            "values": {key: {"mean": statistics.mean(values), "min": min(values), "max": max(values)}
                       for key, values in metrics.items()},
            "cube_actions": dict(Counter(row["optimal_action"] for row in rows
                                         if "optimal_action" in row)),
        })
    return {
        "completed_calls": len(records), "p95_method": "nearest rank",
        "game_plan_perspective": "Source player on roll, including post-move rows (previous mover is opposite)",
        "value_statistics_note": "Descriptive aggregates, not an accuracy score or rollout comparison",
        "groups": summaries,
    }


def print_summary(summary: dict) -> None:
    print("\nAPI                   Level  Calls   Total(s)   Mean(ms) Median(ms)    P95(ms)")
    for group in summary["groups"]:
        if group["mode"] is not None or group["source_game_plan"] is not None:
            continue
        stat = group["timing"]
        print(f"{group['api']:<21} {group['level']:<6} {stat['count']:>5} "
              f"{stat['total_seconds']:>10.3f} {stat['mean_seconds'] * 1000:>10.3f} "
              f"{stat['median_seconds'] * 1000:>10.3f} {stat['p95_seconds'] * 1000:>10.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=LEVELS,
                        type=lambda level: TRUNCATED_LEVELS.get(level.upper(), level),
                        default=["2ply", "3ply"])
    parser.add_argument("--apis", nargs="+", choices=APIS, default=list(APIS))
    parser.add_argument("--threads", type=int, default=2,
                        help="Threads requested per call; 0=engine auto; multi-ply raises 1 to 2")
    parser.add_argument("--repeats", type=int, default=1, help="Complete 100-position passes per API/level")
    parser.add_argument("--model", help="Registered model name; default is current production model")
    parser.add_argument("--extension", type=Path, help="Compiled bgbot_cpp inside this repository")
    parser.add_argument("--python-dir", type=Path, default=ROOT / "python",
                        help="Python package root, for comparison to a frozen in-repository version")
    parser.add_argument("--native-source", type=Path, default=ROOT,
                        help="In-repository native source tree to fingerprint, containing cpp/ and CMakeLists.txt")
    parser.add_argument("--output", type=Path, help="New output directory inside this repository")
    parser.add_argument("--cube-details", action="store_true",
                        help="Include the optional per-roll cube details (requires 2-ply or higher)")
    args = parser.parse_args()
    if args.threads < 0 or args.repeats < 1:
        parser.error("--threads must be >= 0 and --repeats must be >= 1")
    if len(set(args.levels)) != len(args.levels) or len(set(args.apis)) != len(args.apis):
        parser.error("Levels and APIs must not be repeated; use --repeats")
    if args.cube_details and (any(level not in ("2ply", "3ply", "4ply") for level in args.levels)
                              or "cube_action" not in args.apis):
        parser.error("--cube-details requires cube_action and levels >= 2ply")
    return args


def main() -> None:
    args = parse_args()
    datasets = {}
    for name in ("checker_play", "cube_action"):
        path = DATA / f"{name}.json"
        dataset = json.loads(path.read_text(encoding="utf-8"))
        expected_kind = "checker" if name == "checker_play" else "cube"
        if dataset.get("schema_version") != 1 or dataset.get("kind") != expected_kind:
            raise ValueError(f"Unsupported schema or incorrect position kind: {path}")
        validate_set(dataset)
        datasets[name] = dataset
    engine, extension = load_engine(args.extension, args.python_dir)
    model = args.model or engine.PRODUCTION_MODEL
    weights = engine.WeightConfig.from_model(model)
    weights.validate()
    weight_files = [file_identity(Path(p)) for p in weights.weight_paths_list]
    bearoff = ROOT / "data" / "bearoff_1sided.db"
    bearoff_identity = file_identity(bearoff)
    output = inside_repo(args.output or ROOT / "logs" / "multi_ply_benchmark" /
                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    native = inside_repo(args.native_source)
    source_paths = [Path(__file__), ROOT / "scripts" / "build_multi_ply_benchmark.py", native / "CMakeLists.txt"]
    for directory in (inside_repo(args.python_dir) / "bgsage", native / "cpp"):
        if not directory.is_dir():
            raise ValueError(f"Source tree does not contain {directory}")
        source_paths.extend(p for p in directory.rglob("*") if p.is_file()
                            and p.suffix in (".py", ".cpp", ".h", ".cu", ".txt"))
    metadata = {
        "schema_version": 1, "status": "running", "started_utc": utc_now(),
        "git_commit": git_commit, "model": model, "strategy_type": weights.strategy_type,
        "hidden_sizes": weights.hidden_sizes_list, "weights": weight_files,
        "extension": file_identity(extension), "bearoff_database": bearoff_identity,
        "source_files": [file_identity(p) for p in sorted(set(source_paths))],
        "native_source_tree": native.relative_to(ROOT).as_posix(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "levels": args.levels, "apis": args.apis, "threads_requested": args.threads,
        "analyzer_threads_by_level": {},
        "thread_note": "Analyzer setting; actual concurrency varies by API/depth and engine pool limits",
        "repeats": args.repeats, "cubeful": True, "cube_details": args.cube_details,
        "cache_policy": CACHE_POLICY,
        "timing_scope": "perf_counter_ns around public API call; excludes constructor, serialization and disk I/O",
        "ordering": "levels, APIs, repeats, then saved position order; no shuffle",
        "positions": {name: file_identity(DATA / f"{name}.json") for name in datasets},
        "expected_calls": 100 * len(args.levels) * len(args.apis) * args.repeats,
        "environment": {key: os.environ[key] for key in
                        ("BGBOT_MULTIPLY_THREADS", "BGBOT_CUBEFUL_DEEP_KEEP",
                         "BGBOT_CUBEFUL_DEEP_THRESHOLD", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                        if key in os.environ},
    }
    write_json(output / "metadata.json", metadata)
    print(f"Output: {output}\nModel: {model}; requested threads: {args.threads}", flush=True)
    records, active = [], None
    try:
        with (output / "results.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for level in args.levels:
                analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                                                parallel_threads=args.threads, bearoff_db=str(bearoff))
                if analyzer._bearoff_db is None:
                    raise RuntimeError("Configured bearoff database did not load")
                inner = getattr(analyzer._analyzer, "_inner", analyzer._analyzer)
                metadata["analyzer_threads_by_level"][level] = getattr(inner, "_parallel_threads", 1)
                truncated = level in TRUNCATED_LEVELS.values()
                if truncated:
                    cfg = inner._rollout_strategy.config()
                    metadata.setdefault("truncated_presets", {})[level] = {
                        "n_trials": cfg.n_trials, "truncation_depth": cfg.truncation_depth,
                        "decision_ply": cfg.decision_ply, "truncation_ply": cfg.truncation_ply,
                        "late_ply": inner._rollout_config["late_ply"],
                        "late_threshold": inner._rollout_config["late_threshold"], "variance_reduction": cfg.enable_vr,
                        "trial_threads": cfg.n_threads, "prefilter_threshold": inner._prefilter_threshold,
                    }
                    metadata["seed_policy"] = "uint32 from the first four SHA-256 bytes of truncated-benchmark-v1:<source key>, big endian; reset before each call"
                write_json(output / "metadata.json", metadata)
                for api in args.apis:
                    dataset_name = "checker_play" if api == "checker_play" else "cube_action"
                    for repeat in range(1, args.repeats + 1):
                        for index, position in enumerate(datasets[dataset_name]["positions"], 1):
                            inputs = dict(position["post_move_inputs"] if api == "post_move_analytics"
                                          else position["inputs"])
                            if api == "cube_action":
                                inputs["incl_2ply_details"] = args.cube_details
                            active = {"api": api, "level": level, "repeat": repeat,
                                      "position_id": position["id"], "mode": position["mode"],
                                      "source_game_plan": position["game_plan"],
                                      "source": position["source"], "inputs": inputs}
                            if truncated:
                                key = "truncated-benchmark-v1:" + position["source"]["key"]
                                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
                                analyzer.set_seed(seed)
                                active["seed"] = seed
                            call = getattr(analyzer, api)
                            started = time.perf_counter_ns()
                            result = call(**inputs)
                            elapsed = (time.perf_counter_ns() - started) / 1e9
                            analytics = asdict(result)
                            record = {**active, "elapsed_seconds": elapsed, "analytics": analytics}
                            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                            compact = {**active, "elapsed_seconds": elapsed,
                                       "value_metrics": value_metrics(api, analytics)}
                            if api == "cube_action":
                                compact["optimal_action"] = analytics["optimal_action"]
                            records.append(compact)
                            if index % 25 == 0:
                                print(f"{level} {api} pass {repeat}: {index}/100", flush=True)
                        write_json(output / "summary.json", summarize(records))
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_json(output / "error.json", {"call": active, "error": repr(error),
                                           "traceback": traceback.format_exc()})
        raise
    finally:
        summary = summarize(records)
        summary["status"] = metadata["status"]
        write_json(output / "summary.json", summary)
        metadata["finished_utc"] = utc_now()
        metadata["completed_calls"] = len(records)
        write_json(output / "metadata.json", metadata)
        print_summary(summary)
        print(f"Run status: {metadata['status']}; full analytics: {output / 'results.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
