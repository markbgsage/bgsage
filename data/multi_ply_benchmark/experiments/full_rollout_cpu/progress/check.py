# SPDX-License-Identifier: MPL-2.0
"""Verify public full-rollout progress on fixed and adaptive batches."""
import argparse
from dataclasses import asdict
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, inside_repo, load_engine, write_json

# The same pure-race cube benchmark also has enough ND variance for a tiny
# target SE to force a second 37-path batch. Each tuple specifies a separate
# analyzer so callback timing and trial counts cannot carry between cases.
PROFILES = (
    ("serial-fixed", 1, 37, 0, 1, 0),
    ("parallel-fixed", 16, 37, 0, 1, 0),
    ("serial-target", 1, 37, 0, 1, 1e-12),
    ("parallel-target", 16, 37, 0, 1, 1e-12),
    ("full-1296", 16, 1296, 0, 3, 0),
    ("full-five", 16, 5, 0, 1, 0),
    ("truncated-custom", 16, 37, 3, 1, 0),
)
ARMS = ("before", "single_claim", "selected")


def read(path):
    return json.loads(path.read_bytes())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def measure(source, output):
    source = inside_repo(source)
    extension = next(source / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                     if (source / ("bgbot_cpp" + suffix)).is_file())
    engine, _ = load_engine(extension, source / "python")
    from bgsage.weights import bearoff_db_path

    require(engine.PRODUCTION_MODEL == "stage11", "Unexpected production model")
    weights = [file_identity(Path(path)) for path in engine.default_weights().weight_paths_list]
    database = file_identity(Path(bearoff_db_path()))
    path = DATA / "cube_action.json"
    pos = next(p for p in read(path)["positions"] if p["id"] == "cube-money-purerace-01")
    os.environ["BGBOT_MULTIPLY_THREADS"] = "16"
    cases = {}
    for name, threads, trials, depth, ply, target in PROFILES:
        policy = engine.TrialEvalConfig(ply=ply)
        analyzer = engine.BgBotAnalyzer(
            eval_level="rollout", parallel_threads=threads, n_trials=trials,
            truncation_depth=depth, decision_ply=ply, late_ply=ply, checker=policy,
            checker_late=policy, cube=policy, cube_late=policy, seed=42,
            target_se=target, max_batches=2 if target else 1)
        native = analyzer._analyzer._inner._rollout_strategy
        config = native.config()
        require((config.n_trials, config.n_threads, config.truncation_depth, config.decision_ply,
                 config.enable_vr, config.target_se) == (trials, threads, depth, ply, True, target)
                and analyzer._bearoff_db is not None, f"Different native settings at {name}")
        progress = []
        def cube_event(done, total, partial):
            progress.append((done, total))
        result = analyzer.cube_action(**pos["inputs"], progress_callback=cube_event)
        require(result.equity_nd_se is not None and result.equity_nd_se > target, f"Unexpected ND-SE for {name}")
        analyzer.set_seed(42)
        without = analyzer.cube_action(**pos["inputs"])
        cases[name] = dict(result=asdict(result), without_progress=asdict(without), progress=progress)
        if name in ("serial-fixed", "parallel-fixed"):
            cubeless_progress = []
            def cubeless_event(done, total):
                cubeless_progress.append((done, total))
            board = pos["post_move_inputs"]["board"]
            result = native.evaluate_board(board, board, cubeless_event)
            analyzer.set_seed(42)
            without = native.evaluate_board(board, board)
            cases[name + "-cubeless"] = dict(result=result, without_progress=without, progress=cubeless_progress)
    write_json(output, dict(extension=file_identity(extension), python=file_identity(Path(engine.__file__)),
                            model=engine.PRODUCTION_MODEL, weights=weights, bearoff_db=database,
                            position=pos["id"], inputs=pos["inputs"], post_move_inputs=pos["post_move_inputs"],
                            corpus=file_identity(path), profiles=PROFILES, seed=42, cases=cases))


def summarize(rows):
    first = rows["before"]
    profiles = first["profiles"]
    require(profiles == [list(p) for p in PROFILES], "Unexpected full-rollout profiles")
    counts = {}
    all_equal = True
    for arm in ARMS:
        require(all(rows[arm][k] == first[k] for k in ("position", "inputs", "post_move_inputs", "corpus",
                                                       "model", "weights", "bearoff_db", "profiles", "seed")),
                "Mismatched inputs, trial policy or seed")
    for name, threads, trials, depth, ply, target in profiles:
        keys = (name, name + "-cubeless") if name in ("serial-fixed", "parallel-fixed") else (name,)
        batches = 2 if target else 1
        for key in keys:
            cases = {arm: rows[arm]["cases"][key] for arm in ARMS}
            require(cases["single_claim"]["result"] == cases["before"]["result"],
                    f"Changing the work allocation changed the analytics for {key}")
            all_equal &= cases["selected"]["result"] == cases["before"]["result"]
            require(all(case["result"] == case["without_progress"] for case in cases.values()),
                    f"Requesting a progress callback changed the analytics for {key}")
            counts[key] = {}
            for arm, case in cases.items():
                events = case["progress"]
                require(all(1 <= done <= total == trials for done, total in events), f"Invalid callback at {key}")
                require(sum(done == trials for done, _ in events) == batches, f"Missing completion(s) at {key}")
                minimum = 8 if threads > 1 and depth == 0 and arm != "single_claim" else 1
                interval = max(minimum, trials // 100)
                expected = batches * ((trials + interval - 1) // interval)
                require(len(events) == expected, f"{arm} {key}: {len(events)} reports vs {expected}")
                counts[key][arm] = len(events)
    require(all(set(row["cases"]) == set(counts) for row in rows.values()), "Different case set in archive")
    for arm in ARMS:
        for suffix in ("fixed", "target", "fixed-cubeless"):
            cases = rows[arm]["cases"]
            require(cases["parallel-" + suffix]["result"] == cases["serial-" + suffix]["result"],
                    f"Serial and parallel results differ at {arm}/{suffix}")
    return dict(counts=counts, all_analytics_equal_between_builds=all_equal, same_serial_parallel=True,
                same_with_and_without_callback=True, binaries={arm: rows[arm]["extension"] for arm in ARMS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, type=Path, metavar=("SOURCE", "OUTPUT"))
    parser.add_argument("--record", type=Path, help="Write a new evidence directory inside the engine checkout")
    parser.add_argument("--verify", type=Path, default=HERE, help="Audit existing data, defaults to this directory")
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), type=Path)
    args = parser.parse_args()
    if args.worker:
        measure(*args.worker)
        return
    if args.record:
        dest = inside_repo(args.record)
        dest.mkdir(parents=True, exist_ok=False)
        with TemporaryDirectory(dir=dest) as temp:
            rows = {}
            for arm in ARMS:
                source = getattr(args, arm)
                if source is None:
                    parser.error(f"Missing build for {arm}")
                output = Path(temp) / (arm + ".json")
                subprocess.run([sys.executable, "-B", str(__file__), "--worker", str(source), str(output)], check=True)
                rows[arm] = read(output)
            summary = summarize(rows)
            with ZipFile(dest / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
                for arm in ARMS:
                    archive.write(Path(temp) / (arm + ".json"), arm + ".json")
            write_json(dest / "summary.json", summary)
    else:
        dest = inside_repo(args.verify)
        with ZipFile(dest / "full_outputs.zip") as archive:
            rows = {arm: json.loads(archive.read(arm + ".json")) for arm in ARMS}
        summary = summarize(rows)
        require(read(dest / "summary.json") == summary, "Saved callback summary differs from full outputs")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
