# SPDX-License-Identifier: MPL-2.0
"""Compare complete public analytics from isolated Python/native pairs, alternating case order."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def evaluate(conn, binary, python, cores):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    else:
        import psutil
        psutil.Process().cpu_affinity(cores)
    engine, _ = load_engine(binary, python)
    weights = engine.WeightConfig.from_model("stage11")
    analyzers = {}
    while (case := conn.recv()) is not None:
        level, api, inputs = case
        if level not in analyzers:
            analyzers[level] = engine.BgBotAnalyzer(
                weights=weights, eval_level=level, cubeful=True, parallel_threads=len(cores),
                bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        start = perf_counter()
        result = getattr(analyzers[level], api)(**inputs)
        conn.send((perf_counter() - start, asdict(result)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", nargs=3, metavar=("NAME", "BUILD", "PYTHON"), required=True)
    parser.add_argument("--cores", type=int, nargs="+", default=[12, 13, 14, 15])
    parser.add_argument("--levels", nargs="+", default=["2ply", "3ply", "4ply"])
    parser.add_argument("--apis", nargs="+", default=["checker_play"],
                        choices=["checker_play", "cube_action", "post_move_analytics"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--deep-per-group", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = {}
    for api in args.apis:
        dataset = "checker_play" if api == "checker_play" else "cube_action"
        positions = json.loads((ROOT / f"data/multi_ply_benchmark/{dataset}.json").read_bytes())["positions"]
        for pos in positions:
            key = "post_move_inputs" if api == "post_move_analytics" else "inputs"
            groups.setdefault((api, pos["mode"], pos["game_plan"]), []).append((api, pos["id"], pos[key]))
    ctx = get_context("spawn")
    processes, pipes, hashes = [], {}, {}
    for name, build, python in args.arm:
        binary = next(ROOT / build / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                      if (ROOT / build / ("bgbot_cpp" + suffix)).is_file())
        hashes[name] = hashlib.sha256(binary.read_bytes()).hexdigest()
        parent, child = ctx.Pipe()
        process = ctx.Process(target=evaluate, args=(child, binary, ROOT / python, args.cores))
        process.start()
        child.close()
        processes.append(process)
        pipes[name] = parent
    names = list(pipes)
    summary = {}
    try:
        with ZipFile(args.output / "outputs.zip", "x", ZIP_DEFLATED) as z, z.open("results.jsonl", "w") as f:
            for level in args.levels:
                n = args.deep_per_group if level == "4ply" else args.per_group
                for rep in range(args.repeats):
                    selected = [group[(rep * n + i) % len(group)] for i in range(n) for group in groups.values()]
                    for i, (api, pid, inputs) in enumerate(selected):
                        offset = (i + rep) % len(names)
                        outputs = {}
                        for name in names[offset:] + names[:offset]:
                            conn = pipes[name]
                            conn.send((level, api, inputs))
                            seconds, result = conn.recv()
                            outputs[name] = result
                            entry = summary.setdefault(level, {}).setdefault(name, dict(seconds=0, calls=0, exact=0))
                            entry["seconds"] += seconds
                            entry["calls"] += 1
                            row = dict(arm=name, level=level, api=api, repeat=rep, position_id=pid,
                                       inputs=inputs, elapsed_seconds=seconds, analytics=result)
                            f.write((json.dumps(row, separators=(",", ":")) + "\n").encode())
                        for name in names:
                            summary[level][name]["exact"] += outputs[name] == outputs[names[0]]
                    print(level, rep, summary[level], flush=True)
                (args.output / "summary.json").write_text(json.dumps(dict(sha256=hashes, levels=summary), indent=2))
    finally:
        for conn in pipes.values():
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


if __name__ == "__main__":
    main()
