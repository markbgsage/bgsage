# SPDX-License-Identifier: MPL-2.0
"""Alternate complete public calls between named CPU builds on a stratified corpus."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from importlib.machinery import EXTENSION_SUFFIXES
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import sys
from time import perf_counter, process_time
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine


def worker(conn, binary, python, cores, options):
    os.environ.update(options)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    else:
        import psutil
        psutil.Process().cpu_affinity(cores)
    engine, _ = load_engine(binary, python)
    weights = engine.WeightConfig.from_model("stage11")
    analyzers = {}
    while (item := conn.recv()) is not None:
        level, api, inputs = item
        if level not in analyzers:
            analyzers[level] = engine.BgBotAnalyzer(
                weights=weights, eval_level=level, cubeful=True, parallel_threads=len(cores),
                bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        start, cpu = perf_counter(), process_time()
        result = getattr(analyzers[level], api)(**inputs)
        elapsed, used = perf_counter() - start, process_time() - cpu
        conn.send((elapsed, used, asdict(result)))


def sample(api, count, rep):
    dataset = "checker_play" if api == "checker_play" else "cube_action"
    rows = json.loads((ROOT / f"data/multi_ply_benchmark/{dataset}.json").read_bytes())["positions"]
    cells = defaultdict(list)
    for row in rows:
        cells[row["mode"], row["game_plan"]].append(row)
    selected = []
    for cell in cells.values():
        selected.extend(cell[:count])
    random.Random(173 + rep).shuffle(selected)
    return selected


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--arm", nargs=3, action="append", required=True, metavar=("NAME", "BUILD", "PYTHON"))
    args.add_argument("--env", nargs=3, action="append", default=[], metavar=("ARM", "KEY", "VALUE"))
    args.add_argument("--cores", nargs="+", type=int, required=True)
    args.add_argument("--levels", nargs="+", default=["3ply", "4ply"])
    args.add_argument("--apis", nargs="+", default=["checker_play", "cube_action", "post_move_analytics"])
    args.add_argument("--per-cell", default=2, type=int, choices=range(1, 11))
    args.add_argument("--repeats", default=2, type=int)
    args.add_argument("--output", required=True, type=Path)
    cfg = args.parse_args()
    cfg.output.mkdir(parents=True, exist_ok=False)
    arms = {}
    ctx = get_context("spawn")
    for name, build, python in cfg.arm:
        path = next(ROOT / build / ("bgbot_cpp" + s) for s in EXTENSION_SUFFIXES
                    if (ROOT / build / ("bgbot_cpp" + s)).is_file())
        arms[name] = (path, ROOT / python, {k: v for who, k, v in cfg.env if who == name})
    meta = dict(args=vars(cfg) | {"output": str(cfg.output)},
                binaries={k: sha256(v[0].read_bytes()).hexdigest() for k, v in arms.items()})
    (cfg.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    with ZipFile(cfg.output / "outputs.zip", "x", ZIP_DEFLATED, compresslevel=4) as archive:
        for rep in range(cfg.repeats):
            processes, pipes = [], {}
            try:
                for arm, (binary, python, options) in arms.items():
                    parent, child = ctx.Pipe()
                    proc = ctx.Process(target=worker, args=(child, binary, python, cfg.cores, options))
                    proc.start()
                    child.close()
                    pipes[arm] = parent
                    processes.append(proc)
                with archive.open(f"repeat-{rep}.jsonl", "w") as out:
                    for level in cfg.levels:
                        for api in cfg.apis:
                            totals = defaultdict(float)
                            for i, row in enumerate(sample(api, cfg.per_cell, rep)):
                                key = "post_move_inputs" if api == "post_move_analytics" else "inputs"
                                names = list(arms)
                                rotate = (i + rep) % len(names)
                                for arm in names[rotate:] + names[:rotate]:
                                    pipes[arm].send((level, api, row[key]))
                                    seconds, cpu, result = pipes[arm].recv()
                                    record = dict(arm=arm, level=level, api=api, position_id=row["id"],
                                                  inputs=row[key], elapsed_seconds=seconds, cpu_seconds=cpu,
                                                  analytics=result)
                                    out.write((json.dumps(record, separators=(",", ":")) + "\n").encode())
                                    totals[arm] += seconds
                            print(rep, level, api, dict(totals), flush=True)
            finally:
                for pipe in pipes.values():
                    try:
                        pipe.send(None)
                    except (EOFError, BrokenPipeError, OSError):
                        pass
                    pipe.close()
                for proc in processes:
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.terminate()


if __name__ == "__main__":
    main()
