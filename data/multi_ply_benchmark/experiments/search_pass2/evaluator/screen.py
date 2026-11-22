# SPDX-License-Identifier: MPL-2.0
"""Run the same stratified deterministic public calls in rotating binary order."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / "build/det_eval2"
DATA = ROOT / "data/multi_ply_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine, APIS


def binary(arm):
    folder = ROOT / "build/multi_ply_truncated_final" if arm == "control" else WORK / arm
    return next(folder.glob("bgbot_cpp.*.pyd"))


def worker(conn, arm, threads):
    psutil.Process().cpu_affinity([4, 5, 6, 7])
    engine, _ = load_engine(binary(arm), ROOT / "build/pass2_checkpoint/python")
    weights = engine.WeightConfig.from_model("stage11")
    analyzers = {}
    conn.send(hashlib.sha256(binary(arm).read_bytes()).hexdigest())
    while (row := conn.recv()) is not None:
        level = row["level"]
        if level not in analyzers:
            analyzers[level] = engine.BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                parallel_threads=threads, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        start = perf_counter()
        value = getattr(analyzers[level], row["api"])(**row["inputs"])
        conn.send((perf_counter() - start, asdict(value)))


def choose(level, limit, rep):
    rows = []
    for api in APIS:
        dataset = "checker_play" if api == "checker_play" else "cube_action"
        positions = json.loads((DATA / (dataset + ".json")).read_bytes())["positions"]
        groups = defaultdict(list)
        for p in positions:
            groups[p["game_plan"], p["mode"]].append(p)
        for group in groups.values():
            for p in sorted(group, key=lambda p: hashlib.sha256(("det-evaluator-v2:" + p["id"]).encode()).digest())[:limit]:
                inputs = dict(p["post_move_inputs"] if api == "post_move_analytics" else p["inputs"])
                if api == "cube_action":
                    inputs["incl_2ply_details"] = False
                rows.append(dict(level=level, api=api, position_id=p["id"], source=p["source"], inputs=inputs,
                                 source_game_plan=p["game_plan"], mode=p["mode"], repeat=rep, seed=None))
    Random(761112 + rep).shuffle(rows)
    return rows


def main(args):
    dest = args.output.resolve()
    if not dest.is_relative_to(ROOT): raise ValueError("Use in-repo output")
    dest.mkdir(exist_ok=False, parents=True)
    hashes = {}
    states = defaultdict(lambda: dict(seconds=0.0, calls=0, identical=0))
    with ZipFile(dest / "full_outputs.zip", "w", ZIP_DEFLATED) as archive, archive.open("records.jsonl", "w") as handle:
        for rep in range(1, args.repeats + 1):
            ctx = get_context("spawn")
            pipes, processes = {}, []
            try:
                for arm in args.arms:
                    conn, child = ctx.Pipe()
                    proc = ctx.Process(target=worker, args=(child, arm, args.threads))
                    proc.start()
                    child.close()
                    pipes[arm] = conn
                    processes.append(proc)
                    hashes[arm] = conn.recv()
                for level in args.levels:
                    limit = args.deep_per_group if level == "4ply" else args.per_group
                    for i, row in enumerate(choose(level, limit, rep)):
                        start = (i + rep - 1) % len(args.arms)
                        order = args.arms[start:] + args.arms[:start]
                        outputs = {}
                        for arm in order:
                            pipes[arm].send(row)
                            sec, output = pipes[arm].recv()
                            outputs[arm] = output
                            group = states[level, arm]
                            group["seconds"] += sec
                            group["calls"] += 1
                            handle.write((json.dumps({**row, "elapsed_seconds":sec, "arm":arm, "analytics":output},
                                                    separators=(",",":")) + "\n").encode())
                        for arm in order:
                            states[level, arm]["identical"] += outputs[arm] == outputs["control"]
                    summary = [{"level":level, "arm":arm, **s} for (level,arm), s in states.items()]
                    print(rep, level, json.dumps(summary), flush=True)
                    (dest / "summary.json").write_text(json.dumps({"hashes":hashes, "groups":summary}, indent=2) + "\n")
            finally:
                for conn in pipes.values():
                    try: conn.send(None)
                    except (OSError, BrokenPipeError): pass
                for proc in processes:
                    proc.join(timeout=4)
                    if proc.is_alive(): proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=["2ply", "3ply", "4ply"])
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--deep-per-group", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
