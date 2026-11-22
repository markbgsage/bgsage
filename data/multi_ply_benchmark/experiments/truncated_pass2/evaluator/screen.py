# SPDX-License-Identifier: MPL-2.0
"""Compare two frozen engines with identical seeds, alternating order over ten stratified calls."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
DATA = ROOT / "data/multi_ply_benchmark"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine
from compare_truncated_benchmark import score_api
sys.path.insert(0, str(DATA / "experiments/truncated_search/evaluator"))
from measure import panel


def score_outputs(directory):
    truth = {(r["api"], r["position_id"]): r for line in (DATA / "rollout_positions/results.jsonl").read_bytes().splitlines()
             if (r := json.loads(line))}
    by_call = defaultdict(dict)
    with ZipFile(directory / "outputs.zip") as archive:
        for line in archive.read("results.jsonl").splitlines():
            r = json.loads(line)
            by_call[r["level"], r["api"], r["position_id"]][r["arm"]] = r
    by_api = defaultdict(list)
    for (level, api, pid), arms in by_call.items():
        assert arms["baseline"]["seed"] == arms["candidate"]["seed"]
        by_api[level, api].append((arms["baseline"], arms["candidate"], truth[api, pid]))
    report = {"/".join(key): score_api(key[1], group) for key, group in by_api.items()}
    (directory / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")


def run(pipe, folder, cores):
    os.sched_setaffinity(0, cores)
    binary = next(folder.glob("bgbot_cpp.*.so"))
    engine, _ = load_engine(binary, folder / "python")
    models = {}
    pipe.send(hashlib.sha256(binary.read_bytes()).hexdigest())
    while (row := pipe.recv()) is not None:
        level = row["level"]
        if level not in models:
            models[level] = engine.BgBotAnalyzer(
                weights=engine.WeightConfig.from_model("stage11"), eval_level=level, cubeful=True,
                parallel_threads=len(cores), bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        model = models[level]
        model.set_seed(row["seed"])
        start = perf_counter()
        result = getattr(model, row["api"])(**row["inputs"])
        pipe.send((perf_counter() - start, asdict(result)))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    workers = []
    pipes = []
    summary = {"hashes": {}, "levels": {}}
    arms = ["baseline", "candidate"]
    try:
        for arm, folder in zip(arms, [args.baseline, args.candidate]):
            parent, child = context.Pipe()
            process = context.Process(target=run, args=(child, folder.resolve(), args.cores))
            process.start()
            child.close()
            workers.append(process)
            pipes.append(parent)
            summary["hashes"][arm] = parent.recv()
        with (ZipFile(DATA / "experiments/truncated_16t/full_outputs.zip") as archive,
              ZipFile(args.output / "outputs.zip", "x", ZIP_DEFLATED) as outputs,
              outputs.open("results.jsonl", "w") as records):
            for level in args.levels:
                totals = dict(baseline=0, candidate=0, exact=0, pairs=0)
                for i, row in enumerate(panel(archive, level)):
                    pair = {}
                    for index in ([1, 0] if i % 2 else [0, 1]):
                        pipes[index].send(row)
                        elapsed, result = pipes[index].recv()
                        arm = arms[index]
                        totals[arm] += elapsed
                        pair[arm] = result
                        record = dict(row, arm=arm, elapsed_seconds=elapsed, analytics=result)
                        records.write((json.dumps(record, separators=(",", ":")) + "\n").encode())
                    totals["exact"] += pair["baseline"] == pair["candidate"]
                    totals["pairs"] += 1
                summary["levels"][level] = totals
                print(level, totals, flush=True)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        for pipe in pipes:
            pipe.send(None)
        for worker in workers:
            worker.join()
    score_outputs(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--levels", nargs="+", default=["2T", "3T"])
    parser.add_argument("--cores", type=int, nargs="+", default=[4, 5, 6, 7])
    main(parser.parse_args())
