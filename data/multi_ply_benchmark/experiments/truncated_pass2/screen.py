# SPDX-License-Identifier: MPL-2.0
"""Interleave truncated-rollout implementations on a fixed sample of the public corpus."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
from importlib.machinery import EXTENSION_SUFFIXES
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys
from time import perf_counter, process_time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, file_identity, inside_repo, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask


def worker(pipe, binary, source, mask):
    set_cpu_mask(mask)
    engine, _ = load_engine(binary, source / "python")
    weights = engine.WeightConfig.from_model("stage11")
    analyzers = {}
    while (case := pipe.recv()) is not None:
        level = case["level"]
        if level not in analyzers:
            analyzers[level] = engine.BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                parallel_threads=mask.bit_count(), bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        analyzer = analyzers[level]
        analyzer.set_seed(case["seed"])
        start, cpu = perf_counter(), process_time()
        result = getattr(analyzer, case["api"])(**case["inputs"])
        seconds, used = perf_counter() - start, process_time() - cpu
        pipe.send((seconds, used, asdict(result)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs=3, action="append", required=True, metavar=("NAME","BUILD","SOURCE"))
    parser.add_argument("--apis", nargs="+", choices=APIS, default=APIS)
    parser.add_argument("--levels", nargs="+", choices=("1T","2T","3T"), default=["2T","3T"])
    parser.add_argument("--mask", type=lambda x: int(x, 0), default=0xffff)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    archive = DATA / "experiments/search_pass3/release/linux/full_outputs.zip"
    cases = []
    with ZipFile(archive) as frozen:
        for level in args.levels:
            seen = set()
            for line in frozen.read(f"candidate/{level}/results.jsonl").splitlines():
                row = json.loads(line)
                key = row["api"], row["mode"], row["source_game_plan"]
                if row["api"] in args.apis and key not in seen:
                    cases.append((level, row))
                    seen.add(key)
    random.Random(49761).shuffle(cases)
    workers, arms, times = {}, {}, {}
    ctx = get_context("spawn")
    try:
        for name, build, src in args.arm:
            folder = inside_repo(Path(build))
            binary = next(folder / ("bgbot_cpp" + s) for s in EXTENSION_SUFFIXES
                          if (folder / ("bgbot_cpp" + s)).is_file())
            arms[name] = dict(extension=file_identity(binary), source=str(src))
            times[name] = Counter()
            a, b = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(b, binary, inside_repo(Path(src)), args.mask))
            proc.start()
            b.close()
            workers[name] = a, proc
        write_json(out / "settings.json", dict(arms=arms, mask=args.mask, cases=len(cases),
                                               reference=file_identity(archive)))
        with gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as calls:
            for i, (level, case) in enumerate(cases):
                order = list(workers)
                offset = i % len(order)
                for name in order[offset:] + order[:offset]:
                    pipe, _ = workers[name]
                    pipe.send(case)
                    sec, cpu, analytics = pipe.recv()
                    record = {k: case[k] for k in ("position_id","seed","level","mode","source_game_plan","api","inputs")}
                    record.update(arm=name, elapsed_seconds=sec, cpu_seconds=cpu, analytics=analytics)
                    calls.write(json.dumps(record, separators=(",",":"), allow_nan=False) + "\n")
                    times[name][level] += sec
                print(i+1, level, case["position_id"], flush=True)
        write_json(out / "seconds.json", times)
    finally:
        for pipe, proc in workers.values():
            if proc.is_alive():
                pipe.send(None)
            pipe.close()
            proc.join()


if __name__ == "__main__":
    main()
