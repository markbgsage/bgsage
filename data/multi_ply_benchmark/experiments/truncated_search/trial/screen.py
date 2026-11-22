# SPDX-License-Identifier: MPL-2.0
"""Measure rollout variants on identical seeded public calls in randomized order."""
import argparse
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine
from multi_ply_rollout_truth import set_cpu_mask

CASES = (
    ("checker_play", "checker-money-attacking-02"),
    ("checker_play", "checker-match-anchoring-06"),
    ("cube_action", "cube-match-priming-06"),
    ("cube_action", "cube-money-racing-06"),
    ("cube_action", "cube-money-purerace-02"),
    ("post_move_analytics", "cube-money-priming-02"),
    ("post_move_analytics", "cube-match-racing-02"),
)

def timed_worker(channel, ext, level, threads, mask):
    set_cpu_mask(mask)
    engine, _ = load_engine(ext)
    analyzer = engine.BgBotAnalyzer(
        weights=engine.WeightConfig.from_model("stage11"),
        eval_level=level, cubeful=True, parallel_threads=threads,
        bearoff_db=str(ROOT / "data/bearoff_1sided.db"),
    )
    channel.send(True)
    while (request := channel.recv()) is not None:
        api, inputs, seed = request
        analyzer.set_seed(seed)
        start = perf_counter()
        analytics = getattr(analyzer, api)(**inputs)
        elapsed = perf_counter() - start
        channel.send((elapsed, asdict(analytics)))

def main(args):
    set_cpu_mask(args.mask)
    args.output.mkdir(parents=True, exist_ok=False)
    builds = {}
    for spec in args.build:
        name, path = spec.split("=", 1)
        if name in builds:
            raise ValueError(f"Duplicate build name {name}")
        builds[name] = Path(path).resolve()
    metadata = {
        "threads": args.threads, "mask": args.mask, "repeats": args.repeats,
        "builds": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in builds.items()},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    positions = {}
    for name in ("checker_play", "cube_action"):
        dataset = ROOT / "data/multi_ply_benchmark" / (name + ".json")
        src = json.loads(dataset.read_bytes())
        positions.update((p["id"], p) for p in src["positions"])
    ctx = get_context("spawn")
    with (args.output / "results.jsonl").open("x", newline="\n") as out:
        for level in ("truncated2", "truncated3"):
            processes, channels = [], {}
            try:
                for name, ext in builds.items():
                    parent, child = ctx.Pipe()
                    proc = ctx.Process(target=timed_worker, args=(child, ext, level, args.threads, args.mask))
                    proc.start()
                    child.close()
                    processes.append(proc)
                    channels[name] = parent
                    if parent.recv() is not True:
                        raise RuntimeError(f"Cannot prepare {name}")
                rng = Random(7917)
                for repeat in range(1, args.repeats + 1):
                    for api, pid in CASES:
                        position = positions[pid]
                        inputs = position["post_move_inputs" if api == "post_move_analytics" else "inputs"]
                        seed_key = ("truncated-benchmark-v1:" + position["source"]["key"]).encode()
                        seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "big")
                        order = list(builds)
                        rng.shuffle(order)
                        for name in order:
                            channels[name].send((api, inputs, seed))
                            elapsed, analytics = channels[name].recv()
                            row = {
                                "level": level, "repeat": repeat, "api": api, "position_id": pid,
                                "arm": name, "inputs": inputs, "seed": seed, "elapsed": elapsed,
                                "analytics": analytics,
                            }
                            out.write(json.dumps(row, separators=(",", ":")) + "\n")
                            out.flush()
                        print(level, repeat, pid, flush=True)
            finally:
                for channel in channels.values():
                    try:
                        channel.send(None)
                    except (OSError, EOFError):
                        pass
                    channel.close()
                for proc in processes:
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True, metavar="NAME=DLL")
    parser.add_argument("--mask", type=lambda s: int(s, 0), default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
