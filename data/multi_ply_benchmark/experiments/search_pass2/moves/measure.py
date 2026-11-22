# SPDX-License-Identifier: MPL-2.0
"""Time public search with one independently loaded native implementation at a time."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine
from multi_ply_rollout_truth import set_cpu_mask

APIS = ("checker_play", "cube_action", "post_move_analytics")

def worker(pipe, binary, python_dir, threads, mask):
    set_cpu_mask(mask)
    engine, _ = load_engine(binary, python_dir)
    models = {}
    pipe.send(sha256(binary.read_bytes()).hexdigest())
    while (request := pipe.recv()) is not None:
        level, api, inputs = request
        if level not in models:
            models[level] = engine.BgBotAnalyzer(
                weights=engine.WeightConfig.from_model("stage11"), eval_level=level,
                bearoff_db=str(ROOT / "data/bearoff_1sided.db"),
                cubeful=True, parallel_threads=threads,
            )
        started = perf_counter()
        value = getattr(models[level], api)(**inputs)
        elapsed = perf_counter() - started
        pipe.send((elapsed, asdict(value)))

def cases(full_corpus):
    positions = {}
    for name in APIS[:2]:
        p = ROOT / "data/multi_ply_benchmark" / (name + ".json")
        positions[name] = json.loads(p.read_bytes())["positions"]
    rows = []
    if full_corpus:
        for api in APIS:
            name = "checker_play" if api == "checker_play" else "cube_action"
            key = "post_move_inputs" if api == "post_move_analytics" else "inputs"
            rows.extend((api, r["id"], r[key]) for r in positions[name])
        return rows
    for i, plan in enumerate(("attacking", "priming", "anchoring", "racing", "purerace")):
        for j, mode in enumerate(("money", "match")):
            api = APIS[(2 * i + j) % 3]
            pool = positions["checker_play" if api == "checker_play" else "cube_action"]
            eligible = [r for r in pool if (r["game_plan"], r["mode"]) == (plan, mode)]
            row = min(eligible, key=lambda r: sha256(("search-moves-v2:" + r["id"]).encode()).digest())
            inputs = row["post_move_inputs" if api == "post_move_analytics" else "inputs"]
            rows.append((api, row["id"], inputs))
    return rows

def main(args):
    set_cpu_mask(args.mask)
    args.output.mkdir(parents=True, exist_ok=False)
    builds = dict(spec.split("=", 1) for spec in args.build)
    names = list(builds)
    hashes, workers, pipes = {}, [], {}
    ctx = get_context("spawn")
    try:
        for name, path in builds.items():
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(
                child, Path(path).resolve(), args.python_dir.resolve(), args.threads, args.mask))
            proc.start()
            child.close()
            workers.append(proc)
            pipes[name] = parent
            hashes[name] = parent.recv()
        selected = cases(args.all_positions)
        meta = {"hashes": hashes, "mask": args.mask, "threads": args.threads, "levels": args.level,
                "cases_per_repeat": len(selected)}
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        with (args.output / "results.jsonl").open("x", encoding="utf8") as out:
            for spec in args.level:
                level, repeats = spec.split("=")
                for repeat in range(int(repeats)):
                    for i, (api, pid, inputs) in enumerate(selected):
                        offset = (i + repeat // 2) % len(names)
                        order = names[offset:] + names[:offset]
                        if repeat % 2:
                            order.reverse()
                        for name in order:
                            pipes[name].send((level, api, inputs))
                            elapsed, analytics = pipes[name].recv()
                            row = dict(level=level, repeat=repeat, api=api, position_id=pid, arm=name,
                                       inputs=inputs, elapsed=elapsed, analytics=analytics)
                            out.write(json.dumps(row, separators=(",", ":")) + "\n")
                            out.flush()
                    print(level, repeat + 1, flush=True)
    finally:
        for c in pipes.values():
            try:
                c.send(None)
            except (EOFError, OSError):
                pass
            c.close()
        for proc in workers:
            proc.join(3)
            if proc.is_alive():
                proc.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True, metavar="NAME=BINARY")
    parser.add_argument("--level", action="append", required=True, metavar="LEVEL=REPEATS")
    parser.add_argument("--python-dir", default=ROOT / "build/pass2_checkpoint/python", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--all-positions", action="store_true", help="Use all 300 public calls at each level.")
    parser.add_argument("--mask", type=lambda n: int(n, 0), default=15)
    main(parser.parse_args())
