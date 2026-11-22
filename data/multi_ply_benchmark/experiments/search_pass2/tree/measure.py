# SPDX-License-Identifier: MPL-2.0
"""Balanced, case-adjacent public 3P/4P timing; save every full analytics object."""
import argparse
import ctypes
from dataclasses import asdict
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

from prepare import ROOT, BASE

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, file_identity, inside_repo, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask


COUNTERS = ("doubles_endpoints", "non_doubles_endpoints", "batch_candidate_requests", "reused_candidates",
            "terminal_scans_skipped", "reused_batches", "dead_leaves", "dead_candidate_scores", "reader_nodes")


def worker(conn, binary, python_dir, threads, mask, profile=False):
    set_cpu_mask(mask)
    engine, _ = load_engine(binary, python_dir)
    models = {}
    weights = engine.WeightConfig.from_model("stage11")
    if profile:
        dll = ctypes.CDLL(str(binary))
        take = dll.take_tree_counts
        take.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        take.restype = None
    while (req := conn.recv()) is not None:
        level, api, inputs = req
        if level not in models:
            models[level] = engine.BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                parallel_threads=threads, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        cpu_start = time.process_time()
        start = time.perf_counter()
        result = getattr(models[level], api)(**inputs)
        wall = time.perf_counter() - start
        cpu = time.process_time() - cpu_start
        response = wall, cpu, asdict(result)
        if profile:
            values = (ctypes.c_longlong * len(COUNTERS))()
            take(values)
            response += (dict(zip(COUNTERS, values)),)
        conn.send(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=EXTENSION")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--levels", nargs="+", default=["3ply", "4ply"])
    p.add_argument("--apis", nargs="+", choices=APIS, default=APIS)
    p.add_argument("--per-stratum", type=int, default=1)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--counters", action="store_true")
    p.add_argument("--python-dir", type=Path, default=ROOT / "build/pass2_checkpoint/python")
    p.add_argument("--mask", type=lambda s: int(s, 0), default=0xf00)
    a = p.parse_args()
    out = inside_repo(a.output)
    out.mkdir(parents=True, exist_ok=False)
    binaries = {name: inside_repo(Path(path)) for name, path in [a.split("=", 1) for a in a.arm]}
    if "control" not in binaries:
        raise ValueError("Include the published control")
    python_dir = inside_repo(a.python_dir)
    files = {api: json.loads((DATA / ("checker_play.json" if api == "checker_play" else "cube_action.json")).read_bytes())["positions"] for api in a.apis}
    cases = []
    for level in a.levels:
        for api in a.apis:
            counts = {}
            for row in files[api]:
                group = row["mode"], row["game_plan"]
                counts[group] = counts.get(group, 0) + 1
                if counts[group] <= a.per_stratum:
                    inputs = row["post_move_inputs"] if api == "post_move_analytics" else row["inputs"]
                    cases.append((level, api, row["id"], inputs))
    study = dict(binaries={n: file_identity(path) for n, path in binaries.items()}, threads=a.threads, mask=hex(a.mask),
                 python_revision=BASE, per_stratum=a.per_stratum, seed=887450, cases=[])
    ctx = mp.get_context("spawn")
    with gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as full:
        for rep in range(a.rounds):
            rng = random.Random(study["seed"] + rep)
            order = cases[:]
            rng.shuffle(order)
            pool = {}
            try:
                for name, binary in binaries.items():
                    receiver, sender = ctx.Pipe()
                    proc = ctx.Process(target=worker, args=(sender, binary, python_dir, a.threads, a.mask, a.counters))
                    proc.start()
                    sender.close()
                    pool[name] = receiver, proc
                names = list(pool)
                for i, (level, api, pos, inputs) in enumerate(order):
                    if i % len(names) == 0:
                        rng.shuffle(names)
                    offset = i % len(names)
                    run_order = names[offset:] + names[:offset]
                    row = dict(level=level, api=api, position_id=pos, repeat=rep, order=run_order, arms={})
                    for name in run_order:
                        c, _ = pool[name]
                        c.send((level, api, inputs))
                        wall, cpu, result, *extra = c.recv()
                        digest = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                        row["arms"][name] = dict(wall=wall, cpu=cpu, sha256=digest)
                        if extra:
                            row["arms"][name]["counts"] = extra[0]
                        full.write(json.dumps(dict(arm=name, api=api, level=level, position_id=pos, repeat=rep,
                            inputs=inputs, analytics=result, wall=wall, cpu=cpu), separators=(",", ":"), allow_nan=False) + "\n")
                    study["cases"].append(row)
                    write_json(out / "measurements.json", study)
                    print(rep, i + 1, level, api, pos, {k: round(v["wall"],3) for k,v in row["arms"].items()},
                          len({x["sha256"] for x in row["arms"].values()}) == 1, flush=True)
            finally:
                for conn, proc in pool.values():
                    if proc.is_alive():
                        conn.send(None)
                    conn.close()
                    proc.join()


if __name__ == "__main__":
    main()
