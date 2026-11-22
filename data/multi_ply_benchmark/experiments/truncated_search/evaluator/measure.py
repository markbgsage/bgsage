# SPDX-License-Identifier: MPL-2.0
"""Run interleaved, seeded public calls on a balanced timing panel."""
import argparse
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[4]
DATA = ROOT / "data/multi_ply_benchmark"
BASE = ROOT / "build/multi_ply_cpu_fixed"
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import load_engine

def worker(connection, folder):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, [4, 5, 6, 7])
    else:
        import psutil
        psutil.Process().cpu_affinity([4, 5, 6, 7])
    ext = next(folder.glob("bgbot_cpp.*.so" if sys.platform == "linux" else "bgbot_cpp.*.pyd"))
    engine, _ = load_engine(ext)
    models = {}
    connection.send(hashlib.sha256(ext.read_bytes()).hexdigest())
    while (row := connection.recv()) is not None:
        level = row["level"]
        if level not in models:
            models[level] = engine.BgBotAnalyzer(
                weights=engine.WeightConfig.from_model("stage11"), eval_level=level,
                cubeful=True, parallel_threads=4, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        model = models[level]
        model.set_seed(row["seed"])
        start = perf_counter()
        result = getattr(model, row["api"])(**row["inputs"])
        elapsed = perf_counter() - start
        connection.send((elapsed, asdict(result)))


def panel(archive, level):
    rows = [json.loads(line) for line in archive.read(f"{level}-candidate/results.jsonl").splitlines()]
    chosen = []
    plans = ["attacking", "priming", "anchoring", "racing", "purerace"]
    for i, plan in enumerate(plans):
        for j, mode in enumerate(("money", "match")):
            api = ("checker_play", "cube_action", "post_move_analytics")[(i * 2 + j) % 3]
            subset = [r for r in rows if (r["source_game_plan"], r["mode"], r["api"]) == (plan, mode, api)]
            chosen.append(min(subset, key=lambda row: hashlib.sha256(("trunc-evaluator-v1:" + row["position_id"]).encode()).digest()))
    return chosen


def main(args):
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Store experiments inside bgsage")
    out.mkdir(parents=True, exist_ok=False)
    ctx = get_context("spawn")
    pipes, workers, hashes = {}, [], {}
    full, summaries = [], []
    try:
        for arm in args.arms:
            folder = (args.binary_root / arm if args.binary_root else
                      BASE if arm == "control" else ROOT / "build/trunc_eval_study" / arm)
            conn, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, folder))
            proc.start()
            child.close()
            pipes[arm] = conn
            workers.append(proc)
            hashes[arm] = conn.recv()
        with ZipFile(DATA / "experiments/truncated_16t/full_outputs.zip") as reference:
            for level in args.levels:
                for i, row in enumerate(panel(reference, level)):
                    if i >= args.limit:
                        break
                    order = args.arms[i % len(args.arms):] + args.arms[:i % len(args.arms)]
                    summary = {"level": level, "api": row["api"], "position_id": row["position_id"],
                               "elapsed": {}, "matches_published_control": {}}
                    for arm in order:
                        pipes[arm].send(row)
                        seconds, result = pipes[arm].recv()
                        summary["elapsed"][arm] = seconds
                        summary["matches_published_control"][arm] = result == row["analytics"]
                        full.append({**row, "arm": arm, "elapsed_seconds": seconds, "analytics": result})
                    summaries.append(summary)
                    print(json.dumps(summary, separators=(",", ":")), flush=True)
                    (out / "summary.json").write_text(json.dumps({"hashes": hashes, "rows": summaries}, indent=2) + "\n", encoding="utf-8")
    finally:
        for conn in pipes.values():
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for proc in workers:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
        with ZipFile(out / "full_outputs.zip", "w", ZIP_DEFLATED) as archive:
            archive.writestr("records.jsonl", "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--levels", nargs="+", default=["2T","3T"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-root", type=Path, help="Directory with native binaries in arm subdirectories")
    main(parser.parse_args())
