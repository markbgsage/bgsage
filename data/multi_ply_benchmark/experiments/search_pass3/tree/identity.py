# SPDX-License-Identifier: MPL-2.0
"""Check exact public results across cold workers, model switches and thread counts."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from prepare import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask

SNAKE = [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 1, 0, 2, 3, 3, 2, 2, 1, -7, -7, 0]


def worker(conn, ext, python_dir, width, order):
    set_cpu_mask(15)
    engine, _ = load_engine(ext, python_dir)
    rows = json.loads((DATA / "checker_play.json").read_bytes())["positions"]
    checker = next(r for r in rows if r["id"] == "checker-match-attacking-01")
    rows = json.loads((DATA / "cube_action.json").read_bytes())["positions"]
    cube = next(r for r in rows if r["id"] == "cube-money-purerace-01")
    for model in order:
        ana = engine.BgBotAnalyzer(weights=engine.WeightConfig.from_model(model), eval_level="3ply",
            parallel_threads=width, cubeful=True, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
        for api, inputs in [("checker_play", checker["inputs"]), ("cube_action", {"board": SNAKE}),
                            ("post_move_analytics", cube["post_move_inputs"])]:
            result = asdict(getattr(ana, api)(**inputs))
            conn.send(dict(model=model, threads=width, api=api, inputs=inputs, analytics=result))
    conn.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("extension", type=Path)
    cli.add_argument("python_dir", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    args.output.mkdir(exist_ok=False)
    ctx = mp.get_context("spawn")
    canonical = {}
    n = 0
    switches = 0
    with gzip.open(args.output / "calls.jsonl.gz", "wt") as f:
        for width, order in [(1, ["stage11"]), (1, ["stage9"]),
                             (4, ["stage9", "stage11", "stage9"]),
                             (16, ["stage11", "stage9", "stage11"])]:
            rx, tx = ctx.Pipe(False)
            proc = ctx.Process(target=worker, args=(tx, args.extension, args.python_dir, width, order))
            proc.start()
            tx.close()
            for call in range(3 * len(order)):
                row = rx.recv()
                key = row["model"], row["api"]
                value = hashlib.sha256(json.dumps(row["analytics"], sort_keys=True).encode()).hexdigest()
                if key in canonical and canonical[key] != value:
                    raise AssertionError(f"Warm/thread-count result differs: {key}, {width}, {call}")
                canonical[key] = value
                n += 1
                switches += call >= 3
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            proc.join()
            if proc.exitcode:
                raise RuntimeError(proc.exitcode)
            rx.close()
    distinguish = sum(canonical["stage11", api] != canonical["stage9", api]
                      for api in ["checker_play", "cube_action", "post_move_analytics"])
    if not distinguish:
        raise AssertionError("The chosen positions failed to distinguish the two models")
    write_json(args.output / "summary.json", dict(extension=file_identity(args.extension), calls=n,
        controls=6, exact_against_cold=n-6, after_model_switch=switches,
        distinguishing_calls=distinguish, threads=[1, 4, 16], cpu_mask="0xf"))


if __name__ == "__main__":
    main()
