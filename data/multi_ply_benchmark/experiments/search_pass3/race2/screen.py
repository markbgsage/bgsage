# SPDX-License-Identifier: MPL-2.0
"""Compare public 2P calls with blocked, rotating binary order on all sixteen CPU threads."""
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
from time import perf_counter_ns
from zipfile import ZipFile, ZIP_DEFLATED

def root():
    return next(p for p in Path(__file__).resolve().parents if (p/'scripts/benchmark_multi_ply.py').is_file())

ROOT = root()
APIS = ("checker_play", "cube_action", "post_move_analytics")

def worker(pipe, build, python, cap, warm):
    os.environ["BGSAGE_RACE_TEST_THREADS"] = str(cap)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, range(16))
    else:
        import psutil
        psutil.Process().cpu_affinity(list(range(16)))
    sys.path.insert(0, str(ROOT/'scripts'))
    from benchmark_multi_ply import load_engine
    binary = next(p for s in EXTENSION_SUFFIXES if (p := ROOT/build/("bgbot_cpp"+s)).is_file())
    engine, _ = load_engine(binary, ROOT/python)
    analyzer = engine.BgBotAnalyzer(weights=engine.WeightConfig.from_model("stage11"), eval_level="2ply",
        parallel_threads=16, cubeful=True, bearoff_db=str(ROOT/"data/bearoff_1sided.db"))
    analyzer.checker_play(**warm)
    pipe.send(sha256(binary.read_bytes()).hexdigest())
    while (work := pipe.recv()) is not None:
        api, batch = work
        evaluate = getattr(analyzer, api)
        results = []
        for row in batch:
            start = perf_counter_ns()
            value = evaluate(**row["inputs"])
            elapsed = (perf_counter_ns() - start) / 1e9
            results.append((elapsed, value))
        pipe.send([(elapsed, asdict(v)) for elapsed,v in results])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs=4, action="append", metavar=("NAME","BUILD","PYTHON","ENV_CAP"), required=True)
    parser.add_argument("--race-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    options=parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    data=ROOT/"data/multi_ply_benchmark"
    positions={d:json.loads((data/f"{d}.json").read_text())["positions"] for d in ("checker_play","cube_action")}
    warm = next(p["inputs"] for p in positions["checker_play"] if p["game_plan"] == "attacking")
    collection={}
    for api in APIS:
        ds = "checker_play" if api=="checker_play" else "cube_action"
        key = "post_move_inputs" if api=="post_move_analytics" else "inputs"
        collection[api]=[dict(position_id=p["id"], inputs=p[key], mode=p["mode"], game_plan=p["game_plan"])
            for p in positions[ds] if not options.race_only or p["game_plan"]=="purerace"]
    reference=options.arm[0][0]
    meta={}
    output={}
    with ZipFile(options.out/"outputs.zip","x",ZIP_DEFLATED,compresslevel=2) as archive:
        for rep in range(options.repeats):
            ctx=get_context("spawn"); processes={}; pipes={}
            for name,build,python,cap in options.arm:
                parent,child=ctx.Pipe()
                proc=ctx.Process(target=worker,args=(child,build,python,cap,warm))
                proc.start(); child.close()
                processes[name]=proc; pipes[name]=parent
            try:
                for name,pipe in pipes.items():
                    meta[name]=pipe.recv()
                with archive.open(f"repeat-{rep}.jsonl","w") as out:
                    for a,api in enumerate(APIS):
                        rows=list(collection[api])
                        random.Random(10931+rep).shuffle(rows)
                        totals=defaultdict(float); pure=defaultdict(float); exact=defaultdict(int)
                        for start in range(0,len(rows),options.block):
                            part=rows[start:start+options.block]
                            names=list(pipes)
                            offset=(rep+a+start//options.block)%len(names)
                            block={}
                            for name in names[offset:]+names[:offset]:
                                pipes[name].send((api,part)); block[name]=pipes[name].recv()
                            for i,p in enumerate(part):
                                expected=block[reference][i][1]
                                for name,values in block.items():
                                    sec,result=values[i]; totals[name]+=sec
                                    if p["game_plan"]=="purerace": pure[name]+=sec
                                    exact[name]+=int(expected==result)
                                    row=dict(**p,arm=name,api=api,level="2ply",repeat=rep,elapsed_seconds=sec,analytics=result)
                                    out.write((json.dumps(row,separators=(",",":"))+"\n").encode())
                        group=output.setdefault(api,{})
                        for name in pipes:
                            group.setdefault(name,[]).append(dict(seconds=totals[name],pure_seconds=pure[name],
                                exact=exact[name],calls=len(rows)))
                        print(rep,api,{k:round(v,5) for k,v in pure.items()},flush=True)
            finally:
                for pipe in pipes.values():
                    try: pipe.send(None)
                    except (EOFError, BrokenPipeError, OSError): pass
                    pipe.close()
                for proc in processes.values():
                    proc.join(5)
                    if proc.is_alive(): proc.terminate(); proc.join()
            (options.out/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
        (options.out/"metadata.json").write_text(json.dumps(dict(binaries=meta,arms=options.arm,repeats=options.repeats,
            race_only=options.race_only,block=options.block,seed=10931,threads=16,warmup=warm),indent=2)+"\n")

if __name__=="__main__":
    main()
