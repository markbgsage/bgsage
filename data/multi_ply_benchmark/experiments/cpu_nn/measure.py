# SPDX-License-Identifier: MPL-2.0
"""Case-paired CPU timing. One preloaded process per arm, one active arm at a time."""
import argparse
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys
from time import perf_counter
here=Path(__file__).resolve().parent
root=here.parents[3]
sys.path.insert(0,str(root/"scripts"))
from benchmark_multi_ply import APIS, load_engine

def worker(pipe,arm,extension,threads,out):
    engine,ext=load_engine(extension)
    model=engine.BgBotAnalyzer(weights=engine.WeightConfig.from_model("stage11"),eval_level="3ply",
        cubeful=True,parallel_threads=threads,bearoff_db=str(root/"data/bearoff_1sided.db"))
    pipe.send(str(ext))
    with (out/f"{arm}.jsonl").open("x") as f:
        while (request:=pipe.recv()) is not None:
            api,case,inputs=request
            begin=perf_counter()
            ret=getattr(model,api)(**inputs)
            elapsed=perf_counter()-begin
            result=asdict(ret)
            digest=hashlib.sha256(json.dumps(result,separators=(",",":"),sort_keys=True).encode()).hexdigest()
            f.write(json.dumps(dict(api=api,position_id=case,inputs=inputs,analytics=result,elapsed=elapsed),
                               separators=(",",":"))+"\n")
            pipe.send((elapsed,digest))

def main(a):
    builds={}
    for spec in a.build:
        name,sep,path=spec.partition("=")
        if not sep or not name.isidentifier() or name in builds:
            raise ValueError("Each --build must have a distinct NAME=path/to/native-extension")
        builds[name]=Path(path).resolve()
    if len(builds)<2:
        raise ValueError("At least two builds are needed for paired comparisons")
    a.arms=list(builds)
    if not a.output.resolve().is_relative_to(root):
        raise ValueError("Output must be inside the engine checkout")
    a.output.mkdir(parents=True,exist_ok=False)
    rng=Random(a.seed)
    pos={api:json.loads((root/"data/multi_ply_benchmark"/(api+".json")).read_bytes())["positions"]
             for api in ["checker_play","cube_action"]}
    ctx=get_context("spawn")
    conns={}
    workers=[]
    rows=[]
    try:
        for arm in a.arms:
            c,d=ctx.Pipe()
            p=ctx.Process(target=worker,args=(d,arm,builds[arm],a.threads,a.output))
            p.start()
            d.close()
            conns[arm]=c
            workers.append(p)
        metadata = {
            "threads": a.threads,
            "case_order_seed": a.seed,
            "builds": {}
        }
        for arm, connection in conns.items():
            extension = Path(connection.recv())
            metadata["builds"][arm] = {
                "extension": str(extension.relative_to(root)),
                "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            }
        (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        # Every arm sees the same ordered cases, so TT histories match.
        for api in APIS:
            cases=pos["cube_action" if api=="post_move_analytics" else api][:]
            rng.shuffle(cases)
            for idx,p in enumerate(cases):
                inputs=dict(p["post_move_inputs"] if api=="post_move_analytics" else p["inputs"])
                if api=="cube_action":
                    inputs["incl_2ply_details"]=False
                row=dict(api=api,id=p["id"],elapsed={},hash={})
                # Shuffle a fresh order every time all arms have had each slot.
                if idx%len(a.arms)==0:
                    order=a.arms[:]
                    rng.shuffle(order)
                offset=idx%len(a.arms)
                for arm in order[offset:]+order[:offset]:
                    c=conns[arm]
                    c.send((api,p["id"],inputs))
                    elapsed,digest=c.recv()
                    row["elapsed"][arm]=elapsed
                    row["hash"][arm]=digest
                rows.append(row)
            summary={arm:sum(r["elapsed"][arm] for r in rows if r["api"]==api) for arm in a.arms}
            print(api, summary,flush=True)
        (a.output/"paired.json").write_text(json.dumps(rows,separators=(",",":")))
        mismatches=[r["id"] for r in rows if len(set(r["hash"].values()))>1]
        print("mismatches",mismatches,flush=True)
        if mismatches:
            raise RuntimeError("public value parity failed")
    finally:
        for c in conns.values():
            try:
                c.send(None)
            except (BrokenPipeError,EOFError):
                pass
        for p in workers:
            p.join(timeout=3)
        for p in workers:
            if p.is_alive():
                p.terminate()
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--threads",type=int,required=True)
    p.add_argument("--build",action="append",required=True,metavar="NAME=EXTENSION")
    p.add_argument("--seed",type=int,default=293441)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
