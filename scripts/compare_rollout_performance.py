# SPDX-License-Identifier: MPL-2.0
"""Pair ordinary rollout analytics on two engine builds in fresh processes.

The fixed cases cover a match contact position and a money pure race for each
API. The 3T preset is unmodified (360 games). Full rollouts use 3P checker and
cube play all the way through a game, with a smaller fixed 36-game batch for
this performance/regression check. The separate truth run uses 1,296 games.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import multi_ply_rollout_truth as truth

ROOT = truth.ROOT
DATA = truth.DATA
APIS = ("checker_play", "cube_action", "post_move_analytics")
CASES = ("match-attacking-01", "money-purerace-01")


def checked_path(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Expected a path inside the engine repository: {path}")
    return path


def snapshot(output, python_dir, extension, manifest):
    """Freeze exactly the public modules, binary, and pinned NN/DB assets."""
    identities = {}
    def copy(src, relative):
        dst = output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        identities[relative.as_posix()] = hashlib.sha256(dst.read_bytes()).hexdigest()
    extension = checked_path(extension)
    copy(extension, Path("build") / extension.name)
    src = checked_path(python_dir) / "bgsage"
    for module in sorted(src.rglob("*.py")):
        copy(module, Path("python/bgsage") / module.relative_to(src))
    for item in manifest["assets"]:
        rel = Path(item["path"])
        if rel.parts[0] in ("models", "data"):
            copy(truth.FROZEN / rel, rel)
            if identities[rel.as_posix()] != item["sha256"]:
                raise ValueError(f"Snapshot did not reproduce pinned asset {rel}")
    truth.save(output / "hashes.json", identities)


def worker(snapshot_dir, output, threads, profiles, full_trials):
    sys.path[:0] = [str(snapshot_dir / "python"), str(snapshot_dir / "build")]
    import bgsage
    import bgbot_cpp
    if not all(Path(m.__file__).resolve().is_relative_to(snapshot_dir) for m in (bgsage, bgbot_cpp)):
        raise RuntimeError("Unexpected engine import")
    sets = {}
    for kind in ("checker_play", "cube_action"):
        path = snapshot_dir / "data/multi_ply_benchmark" / (kind + ".json")
        sets[kind] = {p["id"]:p for p in truth.read(path)["positions"]}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for profile in profiles:
            kwargs = {}
            if profile == "full":
                config = dict(truth.CONFIG, n_trials=full_trials)
                # Force the usual full-rollout root move prefilter for the public
                # checker call. Inside each game, use full-depth 3P throughout.
                config.pop("prefilter_threshold")
                trial = bgsage.TrialEvalConfig(ply=3)
                kwargs.update(config, checker=trial, checker_late=trial, cube=trial, cube_late=trial)
            analyzer = bgsage.BgBotAnalyzer(eval_level="truncated3" if profile=="3T" else "rollout",
                parallel_threads=threads, weights=bgsage.WeightConfig.from_model("stage11"), **kwargs)
            for api in APIS:
                for case in CASES:
                    checker = api == "checker_play"
                    pid = ("checker-" if checker else "cube-") + case
                    p = sets["checker_play" if checker else "cube_action"][pid]
                    inputs = p["post_move_inputs"] if api == "post_move_analytics" else p["inputs"]
                    seed = truth.seed_for(p)
                    analyzer.set_seed(seed)
                    call = getattr(analyzer, api)
                    start = time.perf_counter_ns()
                    result = call(**inputs)
                    elapsed = (time.perf_counter_ns()-start)/1e9
                    record = dict(profile=profile, api=api, position_id=pid, seconds=elapsed,
                                  seed=seed,inputs=inputs, analytics=asdict(result))
                    stream.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
                    stream.flush()
                    print(f"{profile} {api} {pid} {elapsed:.3f}s", flush=True)


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def compare(base, candidate):
    b, c = base["analytics"], candidate["analytics"]
    diffs = {"max_abs_probability":0., "max_abs_equity":0.}
    def numeric(x, y):
        for name in x:
            if name == "probs":
                diffs["max_abs_probability"] = max(diffs["max_abs_probability"],
                        *(abs(x[name][k]-y[name][k]) for k in x[name]))
            elif "equity" in name and not name.endswith("_se") and isinstance(x[name],(int,float)):
                diffs["max_abs_equity"] = max(diffs["max_abs_equity"], abs(x[name]-y[name]))
    if base["api"] == "checker_play":
        lb = [tuple(m["board"]) for m in b["moves"]]
        lc = [tuple(m["board"]) for m in c["moves"]]
        if set(lb) != set(lc):
            raise ValueError("Legal move endpoints changed")
        diffs.update(best_action_same=lb[0]==lc[0], move_order_same=lb==lc)
        by_board={tuple(m["board"]):m for m in c["moves"]}
        for m in b["moves"]:
            numeric(m,by_board[tuple(m["board"])])
    else:
        numeric(b,c)
        if base["api"]=="cube_action":
            diffs["best_action_same"] = b["optimal_action"]==c["optimal_action"]
    return dict(diffs,exact_fields_equal=b==c)


def paired(args):
    output=checked_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest=truth.read(truth.OUT/"manifest.json")
    baseline_exe=args.baseline_extension or truth.FROZEN / manifest["assets"][0]["path"]
    arms={"baseline": (args.baseline_python,baseline_exe),
          "candidate": (args.candidate_python,args.candidate_extension)}
    for name,(src,exe) in arms.items():
        snapshot(output/name,src,exe,manifest)
    truth.save(output/"settings.json",dict(threads=args.threads,cpu_mask=args.cpu_mask,profiles=args.profiles,
               full_trials=args.full_trials, repeats=args.repeats,cases=CASES,
               reference_manifest_sha256=truth.digest((truth.OUT/"manifest.json").read_bytes())))
    pairs=[]
    for i in range(args.repeats):
        order=("baseline","candidate") if i%2==0 else ("candidate","baseline")
        for arm in order:
            cmd=[sys.executable,str(Path(__file__).resolve()),"worker",
                 "--worker-snapshot",str(output/arm),"--output",str(output/f"{i}-{arm}.jsonl"),
                 "--threads",str(args.threads),"--profiles",*args.profiles,"--full-trials",str(args.full_trials)]
            if args.cpu_mask is not None:
                cmd.extend(["--cpu-mask",hex(args.cpu_mask)])
            subprocess.run(cmd,check=True,cwd=ROOT)
        base=read_rows(output/f"{i}-baseline.jsonl")
        other=read_rows(output/f"{i}-candidate.jsonl")
        for b,c in zip(base,other,strict=True):
            key=("profile","api","position_id","seed","inputs")
            if any(b[k]!=c[k] for k in key):
                raise ValueError("Unpaired inputs")
            pairs.append(dict(repeat=i,profile=b["profile"],api=b["api"],position_id=b["position_id"],
                baseline_seconds=b["seconds"],candidate_seconds=c["seconds"],**compare(b,c)))
        aggregates=[]
        for profile in args.profiles:
            for api in APIS:
                sub=[r for r in pairs if r["profile"]==profile and r["api"]==api]
                a=sum(r["baseline_seconds"] for r in sub)
                z=sum(r["candidate_seconds"] for r in sub)
                aggregates.append(dict(profile=profile,api=api,calls=len(sub),baseline_seconds=a,
                        candidate_seconds=z,speedup=a/z,all_equal=all(r["exact_fields_equal"] for r in sub)))
        truth.save(output/"summary.json",dict(pairs=pairs,groups=aggregates))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("paired","worker"))
    ap.add_argument("--candidate-extension",type=Path)
    ap.add_argument("--baseline-extension",type=Path,
                    help="Use a matched-toolchain control extension")
    ap.add_argument("--baseline-python",type=Path,default=truth.FROZEN/"python",
                    help="Python package root corresponding to the control extension")
    ap.add_argument("--candidate-python",type=Path,default=ROOT/"python")
    ap.add_argument("--worker-snapshot",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--cpu-mask",type=lambda v:int(v,0),help="Optional logical CPU affinity of each timing worker")
    ap.add_argument("--profiles",nargs="+",choices=("3T","full"),default=["3T","full"])
    ap.add_argument("--full-trials",type=int,default=36)
    ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args()
    if min(args.full_trials,args.threads,args.repeats)<1:
        ap.error("Require positive --full-trials, --threads and --repeats")
    if args.action=="worker":
        truth.set_cpu_mask(args.cpu_mask)
        worker(checked_path(args.worker_snapshot),checked_path(args.output),args.threads,args.profiles,args.full_trials)
    else:
        if args.candidate_extension is None:
            ap.error("--candidate-extension required for paired run")
        paired(args)
