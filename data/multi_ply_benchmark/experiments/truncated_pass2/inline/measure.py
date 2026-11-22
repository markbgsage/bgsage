# SPDX-License-Identifier: MPL-2.0
"""Compare 1P and 2P calls on two Linux CPUs with mounted and Linux-local weights."""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, file_identity, inside_repo, load_engine, write_json


def binary(arm):
    if arm == "original":
        pinned = json.loads((STUDY.parent / "search_pass2/control_inputs.json").read_bytes())["original"]
        return ROOT / pinned["linux"]["path"], ROOT / pinned["native_source"] / "python", pinned["linux"]["sha256"]
    audit = json.loads((STUDY / "qualified/linux/verification.json").read_bytes())["identities"]["candidate"]
    return ROOT / audit["binary"], ROOT / audit["source_tree"] / "python", audit["extension_sha256"]


def worker(args):
    extension, source, expected = binary(args.arm)
    if file_identity(extension)["sha256"] != expected:
        raise ValueError("Extension differs from the qualified build")
    engine, _ = load_engine(extension, source)
    model = engine.WeightConfig.from_model("stage11")
    models = Path(args.weights)
    weights = replace(model, paths=[str(models / Path(p).name) for p in model.paths])
    weights.validate()
    analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=args.level, parallel_threads=2,
                                    cubeful=True, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
    if len(os.sched_getaffinity(0)) != 2:
        raise ValueError("Expected a two-CPU affinity mask")
    with args.output.open("w", encoding="utf-8") as out:
        for api in APIS:
            panel = "checker_play" if api == "checker_play" else "cube_action"
            dataset = json.loads((DATA / (panel + ".json")).read_bytes())
            for row in dataset["positions"]:
                inputs = row["post_move_inputs"] if api == "post_move_analytics" else row["inputs"]
                call = getattr(analyzer, api)
                start = perf_counter_ns()
                result = call(**inputs)
                seconds = (perf_counter_ns() - start) / 1e9
                record = dict(arm=args.arm, level=args.level, storage=args.storage, api=api,
                              position_id=row["id"], inputs=inputs, elapsed_seconds=seconds, analytics=asdict(result))
                out.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def measure(args):
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        raise ValueError("Two Linux logical CPUs required")
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ, BGBOT_MULTIPLY_THREADS="2")
    mounted = ROOT / "models"
    archive = STUDY / "qualified/linux/full_outputs.zip"
    with ZipFile(archive) as z:
        metadata = json.loads(z.read("candidate/1P/metadata.json"))
    names = {Path(slot["path"]).name: slot["sha256"] for slot in metadata["weights"]}
    order = []
    rows = {}
    totals = {}
    with TemporaryDirectory(prefix="bgsage-inline-weights-", dir=args.native_parent) as temp:
        local = Path(temp)
        fs = {key: subprocess.check_output(["stat", "-f", "-c", "%T", str(path)], text=True).strip()
              for key, path in (("linux", local), ("mounted", mounted))}
        if fs["linux"] == fs["mounted"]:
            raise ValueError("This experiment needs distinct mounted and native filesystems")
        for name, digest in names.items():
            for path in (mounted / name, Path(shutil.copyfile(mounted / name, local / name))):
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Different weights: {name}")
        configs = [("mounted", mounted, "original"), ("mounted", mounted, "selected"),
                   ("linux", local, "selected"), ("linux", local, "original")]
        for level in ("1ply", "2ply"):
            ordered = configs if level == "1ply" else reversed(configs)
            for storage, path, arm in ordered:
                name = f"{level}-{storage}-{arm}"
                order.append(name)
                output = out / (name + ".jsonl")
                subprocess.run([sys.executable, "-B", __file__, "worker", "--level", level, "--storage", storage,
                                "--arm", arm, "--weights", path, "--output", output], check=True, env=env)
                case_rows = [json.loads(line) for line in output.read_bytes().splitlines()]
                if len(case_rows) != 300:
                    raise ValueError("Incomplete panel")
                by_api = defaultdict(float)
                for r in case_rows:
                    by_api[r["api"]] += r["elapsed_seconds"]
                rows[level, storage, arm] = [(r["api"], r["position_id"], r["inputs"], r["analytics"]) for r in case_rows]
                totals[name] = dict(**by_api, total=sum(by_api.values()))
                print(name, totals[name], flush=True)
    for level in ("1ply", "2ply"):
        for arm in ("original", "selected"):
            if rows[level, "linux", arm] != rows[level, "mounted", arm]:
                raise ValueError(f"Changing model storage changed {arm} {level} results")
    sources = {}
    for arm in ("original", "selected"):
        ext, pkg, _ = binary(arm)
        sources[arm] = dict(extension=file_identity(ext), python_dir=pkg.relative_to(ROOT).as_posix())
    factors = {}
    for level in ("1ply", "2ply"):
        for storage in ("mounted", "linux"):
            original = totals[f"{level}-{storage}-original"]
            selected = totals[f"{level}-{storage}-selected"]
            factors[f"{level}-{storage}"] = {api: original[api] / selected[api] for api in original}
    report = dict(cpu_affinity=cpus, native_threads=2, filesystem=fs, order=order, binaries=sources,
                  weight_slots=metadata["weights"], bearoff=metadata["bearoff_database"], seconds=totals, speedups=factors,
                  identical_inputs_and_outputs_across_storage=1200)
    write_json(out / "summary.json", report)
    with ZipFile(out / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as z:
        for name in order:
            z.write(out / (name + ".jsonl"), name + ".jsonl")
        z.write(out / "summary.json", "summary.json")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    replay = commands.add_parser("measure")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--native-parent", type=Path, default=Path("/tmp"),
                        help="Linux filesystem for temporary benchmark weight copies")
    call = commands.add_parser("worker")
    call.add_argument("--level", required=True, choices=("1ply", "2ply"))
    call.add_argument("--storage", required=True, choices=("mounted", "linux"))
    call.add_argument("--arm", required=True, choices=("original", "selected"))
    call.add_argument("--weights", required=True, type=Path)
    call.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (worker if args.action == "worker" else measure)(args)
