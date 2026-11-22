# SPDX-License-Identifier: MPL-2.0
"""Measure native worker phases on the published, balanced 2T/3T panel."""
import argparse
from collections import Counter
import ctypes
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

from prepare import REVISION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, file_identity, inside_repo, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask

PHASES = ("prefill0", "prefill1", "cube", "movegen", "vr_mean", "select",
          "vr_luck", "truncation", "other", "fan")
REFERENCE = DATA / "experiments/search_pass3/release/windows/full_outputs.zip"


def sample(reference, level):
    seen = set()
    for line in reference.read(f"candidate/{level}/results.jsonl").splitlines():
        row = json.loads(line)
        stratum = (row["api"], row["mode"], row["source_game_plan"])
        if stratum not in seen:
            seen.add(stratum)
            yield row
    if Counter(api for api, _, _ in seen) != {api: 10 for api in APIS}:
        raise ValueError(f"Incomplete {level} benchmark sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--python-dir", type=Path, required=True, help="Python package root of an unmodified control export")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-mask", type=lambda value: int(value, 0), default=0xF00)
    args = parser.parse_args()
    set_cpu_mask(args.cpu_mask)
    engine, extension = load_engine(args.extension, python_dir=args.python_dir)
    read_phase = ctypes.CDLL(str(extension)).take_rollout_profile
    read_phase.argtypes = [ctypes.POINTER(ctypes.c_longlong)] * 2
    read_phase.restype = None
    buffer = ctypes.c_longlong * len(PHASES)
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    weights = engine.WeightConfig.from_model("stage11")
    summary = dict(revision=REVISION, binary=file_identity(extension), archive=file_identity(REFERENCE),
                   threads=args.threads, cpu_mask=hex(args.cpu_mask), cases=[])
    with ZipFile(REFERENCE) as reference, gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as full:
        for level in ("2T", "3T"):
            refs = list(sample(reference, level))
            analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=refs[0]["level"],
                                            cubeful=True, parallel_threads=args.threads,
                                            bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
            for ref in refs:
                ns, visits = buffer(), buffer()
                read_phase(ns, visits)
                analyzer.set_seed(ref["seed"])
                start_cpu = time.process_time()
                start = time.perf_counter()
                result = getattr(analyzer, ref["api"])(**ref["inputs"])
                seconds = time.perf_counter() - start
                cpu_seconds = time.process_time() - start_cpu
                read_phase(ns, visits)
                analytics = asdict(result)
                row = {k: ref[k] for k in ("api", "position_id", "level", "mode", "source_game_plan", "seed")}
                row.update(seconds=seconds, cpu_seconds=cpu_seconds, archived_seconds=ref["elapsed_seconds"],
                           exact=analytics == ref["analytics"], phase_visits=dict(zip(PHASES, visits)),
                           phase_worker_seconds=dict(zip(PHASES, (v / 1e9 for v in ns))))
                summary["cases"].append(row)
                full.write(json.dumps(dict(**row, inputs=ref["inputs"], analytics=analytics),
                                      separators=(",", ":"), allow_nan=False) + "\n")
                print(level, ref["position_id"], ref["api"], f"{seconds:.3f}s", row["exact"], flush=True)
            write_json(out / "measurements.json", summary)
    if not all(r["exact"] for r in summary["cases"]):
        raise ValueError("Issues reproducing published full analytics. Inspect measurements.json.")


if __name__ == "__main__":
    main()
