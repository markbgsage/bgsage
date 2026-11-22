# SPDX-License-Identifier: MPL-2.0
"""Time a balanced one-position-per-stratum sample and read native phase counters."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, APIS, file_identity, inside_repo, load_engine, write_json

PHASES = "prefill0 prefill1 cube movegen vr_mean select vr_luck truncation other".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--memo-counts", action="store_true")
    parser.add_argument("--levels", nargs="+", choices=["2T", "3T"], default=["2T", "3T"])
    parser.add_argument("--apis", nargs="+", choices=APIS, default=APIS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-mask", type=lambda v: int(v, 0), default=0xf00)
    parser.add_argument("--per-stratum", type=int, default=1)
    opts = parser.parse_args()
    from multi_ply_rollout_truth import set_cpu_mask
    set_cpu_mask(opts.cpu_mask)
    engine, extension = load_engine(opts.extension)
    root = inside_repo(opts.output)
    root.mkdir(parents=True, exist_ok=False)
    take = None
    take_hits = None
    if opts.memo_counts:
        memo_dll = ctypes.CDLL(str(extension))
        take_hits = memo_dll.take_vr_hits
        take_hits.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
        take_hits.restype = None
    if opts.profile:
        dll = ctypes.CDLL(str(extension))
        take = dll.take_rollout_profile
        take.argtypes = [ctypes.POINTER(ctypes.c_longlong)] * 2
        take.restype = None
    archive = DATA / "experiments/truncated_16t/full_outputs.zip"
    weights = engine.WeightConfig.from_model("stage11")
    rows = []
    with ZipFile(archive) as original, gzip.open(root / "calls.jsonl.gz", "wt", encoding="utf-8") as full:
        for level in opts.levels:
            refrows = [
                json.loads(line) for line in original.read(f"{level}-candidate/results.jsonl").splitlines()
            ]
            selected = {api: [] for api in opts.apis}
            quotas = Counter()
            for row in refrows:
                key = row["api"], row["mode"], row["source_game_plan"]
                if row["api"] in selected and quotas[key] < opts.per_stratum:
                    selected[row["api"]].append(row)
                    quotas[key] += 1
            analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=refrows[0]["level"], cubeful=True,
                                            parallel_threads=opts.threads,
                                            bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
            for api, refs in selected.items():
                for ref in refs:
                    ns = (ctypes.c_longlong * len(PHASES))()
                    entries = (ctypes.c_longlong * len(PHASES))()
                    if take:
                        take(ns, entries)
                    analyzer.set_seed(ref["seed"])
                    cpu_start = time.process_time()
                    start = time.perf_counter()
                    result = getattr(analyzer, api)(**ref["inputs"])
                    wall = time.perf_counter() - start
                    cpu = time.process_time() - cpu_start
                    if take:
                        take(ns, entries)
                    analytics = asdict(result)
                    row = {key: ref[key] for key in ("position_id", "api", "level", "mode", "source_game_plan", "seed")}
                    row.update(wall_seconds=wall, process_cpu_seconds=cpu, exact=analytics == ref["analytics"])
                    if take:
                        row["phase_worker_seconds"] = dict(zip(PHASES, (v / 1e9 for v in ns)))
                        row["phase_visits"] = dict(zip(PHASES, entries))
                    if take_hits:
                        counters = (ctypes.c_longlong * 2)()
                        take_hits(counters)
                        row["vr_visits"], row["vr_hits"] = counters
                    rows.append(row)
                    full.write(json.dumps(dict(row, inputs=ref["inputs"], analytics=analytics),
                                          separators=(",", ":"), allow_nan=False) + "\n")
                    print(level, api, ref["position_id"], f"{wall:.3f}s", row["exact"], flush=True)
                write_json(root / "measurements.json", dict(binary=file_identity(extension), cpu_mask=hex(opts.cpu_mask),
                           threads=opts.threads, source_archive=file_identity(archive), cases=rows))


if __name__ == "__main__":
    main()
