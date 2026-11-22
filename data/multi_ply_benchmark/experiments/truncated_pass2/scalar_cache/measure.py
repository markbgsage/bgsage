# SPDX-License-Identifier: MPL-2.0
"""Measure a cold analyzer and two warm repeats per case, each arm in a fresh process."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, inside_repo, load_engine, write_json
from multi_ply_rollout_truth import set_cpu_mask

ARCHIVE = DATA / "experiments/search_pass3/release/windows/full_outputs.zip"
CASES = [
    ("1T", "checker_play", "checker-money-attacking-01"),
    ("1T", "cube_action", "cube-match-anchoring-01"),
    ("1T", "post_move_analytics", "cube-money-priming-01"),
    ("2T", "cube_action", "cube-match-anchoring-01"),
    ("3T", "cube_action", "cube-match-anchoring-01"),
]


def memory():
    import psutil
    info = psutil.Process().memory_info()
    return dict(rss=info.rss, peak_rss=info.peak_wset)


def sample(args):
    set_cpu_mask(args.cpu_mask)
    engine, _ = load_engine(args.control if args.arm == "control" else args.candidate, args.python_dir)
    weights = engine.WeightConfig.from_model("stage11")
    level, api, position = CASES[args.case]
    with ZipFile(ARCHIVE) as z:
        refs = (json.loads(line) for line in z.read(f"candidate/{level}/results.jsonl").splitlines())
        ref = next(r for r in refs if r["api"] == api and r["position_id"] == position)
    start_memory = memory()
    start = time.perf_counter()
    analyzer = engine.BgBotAnalyzer(weights=weights, eval_level=ref["level"], cubeful=True,
                                    parallel_threads=args.threads, bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
    init_seconds = time.perf_counter() - start
    result = dict(arm=args.arm, process_repeat=args.repeat, level=level, api=api, position_id=position,
                  seed=ref["seed"], inputs=ref["inputs"], before=start_memory, construction_seconds=init_seconds,
                  after_construction=memory(), calls=[])
    for _ in range(3):
        analyzer.set_seed(ref["seed"])
        start = time.perf_counter()
        ret = getattr(analyzer, api)(**ref["inputs"])
        seconds = time.perf_counter() - start
        m = memory()
        analytics = asdict(ret)
        result["calls"].append(dict(seconds=seconds, **m, exact=analytics == ref["analytics"], analytics=analytics))
    print(json.dumps(result, separators=(",", ":"), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "python-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=["control", "candidate"])
    parser.add_argument("--case", type=int)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--process-repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-mask", type=lambda s: int(s, 0), default=0xF00)
    args = parser.parse_args()
    if args.arm:
        sample(args)
        return
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(threads=args.threads, cpu_mask=hex(args.cpu_mask), control=file_identity(args.control),
                   candidate=file_identity(args.candidate), reference=file_identity(ARCHIVE), cases=[])
    with gzip.open(out / "calls.jsonl.gz", "wt", encoding="utf-8") as archive:
        for repeat in range(args.process_repeats):
            for case in range(len(CASES)):
                for arm in (("control", "candidate") if (repeat + case) % 2 == 0 else ("candidate", "control")):
                    command = [sys.executable, "-B", __file__, *sys.argv[1:], "--case", str(case),
                               "--arm", arm, "--repeat", str(repeat)]
                    result = json.loads(subprocess.check_output(command))
                    archive.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                    result.pop("inputs")
                    for call in result["calls"]:
                        call.pop("analytics")
                    summary["cases"].append(result)
                    print(result["level"], result["api"], arm, repeat,
                          round(result["calls"][0]["peak_rss"] / (1024 ** 2)), "MiB",
                          round(result["construction_seconds"], 3), "s construct", flush=True)
            write_json(out / "measurements.json", summary)
    if not all(call["exact"] for row in summary["cases"] for call in row["calls"]):
        raise ValueError("Full analytics differ from the checkpoint")


if __name__ == "__main__":
    main()
