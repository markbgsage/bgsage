# SPDX-License-Identifier: MPL-2.0
"""Measure cache memory on one corpus case and fingerprint the native match tables."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, inside_repo, load_engine, write_json


def main():
    import psutil
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpus", type=int, nargs="+", help="Optional logical CPU affinity")
    args = p.parse_args()
    proc = psutil.Process()
    if args.cpus:
        proc.cpu_affinity(args.cpus)
    engine, dll = load_engine(args.extension)
    import bgbot_cpp as cpp
    # Include both terminal boundaries and scores above the 25-point table.
    digest = hashlib.sha256()
    n = 0
    for a in range(-2, 29):
        for b in range(-2, 29):
            for phase in (False, True):
                digest.update(struct.pack("<f", cpp.get_met(a, b, phase)))
                n += 1
    with zipfile.ZipFile(DATA / "experiments/truncated_16t/full_outputs.zip") as z:
        reference = next(r for line in z.read("2T-candidate/results.jsonl").splitlines()
                         if (r := json.loads(line))["position_id"] == "cube-match-anchoring-01"
                         and r["api"] == "cube_action")
    analyzer = engine.BgBotAnalyzer(weights=engine.WeightConfig.from_model("stage11"),
                                    eval_level="truncated2", cubeful=True, parallel_threads=4,
                                    bearoff_db=str(ROOT / "data/bearoff_1sided.db"))
    analyzer.set_seed(reference["seed"])
    before = proc.memory_info().rss
    begin = time.perf_counter()
    actual = analyzer.cube_action(**reference["inputs"])
    elapsed = time.perf_counter() - begin
    after = proc.memory_info().rss
    out = inside_repo(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(extension=file_identity(dll), public_position=reference["position_id"],
                         wall_seconds=elapsed, rss_bytes_before=before, rss_bytes_after=after,
                         rss_bytes_change=after - before, threads=4, cpus=proc.cpu_affinity(),
                         match_scores_checked=n, match_table_sha256=digest.hexdigest(),
                         exact=asdict(actual) == reference["analytics"], analytics=asdict(actual)))


if __name__ == "__main__":
    main()
