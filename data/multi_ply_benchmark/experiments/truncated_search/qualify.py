# SPDX-License-Identifier: MPL-2.0
"""Package and verify the two complete paired 16-worker benchmarks."""
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, write_json
from compare_truncated_benchmark import compare

LEVELS = ("2T", "3T")
ARMS = ("baseline", "candidate")
RUNS = ROOT / "logs/truncated_search/final"


def main():
    results = {arm: [RUNS / f"{level}-{arm}/results.jsonl" for level in LEVELS] for arm in ARMS}
    score = compare(results["baseline"], results["candidate"], DATA / "rollout_positions/results.jsonl", LEVELS)
    summary = {}
    study_archive = DATA / "experiments/truncated_16t/full_outputs.zip"
    with ZipFile(study_archive) as prior, ZipFile(HERE / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for level in LEVELS:
            reference = [json.loads(line) for line in prior.read(f"{level}-candidate/results.jsonl").splitlines()]
            detail = score["levels"][level]
            calls = {}
            for arm in ARMS:
                folder = RUNS / f"{level}-{arm}"
                meta = json.loads((folder / "metadata.json").read_bytes())
                if (meta["status"], meta["completed_calls"], meta["threads_requested"]) != ("complete", 300, 16):
                    raise ValueError(f"Incomplete {level} {arm} or wrong concurrency")
                rows = [json.loads(line) for line in (folder / "results.jsonl").read_bytes().splitlines()]
                calls[arm] = rows
                if Counter(row["api"] for row in rows) != dict.fromkeys(APIS, 100):
                    raise ValueError(f"Wrong API mix for {level} {arm}")
                for name in ("results.jsonl", "metadata.json", "summary.json"):
                    archive.write(folder / name, folder.name + "/" + name)
            def key(row):
                return row["api"], row["position_id"], row["inputs"], row["seed"]
            if list(map(key, reference)) != list(map(key, calls["baseline"])):
                raise ValueError("Baseline did not replay published positions")
            vs_published = sum(a["analytics"] == b["analytics"] for a, b in zip(reference, calls["baseline"]))
            counts = Counter()
            for result in detail["by_api"].values():
                for change in result["material_changes_in_rollout_error"]:
                    counts["farther" if change["change_in_absolute_error"] > 0 else "closer"] += 1
            summary[level] = dict(total_call_seconds=detail["total_call_seconds"], speedup=detail["speedup"],
                                  exact_complete_outputs=sum(g["exact_analytics"] for g in detail["by_api"].values()),
                                  baseline_exact_vs_published=vs_published, material_error_changes=dict(counts))
            print(level, json.dumps(summary[level]))
    write_json(HERE / "comparison.json", score)
    write_json(HERE / "qualification.json", summary)
    guard = RUNS.parent / "fullgame-guard"
    if (guard / "summary.json").is_file():
        files = sorted(guard.glob("*-*.jsonl")) + [guard / "settings.json", guard / "summary.json"]
        files.extend(guard / arm / "hashes.json" for arm in ARMS)
        with ZipFile(HERE / "fullgame_guard.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for source in files:
                archive.write(source, source.relative_to(guard).as_posix())
    if any(x["material_error_changes"].get("farther") for x in summary.values()):
        raise SystemExit("Materially worse reference errors require investigation")


if __name__ == "__main__":
    main()
