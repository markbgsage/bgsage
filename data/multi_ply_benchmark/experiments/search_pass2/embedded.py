# SPDX-License-Identifier: MPL-2.0
"""Package the full-game guard and compare Linux truncated presets with the published Linux checkpoint."""
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from qualify import FILES, HERE, ROOT, history

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, write_json
from compare_truncated_benchmark import compare


def main():
    guard = ROOT / "logs/search_pass2_guard"
    cutoff = json.loads((guard / "summary.json").read_bytes())
    if len(cutoff["pairs"]) != 12:
        raise ValueError("Expected two paired rounds of six full-game calls")
    with ZipFile(HERE / "fullgame_guard.zip", "w", ZIP_DEFLATED) as archive:
        for path in [*guard.glob("*-*.jsonl"), guard / "settings.json", guard / "summary.json",
                     guard / "baseline/hashes.json", guard / "candidate/hashes.json"]:
            archive.write(path, path.relative_to(guard).as_posix())
    runs = ROOT / "logs/search_pass2/linux"
    dest = HERE / "linux-truncated"
    dest.mkdir(exist_ok=True)
    levels = ("1T", "2T", "3T")
    prior = HERE.parent / "truncated_search/evaluator/linux_final/full_outputs.zip"
    for level in ("2T", "3T"):
        history(runs, "checkpoint", level, prior, level + "-candidate")
    paths = {name: [runs / name / level / "results.jsonl" for level in levels]
             for name in ("checkpoint", "candidate")}
    report = compare(paths["checkpoint"], paths["candidate"], DATA / "rollout_positions/results.jsonl", levels)
    for side in paths.values():
        for path in side:
            meta = json.loads(path.with_name("metadata.json").read_bytes())
            if (meta["status"], meta["completed_calls"], meta["threads_requested"]) != ("complete", 300, 16):
                raise ValueError(f"Incomplete run or wrong thread count {path}")
    write_json(dest / "comparison.json", report)
    with ZipFile(dest / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for side in paths.values():
            for path in side:
                for name in FILES:
                    archive.write(path.with_name(name), path.with_name(name).relative_to(runs).as_posix())
    for level, data in report["levels"].items():
        scores = list(data["by_api"].values())
        print(level, data["total_call_seconds"], data["speedup"],
              "exact", sum(score["exact_analytics"] for score in scores),
              "materially farther", sum(change["change_in_absolute_error"] > 0 for score in scores
                                        for change in score["material_changes_in_rollout_error"]))


if __name__ == "__main__":
    main()
