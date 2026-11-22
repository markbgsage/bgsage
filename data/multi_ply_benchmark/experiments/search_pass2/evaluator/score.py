# SPDX-License-Identifier: MPL-2.0
"""Score each recorded deterministic candidate against its paired control and full-game truth."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile
from screen import DATA
from compare_truncated_benchmark import score_api


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    truth = {}
    for line in (DATA / "rollout_positions/results.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        truth[r["api"], r["position_id"]] = r
    with ZipFile(args.folder / "full_outputs.zip") as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines()]
    key = lambda row: tuple(row[f] for f in ("level", "repeat", "api", "position_id"))
    control = {key(r): r for r in rows if r["arm"] == "control"}
    groups = defaultdict(list)
    for r in rows:
        if r["arm"] != "control":
            groups[r["arm"], r["level"], r["api"]].append(
                (control[key(r)], r, truth[r["api"], r["position_id"]]))
    report, summary = {}, {}
    for (arm, level, api), pairs in sorted(groups.items()):
        stats = score_api(api, pairs)
        report.setdefault(arm, {}).setdefault(level, {})[api] = stats
        group = summary.setdefault(arm, {}).setdefault(level, dict(control_seconds=0, seconds=0, calls=0,
            identical=0, material_worse=0, material_better=0, changed_decisions=0))
        group["control_seconds"] += stats["timing"]["baseline"]["total_seconds"]
        group["seconds"] += stats["timing"]["candidate"]["total_seconds"]
        group["calls"] += len(pairs)
        group["identical"] += stats["exact_analytics"]
        if "same_best_action" in stats:
            group["changed_decisions"] += len(pairs)-stats["same_best_action"]
        for c in stats["material_changes_in_rollout_error"]:
            group["material_worse" if c["change_in_absolute_error"] > 0 else "material_better"] += 1
        group["speedup"] = group["control_seconds"] / group["seconds"]
    (args.folder / "comparison.json").write_text(json.dumps(report, separators=(",",":")) + "\n")
    (args.folder / "score.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
