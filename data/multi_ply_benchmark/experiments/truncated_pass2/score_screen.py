# SPDX-License-Identifier: MPL-2.0
"""Score each experimental arm against the published inputs and 1,296-game references."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from screen import DATA, HERE, write_json
from compare_truncated_benchmark import level_name, score_api


def main(folder):
    with (DATA / "rollout_positions/results.jsonl").open() as f:
        truth = {(r["api"], r["position_id"]): r for r in map(json.loads, f)}
    with gzip.open(folder / "calls.jsonl.gz", "rt") as f:
        arms = list(map(json.loads, f))
    old = {}
    with ZipFile(HERE.parent / "search_pass3/release/linux/full_outputs.zip") as z:
        for level in {level_name(r["level"]) for r in arms}:
            for line in z.read(f"candidate/{level}/results.jsonl").splitlines():
                r = json.loads(line)
                old[level, r["api"], r["position_id"]] = r
    comparisons = defaultdict(list)
    for r in arms:
        level = level_name(r["level"])
        api, pid = r["api"], r["position_id"]
        control = old[level, api, pid]
        if any(r[key] != control[key] for key in ("inputs","seed","source_game_plan","mode")):
            raise ValueError(f"Different inputs for {pid}")
        comparisons[r["arm"],level,api].append((control, control | r, truth[api,pid]))
    report = {}
    for (arm, level, api), pairs in comparisons.items():
        stats = score_api(api, pairs)
        changes = stats["material_changes_in_rollout_error"]
        report.setdefault(arm, {}).setdefault(level, {})[api] = dict(
            cases=len(pairs), materially_farther=sum(c["change_in_absolute_error"] > 0 for c in changes),
            materially_closer=sum(c["change_in_absolute_error"] < 0 for c in changes),
            all_fields_identical=stats["exact_analytics"], same_action=stats.get("same_best_action"))
    write_json(folder / "accuracy.json", report)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
