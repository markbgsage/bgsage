# SPDX-License-Identifier: MPL-2.0
"""Score complete public responses against the paired control and rollout references."""
import argparse
from collections import defaultdict
import gzip
import itertools
import json
from pathlib import Path
import sys

from prepare import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from compare_multi_ply_accuracy import metrics, projected


def empty():
    return dict(calls=0, exact=0, decision_changes=0, wall=0., control_wall=0.,
                thresholds={"probability": .0025, "equity": .005},
                fields={kind: dict(count=0, max_shift=0., material_shifts=0, references=0,
                                   control_abs_error=0., arm_abs_error=0., materially_farther=0,
                                   materially_closer=0) for kind in ("probability", "equity")})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    truth = {}
    refs = ROOT / "data/multi_ply_benchmark/rollout_positions/results.jsonl"
    for line in refs.read_bytes().splitlines():
        row = json.loads(line)
        for board, item in projected(row["api"], row["analytics"]):
            if row["api"] != "checker_play" or item["eval_level"] == "Rollout":
                truth[row["api"], row["position_id"], board] = dict(metrics(item))
    groups = defaultdict(empty)
    with gzip.open(args.directory / "calls.jsonl.gz", "rt") as f:
        case = lambda row: (row["repeat"], row["level"], row["api"], row["position_id"])
        for (_, level, api, pid), pairs in itertools.groupby(map(json.loads, f), key=case):
            arms = {row["arm"]: row for row in pairs}
            ref = arms.pop("control")
            left = ref["analytics"]
            for name, row in arms.items():
                if ref["inputs"] != row["inputs"]:
                    raise ValueError(pid)
                right = row["analytics"]
                stats = groups[level, api, name]
                stats["calls"] += 1
                stats["wall"] += row["wall"]
                stats["control_wall"] += ref["wall"]
                stats["exact"] += left == right
                if api == "checker_play":
                    stats["decision_changes"] += left["moves"][0]["board"] != right["moves"][0]["board"]
                if api == "cube_action":
                    stats["decision_changes"] += left["optimal_action"] != right["optimal_action"]
                control, candidate = dict(projected(api, left)), dict(projected(api, right))
                if control.keys() != candidate.keys():
                    raise ValueError(f"Endpoint membership changed for {pid}")
                for board, item in control.items():
                    target = truth.get((api, pid, board), {})
                    new = dict(metrics(candidate[board]))
                    for field, old in metrics(item):
                        kind = "probability" if field.startswith("p.") else "equity"
                        s = stats["fields"][kind]
                        limit = stats["thresholds"][kind]
                        d = abs(new[field] - old)
                        s["count"] += 1
                        s["max_shift"] = max(s["max_shift"], d)
                        s["material_shifts"] += d > limit
                        if field in target:
                            a, b = abs(old - target[field]), abs(new[field] - target[field])
                            s["references"] += 1
                            s["control_abs_error"] += a
                            s["arm_abs_error"] += b
                            s["materially_farther"] += b - a > limit
                            s["materially_closer"] += a - b > limit
    rows = []
    for (level, api, arm), stats in sorted(groups.items()):
        rows.append(dict(level=level, api=api, arm=arm, **stats))
    (args.directory / "scores.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
