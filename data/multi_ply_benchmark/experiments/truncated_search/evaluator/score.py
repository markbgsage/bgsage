# SPDX-License-Identifier: MPL-2.0
"""Compare the timing panel with the unchanged control and frozen full-game rollout."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_multi_ply_accuracy import projected, metrics, records


def standard_errors(item):
    for key, value in item.items():
        if key.endswith("_se") or key == "std_error":
            if value is not None:
                yield key, value
        if key == "prob_std_errors" and value is not None:
            yield from enumerate(value)


def compare(folder):
    references = {}
    for row in records(ROOT / "data/multi_ply_benchmark/rollout_positions/results.jsonl"):
        for board, item in projected(row["api"], row["analytics"]):
            if row["api"] != "checker_play" or item["eval_level"] == "Rollout":
                references[row["api"], row["position_id"], board] = dict(metrics(item))
    with ZipFile(folder / "full_outputs.zip") as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines()]
    controls = {(r["level"], r["api"], r["position_id"]): r for r in rows if r["arm"] == "control"}
    groups = {}
    shifts = defaultdict(lambda: defaultdict(list))
    for r in rows:
        arm, level, api, pid = r["arm"], r["level"], r["api"], r["position_id"]
        g = groups.setdefault((level, arm), dict(calls=0, seconds=0, control_seconds=0,
            identical_calls=0, same_best_decisions=0, decisions=0, truth_fields=0, materially_farther=0,
            materially_closer=0, max_se_shift=0))
        baseline = controls[level, api, pid]
        assert (baseline["inputs"], baseline["seed"]) == (r["inputs"], r["seed"])
        a, b = baseline["analytics"], r["analytics"]
        g["calls"] += 1
        g["seconds"] += r["elapsed_seconds"]
        g["control_seconds"] += baseline["elapsed_seconds"]
        g["identical_calls"] += a == b
        if api != "post_move_analytics":
            field = "moves" if api == "checker_play" else "optimal_action"
            best = lambda result: result[field][0]["board"] if field == "moves" else result[field]
            g["decisions"] += 1
            g["same_best_decisions"] += best(a) == best(b)
        aa, bb = dict(projected(api, a)), dict(projected(api, b))
        assert aa.keys() == bb.keys()
        for board, left in aa.items():
            right = bb[board]
            old_se, new_se = dict(standard_errors(left)), dict(standard_errors(right))
            assert old_se.keys() == new_se.keys()
            for key in old_se:
                g["max_se_shift"] = max(g["max_se_shift"], abs(old_se[key]-new_se[key]))
            original, updated = dict(metrics(left)), dict(metrics(right))
            truth = references.get((api, pid, board), {})
            for key, x in original.items():
                field = "probability" if key.startswith("p.") else "equity"
                y = updated[key]
                shifts[level, arm][field].append(abs(y-x))
                if key in truth:
                    change = abs(y-truth[key])-abs(x-truth[key])
                    threshold = 0.0025 if field == "probability" else 0.005
                    g["truth_fields"] += 1
                    g["materially_farther"] += change > threshold
                    g["materially_closer"] += change < -threshold
    rows = []
    for (level, arm), g in sorted(groups.items()):
        g["speedup"] = g["control_seconds"] / g["seconds"]
        for name, values in shifts[level, arm].items():
            g["max_" + name + "_shift"] = max(values)
        rows.append({"level":level, "arm":arm, **g})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    scored = json.dumps(compare(args.folder), indent=2) + "\n"
    (args.folder / "score.json").write_text(scored, encoding="utf-8")
    print(scored)
