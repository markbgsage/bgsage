# SPDX-License-Identifier: MPL-2.0
"""Summarize paired public wall times and exact full-response comparisons."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median

def score(run, destination, partial):
    raw = (run / "results.jsonl").read_bytes()
    meta = json.loads((run / "metadata.json").read_bytes())
    levels = defaultdict(lambda: defaultdict(dict))
    for line in raw.splitlines():
        row = json.loads(line)
        levels[row["level"]][row["repeat"], row["position_id"], row["api"]][row["arm"]] = row
    result = {"metadata": meta, "levels": {}}
    for level, cases in levels.items():
        complete = [v for v in cases.values() if set(v) == set(meta["hashes"])]
        if len(complete) != len(cases) and not partial:
            raise ValueError(f"Incomplete {level} block")
        if not complete:
            continue
        result["levels"][level] = stats = {}
        for arm in meta["hashes"]:
            base = [v["base"]["elapsed"] for v in complete]
            times = [v[arm]["elapsed"] for v in complete]
            stats[arm] = {
                "calls": len(times), "seconds": sum(times),
                "speedup_vs_relink": sum(base) / sum(times),
                "paired_median_speedup": median(x / y for x, y in zip(base, times)),
                "faster_pairs": sum(y < x for x, y in zip(base, times)),
                "exact_vs_relink": sum(v[arm]["analytics"] == v["base"]["analytics"] for v in complete),
                "by_repeat": {
                    repeat: sum(v[arm]["elapsed"] for v in complete if v[arm]["repeat"] == repeat)
                    for repeat in sorted({v[arm]["repeat"] for v in complete})
                },
            }
            print(level, arm, len(times), round(stats[arm]["speedup_vs_relink"], 4),
                  stats[arm]["exact_vs_relink"])
    if not partial:
        for spec in meta["levels"]:
            level, repeats = spec.split("=")
            if len(levels[level]) != meta.get("cases_per_repeat", 10) * int(repeats):
                raise ValueError(f"Missing {level} cases")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        (destination / "calls.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    score(args.run, args.output, args.partial)
