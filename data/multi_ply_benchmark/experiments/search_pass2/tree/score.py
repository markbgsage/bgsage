# SPDX-License-Identifier: MPL-2.0
"""Summarize tree-arm wall time, CPU time and whole-response agreement by API."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    groups = defaultdict(list)
    rows = json.loads(args.measurements.read_text())["cases"]
    for row in rows:
        for arm, metrics in row["arms"].items():
            control = row["arms"]["control"]
            for api in [row["api"], "all"]:
                groups[(row["level"], api, arm)].append((metrics, control))
    scores = []
    for (level, api, arm), pairs in sorted(groups.items()):
        a, b = zip(*pairs)
        wall, cpu = sum(r["wall"] for r in a), sum(r["cpu"] for r in a)
        ref_wall, ref_cpu = sum(r["wall"] for r in b), sum(r["cpu"] for r in b)
        scores.append(dict(level=level, api=api, arm=arm, n=len(a), exact=sum(x["sha256"] == y["sha256"] for x,y in pairs),
            wall=wall, control_wall=ref_wall, wall_saved=1-wall/ref_wall,
            cpu=cpu, control_cpu=ref_cpu, cpu_saved=1-cpu/ref_cpu if ref_cpu else None))
    if args.output:
        args.output.write_text(json.dumps(scores, indent=2) + "\n")
    else:
        print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
