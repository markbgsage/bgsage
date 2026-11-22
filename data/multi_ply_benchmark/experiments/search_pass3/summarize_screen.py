# SPDX-License-Identifier: MPL-2.0
"""Recompute paired public-output parity, CPU time and elapsed time from a screen."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
from zipfile import ZipFile


def summarize(folder):
    groups = {}
    with ZipFile(folder / "outputs.zip") as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                def key(row):
                    return row["level"], row["api"], row["position_id"]
                for (level, api, _), group in itertools.groupby(map(json.loads, stream), key):
                    rows = {r["arm"]: r for r in group}
                    baseline = rows["checkpoint"]
                    cell = groups.setdefault(level, {}).setdefault(api, {})
                    for arm, row in rows.items():
                        if baseline["inputs"] != row["inputs"]:
                            raise ValueError("Unmatched inputs")
                        stats = cell.setdefault(arm, defaultdict(float))
                        stats["seconds"] += row["elapsed_seconds"]
                        stats["cpu_seconds"] += row["cpu_seconds"]
                        stats["calls"] = int(stats["calls"]) + 1
                        stats["exact"] = int(stats["exact"]) + (baseline["analytics"] == row["analytics"])
                        stats["faster"] = int(stats["faster"]) + (row["elapsed_seconds"] < baseline["elapsed_seconds"])
    (folder / "summary.json").write_text(json.dumps(groups, indent=2) + "\n")
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    for folder in parser.parse_args().folders:
        for level, cells in summarize(folder).items():
            print(folder.name, level, {api: {arm: (round(v["seconds"], 3), v["exact"], v["calls"])
                                              for arm, v in cell.items()} for api, cell in cells.items()})
