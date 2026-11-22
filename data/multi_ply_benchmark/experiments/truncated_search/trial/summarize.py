# SPDX-License-Identifier: MPL-2.0
"""Summarize paired native timing records and preserve their complete analytics."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median


def difference(a, b):
    """Return the largest numeric difference; validate all labels, order and shape."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Dictionary shape differs")
        return max((difference(value, b[key]) for key, value in a.items()), default=0.0)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError("Array shape differs")
        return max((difference(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b)
    if a != b:
        raise ValueError(f"Discrete field differs: {a!r} and {b!r}")
    return 0.0


def summarize(run, destination, prefix, reference):
    raw = (run / "results.jsonl").read_bytes()
    metadata = json.loads((run / "metadata.json").read_bytes())
    arms = tuple(metadata["builds"])
    blocks = defaultdict(dict)
    for line in raw.splitlines():
        row = json.loads(line)
        key = row["level"], row["repeat"], row["api"], row["position_id"]
        if row["arm"] in blocks[key]:
            raise ValueError(f"Duplicate arm in paired block {key}")
        blocks[key][row["arm"]] = row
    if any(set(block) != set(arms) for block in blocks.values()):
        raise ValueError("Incomplete timing block")

    levels = {}
    archived_match = {}
    for level, tag in (("truncated2", "2T"), ("truncated3", "3T")):
        pairs = [rows for key, rows in blocks.items() if key[0] == level]
        if not pairs:
            raise ValueError(f"Missing {level} records")
        control = [row["base"]["elapsed"] for row in pairs]
        stats = {}
        for arm in arms:
            elapsed = [row[arm]["elapsed"] for row in pairs]
            stats[arm] = {
                "seconds": sum(elapsed),
                "ratio": sum(control) / sum(elapsed),
                "median_ratio": median(x / y for x, y in zip(control, elapsed)),
                "faster": sum(y < x for x, y in zip(control, elapsed)),
                "exact": sum(row[arm]["analytics"] == row["base"]["analytics"] for row in pairs),
                "count": len(pairs),
            }
        levels[level] = stats
        if reference:
            original = {}
            path = reference / (tag + "-candidate") / "results.jsonl"
            for line in path.read_bytes().splitlines():
                row = json.loads(line)
                original[row["api"], row["position_id"]] = row
            matched = 0
            max_difference = 0.0
            for block in pairs:
                base = block["base"]
                previous = original[base["api"], base["position_id"]]
                inputs = dict(base["inputs"])
                if base["api"] == "cube_action":
                    inputs.setdefault("incl_2ply_details", False)
                if inputs != previous["inputs"] or base["seed"] != previous["seed"]:
                    raise ValueError("Archived call has different inputs or seed")
                matched += base["analytics"] == previous["analytics"]
                max_difference = max(max_difference, difference(base["analytics"], previous["analytics"]))
            archived_match[level] = {"exact": matched, "count": len(pairs),
                                     "max_abs_numeric_delta": max_difference}
    summary = {
        "metadata": metadata,
        "complete_blocks": len(blocks),
        "total_calls": sum(map(len, blocks.values())),
        "levels": levels,
        "archived_16_thread_control": archived_match,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / (prefix + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
    (destination / (prefix + ".jsonl.gz")).write_bytes(gzip.compress(raw, mtime=0))
    for level, stats in levels.items():
        for arm, row in stats.items():
            print(f"{level} {arm}: {row['ratio']:.4f}x, {row['exact']}/{row['count']} exact")
    if reference:
        print("Original 16-thread baseline:", archived_match)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--reference", type=Path, help="Original full 16-thread benchmark output directory")
    args = parser.parse_args()
    summarize(args.run, args.destination, args.prefix, args.reference)
