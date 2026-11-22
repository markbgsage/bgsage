# SPDX-License-Identifier: MPL-2.0
"""Archive completed screening runs and their complete public responses."""
from collections import Counter
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[3]
RUNS = ENGINE / "logs/truncated_search/screen-final"


def main():
    summary = {}
    with ZipFile(HERE / "screen_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for run in sorted(RUNS.iterdir()):
            if run.name == "memory":
                summary["working_set"] = {}
                for source in sorted(run.glob("*.json")):
                    obj = json.loads(source.read_bytes())
                    summary["working_set"][source.stem] = {k: v for k, v in obj.items() if k != "analytics"}
                    archive.write(source, "memory/" + source.name)
                continue
            record_file = run / "results.jsonl"
            records = [json.loads(line) for line in record_file.read_bytes().splitlines()]
            totals = Counter((v["level"], v["api"]) for v in records)
            if len(totals) != 6 or set(totals.values()) != {2}:
                raise ValueError(f"Incomplete experiment: {run.name}")
            stats = json.loads((run / "summary.json").read_bytes())
            levels = {}
            for level, by_api in stats["levels"].items():
                items = list(by_api.values())
                counts = sum((Counter(x["material_counts"]) for x in items), Counter())
                levels[level] = dict(calls=sum(x["calls"] for x in items),
                                     exact=sum(x["exact"] for x in items),
                                     seconds=sum(x["seconds"] for x in items),
                                     material=dict(counts))
            summary[run.name] = dict(binary=stats["extension"], levels=levels)
            for source in (record_file, run / "summary.json"):
                archive.write(source, run.name + "/" + source.name)
    (HERE / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
