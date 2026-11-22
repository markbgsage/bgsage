# SPDX-License-Identifier: MPL-2.0
"""Summarize an independent, reverse-order full-corpus Linux timing pass."""
from zipfile import ZipFile, ZIP_DEFLATED

from qualify import DATA, FILES, HERE, ROOT, compare, write_json


if __name__ == "__main__":
    runs = ROOT / "logs/search_pass2_repeat/linux"
    levels = ("1P", "2P", "3P", "4P")
    paths = {arm: [runs / arm / level / "results.jsonl" for level in levels]
             for arm in ("checkpoint", "candidate")}
    report = compare(paths["checkpoint"], paths["candidate"],
                     DATA / "rollout_positions/results.jsonl", levels)
    out = HERE / "linux-repeat"
    out.mkdir(exist_ok=True)
    with ZipFile(out / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for arm in paths.values():
            for run in arm:
                for name in FILES:
                    path = run.with_name(name)
                    archive.write(path, path.relative_to(runs).as_posix())
    write_json(out / "comparison.json", report)
    for level, stats in report["levels"].items():
        print(level, stats["total_call_seconds"], stats["speedup"])
