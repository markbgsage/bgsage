# SPDX-License-Identifier: MPL-2.0
"""Verify complete benchmark sets, preserve public responses and score against the frozen rollouts."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, write_json
from compare_truncated_benchmark import compare, level_name

ARMS = ("original", "checkpoint", "candidate")
FILES = ("results.jsonl", "metadata.json", "summary.json")


def history(runs, arm, level, archive, member):
    folder = runs / arm / level
    present = [name for name in FILES if (folder / name).exists()]
    if len(present) == len(FILES):
        return  # A local, freshly timed run takes precedence over its historical replay.
    if present:
        raise ValueError(f"Incomplete local run at {folder}; its files were preserved")
    with ZipFile(archive) as src:
        values = {name: src.read(f"{member}/{name}") for name in FILES}
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (folder / name).write_bytes(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("platform", choices=("windows", "linux"))
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--levels", nargs="+", required=True, type=level_name)
    args = ap.parse_args()
    if len(set(args.levels)) != len(args.levels):
        ap.error("Select each level once")
    runs = args.runs.resolve()
    if not runs.is_relative_to(ROOT):
        raise ValueError("Expected engine-local benchmark directories")
    if args.platform == "windows":
        for level in set(args.levels) & {"1T", "2T", "3T"}:
            history(runs, "original", level, HERE.parent / "truncated_16t/full_outputs.zip", level + "-baseline")
            if level != "1T":
                history(runs, "checkpoint", level, HERE.parent / "truncated_search/full_outputs.zip", level + "-candidate")
    dest = HERE / args.platform
    dest.mkdir(exist_ok=True)
    paths = {}
    archive_files = []
    for arm in ARMS:
        paths[arm] = []
        seen = set()
        for folder in sorted((runs / arm).iterdir()):
            if not folder.is_dir():
                continue
            meta = json.loads((folder / "metadata.json").read_bytes())
            selected = set(map(level_name, meta["levels"]))
            if not selected <= set(args.levels):
                continue
            if seen & selected or meta["threads_requested"] != 16 or meta["status"] != "complete":
                raise ValueError(f"Incomplete, duplicate or non-16-thread run {folder}")
            if (meta["expected_calls"] != 300 * len(selected) or
                    meta["completed_calls"] != meta["expected_calls"] or meta["repeats"] != 1):
                raise ValueError(f"Unexpected call count {folder}")
            seen |= selected
            paths[arm].append(folder / "results.jsonl")
            archive_files.extend(folder / name for name in FILES)
        if seen != set(args.levels):
            raise ValueError(f"Missing {set(args.levels) - seen} for {arm}")
    summary = {}
    reports = {}
    for arm in ARMS[:2]:
        report = compare(paths[arm], paths["candidate"], DATA / "rollout_positions/results.jsonl", args.levels)
        reports[arm] = report
        for level, group in report["levels"].items():
            apis = group["by_api"]
            changes = Counter("farther" if row["change_in_absolute_error"] > 0 else "closer"
                              for api in apis.values() for row in api["material_changes_in_rollout_error"])
            item = dict(seconds=group["total_call_seconds"], speedup=group["speedup"],
                        identical=sum(api["exact_analytics"] for api in apis.values()),
                        same_checker_ranking=apis["checker_play"]["same_full_move_order"],
                        same_cube_decision=apis["cube_action"]["same_best_action"],
                        observations=sum(field["n"] for api in apis.values()
                                         for field in api["rollout_error_by_field"].values()),
                        material_farther=changes["farther"], material_closer=changes["closer"],
                        api_seconds={key: [api["timing"][side]["total_seconds"] for side in ("baseline", "candidate")]
                                     for key, api in apis.items()})
            summary.setdefault(level, {})[arm] = item
            print(level, arm, item, flush=True)
    # Ordinary sibling files inherit the destination's access rules. On Windows,
    # moving a file out of a private TemporaryDirectory retains its private ACL.
    # Finish every staged output before replacing the last qualified results.
    with ExitStack() as cleanup:
        staged = {}
        for name in ("full_outputs.zip", *(arm + "_comparison.json" for arm in reports), "summary.json"):
            pending = dest / f".{name}.{uuid4().hex}.tmp"
            pending.touch(exist_ok=False)
            cleanup.callback(pending.unlink, missing_ok=True)
            staged[name] = pending
        with ZipFile(staged["full_outputs.zip"], "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for path in archive_files:
                archive.write(path, path.relative_to(runs).as_posix())
        for arm, report in reports.items():
            write_json(staged[arm + "_comparison.json"], report)
        write_json(staged["summary.json"], summary)
        for name, pending in staged.items():
            pending.replace(dest / name)


if __name__ == "__main__":
    main()
