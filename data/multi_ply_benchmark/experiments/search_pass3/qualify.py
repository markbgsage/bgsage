# SPDX-License-Identifier: MPL-2.0
"""Time a frozen build per preset, or score and archive paired full-corpus runs."""
import argparse
from collections import Counter
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, TRUNCATED_LEVELS, inside_repo, write_json
from compare_truncated_benchmark import compare, level_name


def run(cfg):
    build, source = inside_repo(cfg.build), inside_repo(cfg.source)
    binary = next(build / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                  if (build / ("bgbot_cpp" + suffix)).is_file())
    for level in cfg.levels:
        preset = TRUNCATED_LEVELS.get(level, level[0] + "ply")
        path = inside_repo(cfg.run_dir) / cfg.arm / level
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/benchmark_multi_ply.py"),
                        "--model", "stage11", "--levels", preset, "--threads", "16",
                        "--extension", str(binary), "--python-dir", str(source / "python"),
                        "--native-source", str(source), "--output", str(path)], check=True)


def report(cfg):
    runs, out = inside_repo(cfg.run_dir), inside_repo(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = {a: [] for a in ("baseline", "candidate")}
    with ZipFile(out / "full_outputs.zip", "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for level in cfg.levels:
            identities = []
            for arm in paths:
                folder = runs / arm / level
                meta = json.loads((folder / "metadata.json").read_bytes())
                if (meta["status"], meta["completed_calls"], meta["expected_calls"], meta["threads_requested"],
                    meta["repeats"], list(map(level_name, meta["levels"]))) != (
                        "complete", 300, 300, 16, 1, [level]):
                    raise ValueError(f"Incomplete or mismatched run in {folder}")
                identities.append(([x["sha256"] for x in meta["weights"]],
                                   meta["bearoff_database"]["sha256"],
                                   {k: v["sha256"] for k, v in meta["positions"].items()}))
                paths[arm].append(folder / "results.jsonl")
                for name in ("results.jsonl", "metadata.json", "summary.json"):
                    archive.write(folder / name, (folder / name).relative_to(runs).as_posix())
            if identities[0] != identities[1]:
                raise ValueError(f"Different benchmark inputs or model assets at {level}")
    report = compare(paths["baseline"], paths["candidate"], DATA / "rollout_positions/results.jsonl", cfg.levels)
    write_json(out / "comparison.json", report)
    summary = {}
    for level, data in report["levels"].items():
        apis = data["by_api"]
        changes = Counter("farther" if f["change_in_absolute_error"] > 0 else "closer"
                          for stats in apis.values() for f in stats["material_changes_in_rollout_error"])
        summary[level] = dict(seconds=data["total_call_seconds"], speedup=data["speedup"],
            exact=sum(a["exact_analytics"] for a in apis.values()),
            same_best_checker=apis["checker_play"]["same_best_action"],
            same_checker_order=apis["checker_play"]["same_full_move_order"],
            same_cube_action=apis["cube_action"]["same_best_action"],
            observations=sum(field["n"] for a in apis.values() for field in a["rollout_error_by_field"].values()),
            material_farther=changes["farther"], material_closer=changes["closer"],
            api_seconds={api: [a["timing"][arm]["total_seconds"] for arm in paths] for api, a in apis.items()})
        print(level, summary[level], flush=True)
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    actions = ap.add_subparsers(dest="action", required=True)
    runner = actions.add_parser("run")
    runner.add_argument("arm", choices=("baseline", "candidate"))
    runner.add_argument("--source", required=True, type=Path)
    runner.add_argument("--build", required=True, type=Path)
    reporter = actions.add_parser("report")
    reporter.add_argument("--output", type=Path, required=True)
    for p in (runner, reporter):
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--levels", nargs="+", type=level_name, required=True)
    cfg = ap.parse_args()
    (run if cfg.action == "run" else report)(cfg)
