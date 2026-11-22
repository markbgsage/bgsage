# SPDX-License-Identifier: MPL-2.0
"""Compare all 2,100 non-full-rollout analytics with the verified prior engine."""
import argparse
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent / "search_pass3"))
import qualify
from benchmark_multi_ply import file_identity, write_json

CONTROL = HERE.parent / "truncated_pass2/qualified/linux/full_outputs.zip"
CONTROL_SETTINGS = HERE.parent / "truncated_pass2/app-style/settings.json"
LEVELS = ["1P", "2P", "3P", "4P", "1T", "2T", "3T"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["BGBOT_MULTIPLY_THREADS"] = "16"
    os.sched_setaffinity(0, list(range(16)))
    runs = qualify.inside_repo(args.run_dir)
    runs.mkdir(exist_ok=False, parents=True)
    build = qualify.inside_repo(args.build)
    binary = next(build / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                  if (build / ("bgbot_cpp" + suffix)).is_file())
    frozen = json.loads(CONTROL_SETTINGS.read_bytes())["arms"]["selected"]["extension"]
    with ZipFile(CONTROL) as control:
        for level in LEVELS:
            dest = runs / "baseline" / level
            dest.mkdir(parents=True)
            for name in ("metadata.json", "results.jsonl", "summary.json"):
                (dest / name).write_bytes(control.read(f"candidate/{level}/{name}"))
            meta = json.loads((dest / "metadata.json").read_bytes())
            if meta["extension"]["sha256"] != frozen["sha256"]:
                raise ValueError("Archived control binary differs from the full-rollout control")
    selected = file_identity(binary)
    truth = qualify.DATA / "rollout_positions/results.jsonl"
    bounds = {}
    for level in LEVELS:
        qualify.run(SimpleNamespace(build=build, source=build, run_dir=runs, arm="candidate", levels=[level]))
        candidate = runs / "candidate" / level
        meta = json.loads((candidate / "metadata.json").read_bytes())
        if meta["extension"]["sha256"] != selected["sha256"]:
            raise ValueError(f"Unexpected candidate binary at {level}")
        baseline = runs / "baseline" / level
        result = qualify.compare([baseline / "results.jsonl"], [candidate / "results.jsonl"], truth, [level])
        fields = [(name, stats) for api in result["levels"][level]["by_api"].values()
                  for name, stats in api["rollout_error_by_field"].items()]
        bounds[level] = {}
        for kind in ("equity", "probability"):
            selected_fields = [stats for name, stats in fields if name.startswith("p.") == (kind == "probability")]
            bounds[level][kind] = dict(
                observations=sum(s["n"] for s in selected_fields),
                largest_increase=max(s["worst_change_in_abs_error"] for s in selected_fields),
                largest_decrease=min(s["best_change_in_abs_error"] for s in selected_fields),
                materially_farther=sum(s["materially_farther"] for s in selected_fields),
                materially_closer=sum(s["materially_closer"] for s in selected_fields),
            )
        write_json(runs / "truth_bounds.json", bounds)
        print(level, "change in distance to rollout truth", bounds[level], flush=True)
    out = qualify.inside_repo(args.output)
    qualify.report(SimpleNamespace(run_dir=runs, output=out, levels=LEVELS))
    write_json(out / "truth_bounds.json", bounds)
    write_json(out / "control.json", dict(
        archive=file_identity(CONTROL), member_arm="candidate", control_extension=frozen,
        selected_extension=selected,
        timing_design="The archived control and the new candidate ran in different sessions on the same machine.",
    ))


if __name__ == "__main__":
    main()
