# SPDX-License-Identifier: MPL-2.0
"""Verify and archive the two-repeat embedded 3P full-game performance check."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_rollout_performance as guard
from benchmark_multi_ply import file_identity
from compare_truncated_benchmark import score_api

ARMS = ("baseline", "candidate")


def verify_snapshot(side, hashes, official, manifest):
    expected = {a["path"]: a["sha256"] for a in manifest["assets"]
                if a["path"].startswith(("models/", "data/"))}
    expected.update({"python/bgsage/" + a["path"].split("python/bgsage/")[-1]: a["sha256"]
                     for a in official["source_files"] if "python/bgsage/" in a["path"]})
    native = official["extension"]
    expected["build/" + Path(native["path"]).name] = native["sha256"]
    if hashes != expected:
        raise ValueError(f"{side} binary, Python, model, database or corpus differs from its pinned inputs")
    if (official["status"], official["completed_calls"], official["threads_requested"]) != ("complete", 300, 16):
        raise ValueError(f"Incomplete official {side} qualification")
    return native


def package(run, qualification, output):
    settings = guard.truth.read(run / "settings.json")
    manifest_path = guard.truth.OUT / "manifest.json"
    if settings != dict(threads=16, cpu_mask=65535, profiles=["full"], full_trials=36,
                        repeats=2, cases=list(guard.CASES),
                        reference_manifest_sha256=file_identity(manifest_path)["sha256"]):
        raise ValueError("Unexpected guard settings or reference manifest")

    binaries = {}
    members = ["settings.json", "summary.json"]
    manifest = guard.truth.read(manifest_path)
    with ZipFile(qualification) as official:
        for side in ARMS:
            member = f"{side}/hashes.json"
            members.append(member)
            binaries[side] = verify_snapshot(side, guard.truth.read(run / member),
                                             json.loads(official.read(f"{side}/3T/metadata.json")), manifest)

    corpus = {p["id"]: p for name in ("checker_play", "cube_action")
              for p in guard.truth.read(guard.DATA / (name + ".json"))["positions"]}
    reference_path = guard.DATA / "rollout_positions/results.jsonl"
    references = {(row["api"], row["position_id"]): row for row in guard.read_rows(reference_path)}
    keys = [(api, ("checker-" if api == "checker_play" else "cube-") + case)
            for api in guard.APIS for case in guard.CASES]
    groups = defaultdict(list)
    pairs = []
    for repeat in range(settings["repeats"]):
        rows = {}
        for side in ARMS:
            member = f"{repeat}-{side}.jsonl"
            members.append(member)
            rows[side] = guard.read_rows(run / member)
            if [(r["api"], r["position_id"]) for r in rows[side]] != keys:
                raise ValueError(f"Unexpected cases or execution order in {member}")
        for (api, pid), left, right in zip(keys, rows["baseline"], rows["candidate"], strict=True):
            position = corpus[pid]
            inputs = position["post_move_inputs" if api == "post_move_analytics" else "inputs"]
            for row in (left, right):
                if (row["inputs"], row["seed"], row["profile"]) != (inputs, guard.truth.seed_for(position), "full"):
                    raise ValueError(f"Mismatched inputs, seed or profile for {pid}")
                if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
                    raise ValueError(f"Invalid elapsed time for {pid}")
                row.update(repeat=repeat, elapsed_seconds=row["seconds"], source=position["source"],
                           mode=position["mode"], source_game_plan=position["game_plan"])
            groups[api].append((left, right, references[api, pid]))
            pairs.append(dict(repeat=repeat, profile="full", api=api, position_id=pid,
                              baseline_seconds=left["seconds"], candidate_seconds=right["seconds"],
                              **guard.compare(left, right)))
    if pairs != guard.truth.read(run / "summary.json")["pairs"]:
        raise ValueError("Summary does not describe the raw paired analytics")

    reports = {api: score_api(api, calls) for api, calls in groups.items()}
    totals = {side: sum(p[side + "_seconds"] for p in pairs) for side in ARMS}
    errors = [d["change_in_absolute_error"] for r in reports.values() for d in r["material_changes_in_rollout_error"]]
    summary = dict(pairs=len(pairs), exact=sum(r["exact_analytics"] for r in reports.values()),
                   same_checker_order=reports["checker_play"]["same_full_move_order"],
                   same_checker_best=reports["checker_play"]["same_best_action"],
                   same_cube_action=reports["cube_action"]["same_best_action"],
                   seconds=totals, speedup=totals["baseline"] / totals["candidate"],
                   api_seconds={api: {side: r["timing"][side]["total_seconds"] for side in ARMS}
                                for api, r in reports.items()},
                   probability_threshold=.0025, equity_threshold=.005,
                   reference_observations=sum(f["n"] for r in reports.values()
                                              for f in r["rollout_error_by_field"].values()),
                   material_farther=sum(x > 0 for x in errors), material_closer=sum(x < 0 for x in errors),
                   max_abs_probability_shift=max(p["max_abs_probability"] for p in pairs),
                   max_abs_equity_shift=max(p["max_abs_equity"] for p in pairs))
    config = dict(guard.truth.CONFIG, n_trials=settings["full_trials"])
    del config["prefilter_threshold"]
    config.update(eval_level="rollout", variance_reduction=True, model="stage11",
                  checker={"ply": 3}, checker_late={"ply": 3}, cube={"ply": 3}, cube_late={"ply": 3})
    provenance = dict(binaries=binaries, runner=file_identity(Path(guard.__file__)),
                      accuracy_reference=file_identity(reference_path), full_game_profile=config,
                      arm_order=[["baseline", "candidate"], ["candidate", "baseline"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for member in members:
            archive.write(run / member, member)
        for name, value in (("verification", summary), ("accuracy_by_api", reports), ("provenance", provenance)):
            archive.writestr(name + ".json", json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(output)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "logs/truncated_pass2/final-fullgame")
    parser.add_argument("--qualification", type=Path, default=HERE / "qualified/windows/full_outputs.zip")
    parser.add_argument("--output", type=Path, default=HERE / "qualified/fullgame_guard_16.zip")
    cfg = parser.parse_args()
    package(cfg.run, cfg.qualification, cfg.output)
