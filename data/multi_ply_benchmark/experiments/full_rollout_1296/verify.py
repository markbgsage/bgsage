# SPDX-License-Identifier: MPL-2.0
"""Verify the paired 1,296-path rollout panel and reconstruct its timing and accuracy report."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import geometric_mean, median
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, file_identity, inside_repo, write_json
from compare_multi_ply_accuracy import metrics
from compare_truncated_benchmark import endpoints, score_api, threshold

ARMS = ["original", "selected"]
PLANS = ("attacking", "priming", "anchoring", "racing", "purerace")
TRUTH = DATA / "rollout_positions"
PREVIOUS = HERE.parent / "truncated_pass2"
RAW_FILES = ("settings.json", "calls.jsonl")
SENSITIVITY = (
    (0, "checker_play", "checker-match-attacking-01"),
    (1, "cube_action", "cube-money-anchoring-01"),
    (14, "post_move_analytics", "cube-match-racing-01"),
)


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def check_settings(settings, policy, sensitivity):
    original = read(PREVIOUS / "historical-linux/provenance.json")
    qualified_path = PREVIOUS / "app-style/settings.json"
    qualified = read(qualified_path)["arms"]
    correction = None if sensitivity else original["source"]
    require(settings.get("sensitivity", False) is sensitivity, "Incorrect study identity")
    require(settings["original_commit"] == original["commit"] and settings["original_correction"] == correction
            and settings["selected_code_commit"] == "5de8023cc5e548ef5ab4400488508fd1bbe8aa7b", "Build identity")
    known = qualified_path.read_bytes().replace(b"\r\n", b"\n")
    fingerprint = settings["qualified_identities"]
    require(fingerprint["path"] == qualified_path.relative_to(ROOT).as_posix()
            and any(hashlib.sha256(data).hexdigest() == fingerprint["sha256"] and len(data) == fingerprint["bytes"]
                    for data in (known, known.replace(b"\n", b"\r\n"))), "Build manifest")
    require(settings["public_constructor"] == policy["fixed_public_trial_config"], "Full-game constructor settings")
    require(all(settings[key] == 3 for key in ("early_checker_ply", "late_checker_ply", "early_cube_ply", "late_cube_ply")),
            "Checker or cube depth")
    require(settings["threads"] == 16 and settings["model"] == "stage11" and settings["level"] == "rollout"
            and settings["cubeful"] is True and int(settings["cpu_affinity"], 0) == 65535, "Analyzer settings")
    require((settings["root_prefilter"], settings["root_filter_max_moves"], settings["root_filter_threshold"])
            == (0.15, 5, 0.08), "Checker filter")
    first = ARMS[::-1] if sensitivity else ARMS
    require(settings["even_sequence_order"] == first and settings["odd_sequence_order"] == first[::-1], "Pair ordering")
    for arm, identity in settings["arms"].items():
        for key in ("extension", "python", "analyzer", "database", "weight_slots"):
            expected = qualified[arm][key]
            if sensitivity and arm == "original" and key == "extension":
                expected = original["original_extension"]
            # Independent source exports can place identical assets at different paths.
            actual_files = identity[key] if key == "weight_slots" else [identity[key]]
            pinned_files = expected if key == "weight_slots" else [expected]
            for actual, pinned in zip(actual_files, pinned_files, strict=True):
                require((actual["bytes"], actual["sha256"]) == (pinned["bytes"], pinned["sha256"]), f"Unexpected {arm} {key}")
        config = dict(settings["public_constructor"], n_threads=16, enable_vr=True)
        for key in ("late_threshold", "late_ply"):
            config.pop(key)  # passed to the constructor, absent from the recorded runtime projection
        require(identity["config"] == config and identity["affinity"] == list(range(16)), f"Runtime settings for {arm}")
    require(set(settings["arms"]) == set(ARMS), "Wrong engine arms")


def is_terminal(board):
    positive = sum(max(0, count) for count in board[1:25]) + board[25]
    negative = sum(max(0, -count) for count in board[1:25]) + board[0]
    return (positive == 0) != (negative == 0)


def timing(pairs):
    before, after = [math.fsum(p[arm + "_seconds"] for p in pairs) for arm in ARMS]
    factors = [p["speedup"] for p in pairs]
    return dict(calls=len(pairs), original_seconds=before, selected_seconds=after, speedup=before / after,
                percent_time_saved=100 * (1 - after / before), median_paired_speedup=median(factors),
                geometric_mean_paired_speedup=geometric_mean(factors), faster_pairs=sum(r > 1 for r in factors))


def check_pair(rows, job, reference, source):
    api, pid = job["api"], job["position_id"]
    for row in rows:
        require(all(row[key] == value for key, value in job.items()), f"Input, seed, order or context for {pid}")
        require(math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0, f"Invalid timing for {pid}")
        row.update(source=source, repeat=1)  # fields the existing frozen-reference scorer requires
    before, after = rows
    detail = dict(sequence=job["sequence"], api=api, position_id=pid, mode=job["mode"],
                  game_plan=job["source_game_plan"], seed=job["seed"], reference_origin=reference["reference_origin"],
                  recorded_reference_seed=reference["seed"],
                  original_seconds=before["elapsed_seconds"], selected_seconds=after["elapsed_seconds"],
                  speedup=before["elapsed_seconds"] / after["elapsed_seconds"])
    if detail["reference_origin"] == "frozen_public_rollout":
        require(job["seed"] == reference["seed"], f"Frozen dice seed for {pid}")
    elif detail["reference_origin"] == "archival_rollout":
        p = reference["provenance"]
        require(reference["seed"] is None and not p["rollout_rng_seed_recorded"] and p["actual_batches"] is None,
                f"Unknown historical provenance for {pid}")
    else:
        raise ValueError(f"Unknown reference for {pid}")

    moves = [endpoints(api, row["analytics"]) for row in (*rows, reference)]
    require(moves[0].keys() == moves[1].keys() == moves[2].keys(), f"Legal endpoint sets for {pid}")
    if api == "checker_play":
        detail["legal_endpoints"] = len(moves[0])
        boards = [[board for board, item in arm.items() if item["eval_level"] == "Rollout"] for arm in moves]
        detail["rollout_endpoints"] = {arm: sorted(group) for arm, group in zip([*ARMS, "reference"], boards)}
        detail["same_rollout_shortlist_including_promotions"] = set(boards[0]) == set(boards[1])
        detail["same_full_order"] = list(moves[0]) == list(moves[1])
        detail["same_best_move"] = next(iter(moves[0])) == next(iter(moves[1]))
        detail["count_at_top_equity"] = {}
        for arm, row in zip(ARMS, rows):
            choices = row["analytics"]["moves"]
            detail["count_at_top_equity"][arm] = sum(m["equity"] == choices[0]["equity"] for m in choices)
        detail["changed_levels"] = sum(a["eval_level"] != moves[1][board]["eval_level"] for board, a in moves[0].items())
        detail["fixed_batches"] = {arm: sum(not is_terminal(board) for board in group)
                                   for arm, group in zip(ARMS, boards)}
    else:
        require(all(row["analytics"]["eval_level"] == "Rollout" for row in (*rows, reference)), f"Level for {pid}")
        require(not is_terminal(job["inputs"]["board"]), f"Unexpected terminal public {api} input for {pid}")
        detail["fixed_batches"] = dict.fromkeys(ARMS, 1)
        if api == "cube_action":
            detail["same_cube_action"] = before["analytics"]["optimal_action"] == after["analytics"]["optimal_action"]
    shifts = defaultdict(list)
    for board, old in moves[0].items():
        new = dict(metrics(moves[1][board]))
        for field, value in metrics(old):
            kind = "probability" if field.startswith("p.") else "equity"
            diff = abs(value - new[field])
            require(math.isfinite(diff), f"Invalid {field} for {pid}")
            shifts[kind].append(diff)
    detail["direct_shifts"] = {
        kind: {"observations": len(items), "largest": max(items),
               "material": sum(value > threshold("p.win" if kind == "probability" else "equity") for value in items)}
        for kind, items in shifts.items()
    }
    return detail


def analyze(raw, sensitivity=False, primary=None):
    settings = json.loads(raw["settings.json"])
    rows = [json.loads(line) for line in raw["calls.jsonl"].splitlines()]
    manifest = read(TRUTH / "manifest.json")
    check_settings(settings, manifest["policy"], sensitivity)
    reference_file = TRUTH / "results.jsonl"
    require(file_identity(reference_file)["sha256"] == manifest["outputs"]["results.jsonl"], "Frozen reference digest")
    references = {(r["api"], r["position_id"]): r for r in map(json.loads, reference_file.read_bytes().splitlines())}
    positions = {}
    for api in APIS[:2]:
        path = DATA / (api + ".json")
        require(file_identity(path) == settings["positions"][api]
                and settings["positions"][api]["sha256"] == manifest["fixed_sets"][path.name], "Input corpus digest")
        positions.update((p["id"], p) for p in read(path)["positions"])

    jobs = settings["roster"]
    require(len(rows) == 2 * len(jobs), "Incomplete paired run")
    if sensitivity:
        require(tuple((j["sequence"], j["api"], j["position_id"]) for j in jobs) == SENSITIVITY, "Sensitivity roster")
    else:
        require(len(set(j["position_id"] for j in jobs)) == 10, "Expected 10 distinct benchmark positions")
        require(Counter((j["api"], j["source_game_plan"]) for j in jobs) == {(a, p): 1 for a in APIS for p in PLANS},
                "Expected one request per API and game plan")
    details, scores = [], defaultdict(list)
    for index, job in enumerate(jobs):
        api, pid = job["api"], job["position_id"]
        position = positions[pid]
        mode, plan = position["mode"], position["game_plan"]
        checker_mode = "match" if PLANS.index(plan) % 2 == 0 else "money"
        require(pid.endswith("-01") and mode == job["mode"] and plan == job["source_game_plan"]
                and ((mode == checker_mode) == (api == "checker_play")), "Incorrect stratum")
        expected_seed = int.from_bytes(hashlib.sha256(("rollout-truth-v1:" + position["source"]["key"]).encode()).digest()[:4], "big")
        inputs = position["post_move_inputs" if api == "post_move_analytics" else "inputs"]
        require(job["seed"] == expected_seed and job["inputs"] == inputs and (sensitivity or job["sequence"] == index),
                f"Recorded roster differs from the source for {pid}")
        run = rows[2 * index:2 * index + 2]
        expected_order = settings["odd_sequence_order" if job["sequence"] % 2 else "even_sequence_order"]
        require([r["arm"] for r in run] == expected_order, f"Arm order for {pid}")
        if expected_order[0] != ARMS[0]:
            run.reverse()
        reference = references[api, pid]
        details.append(check_pair(run, job, reference, position["source"]))
        scores[api].append((*run, reference))

    accuracy = {api: score_api(api, cases) for api, cases in scores.items()}
    fields = [stats for api in accuracy.values() for stats in api["rollout_error_by_field"].values()]
    total_fields = sum(f["n"] for f in fields)
    scored_expected = sum(8 * len(p["rollout_endpoints"]["reference"]) if p["api"] == "checker_play"
                          else (10 if p["api"] == "cube_action" else 7) for p in details)
    require(total_fields == scored_expected, "Frozen score includes missing or unrolled fields")
    batches = {arm: sum(p["fixed_batches"][arm] for p in details) for arm in ARMS}
    checkers = [p for p in details if p["api"] == "checker_play"]
    summary = dict(
        old_commit=settings["original_commit"], selected_commit=settings["selected_code_commit"],
        overall=timing(details), by_api={api: timing([p for p in details if p["api"] == api]) for api in APIS},
        by_game_plan={plan: timing([p for p in details if p["game_plan"] == plan])
                      for plan in PLANS if any(p["game_plan"] == plan for p in details)},
        by_mode={mode: timing([p for p in details if p["mode"] == mode]) for mode in ("money", "match")},
        by_first_arm={arm: timing([p for p in details if settings["even_sequence_order"][p["sequence"] % 2] == arm])
                      for arm in ARMS},
        batches=batches, scheduled_trial_paths={arm: count * settings["public_constructor"]["n_trials"]
                                              for arm, count in batches.items()},
        same_checker_rollout_shortlist=sum(p["same_rollout_shortlist_including_promotions"] for p in checkers),
        endpoints_returned=sum(p["legal_endpoints"] for p in checkers),
        checker_rollout_endpoints={arm: sum(len(p["rollout_endpoints"][arm]) for p in checkers)
                                  for arm in (*ARMS, "reference")},
        reference_fields=total_fields, material_farther=sum(f["materially_farther"] for f in fields),
        material_closer=sum(f["materially_closer"] for f in fields), probability_threshold=0.0025, equity_threshold=0.005,
        direct_shifts={kind: dict(
            observations=sum(p["direct_shifts"][kind]["observations"] for p in details),
            material=sum(p["direct_shifts"][kind]["material"] for p in details),
            largest=max(p["direct_shifts"][kind]["largest"] for p in details)) for kind in ("probability", "equity")},
        same_checker_order=accuracy["checker_play"]["same_full_move_order"], same_checker_best=accuracy["checker_play"]["same_best_action"],
        same_cube_action=accuracy["cube_action"]["same_best_action"], complete_analytics_equal=sum(a["exact_analytics"] for a in accuracy.values()),
        reference_origins=dict(Counter(p["reference_origin"] for p in details)), pairs=details,
        reference_sha256=manifest["outputs"]["results.jsonl"],
        source_sha256={name: hashlib.sha256(data).hexdigest() for name, data in raw.items()})
    if sensitivity:
        primary_rows = {(r["api"], r["position_id"], r["arm"]): r for r in primary}
        replay = []
        for job, pair in zip(jobs, details, strict=True):
            key = (job["api"], job["position_id"])
            raw_original, raw_selected, ref = scores[key[0]][0]  # one case per API in the control check
            corrected, first_selected = (primary_rows[(*key, arm)] for arm in ARMS)
            for prior in (corrected, first_selected):
                require(all(prior[k] == v for k, v in job.items()), "Different input in main and sensitivity runs")
            compared = check_pair([raw_original, corrected], job, ref, positions[key[1]]["source"])
            scoring = score_api(key[0], [(raw_original, corrected, ref)])
            errors = scoring["material_changes_in_rollout_error"]
            replay.append(dict(api=key[0], position_id=key[1],
                               unmodified_original_seconds=pair["original_seconds"], corrected_original_seconds=corrected["elapsed_seconds"],
                               first_selected_seconds=first_selected["elapsed_seconds"], repeated_selected_seconds=pair["selected_seconds"],
                               selected_analytics_identical=first_selected["analytics"] == raw_selected["analytics"],
                               original_analytics_identical=corrected["analytics"] == raw_original["analytics"],
                               direct_shifts_raw_to_corrected=compared["direct_shifts"],
                               raw_to_corrected_material_farther=sum(d["change_in_absolute_error"] > 0 for d in errors),
                               raw_to_corrected_material_closer=sum(d["change_in_absolute_error"] < 0 for d in errors)))
        summary["comparison_with_primary_run"] = replay
    return summary, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Package completed engine logs; omit to audit the published archive")
    parser.add_argument("--sensitivity", action="store_true", help="Verify the three unmodified-original pairs")
    args = parser.parse_args()
    dest = HERE / "sensitivity" if args.sensitivity else HERE
    archive = dest / "full_outputs.zip"
    primary = None
    if args.sensitivity:
        with ZipFile(HERE / "full_outputs.zip") as source:
            data = source.read("calls.jsonl")
        require(hashlib.sha256(data).hexdigest() == read(HERE / "summary.json")["source_sha256"]["calls.jsonl"],
                "Primary calls digest")
        primary = list(map(json.loads, data.splitlines()))
    if args.run:
        run = inside_repo(args.run)
        raw = {name: (run / name).read_bytes() for name in RAW_FILES}
        settings = json.loads(raw["settings.json"])
        if "node" in settings["platform"]:
            del settings["platform"]["node"]
            raw["settings.json"] = (json.dumps(settings, indent=2, allow_nan=False) + "\n").encode()
        summary, accuracy = analyze(raw, args.sensitivity, primary)
        dest.mkdir(exist_ok=True)
        temp = archive.with_suffix(".tmp")
        contents = {**raw, "accuracy_by_api.json": json.dumps(accuracy, indent=2, allow_nan=False) + "\n"}
        with ZipFile(temp, "w") as z:
            for name, data in contents.items():
                z.writestr(ZipInfo(name), data, compress_type=ZIP_DEFLATED, compresslevel=9)
        temp.replace(archive)
        write_json(dest / "summary.json", summary)
    else:
        with ZipFile(archive) as z:
            summary, accuracy = analyze({name: z.read(name) for name in RAW_FILES}, args.sensitivity, primary)
            require(json.loads(z.read("accuracy_by_api.json")) == json.loads(json.dumps(accuracy)),
                    "Archived accuracy differs from full outputs")
        require(read(dest / "summary.json") == json.loads(json.dumps(summary)), "Published summary differs from full outputs")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
