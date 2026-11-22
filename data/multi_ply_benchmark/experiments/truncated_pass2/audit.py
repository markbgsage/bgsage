# SPDX-License-Identifier: MPL-2.0
"""Recompute the paired timings, rollout accuracy, coverage and input provenance."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
from statistics import fmean
import subprocess
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "search_pass3"
sys.path.insert(0, str(PREVIOUS))
import verify as prior
from benchmark_multi_ply import APIS, DATA, ROOT, inside_repo, write_json
from compare_multi_ply_accuracy import metrics
from compare_truncated_benchmark import endpoints, score_api, threshold
from measure_multi_ply_pair import verify_control_sources

BASE = "1ecca3902fc7d04e35c43b9fcd3ec45920d0fcb5"
SELECTED = "5de8023cc5e548ef5ab4400488508fd1bbe8aa7b"
INTERMEDIATE = "82e63972379b77ab52ebafc716620e813ff7baa1"
ORIGINAL = "139427d826dd67798226e7d23c003c90315306ba"
LEVELS = ("1P", "2P", "3P", "4P", "1T", "2T", "3T")
ARMS = ("baseline", "candidate")
MATCHED_SETTINGS = ("schema_version", "model", "strategy_type", "hidden_sizes", "apis", "cache_policy",
                    "threads_requested", "analyzer_threads_by_level", "thread_note", "repeats", "cubeful",
                    "cube_details", "ordering", "timing_scope", "seed_policy", "expected_calls", "completed_calls",
                    "truncated_presets", "environment", "levels", "platform", "machine", "logical_cpu_count")


def read_json(path):
    return json.loads(path.read_bytes())


def archived_json(z, member):
    return json.loads(z.read(member))


def pinned_sources(tree, candidate, checkout):
    files = prior.source_paths(tree)
    revision = SELECTED if candidate else BASE
    verify_control_sources(revision, tree, tree / "python")
    cmake = subprocess.check_output(["git", "show", revision + ":CMakeLists.txt"], cwd=ROOT)
    prior.require(prior.text_digest(tree / "CMakeLists.txt") == prior.digest(cmake.replace(b"\r\n", b"\n")),
                  "CMake differs from " + revision)
    if candidate:
        shipped = prior.source_paths(checkout)
        prior.require(files.keys() == shipped.keys(), "Candidate source inventory differs from the checkout")
        for name, file in files.items():
            prior.require(prior.text_digest(file) == prior.text_digest(shipped[name]), f"Unshipped candidate code in {name}")
    return files


def check_runtime(meta, published, platform, arm, key, checked, checkout=ROOT):
    for f in MATCHED_SETTINGS:
        prior.require(meta.get(f) == published.get(f), key + " differs from checkpoint metadata at " + f)
    prior.require(meta["threads_requested"] == 16 and meta["completed_calls"] == 300
                  and meta["status"] == "complete" and meta["git_commit"] in {BASE, SELECTED}
                  and platform in meta["platform"].lower(), key + " engine, corpus, or platform mismatch")
    ext = inside_repo(ROOT / PurePosixPath(meta["extension"]["path"]))
    tree = inside_repo(ROOT / PurePosixPath(meta["native_source_tree"]))
    config = str(ext), str(tree)
    if config not in checked:
        checked[config] = pinned_sources(tree, arm == "candidate", checkout)
    sources = checked[config].keys() | {"scripts/benchmark_multi_ply.py", "scripts/build_multi_ply_benchmark.py"}
    identity = prior.runtime(meta, tree, ext, published, sources)
    if arm == "baseline":
        prior.require(identity["extension_sha256"] == published["extension"]["sha256"], key + " checkpoint binary")
    identity["binary"] = ext.relative_to(ROOT).as_posix()
    identity["source_tree"] = tree.relative_to(ROOT).as_posix()
    return identity


def value_group(field):
    return "probability" if field.startswith("p.") else "equity"


def error_summary(data):
    result = {}
    for kind, observations in sorted(data.items()):
        # Each observation has old prediction, new prediction and reference.
        old = [abs(a - target) for a, _, target in observations]
        new = [abs(b - target) for _, b, target in observations]
        change = [b - a for a, b in zip(old, new)]
        cutoff = 0.0025 if kind == "probability" else 0.005
        result[kind] = dict(fields=len(new), baseline_mae=fmean(old), candidate_mae=fmean(new),
                            mean_change_in_error=fmean(change), max_increase=max(change), max_decrease=min(change),
                            material_farther=sum(v > cutoff for v in change), material_closer=sum(v < -cutoff for v in change))
    return result


def summarize(rows, truth, api_scores):
    groups = defaultdict(lambda: defaultdict(list))
    costs = defaultdict(lambda: {arm: [] for arm in ARMS})
    direct = defaultdict(list)
    rolled = Counter()
    checker_endpoints = 0
    for left, right in zip(rows["baseline"], rows["candidate"], strict=True):
        api, pid = left["api"], left["position_id"]
        splits = ("all", "api/" + api, "mode/" + left["mode"], "plan/" + left["source_game_plan"])
        for group in splits:
            for row, arm in zip((left, right), ARMS):
                costs[group][arm].append(row["elapsed_seconds"])
        baseline, candidate, reference = [endpoints(api, r["analytics"]) for r in (left, right, truth[api, pid])]
        prior.require(baseline.keys() == candidate.keys() == reference.keys(), f"Missing/extra legal endpoint for {pid}")
        if api == "checker_play":
            checker_endpoints += len(reference)
        for board, original in reference.items():
            a, b = dict(metrics(baseline[board])), dict(metrics(candidate[board]))
            prior.require(a.keys() == b.keys() == dict(metrics(original)).keys(), f"Missing predicted field at {pid}")
            for field in a:
                direct[value_group(field)].append(abs(b[field] - a[field]))
            if original["eval_level"] == "Rollout":
                rolled[api] += 1
                for field, target in metrics(original):
                    for group in splits:
                        groups[group][value_group(field)].append((a[field], b[field], target))
    prior.require(checker_endpoints == 1709 and rolled == {"checker_play": 378, "cube_action": 100,
                  "post_move_analytics": 100}, "Legal checker or rolled endpoint coverage changed")
    grouped = {}
    for group, times in sorted(costs.items()):
        pairs = list(zip(times["baseline"], times["candidate"]))
        grouped[group] = dict(calls=len(pairs), seconds={arm: sum(t) for arm, t in times.items()},
                              faster_calls=sum(b < a for a, b in pairs), errors=error_summary(groups[group]))
    direct = {k: dict(fields=len(v), max_abs_change=max(v), mean_abs_change=fmean(v),
                      material=sum(x > threshold("p.win" if k == "probability" else "equity") for x in v))
              for k, v in direct.items()}
    apis = api_scores
    all_errors = grouped["all"]["errors"].values()
    compact = dict(seconds={a: sum(apis[api]["timing"][a]["total_seconds"] for api in APIS) for a in ARMS},
                   exact=sum(s["exact_analytics"] for s in apis.values()),
                   same_best_checker=apis["checker_play"]["same_best_action"],
                   same_checker_order=apis["checker_play"]["same_full_move_order"],
                   same_cube_action=apis["cube_action"]["same_best_action"],
                   observations=sum(e["fields"] for e in all_errors),
                   material_farther=sum(e["material_farther"] for e in all_errors),
                   material_closer=sum(e["material_closer"] for e in all_errors),
                   api_seconds={api: [apis[api]["timing"][a]["total_seconds"] for a in ARMS] for api in APIS})
    compact["speedup"] = compact["seconds"]["baseline"] / compact["seconds"]["candidate"]
    prior.require(compact["observations"] == 4724, "Wrong number of rollout-scored fields")
    decisions = {a: apis[a]["decision_accuracy"] for a in ("checker_play", "cube_action")}
    return compact, dict(grouped=grouped, direct=direct, all_checker_endpoints=checker_endpoints,
                         rolled_endpoints=dict(rolled), decisions_vs_rollout=decisions)


def verify_original(platform):
    study = HERE.parent / "search_pass2"
    checked = read_json(study / "control_inputs.json")["original"]
    prior.require(checked["commit"] == ORIGINAL and checked["verified_native_and_python_files"] == 73,
                  "Incorrect historical original")
    verify_control_sources(ORIGINAL, ROOT / checked["native_source"], ROOT / checked["python_source"])
    prior.verify_id(ROOT / checked[platform]["path"], checked[platform])
    return checked[platform]["sha256"], read_json(study / platform / "original_comparison.json")


def check_corrected_original(tree, metadata, archive):
    provenance = read_json(archive.parent / "provenance.json")
    proof = provenance["source"]
    prior.require(provenance["commit"] == ORIGINAL and proof["correction_commit"].startswith("60f9355")
                  and (proof["inserted_lines"], proof["deleted_lines"]) == (10, 0)
                  and provenance["extension"] == metadata["extension"], "Unexpected corrected Linux control")
    native = "cpp/src/rollout.cpp"
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT)
    fixed = git("show", proof["correction_commit"] + ":" + native).decode()
    start = fixed.index("        // Reserve every scheduled entry before any worker can start a trial.\n")
    end = fixed.index("        // Use persistent thread pool — same rationale as cubeless path.\n", start)
    inserted = fixed[start:end].encode()
    prior.require(len(inserted.splitlines()) == 10 and inserted.count(b".store(1,") == 2, "Unexpected publication fix")
    blobs = git("ls-tree", "-rz", ORIGINAL, "--", "cpp", "python").rstrip(b"\0").split(b"\0")
    for entry in blobs:
        meta, name = entry.split(b"\t", 1)
        name = name.decode()
        data = (tree / name).read_bytes().replace(b"\r\n", b"\n")
        if name == native:
            prior.require(data.count(inserted) == 1 and prior.digest(data) == proof["corrected_source_sha256"], name + " corrected body")
            data = data.replace(inserted, b"", 1)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        prior.require(blob == meta.split()[2].decode(), "Unexpected additional historic code edits in " + name)
    prior.require(len(blobs) == proof["verified_git_blobs"] == 73, "Missing original source blobs")
    historical_cmake = ROOT / read_json(HERE.parent / "search_pass2/control_inputs.json")["original"]["linux_native_source"] / "CMakeLists.txt"
    prior.verify_id(tree / "CMakeLists.txt", proof["original_linux_cmake_wrapper"])
    prior.require(prior.text_digest(tree / "CMakeLists.txt") == prior.text_digest(historical_cmake), "Original Linux build wrapper")
    source = prior.source_paths(tree).keys() | {"scripts/benchmark_multi_ply.py", "scripts/build_multi_ply_benchmark.py"}
    return dict(source_files=source, verified_blobs=len(blobs), provenance_sha256=prior.digest((archive.parent / "provenance.json").read_bytes()))


def original_comparison(old_rows, now, truth):
    sides = dict(baseline=old_rows, candidate=now)
    scores = {}
    for api in APIS:
        pairs = [(a, b, truth[api, a["position_id"]]) for a, b in zip(old_rows, now) if a["api"] == api]
        scores[api] = score_api(api, pairs)
    compact, extra = summarize(sides, truth, scores)
    changes = {api: values["material_changes_in_rollout_error"] for api, values in scores.items()}
    return dict(**compact, **extra, material_changes_against_original=changes)


def truth_policy(result_digest):
    path = DATA / "rollout_positions/manifest.json"
    original = subprocess.check_output(["git", "show", BASE + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT)
    prior.require(json.loads(original) == read_json(path), "Reference manifest changed since checkpoint")
    manifest = json.loads(original)
    policy = manifest["policy"]
    fixed = policy["fixed_public_trial_config"]
    prior.require(manifest["outputs"]["results.jsonl"] == result_digest and policy["variance_reduction"] is True
                  and set(policy["checker_and_cube_phase_ply"].values()) == {3}
                  and (fixed["n_trials"], fixed["truncation_depth"], fixed["max_batches"]) == (1296, 0, 1)
                  and policy["source_policy"]["paths_per_batch"] == 1296, "Incorrect full-game reference policy")
    return dict(frozen_engine=manifest["frozen_engine"]["revision"], games_per_batch=1296, variance_reduction=True,
                early_late_ply=policy["checker_and_cube_phase_ply"], archived_settings=policy["source_policy"],
                manifest_sha256=prior.digest(path.read_bytes()))


def validate_control(z, level, record, source):
    prefix = "baseline/" + level + "/"
    path = PurePosixPath(record["path"])
    prior.require(path.parts[-3:] == ("baseline", level, "results.jsonl"), f"Unknown origin of reused {level} control")
    if source is None:
        original = inside_repo(ROOT / path).parent
        read = lambda name: (original / name).read_bytes()
    else:
        read = lambda name: source.read(prefix + name)
    for name in ("results.jsonl", "summary.json", "metadata.json"):
        prior.require(z.read(prefix + name) == read(name), prefix + name + " differs from matched control")
    prior.checksum(z.read(prefix + "results.jsonl"), record, prefix + "reused control checksum")


def audit(platform, folder=None, checkout=ROOT, control_archive=None, linux_original_truncated=None,
          levels=LEVELS, output=None):
    folder = inside_repo(folder or HERE / "qualified" / platform)
    checkout = inside_repo(checkout)
    earlier = PREVIOUS / "release" / platform
    default_original = HERE / "historical-linux/full_outputs.zip"
    if platform == "linux" and not linux_original_truncated and default_original.is_file():
        linux_original_truncated = default_original
    published, score = read_json(folder / "summary.json"), read_json(folder / "comparison.json")
    prior.require(len(set(levels)) == len(levels), "Duplicate qualification level")
    required = {f"{a}/{l}/{f}" for a in ARMS for l in levels for f in ("results.jsonl", "metadata.json", "summary.json")}
    cases, truth = prior.corpus()
    prior.verify_id(DATA / "rollout_positions/results.jsonl", score["inputs"]["truth"])
    prior.require((score["material_probability"], score["material_equity"]) == (0.0025, 0.005), "Wrong materiality")
    prior.compare_values(dict(Counter(r["reference_origin"] for r in truth.values())), score["reference_positions_by_origin"], "rollout provenance")
    prior.require(set(published) == set(score["levels"]) == set(levels), "Missing or unexpected presets")
    latest = folder / "full_outputs.zip"
    previous = earlier / "full_outputs.zip"
    prior_verification = read_json(earlier / "verification.json")
    prior.require(prior.digest(previous.read_bytes()) == prior_verification["archive_sha256"]
                  and prior_verification["baseline_commit"] == INTERMEDIATE, "Unexpected reference archive")
    provenance = folder / "truncated_controls.json"
    reused = read_json(provenance) if provenance.exists() else {}
    prior.require(set(reused) <= set(levels), "Unknown reused preset")
    if control_archive:
        prior.require(reused, "A reused-control archive requires declared controls")
    packet = dict(platform=platform, checkpoint=BASE, selected_commit=SELECTED,
                  source_verification="checked native and Python Git blobs for each build, compiled extension files and checkout",
                  rollout_truth_sha256=score["inputs"]["truth"]["sha256"], archive_sha256=prior.digest(latest.read_bytes()),
                  previous_archive_sha256=prior.digest(previous.read_bytes()), identities={}, levels={})
    packet["rollout_reference_policy"] = truth_policy(packet["rollout_truth_sha256"])
    packet["original_comparisons"] = {}
    if platform == "linux":
        packet["intermediate_commit"] = INTERMEDIATE
        packet["intermediate_comparisons"] = {}
    original_ext, historical_report = verify_original(platform)
    checked = {}
    fresh_intervals = []
    with ExitStack() as session:
        old = session.enter_context(ZipFile(previous))
        z = session.enter_context(ZipFile(latest))
        matched = session.enter_context(ZipFile(inside_repo(control_archive))) if control_archive else None
        original_zip = HERE.parent / "search_pass2" / platform / "full_outputs.zip"
        history = session.enter_context(ZipFile(original_zip))
        corrected_identity = None
        if linux_original_truncated:
            prior.require(platform == "linux", "Race-corrected historical control is Linux only")
            corrected_history = session.enter_context(ZipFile(inside_repo(linux_original_truncated)))
        else:
            corrected_history = None
        primary = None
        if folder.parent == HERE / "qualified" and folder.name.startswith(platform + "-"):
            primary_path = folder.parent / platform / "full_outputs.zip"
            primary_audit = read_json(primary_path.parent / "verification.json")
            prior.require(prior.digest(primary_path.read_bytes()) == primary_audit["archive_sha256"], "Main archive checksum")
            primary = session.enter_context(ZipFile(primary_path))
            packet["repeat_of"] = primary_audit["archive_sha256"]
        prior.require(len(z.namelist()) == len(required) and set(z.namelist()) == required, "Unexpected or duplicate zip members")
        for level in levels:
            checkpoint_dir = "candidate/" + level + "/"
            prior_meta = archived_json(old, checkpoint_dir + "metadata.json")
            prior_rows, old_id, _ = prior.read_calls(old, "candidate", level, cases)
            rows, recorded = {}, {}
            for arm in ARMS:
                key = arm + "/" + level + "/"
                meta = archived_json(z, key + "metadata.json")
                identity = check_runtime(meta, prior_meta, platform, arm, key, checked, checkout)
                prior.require(arm not in packet["identities"] or packet["identities"][arm] == identity, "Build changed between presets")
                packet["identities"][arm] = identity
                rows[arm], result_id, _ = prior.read_calls(z, arm, level, cases)
                if primary is not None:
                    first_meta = archived_json(primary, key + "metadata.json")
                    for name in ("source_files", "weights", "bearoff_database", "positions", "extension", "truncated_presets",
                                 "analyzer_threads_by_level", "environment"):
                        prior.require(meta.get(name) == first_meta.get(name), f"{key} repeat changed {name}")
                    prior.fresh_recording(primary, key, meta, result_id)
                    drift = prior.changed_analytics(primary, key, rows[arm])
                    prior.require(not drift, f"{key} repeat response drift at {drift}")
                record = [r for r in score["inputs"][arm] if r["path"].endswith(key + "results.jsonl")]
                prior.require(len(record) == 1 and all(record[0][k] == v for k, v in result_id.items()), key + " full-result checksum")
                prior.require(datetime.fromisoformat(meta["started_utc"]) >= datetime.fromisoformat(prior_meta["finished_utc"])
                              and result_id["sha256"] != old_id["sha256"], key + " expected a fresh run")
                origin = "new measurement"
                if arm == "baseline" and level in reused:
                    validate_control(z, level, reused[level], matched)
                    origin = "verified earlier matched control from this study"
                else:
                    start = datetime.fromisoformat(meta["started_utc"])
                    finish = datetime.fromisoformat(meta["finished_utc"])
                    prior.require(finish > start, key + " invalid worker time interval")
                    fresh_intervals.append((start, finish, key))
                recorded[arm] = dict(**result_id, started=meta["started_utc"], finished=meta["finished_utc"],
                                     runner_checkout=meta["git_commit"], origin=origin)
            for r, p in zip(rows["baseline"], prior_rows, strict=True):
                prior.require(r["analytics"] == p["analytics"],
                              f"Checkpoint output mismatch at {level}/{r['position_id']}/{r['api']}")
            by_api = {}
            for api in APIS:
                pairs = [(a, b, truth[api, a["position_id"]])
                         for a, b in zip(rows["baseline"], rows["candidate"]) if a["api"] == api]
                by_api[api] = score_api(api, pairs)
                prior.compare_values(prior.normalized(by_api[api]), score["levels"][level]["by_api"][api], level + "/" + api)
            compact, extra = summarize(rows, truth, by_api)
            prior.compare_values(compact, published[level], level + "/summary")
            prior.compare_values(compact["seconds"], score["levels"][level]["total_call_seconds"], level + "/seconds")
            prior.compare_values(compact["speedup"], score["levels"][level]["speedup"], level + "/speedup")
            packet["levels"][level] = dict(**compact, **extra, recorded=recorded, exact_checkpoint_controls=300,
                                           prior_published_seconds=sum(r["elapsed_seconds"] for r in prior_rows))
            if platform == "linux":
                middle, middle_id, _ = prior.read_calls(old, "baseline", level, cases)
                middle_meta = archived_json(old, "baseline/" + level + "/metadata.json")
                prior.require(middle_meta["extension"]["sha256"] == prior_verification["identities"]["baseline"]["extension_sha256"],
                              "Unexpected intermediate extension")
                for field in MATCHED_SETTINGS:
                    prior.require(middle_meta.get(field) == prior_meta.get(field), level + " intermediate setting " + field)
                middle_score = original_comparison(middle, rows["candidate"], truth)
                prior.compare_values(middle_score["seconds"]["baseline"],
                                     prior_verification["levels"][level]["seconds"]["baseline"], level + " intermediate timing")
                packet["intermediate_comparisons"][level] = dict(**middle_score, full_results=middle_id,
                                                                 started=middle_meta["started_utc"], finished=middle_meta["finished_utc"])
            historical_dir = "original/" + level + "/"
            member = historical_dir + "results.jsonl"
            if member in history.namelist():
                historical_rows, data_id, _ = prior.read_calls(history, "original", level, cases)
                old_meta = archived_json(history, historical_dir + "metadata.json")
                prior.require(old_meta["extension"]["sha256"] == original_ext, member + " original source/binary")
                record = next(r for r in historical_report["inputs"]["baseline"] if r["path"].endswith(member))
                prior.checksum(history.read(member), record, member)
                origin = "archived unmodified 139427d"
            elif corrected_history and level.endswith("T"):
                historical_rows, data_id, _ = prior.read_calls(corrected_history, "original", level, cases)
                old_meta = archived_json(corrected_history, historical_dir + "metadata.json")
                hist_tree = inside_repo(ROOT / PurePosixPath(old_meta["native_source_tree"]))
                hist_ext = inside_repo(ROOT / PurePosixPath(old_meta["extension"]["path"]))
                if corrected_identity is None:
                    corrected_identity = check_corrected_original(hist_tree, old_meta, inside_repo(linux_original_truncated))
                runtime = prior.runtime(old_meta, hist_tree, hist_ext, prior_meta, corrected_identity["source_files"])
                packet["corrected_original"] = dict(**runtime, verified_git_files=corrected_identity["verified_blobs"],
                                                    provenance_sha256=corrected_identity["provenance_sha256"],
                                                    archive_sha256=prior.digest(inside_repo(linux_original_truncated).read_bytes()))
                origin = "new measurement of 139427d with isolated parallel publication fix"
            else:
                prior.require(platform == "linux" and level.endswith("T"), "Missing historical level " + level)
                continue
            for field in (*MATCHED_SETTINGS, "positions", "bearoff_database"):
                prior.require(old_meta.get(field) == prior_meta.get(field), member + " historical inputs " + field)
            prior.require([(p["sha256"], p["bytes"]) for p in old_meta["weights"]] ==
                          [(p["sha256"], p["bytes"]) for p in prior_meta["weights"]], member + " original weight slots")
            relative = original_comparison(historical_rows, rows["candidate"], truth)
            packet["original_comparisons"][level] = dict(**relative, origin=origin, full_results=data_id,
                                                         started=old_meta["started_utc"], finished=old_meta["finished_utc"])
            if origin.startswith("archived"):
                prior.compare_values(relative["seconds"]["baseline"], historical_report["levels"][level]["total_call_seconds"]["baseline"], member + " historical timing")
    fresh_intervals.sort()
    for first, second in zip(fresh_intervals, fresh_intervals[1:]):
        prior.require(first[1] <= second[0], f"Timing runs overlap: {first[2]} and {second[2]}")
    packet["fresh_public_run_order"] = [key.strip("/") for _, _, key in fresh_intervals]
    if "repeat_of" in packet:
        first = "baseline" if folder.name.endswith("control-first") else "candidate"
        prior.require(fresh_intervals[0][2].startswith(first + "/"), "Unexpected order for timing repeat")
    write_json(inside_repo(output or folder / "verification.json"), packet)
    print(f"Verified {len(levels)*len(ARMS)*300} archived {platform} calls, including {300*len(reused)} "
          f"reused controls, {300*len(levels)} exact checkpoint control outputs and "
          f"{300*len(packet['original_comparisons'])} original-engine comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=("linux", "windows"))
    parser.add_argument("--directory", type=Path, help="Paired benchmark archive and comparison directory")
    parser.add_argument("--checkout", type=Path, default=ROOT, help="Qualified source tree corresponding to the final build")
    parser.add_argument("--control-archive", type=Path, help="Earlier paired archive supplying the declared reused controls")
    parser.add_argument("--linux-original-truncated", type=Path, help="Race-corrected original Linux truncated archive")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=LEVELS)
    parser.add_argument("--output", type=Path, help="Override the output verification.json location")
    args = parser.parse_args()
    audit(args.platform, args.directory, args.checkout, args.control_archive, args.linux_original_truncated,
          args.levels, args.output)
