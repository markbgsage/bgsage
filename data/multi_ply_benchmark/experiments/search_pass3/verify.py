# SPDX-License-Identifier: MPL-2.0
"""Verify a completed CPU qualification directly from its saved calls and frozen builds."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import APIS, DATA, TRUNCATED_LEVELS, inside_repo, summarize, value_metrics, write_json
from build_multi_ply_benchmark import validate_set
from compare_truncated_benchmark import endpoints, level_name, score_api
from compare_multi_ply_accuracy import metrics
from measure_multi_ply_pair import verify_control_sources

BASE = "82e63972379b77ab52ebafc716620e813ff7baa1"
ALL = ("1P", "2P", "3P", "4P", "1T", "2T", "3T")
ARMS = ("baseline", "candidate")
SOURCES = {
    "windows": (("build/pass3_clean", "build/multi_ply_second_final"),
                ("build/pass3_lanes_source", "build/multi_ply_third_lanes")),
    "linux": (("build/det_eval2/linux_release/guarded", "build/det_eval2/linux_release/guarded"),
              ("build/search_pass3/evaluator/lanes_final", "build/search_pass3/evaluator/lanes_final")),
}
RELEASE_CANDIDATE = {
    "windows": ("build/pass3_release_source", "build/multi_ply_third_release"),
    "linux": ("build/search_pass3/evaluator/race1linux", "build/search_pass3/evaluator/race1linux"),
}


def require(ok, where):
    if not ok:
        raise ValueError(where)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    # Round-trip tuples produced by the scoring library to on-disk JSON lists.
    return json.loads(json.dumps(value, allow_nan=False))


def compare_values(actual, published, where):
    """Report the specific mismatched field. Allow platform rounding in descriptive statistics."""
    if isinstance(actual, dict) and isinstance(published, dict):
        require(actual.keys() == published.keys(), f"{where}: missing/extra fields {actual.keys() ^ published.keys()}")
        for key in actual:
            compare_values(actual[key], published[key], f"{where}/{key}")
    elif isinstance(actual, list) and isinstance(published, list):
        require(len(actual) == len(published), f"{where}: different list length")
        for i, (a, b) in enumerate(zip(actual, published)):
            compare_values(a, b, f"{where}/{i}")
    elif isinstance(actual, float) and type(published) in (float, int):
        require(math.isclose(actual, published, rel_tol=1e-12, abs_tol=1e-15), f"{where}: {actual} != {published}")
    else:
        require(actual == published and type(actual) is type(published), f"{where}: {actual!r} != {published!r}")


def checksum(data, record, where):
    require(len(data) == record["bytes"] and digest(data) == record["sha256"], f"{where}: raw bytes/SHA-256 mismatch")


def source_paths(tree):
    result = {"CMakeLists.txt": tree / "CMakeLists.txt"}
    for folder in ("cpp", "python/bgsage"):
        for p in (tree / folder).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".cu", ".txt"):
                result[p.relative_to(tree).as_posix()] = p
    return result


def text_digest(path):
    return digest(path.read_bytes().replace(b"\r\n", b"\n"))


def pinned_text(path, relative):
    blob = subprocess.check_output(["git", "show", f"{BASE}:{relative}"], cwd=ROOT)
    require(text_digest(path) == digest(blob.replace(b"\r\n", b"\n")), f"{relative} differs from pinned Git input")


def binary(path, platform):
    if path.is_file():
        return path
    found = list(path.glob("bgbot_cpp*.so" if platform == "linux" else "bgbot_cpp*.pyd"))
    require(len(found) == 1, f"{path}: expected one extension, found {found}")
    return found[0]


@lru_cache(maxsize=None)
def check_file(path, sha, size):
    checksum(path.read_bytes(), dict(bytes=size, sha256=sha), path)


def verify_id(path, row):
    check_file(path, row["sha256"], row["bytes"])


def runtime(meta, tree, ext, prior, expected_sources):
    verify_id(ext, meta["extension"])
    raw_tree = PurePosixPath(meta["native_source_tree"])
    observed = {}
    for row in meta["source_files"]:
        p = PurePosixPath(row["path"])
        if p.as_posix() in ("scripts/benchmark_multi_ply.py", "scripts/build_multi_ply_benchmark.py"):
            name = p.as_posix()
            actual = ROOT / name
        else:
            name = p.relative_to(raw_tree).as_posix()
            actual = tree / name
        require(name not in observed, f"Duplicate source record {name}")
        verify_id(actual, row)
        observed[name] = row["sha256"]
    require(observed.keys() == expected_sources, f"{tree}: recorded source file set differs")
    require(meta["strategy_type"] == prior["strategy_type"] and meta["hidden_sizes"] == prior["hidden_sizes"],
            f"{tree}: different network layout")
    names = []
    for w, frozen in zip(meta["weights"], prior["weights"], strict=True):
        name = PurePosixPath(w["path"]).name
        require((name, w["sha256"], w["bytes"]) ==
                (PurePosixPath(frozen["path"]).name, frozen["sha256"], frozen["bytes"]), f"Stage 11 weight slot {name}")
        verify_id(inside_repo(ROOT / PurePosixPath(w["path"])), w)
        names.append((name, w["sha256"]))
    require(len(names) == 24, "Expected 24 ordered Stage 11 slots")
    verify_id(ROOT / "data/bearoff_1sided.db", meta["bearoff_database"])
    require(meta["bearoff_database"]["sha256"] == prior["bearoff_database"]["sha256"], "Published bearoff differs")
    require(meta["positions"].keys() == {"checker_play", "cube_action"}, "Different benchmark manifests")
    for api, record in meta["positions"].items():
        verify_id(DATA / f"{api}.json", record)
        require(prior["positions"][api]["sha256"] == record["sha256"], f"Published {api} manifest differs")
    return dict(source_files=len(observed), extension_sha256=meta["extension"]["sha256"],
                raw_source_manifest_sha256=digest(json.dumps(observed, sort_keys=True).encode()),
                ordered_weights_sha256=digest(json.dumps(names).encode()),
                bearoff_sha256=meta["bearoff_database"]["sha256"],
                position_sha256={a: r["sha256"] for a, r in meta["positions"].items()})


def corpus():
    inputs = {}
    for name in ("checker_play.json", "cube_action.json", "rollout_positions/results.jsonl"):
        path = DATA / name
        pinned_text(path, path.relative_to(ROOT).as_posix())
    for api in APIS:
        manifest = json.loads((DATA / ("checker_play.json" if api == "checker_play" else "cube_action.json")).read_bytes())
        validate_set(manifest)
        for p in manifest["positions"]:
            args = dict(p["post_move_inputs"] if api == "post_move_analytics" else p["inputs"])
            if api == "cube_action":
                args["incl_2ply_details"] = False
            inputs[api, p["id"]] = p["mode"], p["game_plan"], p["source"], args
    truth_path = DATA / "rollout_positions/results.jsonl"
    truth = {}
    for line in truth_path.read_bytes().splitlines():
        r = json.loads(line)
        key = r["api"], r["position_id"]
        require(key not in truth and key in inputs, f"Unexpected rollout reference {key}")
        mode, plan, source, args = inputs[key]
        actual = dict(r["inputs"])
        if key[0] == "cube_action":
            actual.setdefault("incl_2ply_details", False)
        require((mode, plan, source, args) == (r["mode"], r["source_game_plan"], r["source"], actual), f"Rollout inputs {key}")
        truth[key] = r
    require(truth.keys() == inputs.keys() and len(inputs) == 300, "Wrong reference/corpus coverage")
    return inputs, truth


def changed_analytics(prior, prefix, rows):
    earlier = [json.loads(line) for line in prior.read(prefix + "results.jsonl").splitlines()]
    require(len(earlier) == len(rows), f"Different call count at {prefix}")
    changed = []
    for a, b in zip(rows, earlier):
        fixed = ("api", "position_id", "inputs", "source", "mode", "source_game_plan", "level", "repeat")
        require(all(a[k] == b[k] for k in fixed) and a.get("seed") == b.get("seed"), f"Different inputs at {prefix}")
        if a["analytics"] != b["analytics"]:
            changed.append([a["api"], a["position_id"]])
    return changed


def read_calls(z, arm, level, cases, past=None):
    prefix = f"{arm}/{level}/"
    raw = z.read(prefix + "results.jsonl")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = list(cases)
    require([(r["api"], r["position_id"]) for r in rows] == expected, f"{prefix}: position set/order/duplicate")
    compact = []
    for key, r in zip(expected, rows):
        require((r["mode"], r["source_game_plan"], r["source"], r["inputs"]) == cases[key],
                f"{prefix}{key}: position or public API arguments")
        source = r["source"]
        require(r["repeat"] == 1 and level_name(r["level"]) == level, f"{prefix}{key}: preset/repeat")
        require(math.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > 0, f"{prefix}{key}: bad timing")
        if level in TRUNCATED_LEVELS:
            seed = int.from_bytes(hashlib.sha256(("truncated-benchmark-v1:" + source["key"]).encode()).digest()[:4], "big")
            require(type(r.get("seed")) is int and r["seed"] == seed, f"{prefix}{key}: wrong rollout seed")
        else:
            require("seed" not in r, f"{prefix}{key}: unexpected seed")
        v = dict(r, value_metrics=value_metrics(key[0], r["analytics"]))
        if key[0] == "cube_action":
            v["optimal_action"] = r["analytics"]["optimal_action"]
        compact.append(v)
    drift = changed_analytics(past, prefix, rows) if past is not None else []
    actual = summarize(compact)
    actual["status"] = "complete"
    compare_values(normalized(actual), json.loads(z.read(prefix+"summary.json")), prefix+"summary")
    return rows, dict(sha256=digest(raw), bytes=len(raw)), drift


def fresh_recording(prior, prefix, meta, identity, *, after=True):
    earlier = json.loads(prior.read(prefix + "metadata.json"))
    start = datetime.fromisoformat(meta["started_utc"])
    if after:
        require(start >= datetime.fromisoformat(earlier["finished_utc"]),
                prefix + " recording did not start after its previous comparison")
    else:
        require(start != datetime.fromisoformat(earlier["started_utc"]), prefix + " reused the primary start time")
    require(identity["sha256"] != digest(prior.read(prefix + "results.jsonl")),
            prefix + " repeated the previous timing file")


def release_recording(z, earlier, declared, arm, level, meta, identity):
    prefix = f"{arm}/{level}/"
    if arm == "baseline" and level in TRUNCATED_LEVELS:
        for name in ("metadata.json", "results.jsonl", "summary.json"):
            require(z.read(prefix + name) == earlier.read(prefix + name), prefix + name + " differs from the reused control")
        require(all(declared[level][k] == v for k, v in identity.items()), prefix + " reused result fingerprint")
        return "same-pass truncated control, exact archived bytes"
    fresh_recording(earlier, prefix, meta, identity)
    return "new measurement"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=("linux", "windows", "linux-repeat"))
    p.add_argument("--release", action="store_true", help="Verify the final release against its source and prior control recordings")
    p.add_argument("--levels", nargs="+", type=level_name)
    p.add_argument("--directory", type=Path)
    p.add_argument("--checkout", type=Path, help="Expected source checkout (release defaults to current, earlier suite to its frozen source)")
    p.add_argument("--output", type=Path)
    for arm in ARMS:
        p.add_argument("--" + arm + "-source", type=Path)
        p.add_argument("--" + arm + "-build", type=Path)
    a = p.parse_args()
    study = HERE / "release" if a.release else HERE
    directory = inside_repo(a.directory or study / a.suite)
    platform = "windows" if a.suite == "windows" else "linux"
    archive = directory / "full_outputs.zip"
    report = json.loads((directory / "comparison.json").read_bytes())
    published = json.loads((directory / "summary.json").read_bytes())
    levels = list(a.levels or (ALL[:4] if a.suite == "linux-repeat" else ALL))
    require(len(set(levels)) == len(levels) and set(levels) == published.keys() == report["levels"].keys(),
            "The requested presets differ from the completed comparison")
    require(all(len(report["inputs"][arm]) == len(levels) for arm in ARMS), "Unexpected number of archived result identities")
    cases, truth = corpus()
    truth_record = report["inputs"]["truth"]
    verify_id(DATA / "rollout_positions/results.jsonl", truth_record)
    require((report["material_probability"], report["material_equity"]) == (.0025, .005), "Wrong materiality thresholds")
    compare_values(dict(Counter(r.get("reference_origin", "unspecified") for r in truth.values())),
                   report["reference_positions_by_origin"], "reference origins")
    checked = {}
    configs = {}
    baseline, original_candidate = SOURCES[platform]
    candidate = RELEASE_CANDIDATE[platform] if a.release else original_candidate
    checkout = inside_repo(a.checkout or (ROOT if a.release else ROOT / original_candidate[0]))
    for arm, defaults in zip(ARMS, (baseline, candidate)):
        tree = inside_repo(getattr(a, arm + "_source") or ROOT / defaults[0])
        ext = binary(inside_repo(getattr(a, arm + "_build") or ROOT / defaults[1]), platform)
        files = source_paths(tree)
        if arm == "baseline":
            verify_control_sources(BASE, tree, tree / "python")
            pinned_text(tree / "CMakeLists.txt", "CMakeLists.txt")
        else:
            shipped = source_paths(checkout)
            require(files.keys() == shipped.keys(), "Frozen candidate and intended checkout contain different source file sets")
            for name in files:
                require(text_digest(files[name]) == text_digest(shipped[name]), f"Frozen candidate differs from intended checkout: {name}")
        configs[arm] = (tree, ext, files.keys() | {"scripts/benchmark_multi_ply.py", "scripts/build_multi_ply_benchmark.py"})
    results = {}
    with ExitStack() as archives:
        repeat = archives.enter_context(ZipFile(study / "linux/full_outputs.zip")) if a.suite == "linux-repeat" else None
        if a.release:
            original = HERE / a.suite
            earlier = archives.enter_context(ZipFile(original / "full_outputs.zip"))
            reused = set(levels) & TRUNCATED_LEVELS.keys()
            provenance = directory / "truncated_controls.json"
            declared = json.loads(provenance.read_bytes()) if provenance.is_file() else {}
            require(declared.keys() == reused, "Incorrect set of declared reused controls")
            previous_inputs = json.loads((original / "comparison.json").read_bytes())["inputs"]["baseline"]
            for level in reused:
                require(declared[level] in previous_inputs and
                        PurePosixPath(declared[level]["path"]).parts[-3:] == ("baseline", level, "results.jsonl"),
                        f"{level}: declared control does not identify the corresponding original recording")
        with ZipFile(archive) as z, ZipFile(HERE.parent / "search_pass2" / platform / "full_outputs.zip") as older:
            previous = json.loads(older.read("candidate/3P/metadata.json"))
            files = {f"{arm}/{level}/{name}" for arm in ARMS for level in levels
                     for name in ("metadata.json", "results.jsonl", "summary.json")}
            require(set(z.namelist()) == files and len(z.namelist()) == len(files), "Unexpected, missing or duplicate archive members")
            for level in levels:
                historical = f"candidate/{level}/metadata.json"
                if historical in older.namelist():
                    preset = json.loads(older.read(historical))
                else:
                    require(platform == "linux" and level in TRUNCATED_LEVELS, f"Missing published {level} settings")
                    with ZipFile(HERE.parent / "search_pass2/linux-truncated/full_outputs.zip") as tr:
                        preset = json.loads(tr.read(historical))
                sides = {}
                repeat_drift = {}
                recording_origin = {}
                metas = []
                for arm in ARMS:
                    tree, ext, expected_sources = configs[arm]
                    prefix = f"{arm}/{level}/"
                    meta = json.loads(z.read(prefix+"metadata.json"))
                    metas.append(meta)
                    require((meta["status"], meta["completed_calls"], meta["expected_calls"], meta["repeats"],
                             meta["threads_requested"], meta["model"], meta["cubeful"], meta["cube_details"],
                             meta["apis"], list(map(level_name, meta["levels"]))) ==
                            ("complete", 300, 300, 1, 16, "stage11", True, False, list(APIS), [level]), prefix+"metadata")
                    require(platform in meta["platform"].lower(), prefix+"operating system")
                    require(meta.get("truncated_presets") == preset.get("truncated_presets"),
                            prefix+" trial settings differ from the pinned preset")
                    require(meta["analyzer_threads_by_level"] == preset["analyzer_threads_by_level"],
                            prefix+" analyzer thread settings differ from the paired qualification")
                    actual = runtime(meta, tree, ext, previous, expected_sources)
                    if arm == "baseline":
                        require(actual["extension_sha256"] == previous["extension"]["sha256"], "Baseline differs from published binary")
                    if repeat:
                        origin_meta = json.loads(repeat.read(prefix+"metadata.json"))
                        for field in ("source_files", "weights", "extension", "bearoff_database", "positions", "environment", "analyzer_threads_by_level", "truncated_presets"):
                            require(meta.get(field) == origin_meta.get(field), prefix+" primary/repeat "+field)
                    require(arm not in checked or checked[arm] == actual, prefix+"identities vary between presets")
                    checked[arm] = actual
                    sides[arm], identity, repeat_drift[arm] = read_calls(z, arm, level, cases, repeat)
                    if a.release:
                        recording_origin[arm] = release_recording(z, earlier, declared, arm, level, meta, identity)
                        if repeat:
                            fresh_recording(repeat, prefix, meta, identity, after=False)
                    inputs = report["inputs"][arm]
                    match = [r for r in inputs if PurePosixPath(r["path"]).parts[-3:] == (arm, level, "results.jsonl")]
                    require(len(match) == 1 and all(match[0][k] == v for k, v in identity.items()), prefix+"comparison points to different results")
                for field in ("environment", "analyzer_threads_by_level", "truncated_presets", "ordering", "cache_policy", "timing_scope"):
                    require(metas[0].get(field) == metas[1].get(field), f"{level}: different paired {field}")
                by_api = {}
                maxima = {"probability": 0., "equity": 0.}
                changes = {"checker_play": [], "cube_action": []}
                for api in APIS:
                    pairs = [(base, cand, truth[api, base["position_id"]]) for base, cand in zip(sides["baseline"], sides["candidate"]) if base["api"] == api]
                    stats = score_api(api, pairs)
                    compare_values(normalized(stats), report["levels"][level]["by_api"][api], f"{level}/{api}/comparison")
                    by_api[api] = stats
                    for left, right, ref in pairs:
                        l, r = (endpoints(api, row["analytics"]) for row in (left, right))
                        if api == "checker_play" and next(iter(l)) != next(iter(r)):
                            changes[api].append(left["position_id"])
                        if api == "cube_action" and l[None]["optimal_action"] != r[None]["optimal_action"]:
                            changes[api].append(left["position_id"])
                        for board, item in l.items():
                            b = dict(metrics(r[board]))
                            for f, x in metrics(item):
                                k = "probability" if f.startswith("p.") else "equity"
                                maxima[k] = max(maxima[k], abs(x - b[f]))
                timing = {arm: sum(s["timing"][arm]["total_seconds"] for s in by_api.values()) for arm in ARMS}
                delta = [r["change_in_absolute_error"] for s in by_api.values() for r in s["material_changes_in_rollout_error"]]
                compact = dict(seconds=timing, speedup=timing["baseline"]/timing["candidate"],
                               exact=sum(s["exact_analytics"] for s in by_api.values()),
                               same_best_checker=100-len(changes["checker_play"]), same_cube_action=100-len(changes["cube_action"]),
                               same_checker_order=by_api["checker_play"]["same_full_move_order"],
                               observations=sum(f["n"] for s in by_api.values() for f in s["rollout_error_by_field"].values()),
                               material_farther=sum(v>0 for v in delta), material_closer=sum(v<0 for v in delta),
                               api_seconds={api: [s["timing"][arm]["total_seconds"] for arm in ARMS] for api,s in by_api.items()})
                compare_values(compact, published[level], level+" published summary")
                compare_values(timing, report["levels"][level]["total_call_seconds"], level+" comparison totals")
                compare_values(compact["speedup"], report["levels"][level]["speedup"], level+" comparison speedup")
                results[level] = dict(**compact, changed_decision_ids=changes, max_prediction_shifts=maxima)
                if a.release:
                    results[level]["recording_origin"] = recording_origin
                    rows = sides["candidate"]
                    changed = changed_analytics(earlier, f"candidate/{level}/", rows)
                    results[level]["candidate_vs_wide_race"] = {
                        "calls": len(rows), "exact_analytics": len(rows) - len(changed),
                        "changed_calls": changed,
                    }
                if repeat:
                    results[level]["changed_full_outputs_from_primary"] = repeat_drift
    output = inside_repo(a.output or directory / "verification.json")
    suite_name = f"release/{a.suite}" if a.release else a.suite
    write_json(output, dict(suite=suite_name, archive_sha256=digest(archive.read_bytes()), baseline_commit=BASE,
               rollout_truth_sha256=truth_record["sha256"], identities=checked, levels=results))
    print(f"Verified {len(levels)*600} archived calls in {suite_name}: {output}")


if __name__ == "__main__":
    main()
