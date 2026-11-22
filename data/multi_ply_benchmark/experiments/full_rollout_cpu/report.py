# SPDX-License-Identifier: MPL-2.0
"""Verify full-rollout pairs and package their complete responses, timings and phase profiles."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
from run import full_roster
sys.path.insert(0, str(HERE.parent / "full_rollout_1296"))
import measure
from verify import check_pair
from compare_truncated_benchmark import score_api
from benchmark_multi_ply import DATA, inside_repo, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def totals(pairs):
    original = math.fsum(p["original_seconds"] for p in pairs)
    selected = math.fsum(p["selected_seconds"] for p in pairs)
    return dict(calls=len(pairs), baseline_seconds=original, candidate_seconds=selected, speedup=original/selected,
                time_saved_percent=(1-selected/original)*100, median_paired_speedup=median(p["speedup"] for p in pairs),
                faster_pairs=sum(p["speedup"] > 1 for p in pairs))


def analyze(files):
    cfg = json.loads(files["settings.json"])
    rows = [json.loads(line) for line in files["calls.jsonl"].splitlines()]
    arms, jobs = cfg["arm_order"], cfg["roster"]
    require(len(rows) == len(jobs)*len(arms)*cfg["repeats"], "Incomplete panel")
    full = {(r["api"], r["position_id"]): r for r in full_roster(cfg.get("case_number", 1))}
    ref_path = DATA / "rollout_positions/results.jsonl"
    truth = {(r["api"], r["position_id"]): r for r in map(json.loads, ref_path.read_bytes().splitlines())}
    with ZipFile(HERE.parent / "full_rollout_1296/full_outputs.zip") as z:
        historical_rows = list(map(json.loads, z.read("calls.jsonl").splitlines()))
        checkpoint = {(r["api"], r["position_id"]): r for r in historical_rows
                      if r["arm"] == "selected"}
    checkpoint_cfg = measure.read(measure.KNOWN)["arms"]["selected"]
    for key, info in cfg["arms"].items():
        require(info["affinity"] == list(range(16)) and info["trial_policy_ply"] == 3, "Affinity or strength differs")
        require(info["constructor"] == dict(measure.OPTIONS, n_trials=cfg["trials"]), "Full-game configuration differs")
        for f1, f2 in (("analyzer","analyzer"),("python","python"),("database","database")):
            require(info[f1]["sha256"] == checkpoint_cfg[f2]["sha256"], f"Changed benchmark {f1}")
        require([x["sha256"] for x in info["weights"]] == [x["sha256"] for x in checkpoint_cfg["weight_slots"]],
                "Weights differ from the checkpoint")
    if len(arms) > 1:
        require(cfg["arms"][arms[0]]["binary"]["sha256"] == checkpoint_cfg["extension"]["sha256"],
                "First arm must be the actual qualified baseline")
    stream = iter(rows)
    pairs = defaultdict(list)
    scores = defaultdict(lambda: defaultdict(list))
    profile = []
    checkpoint_checks = []
    short_counts = defaultdict(int)
    for repeat in range(cfg["repeats"]):
        for index, job in enumerate(jobs):
            k = (job["api"], job["position_id"])
            require(job == full[k], f"Unexpected corpus input or seed: {k}")
            pos = index + repeat
            if cfg.get("reverse_full_panel_order", False):
                require(len(arms) >= 2, "Reverse-order repeat must compare at least two arms")
                previous = cfg.get("previous_panel_order", arms)
                offset = job["sequence"] % len(previous)
                prior = previous[offset:] + previous[:offset]
                order = [name for name in reversed(prior) if name in arms]
                pos = repeat
            else:
                order = arms
            order = order[pos%len(order):] + order[:pos%len(order)]
            run = {}
            for arm in order:
                r = next(stream)
                require(r["arm"] == arm and r["repeat"] == repeat+1 and all(r[f] == v for f,v in job.items()),
                        f"Different paired input/arm at {k}")
                require(r["elapsed_seconds"] > 0 and math.isfinite(r["elapsed_seconds"]), "Invalid timing")
                r["source"] = truth[k]["source"]
                run[arm] = r
                if arm == arms[0] and cfg["trials"] == 1296 and k in checkpoint:
                    archived = checkpoint[k]
                    comparison = check_pair([dict(archived), dict(r)], job, truth[k], truth[k]["source"])
                    checkpoint_checks.append(dict(api=k[0], position_id=k[1],
                        exact=r["analytics"] == archived["analytics"],
                        same_decision=comparison.get("same_best_move", comparison.get("same_cube_action", True)),
                        direct=comparison["direct_shifts"]))
                if cfg["profile"]:
                    profile.append({f:r[f] for f in ("api", "position_id","elapsed_seconds","cpu_seconds","phase_visits","phase_worker_seconds")})
            for arm in arms[1:]:
                a, b, f = run[arms[0]], run[arm], truth[k]
                d = check_pair([dict(a),dict(b)], job, f, f["source"])
                if "same_rollout_shortlist_including_promotions" in d:
                    short_counts[arm] += d["same_rollout_shortlist_including_promotions"]
                pairs[arm].append(d)
                scores[arm][job["api"]].append((a,b,f))
    require(all(p["same_decision"] and not any(v["material"] for v in p["direct"].values())
                for p in checkpoint_checks), "Full-budget baseline materially differs from the published control")
    summary = dict(arm_order=arms, trials=cfg["trials"], repeats=cfg["repeats"], cases=len(jobs),
                   full_budget_baseline_outputs_matching_checkpoint=sum(p["exact"] for p in checkpoint_checks),
                   full_budget_baseline_checkpoint_comparison=checkpoint_checks, comparisons={})
    accuracy = {}
    for arm, group in pairs.items():
        accuracy[arm] = {api: score_api(api, records) for api,records in scores[arm].items()}
        scored = [field for a in accuracy[arm].values() for field in a["rollout_error_by_field"].values()]
        compare = dict(overall=totals(group), by_api={}, by_plan={}, pairs=group,
                       exact_complete=sum(a["exact_analytics"] for a in accuracy[arm].values()),
                       checker_order=sum(d.get("same_full_order",False) for d in group),
                       checker_shortlists=short_counts[arm], cube_agree=sum(d.get("same_cube_action",False) for d in group),
                       scored=sum(f["n"] for f in scored), material_farther=sum(f["materially_farther"] for f in scored),
                       material_closer=sum(f["materially_closer"] for f in scored),
                       fixed_batches={a:sum(d["fixed_batches"][a] for d in group) for a in ("original","selected")},
                       direct_max={m:max(d["direct_shifts"][m]["largest"] for d in group) for m in ("equity","probability")})
        for field, out in (("api","by_api"),("game_plan","by_plan")):
            compare[out] = {v:totals([d for d in group if d[field]==v]) for v in sorted({d[field] for d in group})}
        if (cfg["trials"] == 1296 and cfg.get("case_number", 1) == 1
                and len(jobs) == 15 and cfg["repeats"] == 1):
            cumulative = {}
            for api in ["all", *measure.APIS]:
                old = [r for r in historical_rows if r["api"] == api or api == "all"]
                initial, last = [math.fsum(r["elapsed_seconds"] for r in old if r["arm"] == label)
                                 for label in ("original", "selected")]
                fresh = compare["overall"] if api == "all" else compare["by_api"][api]
                cumulative[api] = dict(historical_initial_seconds=initial, prior_published_seconds=last,
                                       fresh_baseline_seconds=fresh["baseline_seconds"],
                                       candidate_seconds=fresh["candidate_seconds"],
                                       initial_over_candidate=initial/fresh["candidate_seconds"])
            compare["cumulative_across_sessions"] = cumulative
        summary["comparisons"][arm] = compare
    if profile:
        total = {phase:math.fsum(r["phase_worker_seconds"][phase] for r in profile)
                 for phase in profile[0]["phase_worker_seconds"]}
        total_s = math.fsum(total.values())
        summary["profile"] = dict(phases_worker_seconds=total, shares={k:v/total_s for k,v in total.items()}, cases=profile)
    return summary, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    ns = parser.parse_args()
    out = inside_repo(ns.output)
    zip_path = out / "full_outputs.zip"
    files = ("settings.json","calls.jsonl")
    if ns.run:
        raw = {n:(inside_repo(ns.run)/n).read_bytes() for n in files}
        s, a = analyze(raw)
        out.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path.with_suffix(".tmp"), "w", ZIP_DEFLATED, compresslevel=9) as z:
            for n, data in raw.items():
                z.writestr(n,data)
            z.writestr("accuracy.json", json.dumps(a, indent=2, allow_nan=False)+"\n")
        zip_path.with_suffix(".tmp").replace(zip_path)
        write_json(out/"summary.json", s)
    else:
        with ZipFile(zip_path) as z:
            require(z.testzip() is None, "Damaged archive")
            s, a = analyze({n:z.read(n) for n in files})
            require(json.loads(z.read("accuracy.json")) == json.loads(json.dumps(a)), "Accuracy report differs")
        require(json.loads(json.dumps(s)) == json.loads((out/"summary.json").read_bytes()), "Timing report differs")
    for arm, value in s["comparisons"].items():
        print(arm, value["overall"], "exact",value["exact_complete"],"farther",value["material_farther"])
    if "profile" in s:
        print("Profile", s["profile"]["shares"])


if __name__ == "__main__":
    main()
