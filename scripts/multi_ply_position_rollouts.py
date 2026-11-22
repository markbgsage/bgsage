# SPDX-License-Identifier: MPL-2.0
"""Build 300 position-level references using the source rollouts and frozen public analytics."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time

import multi_ply_rollout_truth as pinned

OUT = pinned.DATA / "rollout_positions"
WORK = pinned.ROOT / "logs/multi_ply_position_rollouts"
CASES = WORK / "checker"
MISSES = WORK / "native"
PROBS = ("win", "gammon_win", "backgammon_win", "gammon_loss", "backgammon_loss")
ENGINE = None


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def fixed_sets():
    frozen = pinned.FROZEN / pinned.DATA.relative_to(pinned.ROOT)
    sets = [pinned.read(frozen / (name + ".json"))["positions"] for name in ("checker_play", "cube_action")]
    return tuple({p["id"]: p for p in group} for group in sets)


def setup(threads):
    global ENGINE
    analyzer, native, owner, _ = pinned.make_analyzer(threads, public_filter=True)
    inner = analyzer._analyzer._inner
    if (inner._prefilter_threshold, inner.FILTER_MAX_MOVES, inner.FILTER_THRESHOLD) != (0.15, 5, 0.08):
        raise ValueError("Frozen normal full-rollout shortlist changed")
    frozen = pinned.FROZEN / pinned.BASE.relative_to(pinned.ROOT) / "results.jsonl"
    predicts = {r["position_id"]: {tuple(m["board"]): m for m in r["analytics"]["moves"]}
                for r in rows(frozen) if r["level"] == "3ply" and r["api"] == "checker_play"}
    units = defaultdict(dict)
    for name, api, p, board in pinned.jobs():
        if api == "checker_play":
            units[p["id"]][tuple(board)] = name
    ENGINE = analyzer, native, owner, units, predicts


class Replay:
    """Intercept only the frozen cubeful per-move trial call; filtering and ranking stay public."""
    def __init__(self, p, native, owner, units, predicts, dry):
        self.p = p
        self.native = native
        self.units = units
        self.predicts = predicts
        self.dry = dry
        self.seed = pinned.seed_for(p)
        self.manifest = pinned.digest((pinned.OUT / "manifest.json").read_bytes())
        inputs = p["inputs"]
        self.context = {k: inputs[k] for k in ("cube_value", "away1", "away2", "is_crawford",
                                                "jacoby", "beaver", "max_cube_value")}
        self.context["owner"] = owner(inputs["cube_owner"])
        self.events = []
        self.native_se = []

    def __getattr__(self, key):
        return getattr(self.native, key)

    def cubeful_evaluate_board(self, board, pre_move_board, **kwargs):
        b = tuple(board)
        name = self.units[b]
        context = dict(kwargs)
        context.pop("progress", None)
        context.setdefault("max_cube_value", 0)
        if context != self.context:
            raise ValueError(f"Cube/rule context changed for {name}")
        # The pinned pybind implementation names its second board argument
        # /*pre_move_board*/ and passes only the endpoint and CubeInfo to
        # cubeful_rollout_position. The caller supplies the original board
        # for initial survivors and the endpoint itself for promotions.
        if list(pre_move_board) == self.p["inputs"]["board"]:
            phase = "shortlist"
        elif list(pre_move_board) == list(board):
            phase = "promotion"
        else:
            raise ValueError(f"Unexpected checker call parent for {name}")
        raw = None
        source = None
        for path, origin in ((pinned.WORK / "units" / (name + ".json"), "supplemental_checkpoint"),
                             (MISSES / (name + ".json"), "position_checkpoint")):
            if not path.is_file():
                continue
            record = pinned.read(path)
            if (record["manifest_sha256"], record["position_id"], record["seed"], record["unit"], record["api"],
                    record["analytics"]["board"]) != (self.manifest, self.p["id"], self.seed, name,
                                                       "checker_play", list(board)):
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if origin == "position_checkpoint" and record["inputs"] != self.p["inputs"]:
                raise ValueError(f"Checkpoint inputs changed: {path}")
            raw = record["native_checker"]
            source = origin
            break
        if raw is None and self.dry:
            # Only to plan tentative promotions. A call that uses any surrogate
            # cannot be published as an actual reference row.
            m = self.predicts[b]
            raw = dict(cubeful_equity=m["equity"], equity=m["cubeless_equity"], std_error=0.,
                       cubeful_se=0., prob_std_errors=[0.]*5, probs=[m["probs"][k] for k in PROBS])
            source = "projected_miss"
        elif raw is None:
            start = time.perf_counter()
            raw = self.native.cubeful_evaluate_board(board, pre_move_board, **kwargs)
            pinned.save(MISSES / (name + ".json"), dict(unit=name, api="checker_play", position_id=self.p["id"],
                inputs=self.p["inputs"], seed=self.seed, manifest_sha256=self.manifest, analytics={"board":list(board)},
                seconds=time.perf_counter()-start, native_checker=raw))
            source = "new_rollout"
        self.events.append(dict(unit=name, source=source, phase=phase))
        if source != "projected_miss":
            self.native_se.append(dict(position_id=self.p["id"], board=list(board), cubeful_se=raw["cubeful_se"]))
        return raw


def checker(p, dry):
    analyzer, native, owner, units, predicts = ENGINE
    inner = analyzer._analyzer._inner
    replay = Replay(p, native, owner, units[p["id"]], predicts[p["id"]], dry)
    analyzer.set_seed(replay.seed)
    inner._rollout_strategy = replay
    try:
        actual = asdict(analyzer.checker_play(**p["inputs"]))
    finally:
        inner._rollout_strategy = native
    missing = sum(e["source"] == "projected_miss" for e in replay.events)
    if not missing:
        row = dict(api="checker_play", position_id=p["id"], mode=p["mode"], source_game_plan=p["game_plan"],
                   source=p["source"], inputs=p["inputs"], seed=replay.seed, analytics=actual,
                   reference_origin="frozen_public_rollout", provenance=dict(
                       frozen_manifest_sha256=replay.manifest, trial_batch_size=1296, trial_batches=1,
                       variance_reduction=True, checker_ply=3, cube_ply=3, truncation_depth=0,
                       normal_root_filter=True, move_evaluation_levels=dict(Counter(m["eval_level"] for m in actual["moves"]))))
        pinned.save(CASES / (p["id"] + ".json"), dict(row=row, cubeful_se=replay.native_se, events=replay.events))
    return dict(position_id=p["id"], projected_misses=missing, events=replay.events)


def prepare():
    for path in (OUT, WORK, CASES, MISSES):
        path.mkdir(parents=True, exist_ok=True)


def dry_plan():
    prepare()
    setup(1)
    positions, _ = fixed_sets()
    reports = [checker(p, True) for p in positions.values() if p["source"]["tier"] != "rollout"]
    events = [e for r in reports for e in r["events"]]
    result = dict(positions=len(reports), fully_replayed=sum(r["projected_misses"] == 0 for r in reports),
                  calls_by_source=dict(Counter(e["source"] for e in events)),
                  projected_misses_by_phase=dict(Counter(e["phase"] for e in events if e["source"]=="projected_miss")),
                  per_position=reports)
    pinned.save(WORK / "dry_plan.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "per_position"}), flush=True)


def archive_and_cube_rows():
    from multi_ply_rollout_sources import load_archived
    archived, supporting = load_archived()
    _, cubes = fixed_sets()
    stacked = []
    manifest = pinned.digest((pinned.OUT / "manifest.json").read_bytes())
    for r in rows(pinned.OUT / "results.jsonl"):
        if r["api"] not in ("cube_action","post_move_analytics"):
            continue
        p = cubes[r["position_id"]]
        expected = p["post_move_inputs"] if r["api"]=="post_move_analytics" else p["inputs"]
        if (r["inputs"] != expected or r["source"] != p["source"] or r["mode"] != p["mode"] or
                r["source_game_plan"] != p["game_plan"] or r["seed"] != pinned.seed_for(p) or
                r["analytics"]["eval_level"] != "Rollout"):
            raise ValueError("Unexpected completed cube or post-move context")
        stacked.append(dict(r,reference_origin="frozen_public_rollout",provenance=dict(
            frozen_manifest_sha256=manifest, trial_batch_size=1296, trial_batches=1, variance_reduction=True,
            checker_ply=3, cube_ply=3, truncation_depth=0, archived_cube_support=r["position_id"] in supporting)))
    if len(archived) != 37 or len(supporting) != 10 or len({(r["api"],r["position_id"]) for r in stacked}) != 200:
        raise ValueError("Incomplete source or frozen cube/post collection")
    return archived, supporting, stacked


def assemble():
    prepare()
    with pinned.unit_lease(WORK / "publish.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another position-level publication is active")
        return publish()


def publish():
    archived, supporting, cubes = archive_and_cube_rows()
    positions, _ = fixed_sets()
    result, errors, coverage = [], [], []
    calls = Counter()
    for pid, p in positions.items():
        path = CASES / (pid + ".json")
        if pid in archived:
            row = archived[pid]
        elif path.is_file():
            data = pinned.read(path)
            row = data["row"]
            if row["inputs"] != p["inputs"] or row["seed"] != pinned.seed_for(p):
                raise ValueError(f"Unexpected checker result: {pid}")
            errors.extend(data["cubeful_se"])
            calls.update(e["source"] for e in data["events"])
        else:
            coverage.append(dict(api="checker_play",position_id=pid,complete=False))
            continue
        result.append(row)
        coverage.append(dict(api="checker_play",position_id=pid,complete=True,reference_origin=row["reference_origin"],
                             move_eval_levels=dict(Counter(m["eval_level"] for m in row["analytics"]["moves"]))))
    result.extend(cubes)
    coverage.extend(dict(api=r["api"],position_id=r["position_id"],complete=True,reference_origin=r["reference_origin"])
                    for r in cubes)
    pinned.save_rows(OUT / "results.jsonl", result)
    pinned.save_rows(OUT / "checker_cubeful_se.jsonl", errors)
    pinned.save_rows(OUT / "archived_cube_support.jsonl", list(supporting.values()))
    pinned.save_rows(OUT / "position_coverage.jsonl", coverage)
    report = dict(complete_cases=len(result), total_cases=300,
                  cases_by_api=dict(Counter(r["api"] for r in result)),
                  cases_by_origin=dict(Counter(r["reference_origin"] for r in result)),
                  available_checker_native_cubeful_ses=len(errors), supplemental_cube_rows=len(supporting),
                  fresh_checker_calls_by_source=dict(calls))
    frozen = pinned.read(pinned.OUT/"manifest.json")
    source_rows = pinned.DATA/"historical_rollouts/rows.jsonl.gz"
    source_sets = pinned.read(pinned.DATA/"checker_play.json")["sources"]
    manifest = dict(schema_version=1, fixed_sets={
        name:pinned.digest((pinned.DATA/name).read_bytes()) for name in ("checker_play.json","cube_action.json")},
        frozen_engine=dict(revision=frozen["git_commit"], model=frozen["model"], native=frozen["assets"][0],
                           supplemental_manifest_sha256=pinned.digest((pinned.OUT/"manifest.json").read_bytes())),
        historical_archives={name:dict(path=meta["path"],sha256=meta["sha256"]) for name,meta in source_sets.items()},
        historical_rollout_extract_sha256=pinned.digest(source_rows.read_bytes()),
        policy=dict(archive_rollout_checker_positions=37, fresh_public_checker_positions=63,
                    reuse_public_cube_positions=100, reuse_public_post_move_positions=100,
                    historical_cube_support_positions=10,
                    source_policy=dict(paths_per_batch=1296, checker_ply=3, cube_ply=3, variance_reduction=True,
                                       target_se=0.00255,max_batches=16,actual_batches_recorded_per_position=False),
                    fixed_public_trial_config={k:v for k,v in frozen["config"].items() if k!="prefilter_threshold"},
                    checker_and_cube_phase_ply=frozen["trial_configs"], variance_reduction=True,
                    checker_root_filter=dict(one_ply_equity_gap=0.15,two_ply_equity_gap=0.08,
                        max_initial_survivors=5,min_candidates_if_available=2,promote_shallow_leader=True),
                    rng_seed=frozen["seed"]),
        outputs={name:pinned.digest((OUT/name).read_bytes()) for name in
                 ("results.jsonl","checker_cubeful_se.jsonl","position_coverage.jsonl","archived_cube_support.jsonl")})
    pinned.save(OUT/"manifest.json",manifest)
    pinned.save(OUT / "coverage.json", report)
    return report


def run(processes, threads):
    prepare()
    positions, _ = fixed_sets()
    todo = [p for p in positions.values() if p["source"]["tier"] != "rollout" and not (CASES/(p["id"]+".json")).exists()]
    plan_path = WORK / "dry_plan.json"
    if plan_path.exists():
        missing = {r["position_id"]:r["projected_misses"] for r in pinned.read(plan_path)["per_position"]}
        todo.sort(key=lambda p:-missing[p["id"]])
    print(f"{len(todo)} remaining public checker positions, {processes} workers x {threads} trial threads",flush=True)
    with ProcessPoolExecutor(max_workers=processes, initializer=setup, initargs=(threads,)) as pool:
        futures = {pool.submit(checker, p, False):p["id"] for p in todo}
        for future in as_completed(futures):
            trace = future.result()
            print(f"{trace['position_id']} {dict(Counter(e['source'] for e in trace['events']))}",flush=True)
    print(json.dumps(assemble()),flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("dry","run","assemble"))
    ap.add_argument("--processes",type=int,default=4)
    ap.add_argument("--threads",type=int,default=4)
    args=ap.parse_args()
    if min(args.processes,args.threads)<1:
        ap.error("Use positive process and thread counts")
    if args.action=="assemble":
        print(json.dumps(assemble()),flush=True)
    else:
        prepare()
        with pinned.unit_lease(WORK / "runner.lock") as acquired:
            if not acquired:
                raise RuntimeError("Another position-level rollout or dry replay is active")
            if args.action=="dry":
                dry_plan()
            else:
                run(args.processes,args.threads)
