# SPDX-License-Identifier: MPL-2.0
"""Compare two qualified Linux engines on public 1,296-path full rollouts."""
import argparse
from dataclasses import asdict
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_multi_ply import DATA, file_identity, inside_repo, load_engine, write_json
from multi_ply_rollout_truth import CONFIG, read, seed_for, set_cpu_mask

ARMS = ("original", "selected")
PLANS = ("attacking", "priming", "anchoring", "racing", "purerace")
APIS = ("checker_play", "cube_action", "post_move_analytics")
THREADS = 16
AFFINITY = (1 << THREADS) - 1
OPTIONS = {key: value for key, value in CONFIG.items() if key != "prefilter_threshold"}
KNOWN = HERE.parent / "truncated_pass2/app-style/settings.json"
HISTORY = HERE.parent / "truncated_pass2/historical-linux/provenance.json"
SENSITIVITY = {("checker_play", "checker-match-attacking-01"),
               ("cube_action", "cube-money-anchoring-01"),
               ("post_move_analytics", "cube-match-racing-01")}


def worker(pipe, source, binary):
    set_cpu_mask(AFFINITY)
    engine, _ = load_engine(binary, source / "python")
    from bgsage.weights import bearoff_db_path
    weights = engine.WeightConfig.from_model("stage11")
    trial = engine.TrialEvalConfig(ply=3)
    analyzer = engine.BgBotAnalyzer(weights=weights, eval_level="rollout", cubeful=True,
                                    parallel_threads=THREADS, checker=trial, checker_late=trial,
                                    cube=trial, cube_late=trial, **OPTIONS)
    inner = analyzer._analyzer._inner
    config = inner._rollout_strategy.config()
    expected = {key: OPTIONS[key] for key in ("n_trials", "truncation_depth", "decision_ply",
                "truncation_ply", "ultra_late_threshold", "cubeful_trial_moves",
                "cubeful_late_threshold", "target_se", "max_batches")}
    expected.update(enable_vr=True, n_threads=THREADS)
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected or analyzer._bearoff_db is None or engine.PRODUCTION_MODEL != "stage11":
        raise ValueError(f"Full rollout config differs from requested profile: {actual}")
    if (inner._prefilter_threshold, inner.FILTER_MAX_MOVES, inner.FILTER_THRESHOLD) != (0.15, 5, 0.08):
        raise ValueError("A nonstandard root checker filter is in use")
    pipe.send(dict(extension=file_identity(binary), python=file_identity(Path(engine.__file__)),
                   analyzer=file_identity(Path(inspect.getfile(type(analyzer)))),
                   weight_slots=[file_identity(Path(path)) for path in weights.weight_paths_list],
                   database=file_identity(Path(bearoff_db_path())), config=actual,
                   affinity=sorted(os.sched_getaffinity(0))))
    while (request := pipe.recv()) is not None:
        api, inputs, seed = request
        analyzer.set_seed(seed)
        call = getattr(analyzer, api)
        start = perf_counter_ns()
        result = call(**inputs)
        seconds = (perf_counter_ns() - start) / 1e9
        pipe.send((seconds, asdict(result)))
    pipe.close()


def roster():
    cases = {}
    for kind in APIS[:2]:
        positions = read(DATA / (kind + ".json"))["positions"]
        cases[kind] = {p["id"]: p for p in positions}
    # Stagger APIs and plans so each pair order appears across the panel.
    jobs = []
    for group in range(len(PLANS)):
        for index, api in enumerate(APIS):
            plan_index = (group + 2 * index) % len(PLANS)
            plan = PLANS[plan_index]
            checker = api == "checker_play"
            mode = "match" if (plan_index % 2 == 0) == checker else "money"
            prefix, kind = ("checker", APIS[0]) if checker else ("cube", APIS[1])
            pos = cases[kind][f"{prefix}-{mode}-{plan}-01"]
            inputs = pos["post_move_inputs"] if api == APIS[2] else pos["inputs"]
            jobs.append(dict(sequence=len(jobs), api=api, position_id=pos["id"], mode=mode,
                             source_game_plan=plan, inputs=inputs, seed=seed_for(pos)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "build/truncated_pass2/original-reserved")
    parser.add_argument("--selected", type=Path, default=ROOT / "build/truncated_pass2/shipped")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Compare the unmodified original on three specified cases, reversing build order")
    args = parser.parse_args()
    output = inside_repo(args.output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["BGBOT_MULTIPLY_THREADS"] = str(THREADS)
    previous = read(KNOWN)["arms"]
    historical = read(HISTORY)
    jobs = roster()
    even_order = ARMS
    if args.sensitivity:
        jobs = [job for job in jobs if (job["api"], job["position_id"]) in SENSITIVITY]
        even_order = ARMS[::-1]
        previous["original"]["extension"] = historical["original_extension"]
    identities = {}
    workers = {}
    ctx = get_context("spawn")
    try:
        for arm in ARMS:
            source = inside_repo(getattr(args, arm))
            if args.sensitivity and arm == "original":
                binary = inside_repo(ROOT / historical["original_extension"]["path"])
            else:
                binary = next(source / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                              if (source / ("bgbot_cpp" + suffix)).is_file())
            conn, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, source, binary))
            process.start()
            child.close()
            workers[arm] = (conn, process)
            info = conn.recv()
            for key in ("extension", "python", "analyzer", "database", "weight_slots"):
                if info[key] != previous[arm][key]:
                    raise ValueError(f"{arm} {key} differs from the qualified full-panel build")
            identities[arm] = info
        if identities["original"]["config"] != identities["selected"]["config"]:
            raise ValueError("Engine settings differ")
        settings = dict(arms=identities, qualified_identities=file_identity(KNOWN),
                        original_commit="139427d826dd67798226e7d23c003c90315306ba",
                        original_correction=None if args.sensitivity else historical["source"],
                        sensitivity=args.sensitivity,
                        selected_code_commit="5de8023cc5e548ef5ab4400488508fd1bbe8aa7b",
                        platform={key: value for key, value in platform.uname()._asdict().items()
                                  if key != "node"}, python=sys.version,
                        level="rollout", public_constructor=OPTIONS, early_checker_ply=3,
                        late_checker_ply=3, early_cube_ply=3, late_cube_ply=3,
                        cubeful=True, model="stage11", threads=THREADS, cpu_affinity=hex(AFFINITY),
                        root_prefilter=0.15, root_filter_max_moves=5, root_filter_threshold=0.08,
                        positions={api: file_identity(DATA / (api + ".json")) for api in APIS[:2]},
                        roster=jobs, even_sequence_order=even_order, odd_sequence_order=even_order[::-1],
                        timing="one public call at a time, reseeded per position, no callbacks, excluding construction and serialization")
        write_json(output / "settings.json", settings)
        with (output / "calls.jsonl").open("x", encoding="utf-8", newline="\n") as out:
            for ordinal, job in enumerate(jobs, start=1):
                order = even_order if job["sequence"] % 2 == 0 else even_order[::-1]
                for arm in order:
                    print(f"Starting {ordinal}/{len(jobs)} {arm} {job['api']} {job['position_id']}", flush=True)
                    conn, _ = workers[arm]
                    conn.send((job["api"], job["inputs"], job["seed"]))
                    seconds, result = conn.recv()
                    row = dict(job, arm=arm, elapsed_seconds=seconds, analytics=result)
                    out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    out.flush()
                    print(f"Finished in {seconds:.3f} s", flush=True)
    finally:
        for conn, process in workers.values():
            if process.is_alive():
                conn.send(None)
            conn.close()
            process.join()


if __name__ == "__main__":
    main()
