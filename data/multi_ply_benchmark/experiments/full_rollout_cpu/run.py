# SPDX-License-Identifier: MPL-2.0
"""Time one full rollout at a time on 16 Linux CPUs with isolated analyzers per build."""
import argparse
import ctypes
from dataclasses import asdict
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from profile import PHASES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "full_rollout_1296"))
import measure
from benchmark_multi_ply import file_identity, inside_repo, load_engine, write_json


def cpu_description():
    entries = []
    for block in Path("/proc/cpuinfo").read_text().strip().split("\n\n"):
        fields = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields and int(fields["processor"]) < measure.THREADS:
            entries.append(fields)
    cores = {(fields["physical id"], fields["core id"]) for fields in entries
             if "physical id" in fields and "core id" in fields}
    flags = entries[0].get("flags", "").split()
    return dict(cpu_model=entries[0].get("model name"), logical_cpus=len(entries),
                physical_cores=len(cores) if cores else None,
                relevant_isa=[flag for flag in ("fma", "avx2", "avx512f", "avx512dq", "avx512bw", "avx512vl")
                              if flag in flags])


def full_roster(number):
    primary = measure.roster()
    if number == 1:
        return primary
    if not 1 <= number <= 10:
        raise ValueError("The timing corpus has cases 01 through 10 per stratum")
    positions = {p["id"]: p for name in measure.APIS[:2]
                 for p in measure.read(measure.DATA / (name + ".json"))["positions"]}
    roster = []
    for primary_job in primary:
        pid = primary_job["position_id"][:-2] + f"{number:02d}"
        p = positions[pid]
        inputs = p["post_move_inputs" if primary_job["api"] == "post_move_analytics" else "inputs"]
        roster.append(dict(primary_job, position_id=pid, inputs=inputs, seed=measure.seed_for(p)))
    return roster


def worker(conn, directory, env, trials, profile):
    try:
        os.environ.update(env)
        measure.set_cpu_mask(measure.AFFINITY)
        binary = next(directory / ("bgbot_cpp" + suffix) for suffix in EXTENSION_SUFFIXES
                      if (directory / ("bgbot_cpp" + suffix)).is_file())
        engine, _ = load_engine(binary, directory / "python")
        from bgsage.weights import bearoff_db_path
        weights = engine.WeightConfig.from_model("stage11")
        policy = engine.TrialEvalConfig(ply=3)
        options = dict(measure.OPTIONS, n_trials=trials)
        analyzer = engine.BgBotAnalyzer(weights=weights, eval_level="rollout", cubeful=True,
                                        parallel_threads=16, checker=policy, checker_late=policy,
                                        cube=policy, cube_late=policy, **options)
        inner = analyzer._analyzer._inner
        cfg = inner._rollout_strategy.config()
        native_options = {k: v for k, v in options.items() if k not in ("late_ply", "late_threshold")}
        actual = {k: getattr(cfg, k) for k in [*native_options, "enable_vr", "n_threads"]}
        if actual != dict(native_options, enable_vr=True, n_threads=16) or analyzer._bearoff_db is None:
            raise ValueError(f"Unexpected native configuration {actual}")
        if (inner._prefilter_threshold, inner.FILTER_MAX_MOVES, inner.FILTER_THRESHOLD) != (0.15, 5, 0.08):
            raise ValueError("Unexpected public checker filter")
        conn.send(dict(binary=file_identity(binary), analyzer=file_identity(Path(inspect.getfile(type(analyzer)))),
                       python=file_identity(Path(engine.__file__)), config=actual, constructor=options, trial_policy_ply=3,
                       affinity=sorted(os.sched_getaffinity(0)), database=file_identity(Path(bearoff_db_path())),
                       weights=[file_identity(Path(s)) for s in weights.weight_paths_list], experiment_environment=env))
        if profile:
            timer = ctypes.CDLL(str(binary)).take_rollout_profile
            timer.argtypes = [ctypes.POINTER(ctypes.c_longlong)] * 2
            timer.restype = None
            values = ctypes.c_longlong * len(PHASES)
        dice_counts = getattr(ctypes.CDLL(str(binary)), "bgsage_full_dice_screen_counts", None)
        if dice_counts is not None:
            dice_counts.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
            dice_counts.restype = None
            counts = ctypes.c_uint64 * 2
        while (job := conn.recv()) is not None:
            analyzer.set_seed(job["seed"])
            call = getattr(analyzer, job["api"])
            if profile:
                ns, visits = values(), values()
                timer(ns, visits)
            if dice_counts is not None:
                old_counts = counts()
                dice_counts(old_counts)
            cpu = time.process_time()
            start = time.perf_counter_ns()
            value = call(**job["inputs"])
            elapsed = (time.perf_counter_ns() - start) / 1e9
            stats = dict(elapsed_seconds=elapsed, cpu_seconds=time.process_time() - cpu)
            if profile:
                timer(ns, visits)
                stats.update(phase_worker_seconds=dict(zip(PHASES, (v/1e9 for v in ns))), phase_visits=dict(zip(PHASES, visits)))
            if dice_counts is not None:
                new_counts = counts()
                dice_counts(new_counts)
                stats["partial_roll_screens"] = dict(zip(("considered", "accepted"),
                    (int(b)-int(a) for a,b in zip(old_counts, new_counts))))
            conn.send(dict(**stats, analytics=asdict(value)))
    except BaseException:
        conn.send({"worker_error": traceback.format_exc()})
        raise
    finally:
        conn.close()


def receive(conn):
    data = conn.recv()
    if "worker_error" in data:
        raise RuntimeError(data["worker_error"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", nargs=2, metavar=("LABEL", "BUILD_DIR"), action="append", required=True)
    parser.add_argument("--env", nargs=3, metavar=("LABEL", "NAME", "VALUE"), action="append", default=[])
    parser.add_argument("--jobs", type=int, nargs="+", default=list(range(15)),
                        help="Zero-based indices in the full_rollout_1296 roster")
    parser.add_argument("--trials", type=int, default=1296)
    parser.add_argument("--case-number", type=int, default=1,
                        help="Use the nth fixed corpus position within each stratum")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--reverse-full-panel-order", action="store_true",
                        help="Reverse the primary panel's arm order separately for each selected case")
    parser.add_argument("--primary-settings", type=Path,
                        help="Settings for an earlier primary run with the same arm names and binaries")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    arms = {label: inside_repo(Path(directory)) for label, directory in args.arm}
    if len(arms) != len(args.arm) or len(args.jobs) != len(set(args.jobs)) or min(args.jobs) < 0:
        parser.error("Repeated labels, duplicate jobs or invalid indices")
    if args.reverse_full_panel_order and len(arms) < 2:
        parser.error("The reverse-order check requires at least two arms")
    if args.primary_settings and not args.reverse_full_panel_order:
        parser.error("Use --primary-settings for a reverse-order comparison")
    env = {arm: {} for arm in arms}
    for arm, key, val in args.env:
        env[arm][key] = val
    os.environ["BGBOT_MULTIPLY_THREADS"] = "16"
    roster = full_roster(args.case_number)
    jobs = [roster[i] for i in args.jobs]
    output = inside_repo(args.output)
    output.mkdir(exist_ok=False, parents=True)
    ctx = get_context("spawn")
    workers, identities = {}, {}
    try:
        for arm, directory in arms.items():
            conn, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, directory, env[arm], args.trials, args.profile))
            proc.start()
            child.close()
            workers[arm] = (conn, proc)
            identities[arm] = receive(conn)
        first_arm = next(iter(arms))
        for arm, identity in identities.items():
            for key in ("weights", "database", "config", "affinity"):
                if identity[key] != identities[first_arm][key]:
                    raise ValueError(f"Inconsistent {key} for build {arm}")
        original_order = list(arms)
        prior_identity = None
        if args.primary_settings:
            previous = inside_repo(args.primary_settings)
            previous_settings = measure.read(previous)
            prior_identity = file_identity(previous)
            original_order = previous_settings["arm_order"]
            if previous_settings["repeats"] != 1 or len(previous_settings["roster"]) != 15:
                raise ValueError("Expected a completed single-pass primary panel")
            for arm, identity in identities.items():
                if identity["binary"]["sha256"] != previous_settings["arms"][arm]["binary"]["sha256"]:
                    raise ValueError(f"New {arm} binary differs from primary run")
        write_json(output / "settings.json", dict(arms=identities, trials=args.trials, repeats=args.repeats,
                   case_number=args.case_number,
                   reverse_full_panel_order=args.reverse_full_panel_order,
                   previous_panel_order=original_order, previous_panel_settings=prior_identity,
                   cpu=cpu_description(),
                   roster=jobs, arm_order=list(arms), platform=platform.system(), architecture=platform.machine(),
                   profile=args.profile, python=sys.version, timer="Public API call, analyzer already constructed, no callback"))
        with (output / "calls.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for repeat in range(args.repeats):
                for index, job in enumerate(jobs):
                    order = list(arms)
                    first = (index + repeat) % len(order)
                    if args.reverse_full_panel_order:
                        offset = job["sequence"] % len(original_order)
                        prior = original_order[offset:] + original_order[:offset]
                        order = [name for name in reversed(prior) if name in arms]
                        first = repeat % len(order)
                    order = order[first:] + order[:first]
                    for arm in order:
                        conn, _ = workers[arm]
                        conn.send(job)
                        row = dict(job, arm=arm, repeat=repeat + 1, **receive(conn))
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                        stream.flush()
                        print(arm, job["api"], job["position_id"], f"{row['elapsed_seconds']:.3f} s", flush=True)
    finally:
        for conn, proc in workers.values():
            if proc.is_alive():
                try:
                    conn.send(None)
                except BrokenPipeError:
                    pass
            conn.close()
            proc.join()


if __name__ == "__main__":
    main()
