# SPDX-License-Identifier: MPL-2.0
"""Compare process/trial-thread splits for the same frozen, full reference units."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

import multi_ply_rollout_truth as truth


def worker(args):
    truth.set_cpu_mask(args.cpu_mask)
    truth.WORK = args.output.resolve()
    (truth.WORK / "units").mkdir(parents=True)
    analyzer = truth.make_analyzer(args.threads[0])
    truth.save(truth.WORK / "ready.json", {"pid": os.getpid()})
    go = truth.WORK.parent / "GO"
    while not go.exists():
        time.sleep(0.025)
    start = time.perf_counter()
    completions = {}
    def completed(name):
        completions[name] = time.perf_counter()
    truth.run(0, 1, args.threads[0], unit_names=args.units, initialized=analyzer,
              claims=truth.WORK.parent / "claims" if args.pull else None, on_done=completed)
    truth.save(truth.WORK / "done.json", {"seconds": time.perf_counter() - start,
                                         "finished": time.perf_counter(), "completions": completions})


def check_processes(workers):
    for proc in workers:
        if proc.poll() not in (0, None):
            raise RuntimeError(f"Worker {proc.pid} exited with {proc.returncode}")


def sample_progress(output, folder, worker_dirs):
    trials = {}
    for path in worker_dirs:
        heartbeat = path / "progress-0.json"
        if heartbeat.exists():
            try:
                row = truth.read(heartbeat)
                trials[row["unit"]] = row["done"]
            except PermissionError:
                pass
        trials.update((p.stem, 1296) for p in (path / "units").glob("*.json"))
    with (output / "trial-progress.jsonl").open("a", encoding="utf-8") as f:
        print(json.dumps(dict(time=time.time(), arm=folder.name, trials=trials), separators=(",", ":")), file=f)


def arm(output, units, threads, mask, pull, minimum_available):
    count = mask.bit_count() // threads
    folder = output / f"{count}x{threads}"
    folder.mkdir()
    if pull:
        (folder / "claims").mkdir()
    workers, logs = [], []
    worker_dirs = [folder / str(i) for i in range(count)]
    start = time.perf_counter()
    try:
        for i, path in enumerate(worker_dirs):
            log = (folder / f"worker-{i}.log").open("x", encoding="utf-8")
            logs.append(log)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                   "--output", str(path), "--threads", str(threads), "--cpu-mask", hex(mask),
                   "--units", *(units if pull else units[i::count])]
            if pull:
                cmd.append("--pull")
            workers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        min_available = psutil.virtual_memory().available
        while not all((p / "ready.json").exists() for p in worker_dirs):
            check_processes(workers)
            min_available = min(min_available, psutil.virtual_memory().available)
            if min_available < minimum_available:
                raise MemoryError("Available physical memory dropped below the configured minimum")
            time.sleep(0.1)
        native = [psutil.Process(truth.read(p / "ready.json")["pid"]) for p in worker_dirs]
        begin = time.perf_counter()
        (folder / "GO").touch()
        peak_rss, peak_private = 0, 0
        next_sample = begin
        while True:
            if time.perf_counter() >= next_sample:
                sample_progress(output, folder, worker_dirs)
                next_sample = time.perf_counter() + 5
            min_available = min(min_available, psutil.virtual_memory().available)
            if min_available < minimum_available:
                raise MemoryError("Available physical memory dropped below the configured minimum")
            rss = private = 0
            for proc in native:
                try:
                    mem = proc.memory_info()
                except psutil.NoSuchProcess:
                    continue
                rss += mem.rss
                private += getattr(mem, "private", mem.vms)
            peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
            check_processes(workers)
            if all((p / "done.json").exists() for p in worker_dirs):
                break
            time.sleep(0.25)
        elapsed = max(truth.read(p / "done.json")["finished"] for p in worker_dirs) - begin
        for proc in workers:
            if proc.wait() != 0:
                raise RuntimeError(f"Worker {proc.pid} failed during shutdown")
    finally:
        for proc in workers:
            if proc.poll() is None:
                descendants = psutil.Process(proc.pid).children(recursive=True)
                for child in descendants:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                proc.terminate()
                proc.wait()
        for log in logs:
            log.close()
    files = {path.stem: path for d in worker_dirs for path in (d / "units").glob("*.json")}
    if set(files) != set(units):
        raise ValueError(f"Incomplete arm in {folder}")
    durations = {key: truth.read(path)["seconds"] for key, path in files.items()}
    completion = {name: t - begin for path in worker_dirs
                  for name, t in truth.read(path / "done.json")["completions"].items()}
    # Identical fixed-priority prefixes let another topology use a different
    # completion order while comparing wall time to finish the same endpoints.
    milestones = {str(n): max(completion[name] for name in units[:n])
                  for n in sorted({max(1, len(units) * q // 4) for q in (1, 2, 3, 4)})}
    timing = dict(processes=count, threads=threads, startup_seconds=begin-start,
                  makespan_seconds=elapsed, unit_seconds=durations,
                  completion_seconds=completion, fixed_prefix_seconds=milestones,
                  sum_unit_seconds_per_worker=sum(durations.values()) / count,
                  trials_per_second=len(units) * 1296 / elapsed,
                  minimum_machine_available_bytes=min_available,
                  peak_simultaneous_rss_bytes=peak_rss, peak_simultaneous_private_bytes=peak_private)
    truth.save(folder / "timing.json", timing)
    print(f"{count}x{threads}: setup {begin-start:.2f}s, batch {elapsed:.2f}s, "
          f"per-worker mean busy time {timing['sum_unit_seconds_per_worker']:.2f}s", flush=True)
    return timing, files


def paired(args):
    output = args.output.resolve()
    if not output.is_relative_to(truth.ROOT):
        raise ValueError("Select an output directory inside the engine checkout")
    plan = {j[0] for j in truth.jobs()}
    if len(set(args.units)) != len(args.units) or set(args.units) - plan:
        raise ValueError("Every unit must identify one distinct frozen reference endpoint")
    slots = args.cpu_mask.bit_count()
    if (not slots or len(set(args.threads)) != len(args.threads)
            or any(t < 1 or slots % t or slots // t > len(args.units) for t in args.threads)):
        raise ValueError("Select thread counts dividing the mask and enough units to fill all workers")
    if args.publish and (4 not in args.threads or len(args.threads) < 2):
        raise ValueError("Publishing requires a four-thread reference and a paired topology")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(cpu_mask=hex(args.cpu_mask), units=args.units, arms=[], pull_queue=args.pull,
                  manifest_sha256=truth.digest((truth.OUT / "manifest.json").read_bytes()))
    files = {}
    for threads in args.threads:
        timing, files[threads] = arm(output, args.units, threads, args.cpu_mask, args.pull,
                                     int(args.min_available_gib * 2**30))
        report["arms"].append(timing)
        truth.save(output / "summary.json", report)
    baseline = files[4 if 4 in files else args.threads[0]]
    compare_keys = ("unit", "api", "position_id", "seed", "manifest_sha256", "analytics", "native_checker")
    for unit, path in baseline.items():
        ref = truth.read(path)
        for by_unit in files.values():
            candidate = truth.read(by_unit[unit])
            if any(ref[key] != candidate[key] for key in compare_keys):
                raise ValueError(f"Cross-topology field mismatch in {by_unit[unit]}")
    report["full_fields_equal"] = True
    report["published_units"] = []
    if args.publish:
        for unit, path in baseline.items():
            target = truth.WORK / "units" / path.name
            try:
                os.link(path, target)
            except FileExistsError:
                prior, new = truth.read(target), truth.read(path)
                if any(prior[key] != new[key] for key in compare_keys):
                    raise ValueError(f"Conflicting existing reference: {target}")
            else:
                report["published_units"].append(unit)
    truth.save(output / "summary.json", report)
    if (output / "trial-progress.jsonl").exists():
        summarize_progress(output)


def summarize_progress(output):
    """Time equal fractions of the batch, weighting unequal trial costs by the control unit."""
    report = truth.read(output / "summary.json")
    baseline = next((a for a in report["arms"] if a["threads"] == 4), report["arms"][0])
    weights = baseline["unit_seconds"]
    total = sum(weights.values())
    samples = [json.loads(s) for s in (output / "trial-progress.jsonl").read_bytes().splitlines()]
    estimates = {}
    for arm in report["arms"]:
        key = f"{arm['processes']}x{arm['threads']}"
        start = (output / key / "GO").stat().st_mtime
        points = [(0., 0.)]
        for sample in samples:
            if sample["arm"] == key:
                fraction = sum(weights[unit] * min(1296, n) / 1296 for unit, n in sample["trials"].items()) / total
                points.append((min(arm["makespan_seconds"], max(0., sample["time"]-start)),
                               max(points[-1][1], min(1., fraction))))
        points.append((arm["makespan_seconds"], 1.))
        times = {}
        for q in (0.25, 0.5, 0.75):
            for (t0, f0), (t1, f1) in zip(points, points[1:]):
                if f0 < q <= f1:
                    times[q] = t0 + (t1 - t0) * (q - f0) / (f1 - f0)
                    break
        estimates[key] = dict(time_to_fraction_seconds=times,
                              middle_half_seconds=times[0.75]-times[0.25])
    truth.save(output / "sustained.json", estimates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("paired", "worker", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-mask", type=lambda s: int(s, 0))
    unit_spec = parser.add_mutually_exclusive_group()
    unit_spec.add_argument("--units", nargs="+")
    unit_spec.add_argument("--units-file", type=Path, help="JSON array of frozen unit names in priority order")
    parser.add_argument("--threads", nargs="+", type=int, default=[4, 2, 1])
    parser.add_argument("--publish", action="store_true", help="Add matching four-thread units to the running reference")
    parser.add_argument("--pull", action="store_true", help="Both arms pull distinct units from the same priority order")
    parser.add_argument("--min-available-gib", type=float, default=0,
                        help="Abort if observed machine-wide available RAM falls below this amount")
    args = parser.parse_args()
    if args.action == "summarize":
        summarize_progress(args.output.resolve())
        sys.exit(0)
    if args.units_file:
        args.units = truth.read(args.units_file)
    if not args.units or args.cpu_mask is None or args.cpu_mask <= 0:
        parser.error("Select units and a positive CPU mask for an evaluation")
    if args.action == "worker":
        worker(args)
    else:
        paired(args)
