# SPDX-License-Identifier: MPL-2.0
"""Resumable 1,296-trial full rollout reference for the fixed timing corpus.

Run prepare, then run (one or more disjoint --shard I --shards N workers).
Use assemble at any time to export complete API records and a coverage report.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import errno
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/multi_ply_benchmark"
BASE = DATA / "baseline_stage11"
WORK = ROOT / "logs/multi_ply_rollout_truth"
FROZEN = WORK / "frozen"
OUT = DATA / "rollout_3p_vr1296"
CONFIG = dict(n_trials=1296, truncation_depth=0, decision_ply=3, truncation_ply=3,
              late_ply=3, late_threshold=9999, ultra_late_threshold=9999,
              cubeful_late_threshold=9999, cubeful_trial_moves=True,
              target_se=0.0, max_batches=1, prefilter_threshold=0.0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    replace(temp, path)


def replace(temp, path):
    # On Windows a concurrent reader can briefly deny rename access.
    for attempt in range(9):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(0.02 * (attempt + 1))


def seed_for(position):
    return int(digest(("rollout-truth-v1:" + position["source"]["key"]).encode())[:8], 16)


def historical_python(commit, asset):
    # Git normally stores LF. The original Windows package may have had CRLF.
    # Admit only bytes that match the historical benchmark's saved SHA-256.
    blob = subprocess.run(["git","show",f"{commit}:{asset['path']}"], cwd=ROOT,
                          check=True,capture_output=True).stdout
    lf = blob.replace(b"\r\n", b"\n")
    for candidate in (blob, lf, lf.replace(b"\n", b"\r\n")):
        if digest(candidate) == asset["sha256"]:
            return candidate
    raise ValueError(f"Pinned Git blob does not reproduce original bytes: {asset['path']}")


def prepare(extension=None):
    meta = read(BASE / "metadata.json")
    assets = [meta["extension"], meta["bearoff_database"], *meta["weights"],
              *(a for a in meta["source_files"] if a["path"].startswith("python/")),
              *meta["positions"].values()]
    # Also freeze the incumbent checker ordering used for scheduling. It
    # changes only which jobs finish first, never the set of legal candidates.
    ranking = BASE / "results.jsonl"
    assets.append(dict(path=ranking.relative_to(ROOT).as_posix(),
                       sha256=digest(ranking.read_bytes())))
    for asset in assets:
        dest = FROZEN / asset["path"]
        if not dest.exists():
            src = ROOT / asset["path"]
            if extension is not None and asset == meta["extension"]:
                src = extension.resolve()
                if not src.is_relative_to(ROOT):
                    raise ValueError("Reference extension must be inside this engine checkout")
            data = src.read_bytes() if src.is_file() else None
            if data is None or digest(data) != asset["sha256"]:
                if not asset["path"].startswith("python/"):
                    raise ValueError(f"Exact baseline asset unavailable: {src} (SHA-256 {asset['sha256']})")
                data = historical_python(meta["git_commit"],asset)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        if digest(dest.read_bytes()) != asset["sha256"]:
            raise ValueError(f"Frozen asset differs: {dest}")
    for folder in (OUT, WORK / "units"):
        folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=1, git_commit=meta["git_commit"], model="stage11",
        config=CONFIG, trial_configs={k: 3 for k in ("checker", "checker_late", "cube", "cube_late")},
        variance_reduction=True, seed="uint32(SHA256('rollout-truth-v1:' + source_key)[:8], 16)",
        checker_candidates="all legal endpoints; same seed across moves",
        assets=assets)
    if (OUT / "manifest.json").exists() and read(OUT / "manifest.json") != manifest:
        raise ValueError("Reference manifest changed; retain prior run and use a separate workspace")
    save(OUT / "manifest.json", manifest)


def jobs():
    source = FROZEN / DATA.relative_to(ROOT)
    checker = read(source / "checker_play.json")["positions"]
    cubes = read(source / "cube_action.json")["positions"]
    incumbent = {}
    for line in (source / "baseline_stage11/results.jsonl").read_bytes().splitlines():
        r = json.loads(line)
        if r["level"] == "3ply" and r["api"] == "checker_play":
            incumbent[r["position_id"]] = [m["board"] for m in r["analytics"]["moves"]]
    # First cover the ten mode/plan cells at rank one, rotating cases across
    # cells. Start in the race stratum for an early pipeline end-to-end check.
    order = sorted(range(100), key=lambda i: (i % 10, (i // 10 - 4) % 10))
    for rank in range(max(map(len, incumbent.values()))):
        for i in order:
            c, p = cubes[i], checker[i]
            if rank == 0:
                for api, key in (("cube_action", "cube"), ("post_move_analytics", "post")):
                    yield f"{key}-{c['id']}", api, c, None
            boards = incumbent[p["id"]]
            if rank < len(boards):
                yield f"move-{p['id']}-{rank:02}", "checker_play", p, boards[rank]


def make_analyzer(threads, *, public_filter=False):
    manifest = read(OUT / "manifest.json")
    for a in manifest["assets"]:
        if digest((FROZEN / a["path"]).read_bytes()) != a["sha256"]:
            raise ValueError("Frozen asset differs: " + a["path"])
    sys.path[:0] = [str(FROZEN / "python"), str((FROZEN / manifest["assets"][0]["path"]).parent)]
    import bgbot_cpp
    import bgsage
    from bgsage.analyzer import resolve_owner
    if not all(Path(m.__file__).resolve().is_relative_to(FROZEN) for m in (bgsage, bgbot_cpp)):
        raise ValueError("Imported unfrozen engine")
    trial = bgsage.TrialEvalConfig(ply=3)
    config = dict(CONFIG)
    if public_filter:
        config.pop("prefilter_threshold")  # Let the frozen public constructor choose its normal full-rollout filter.
    analyzer = bgsage.BgBotAnalyzer(eval_level="rollout", weights=bgsage.WeightConfig.from_model("stage11"),
        cubeful=True, parallel_threads=threads, checker=trial, checker_late=trial,
        cube=trial, cube_late=trial, **config)
    strategy = analyzer._analyzer._inner._rollout_strategy
    cfg = strategy.config()
    if (analyzer._bearoff_db is None or not cfg.enable_vr or cfg.n_trials != 1296
            or cfg.truncation_depth != 0 or cfg.decision_ply != 3 or cfg.n_threads != threads
            or cfg.max_batches != 1 or cfg.target_se != 0.0 or not cfg.cubeful_trial_moves):
        raise ValueError("Rollout configuration did not initialize as requested")
    return analyzer, strategy, resolve_owner, bgsage


@contextmanager
def unit_lease(path):
    """Keep one unit exclusive through its atomic checkpoint, including on worker exit."""
    with path.open("a+b") as lease:
        lease.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            yield False
        else:
            yield True


def pooled_units(plan):
    leases = WORK / "leases"
    leases.mkdir(exist_ok=True)
    waiting = plan
    while waiting:
        busy = []
        for item in waiting:
            if (WORK / "STOP").exists():
                return
            name = item[0]
            if (WORK / "units" / (name + ".json")).is_file():
                yield item  # The caller validates the existing checkpoint.
                continue
            with unit_lease(leases / (name + ".lock")) as acquired:
                if acquired:
                    # The caller rechecks the checkpoint under the lease before
                    # computing, then saves its result before the generator resumes.
                    yield item
                else:
                    busy.append(item)
        waiting = busy
        if waiting:
            time.sleep(2)


def run(shard, shards, threads, *, unit_names=None, initialized=None, claims=None, on_done=None, pooled=False):
    analyzer, strategy, owner, engine = initialized or make_analyzer(threads)
    plan = list(jobs())
    candidates = {}
    for _, api, p, b in plan:
        if api == "checker_play":
            candidates.setdefault(p["id"], []).append(tuple(b))
    if unit_names is not None:
        by_name = {item[0]: item for item in plan}
        plan = [by_name[name] for name in unit_names]
    validated = set()
    manifest_hash = digest((OUT / "manifest.json").read_bytes())
    assigned = pooled_units(plan) if pooled else plan[shard::shards]
    for name, api, p, board in assigned:
        if (WORK / "STOP").exists():
            return
        target = WORK / "units" / (name + ".json")
        if target.exists():
            if read(target)["manifest_sha256"] != manifest_hash:
                raise ValueError(f"Wrong manifest for {name}")
            continue
        if claims is not None:
            try:
                (claims / name).touch(exist_ok=False)
            except FileExistsError:
                continue
        inputs = p["post_move_inputs"] if api == "post_move_analytics" else p["inputs"]
        if api == "checker_play" and p["id"] not in validated:
            legal = engine.possible_moves(inputs["board"], inputs["die1"], inputs["die2"])
            if sorted(candidates[p["id"]]) != sorted(map(tuple, legal)):
                raise ValueError("Fixed checker ordering does not cover every legal endpoint")
            validated.add(p["id"])
        seed = seed_for(p)
        analyzer.set_seed(seed)
        started = time.monotonic()
        last = [0.0]
        def progress(done, total, *unused):
            now = time.monotonic()
            if now - last[0] > 6 or done == total:
                last[0] = now
                try:
                    save(WORK / f"progress-{shard}.json", dict(pid=os.getpid(), unit=name, done=done,
                         total=total, elapsed_seconds=now-started, unix_time=time.time(), threads=threads))
                except PermissionError:
                    # A locked telemetry file must not discard 1,296 expensive
                    # trials. Permanent checkpoints still fail loudly on error.
                    pass
        progress(0, 1296)
        if api == "checker_play":
            # This is the exact per-survivor call inside public checker_play.
            # Checkpoint after each move and retain even moves it normally prunes.
            kw = {k: v for k, v in inputs.items() if k not in ("board", "die1", "die2", "cube_owner")}
            raw = strategy.cubeful_evaluate_board(board, inputs["board"], owner=owner(inputs["cube_owner"]),
                                                  progress=progress, **kw)
            probs = raw["probs"]
            if analyzer._bearoff_db.is_bearoff(board):
                probs = analyzer._bearoff_db.lookup_probs(board, post_move=True)
            analytics = asdict(engine.MoveAnalysis(board=board, equity=raw["cubeful_equity"],
                cubeless_equity=raw["equity"], probs=engine.Probabilities.from_list(probs),
                equity_diff=0.0, eval_level="Rollout", std_error=raw["std_error"],
                prob_std_errors=list(raw["prob_std_errors"])))
        else:
            raw = None
            analytics = asdict(getattr(analyzer, api)(**inputs, progress_callback=progress))
        save(target, dict(unit=name, api=api, position_id=p["id"], seed=seed, threads=threads,
                          manifest_sha256=manifest_hash, seconds=time.monotonic()-started,
                          analytics=analytics, native_checker=raw))
        if on_done is not None:
            on_done(name)
        print(f"{name}: complete in {time.monotonic()-started:.1f}s", flush=True)


def assemble():
    plan = list(jobs())
    groups = {}
    for name, api, p, board in plan:
        groups.setdefault((api,p["id"]), []).append((name,p,board))
    rows = []
    checker_se = []
    coverage = Counter()
    completed_units = 0
    manifest_hash = digest((OUT / "manifest.json").read_bytes())
    for (api,pid), items in groups.items():
        records = []
        for name, p, board in items:
            path = WORK / "units" / (name + ".json")
            if path.exists():
                unit = read(path)
                if (unit["unit"],unit["position_id"],unit["api"],unit["manifest_sha256"],
                    unit["seed"],unit["analytics"]["eval_level"]) != (name,pid,api,manifest_hash,seed_for(p),"Rollout"):
                    raise ValueError(f"Invalid checkpoint {path}")
                if board is not None and unit["analytics"]["board"] != board:
                    raise ValueError(f"Wrong checker endpoint in {path}")
                if board is not None:
                    checker_se.append(dict(position_id=pid,board=board,
                                           cubeful_se=unit["native_checker"]["cubeful_se"]))
                records.append(unit)
        completed_units += len(records)
        if len(records) != len(items):
            continue
        inputs = p["post_move_inputs"] if api == "post_move_analytics" else p["inputs"]
        if api == "checker_play":
            moves = sorted((r["analytics"] for r in records), key=lambda m: (-m["equity"],-m["cubeless_equity"],m["board"]))
            for move in moves:
                move["equity_diff"] = move["equity"] - moves[0]["equity"]
            analytics = dict(moves=moves, board=inputs["board"], die1=inputs["die1"], die2=inputs["die2"],eval_level="Rollout")
        else:
            analytics = records[0]["analytics"]
        rows.append(dict(api=api, position_id=pid, mode=p["mode"], source_game_plan=p["game_plan"],
                         source=p["source"], inputs=inputs, seed=seed_for(p), analytics=analytics))
        coverage[api] += 1
    save_rows(OUT/"results.jsonl",rows)
    save_rows(OUT/"checker_cubeful_se.jsonl",checker_se)
    report = dict(completed_units=completed_units,total_units=len(plan),complete_positions=dict(coverage),
                  total_positions=len(groups),checker_endpoints=len(checker_se),manifest_sha256=manifest_hash)
    save(OUT / "coverage.json", report)
    print(json.dumps(report))


def save_rows(dest, rows):
    tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
    replace(tmp, dest)


def status(shards, pooled=False):
    """Estimate each shard's remaining wall time from completed units."""
    plan = list(jobs())
    pauses = []
    for path in WORK.glob("window-*.json"):
        restored = path.with_suffix(".restored")
        if restored.exists() and read(path)["mask"] is None:
            pauses.append((path.stat().st_mtime, restored.stat().st_mtime))
    excluded_pause_seconds = 0.
    sampled = {}
    missing = [[] for _ in range(shards)]
    counts = Counter()
    for index, (unit, api, position, _) in enumerate(plan):
        shard = index % shards
        path = WORK / "units" / (unit + ".json")
        keys = ((api, position["mode"], position["game_plan"], position["id"]),
                (api, position["mode"], position["game_plan"]),
                (api, position["game_plan"]), (api,), ())
        if path.exists():
            seconds = read(path)["seconds"]
            # Native monotonic call times include suspension. A parked call
            # spans both window boundaries; a comparison call inside the
            # window does not. Keep original checkpoint timings intact.
            end = path.stat().st_mtime
            parked = sum(b-a for a, b in pauses if end-seconds <= a <= b <= end)
            seconds = max(0., seconds - parked)
            excluded_pause_seconds += parked
            counts[shard] += 1
            sampled.setdefault(keys, []).append(seconds)
        else:
            missing[shard].append(keys)

    # Every represented source position contributes once to a fallback
    # stratum, even if several of its checker endpoints were selected for
    # a supplemental paired experiment.
    means, medians = {}, {}
    for keys, times in sampled.items():
        for key in keys:
            means.setdefault(key, []).append(mean(times))
            medians.setdefault(key, []).append(median(times))
    median_time = {key: median(times) for key, times in medians.items()}
    mean_time = {key: mean(times) for key, times in means.items()}
    now = time.time()
    workers = []
    for shard, remaining in enumerate(missing):
        row = dict(shard=shard, completed=counts[shard], remaining=len(remaining))
        if sampled or not remaining:
            strata = [next(key for key in keys if key in mean_time) for keys in remaining]
            row["estimated_remaining_hours"] = round(sum(mean_time[key] for key in strata) / 3600, 2)
            row["median_based_remaining_hours"] = round(sum(median_time[key] for key in strata) / 3600, 2)
            row["units_with_same_position_timing"] = sum(keys[0] in mean_time for keys in remaining)
            row["units_with_matched_timing_stratum"] = sum(keys[1] in mean_time for keys in remaining)
        heartbeat = WORK / f"progress-{shard}.json"
        if (remaining or (pooled and any(missing))) and heartbeat.exists():
            current = read(heartbeat)
            row["heartbeat_age_seconds"] = round(now - current["unix_time"], 1)
            row["current"] = {key: current[key] for key in ("pid", "unit", "done", "total", "threads")}
        workers.append(row)
    result = dict(completed_units=sum(counts.values()), total_units=len(plan), workers=workers,
                  excluded_worker_suspension_seconds=round(excluded_pause_seconds, 2))
    if pooled:
        # The modulo split used to project job costs does not assign owners in
        # a pull queue. Pooling the costs estimates aggregate remaining time.
        if all("estimated_remaining_hours" in row for row in workers):
            result["estimated_pooled_hours"] = round(mean(row["estimated_remaining_hours"] for row in workers), 2)
            result["median_based_pooled_hours"] = round(mean(row["median_based_remaining_hours"] for row in workers), 2)
            result["units_with_same_position_timing"] = sum(row["units_with_same_position_timing"] for row in workers)
        for row in workers:
            for key in ("completed", "remaining", "estimated_remaining_hours", "median_based_remaining_hours",
                        "units_with_matched_timing_stratum", "units_with_same_position_timing"):
                row.pop(key, None)
    print(json.dumps(result, indent=2))


def supervise(args):
    # Kernel-owned lease is released automatically on process exit.
    with (WORK / "supervisor.lock").open("a+b") as lease:
        lease.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch_workers(args)


def set_cpu_mask(mask):
    if mask is None:
        return
    if mask <= 0:
        raise ValueError("CPU mask must select at least one logical processor")
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not k32.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, {i for i in range(mask.bit_length()) if mask & (1 << i)})


def watch_workers(args):
    """Adopt explicit existing workers, restart a failed shard, and assemble."""
    import psutil  # only this optional multi-process supervisor needs psutil
    if args.adopt and len(args.adopt) != args.shards:
        raise ValueError("Provide one existing native Python worker PID per shard")
    workers, restarts, last_count = {}, Counter(), -1
    for shard, pid in enumerate(args.adopt):
        proc = psutil.Process(pid)
        command = proc.cmdline()
        if (not any(Path(item).name == Path(__file__).name for item in command)
                or command[command.index("--shard")+1] != str(shard)
                or command[command.index("--shards")+1] != str(args.shards)
                or Path(proc.cwd()).resolve() != ROOT or ("--pooled" in command) != args.pooled):
            raise ValueError(f"PID {pid} does not match the intended rollout worker")
        workers[shard] = proc
    plan = list(jobs())
    expected = {name for name, *_ in plan}
    while True:
        completed = {p.stem for p in (WORK/"units").glob("*.json")}
        if len(completed) != last_count:
            assemble()
            last_count = len(completed)
        unfinished = []
        for shard in range(args.shards):
            left = (len(expected - completed) if args.pooled else
                    sum(name not in completed for name, *_ in plan[shard::args.shards]))
            if not left:
                continue
            unfinished.append(shard)
            if (WORK / "STOP").exists():
                continue
            previous = workers.get(shard)
            if previous is not None and previous.is_running() and previous.status() != psutil.STATUS_ZOMBIE:
                continue
            if previous is not None:
                restarts[shard] += 1
            if restarts[shard] > 3:
                raise RuntimeError(f"Shard {shard} repeatedly failed; inspect worker output before restarting")
            cmd = [sys.executable,"-u",str(Path(__file__).resolve()),"run",
                   "--shards",str(args.shards),"--shard",str(shard),"--threads",str(args.threads)]
            if args.cpu_mask is not None:
                cmd.extend(["--cpu-mask",hex(args.cpu_mask)])
            if args.pooled:
                cmd.append("--pooled")
            log = WORK / f"worker-{shard}-{int(time.time())}.log"
            with log.open("x",encoding="utf-8") as output:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            workers[shard] = psutil.Process(proc.pid)
        if not unfinished or (WORK / "STOP").exists():
            return
        time.sleep(30)


def repartition(args):
    """Drain identified old workers, then run a new disjoint shard plan."""
    import psutil
    if not args.adopt:
        raise ValueError("Identify the current workers in shard order using --adopt")
    if (WORK / "STOP").exists():
        raise ValueError("A stop is already active")
    old = {}
    for shard, pid in enumerate(args.adopt):
        proc = psutil.Process(pid)
        command = proc.cmdline()
        if (not any(Path(item).name == Path(__file__).name for item in command)
                or command[command.index("--shard")+1] != str(shard)
                or command[command.index("--shards")+1] != str(len(args.adopt))
                or Path(proc.cwd()).resolve() != ROOT):
            raise ValueError(f"PID {pid} does not match existing shard {shard}")
        progress = read(WORK / f"progress-{shard}.json")
        if progress["pid"] != pid:
            raise ValueError(f"Unexpected PID in shard {shard} progress")
        old[shard] = (proc, progress["unit"])
    token=f"repartition-{os.getpid()}"
    with (WORK / "STOP").open("x",encoding="ascii") as f:
        f.write(token)
    print("Draining",[(s,u) for s,(_,u) in old.items()],flush=True)
    # Current workers honor STOP between units. Let each exit on its own even
    # if a new unit started between reading its heartbeat and creating STOP.
    while old:
        for shard,(proc,_) in list(old.items()):
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                del old[shard]
        if old:
            time.sleep(0.5)
    # The previous supervisor checks STOP once per 30s. Wait for its kernel
    # lease before removing the drain marker, avoiding a competing respawn.
    with (WORK / "supervisor.lock").open("a+b") as lease:
        while True:
            lease.seek(0)
            try:
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(lease.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (OSError,BlockingIOError):
                time.sleep(1)
        stop=WORK/"STOP"
        if stop.read_text(encoding="ascii")!=token:
            raise ValueError("Stop marker ownership changed; leaving queue stopped")
        stop.unlink()
        args.adopt=[]
        watch_workers(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare","run","assemble","status","supervise","repartition"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0, help="Zero-based worker index")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--pooled", action="store_true", help="Claim unfinished units using crash-safe per-unit OS locks")
    ap.add_argument("--cpu-mask", type=lambda v: int(v, 0),
                    help="Optional process CPU affinity bit mask (e.g. 0xff)")
    ap.add_argument("--adopt", type=int,nargs="*",default=[],help="Existing worker PIDs, in shard order, for supervise")
    ap.add_argument("--extension", type=Path, help="For prepare: alternate in-repository location of the exact original binary")
    args=ap.parse_args()
    if args.threads<1 or not 0<=args.shard<args.shards:
        ap.error("Require positive threads and 0 <= shard < shards")
    if args.action=="prepare":
        prepare(args.extension)
    elif args.action=="run":
        set_cpu_mask(args.cpu_mask)
        run(args.shard,args.shards,args.threads,pooled=args.pooled)
    elif args.action=="assemble":
        assemble()
    elif args.action=="status":
        status(args.shards,args.pooled)
    elif args.action=="repartition":
        repartition(args)
    else:
        supervise(args)
