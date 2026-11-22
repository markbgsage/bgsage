# SPDX-License-Identifier: MPL-2.0
"""Run one bounded Python benchmark with the four local truth shards suspended
or restricted to a CPU mask. Restore prior scheduling even if the owner exits.

python scripts/with_rollout_window.py --dry-run
python scripts/with_rollout_window.py --timeout 600 -- scripts/measure_multi_ply_pair.py ...
python scripts/with_rollout_window.py --cpu-mask 0xff --timeout 300 -- scripts/compare_rollout_performance.py ...
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "logs/multi_ply_rollout_truth"


def identified(shard, shards, pid, birth=None):
    p = psutil.Process(pid)
    if birth is not None and p.create_time() != birth:
        raise ValueError(f"PID {pid} was recycled")
    argv = p.cmdline()
    if (Path(p.cwd()).resolve() != ROOT or not any(Path(arg).name=="multi_ply_rollout_truth.py" for arg in argv)
            or "run" not in argv or argv[argv.index("--shard")+1] != str(shard)
            or argv[argv.index("--shards")+1] != str(shards)):
        raise ValueError(f"PID {pid} is not the intended reference shard")
    return p


def collect(n):
    saved = []
    for shard in range(n):
        data = json.loads((WORK / f"progress-{shard}.json").read_bytes())
        p = identified(shard,n,data["pid"])
        if p.status() == psutil.STATUS_STOPPED:
            raise ValueError(f"Shard {shard} is already suspended")
        saved.append(dict(shard=shard,pid=p.pid,birth=p.create_time(),affinity=p.cpu_affinity()))
    return saved


def restore(meta):
    errors = []
    for s in meta["shards"]:
        try:
            p = identified(s["shard"],len(meta["shards"]),s["pid"],s["birth"])
        except (psutil.NoSuchProcess,ValueError):
            # Never adjust a process that inherited an exited worker's PID.
            continue
        try:
            if meta["mask"] is None:
                # Each original worker was running before this scheduling window.
                if p.status()==psutil.STATUS_STOPPED:
                    p.resume()
            else:
                p.cpu_affinity(s["affinity"])
        except psutil.NoSuchProcess:
            continue
        except (OSError,psutil.Error) as exc:
            errors.append(f"{s['pid']}: {exc}")
    if errors:
        raise RuntimeError("Could not restore all scheduling settings: " + ", ".join(errors))


def watchdog(path):
    meta = json.loads(path.read_bytes())
    signal = path.with_suffix(".restored")
    while not signal.exists():
        owner_alive = False
        try:
            owner_alive = psutil.Process(meta["owner"]).create_time()==meta["owner_birth"]
        except psutil.NoSuchProcess:
            pass
        if not owner_alive or time.time() >= meta["deadline"]:
            restore(meta)
            signal.write_text("watchdog restored\n",encoding="ascii")
            return
        time.sleep(1)


def locked_window(args):
    state = collect(args.shards)
    print("Identified rollout workers:",[s["pid"] for s in state],flush=True)
    if args.dry_run:
        return 0
    argv=args.command
    if argv and argv[0]=="--":
        argv=argv[1:]
    if (not argv or not Path(argv[0]).resolve().is_relative_to(ROOT)
            or Path(argv[0]).suffix.lower()!=".py" or not Path(argv[0]).is_file()):
        raise ValueError("Supply -- <engine-local Python script> [args...]")
    if args.cpu_mask is not None and not 0<args.cpu_mask<(1<<len(psutil.Process().cpu_affinity())):
        raise ValueError("Invalid CPU mask")
    if args.timeout<1:
        raise ValueError("Timeout must be positive")
    meta=dict(shards=state,owner=os.getpid(),owner_birth=psutil.Process().create_time(),
              mask=args.cpu_mask,deadline=time.time()+args.timeout+8)
    path=WORK/f"window-{os.getpid()}-{time.time_ns()}.json"
    path.write_text(json.dumps(meta),encoding="utf-8")
    with path.with_suffix(".log").open("x",encoding="utf-8") as log:
        subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--watch",str(path)],
                         stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    child=None
    try:
        for s in state:
            p=identified(s["shard"],len(state),s["pid"],s["birth"])
            if args.cpu_mask is None:
                p.suspend()
            else:
                p.cpu_affinity([i for i in range(args.cpu_mask.bit_length()) if args.cpu_mask & (1<<i)])
        print("Window open",flush=True)
        child=subprocess.Popen([sys.executable,"-u",*argv])
        return child.wait(timeout=args.timeout)
    finally:
        try:
            if child is not None and child.poll() is None:
                # The bounded benchmark can itself have spawned Python workers.
                children=psutil.Process(child.pid).children(recursive=True)
                child.kill()
                for sub in children:
                    try:
                        sub.kill()
                    except psutil.NoSuchProcess:
                        pass
        finally:
            restore(meta)
            path.with_suffix(".restored").write_text("owner restored\n",encoding="ascii")
            print("Restored rollout workers",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch",type=Path,help=argparse.SUPPRESS)
    ap.add_argument("--timeout",type=int,default=600)
    ap.add_argument("--shards",type=int,default=4)
    ap.add_argument("--cpu-mask",type=lambda v:int(v,0),help="Restrict rollout workers to mask instead of suspending")
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument("command",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    if args.watch:
        watchdog(args.watch)
    else:
        with (WORK/"timing-window.lock").open("a+b") as lock:
            lock.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            sys.exit(locked_window(args))
