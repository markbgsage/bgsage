#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Run eXtreme Gammon's Batch Analyze over the folder-benchmark shards, one XG
level at a time, and harvest the scores.

``export_folder_benchmark_xg.py generate`` writes the thirteen family
benchmarks as native ``.xg`` shards under ``data/backgame_xg/<folder-slug>/``.
XG's Batch Analyze takes a set of files chosen in one Open dialog and analyses
them in place, so this script

* ``stamp``   copies every family's shards into ONE flat folder per XG level,
              ``data/backgame_xg_flat/<tag>/<folder-slug>__bench_shard_NNN.xg``
              (sidecars alongside), so a single Batch Analyze selection covers
              all 81 shards of a level; the pristine tree is never analysed;
* ``analyze`` drives XG's Batch Analyze dialog deterministically through
              ``xg_batch_win.py`` (pywinauto; XG must be running on this
              desktop): Analyze > Batch Analyze, choose every file of the level
              folder, select the level's analysis profile for every player,
              "Save Games after analyze" on, Start -- and returns once the run
              has started;
* ``wait``    polls the level folder until every shard has been rewritten by
              XG and nothing has changed for a few minutes;
* ``score``   hands the folder to ``export_folder_benchmark_xg.py score``.

Level tags follow ``stamp_xg_levels.py``: xg2ply, xg3ply, xg4ply, xgroller,
xgrollerplus, xgrollerpp (XG's 2-ply, 3-ply, 4-ply, XG Roller, XG Roller+,
XG Roller++ -- Sage 2P, 3P, 4P, 1T, 2T, 3T).

    py -3.14 scripts/xg_folder_batch.py stamp
    py -3.14 scripts/xg_folder_batch.py analyze --level xg3ply
    py -3.14 scripts/xg_folder_batch.py wait --level xg3ply
    py -3.14 scripts/xg_folder_batch.py score --level xg3ply
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

MASTER = _PROJECT_ROOT / "data" / "backgame_xg"
FLAT = _PROJECT_ROOT / "data" / "backgame_xg_flat"
SIDECAR = ".sidecar.jsonl"
#: tag -> the XG analysis PROFILE selected in Batch Analyze's "Analyze Level"
#: combo. XG lists PROFILES there, not raw levels, and each profile analyses at
#: 3-ply and escalates to its own level on close decisions -- the convention the
#: money / match / Paskogammon XG columns used. The report prints the levels XG
#: actually stamped on the records, so a wrong profile is visible.
LEVELS: dict[str, str] = {
    "xg2ply": "Sage2P",      # scored 2026-09-06; the profile was replaced afterwards
    "xg3ply": "Sage3P",
    "xg4ply": "Custom Setting 4Ply",
    "xgroller": "Custom Setting Roller",
    "xgrollerplus": "Custom Setting Roller +",
    "xgrollerpp": "Custom Setting Roller ++",
}


def stamp(levels: list[str], force: bool) -> None:
    shards = sorted(MASTER.glob("*/bench_shard_*.xg"))
    if not shards:
        raise SystemExit(f"no shards under {MASTER}; run export_folder_benchmark_xg.py generate first")
    for tag in levels:
        out = FLAT / tag
        if out.exists() and any(out.iterdir()) and not force:
            print(f"{tag}: {out} exists, skipped (--force to rebuild; never re-stamp an analysed folder)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            name = f"{shard.parent.name}__{shard.name}"
            shutil.copy2(shard, out / name)
            shutil.copy2(Path(str(shard) + SIDECAR), out / (name + SIDECAR))
        print(f"{tag}: {len(shards)} shards -> {out}")


def analyze(tag: str, no_start: bool = False) -> None:
    """Drive XG's Batch Analyze dialog deterministically (``xg_batch_win.py``):
    choose every shard of the level folder, select the level's profile for
    every player row, set the check boxes, Start."""
    import xg_batch_win as win  # noqa: E402  (pywinauto driver, same dir)
    folder = FLAT / tag
    files = sorted(folder.glob("*.xg"))
    if not files:
        raise SystemExit(f"{folder} has no shards; run `stamp` first")
    print(f"{tag}: Batch Analyze on {len(files)} files with profile {LEVELS[tag]!r} ...", flush=True)
    win.dismiss_message()
    dlg = win.open_batch_dialog()
    n = win.choose_files(dlg, folder.resolve())
    win.set_level(dlg, LEVELS[tag])
    win.set_checkboxes(dlg)
    if no_start:
        print(f"{tag}: {n} files chosen, profile set -- not started"); return
    win.start(dlg)
    print(f"{tag}: started on {n} files at {time.strftime('%H:%M:%S')}", flush=True)


def wait(tag: str, quiet_minutes: float, poll_seconds: float, started_after: float | None) -> None:
    """Block until every shard of the level has been rewritten (mtime after the
    run started) and no shard has changed for ``quiet_minutes``."""
    folder = FLAT / tag
    files = sorted(folder.glob("*.xg"))
    t0 = started_after if started_after is not None else time.time()
    last_change = time.time()
    seen: dict[Path, float] = {f: f.stat().st_mtime for f in files}
    while True:
        changed = 0
        for f in files:
            m = f.stat().st_mtime
            if m != seen[f]:
                seen[f] = m; changed += 1
        if changed:
            last_change = time.time()
        done = sum(1 for f in files if seen[f] > t0)
        quiet = (time.time() - last_change) / 60
        print(f"{time.strftime('%H:%M:%S')} {tag}: {done}/{len(files)} shards rewritten, "
              f"quiet {quiet:.1f} min", flush=True)
        if done == len(files) and quiet >= quiet_minutes:
            print(f"{tag}: complete"); return
        time.sleep(poll_seconds)


def score(tag: str) -> None:
    cmd = [sys.executable, str(_SCRIPT_DIR / "export_folder_benchmark_xg.py"), "score",
           "--xg-dir", str(FLAT / tag), "--level", tag]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(_PROJECT_ROOT)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stamp", help="flat per-level copies of the generated shards")
    s.add_argument("--levels", nargs="+", default=list(LEVELS))
    s.add_argument("--force", action="store_true")
    for name in ("analyze", "wait", "score"):
        p = sub.add_parser(name)
        p.add_argument("--level", required=True, choices=list(LEVELS))
        if name == "analyze":
            p.add_argument("--no-start", action="store_true")
        if name == "wait":
            p.add_argument("--quiet-minutes", type=float, default=5.0)
            p.add_argument("--poll-seconds", type=float, default=60.0)
            p.add_argument("--started-after", type=float, default=None,
                           help="epoch seconds the run started (default: now)")
    args = ap.parse_args(argv)
    if args.cmd == "stamp":
        stamp(args.levels, args.force)
    elif args.cmd == "analyze":
        analyze(args.level, args.no_start)
    elif args.cmd == "wait":
        wait(args.level, args.quiet_minutes, args.poll_seconds, args.started_after)
    else:
        score(args.level)


if __name__ == "__main__":
    main()
