#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Run the remaining XG folder-benchmark levels back to back, unattended.

``xg_folder_batch.py`` does one level at a time and each level takes hours, so
finishing a run means sitting at the machine to start the next one. This chains
them: wait out the level XG is analysing now, score it, then for every level
still to do drive Batch Analyze, wait, and score -- logging each step so an
unattended run can be read back afterwards.

Two things it adds beyond the per-level commands, both learned the hard way:

* **A stall guard.** A level that stops progressing (XG crashed, a modal dialog
  nobody dismissed, the machine slept) would otherwise hold the chain forever
  and silently cost a night; here it aborts loudly, leaving the analysed shards
  on disk so the level can be resumed by hand.

* **It waits for an unlocked desktop.** XG keeps *analysing* while the
  workstation is locked, but ``xg_batch_win.py`` drives the GUI with real mouse
  input, and Windows refuses that when the input desktop is the lock screen
  ("There is no active desktop required for moving mouse cursor"). That is not
  an error to retry a couple of times and give up on -- it is a wait for the
  person to come back -- so a locked desktop costs no retry and the chain simply
  holds until the session is interactive again.

    py -3.14 scripts/xg_folder_chain.py --running xgrollerpp --then xgroller xgrollerplus
    py -3.14 scripts/xg_folder_chain.py --then xgroller          # nothing running now
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from xg_folder_batch import FLAT, LEVELS  # noqa: E402

#: pywinauto's set_focus() raises this when the input desktop is the lock screen.
_LOCKED_MARKERS = ("no active desktop", "OpenInputDesktop", "SetCursorPos")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def desktop_interactive() -> bool:
    """Whether synthetic mouse input works RIGHT NOW.

    This probes the capability pywinauto actually needs -- it moves the cursor
    to where it already is, which no one can see -- rather than inferring it
    from lock state. Inference was wrong in both directions on this machine:
    ``GetForegroundWindow()`` returns 0 on a session that still accepts input,
    and returns non-zero when the lock screen briefly wakes, which burned an
    analyze attempt (and left a dialog behind) every time the screen flickered.
    The input desktop's NAME is no better: it reads "Default", not "Winlogon",
    while ``SetCursorPos`` is still refused.
    """
    try:
        u = ctypes.windll.user32
        pt = wintypes.POINT()
        if not u.GetCursorPos(ctypes.byref(pt)):
            return False
        return bool(u.SetCursorPos(pt.x, pt.y))
    except Exception:
        return True                       # can't tell: let the attempt decide


def _steady(checks: int, gap: float) -> bool:
    """True only if the input probe passes ``checks`` times ``gap`` apart."""
    for i in range(checks):
        if not desktop_interactive():
            return False
        if i < checks - 1:
            time.sleep(gap)
    return True


def wait_for_desktop(max_hours: float, poll_seconds: float = 60.0) -> bool:
    """Hold until the workstation is unlocked. Returns False on giving up."""
    if desktop_interactive():
        return True
    log("desktop is LOCKED -- XG keeps analysing, but its GUI cannot be driven; "
        f"holding for up to {max_hours:.0f} h until the session is unlocked")
    deadline = time.time() + max_hours * 3600
    announced = time.time()
    while time.time() < deadline:
        time.sleep(poll_seconds)
        # A sleeping machine flickers: input is accepted for a few seconds at a
        # time (a monitor waking, the cursor being parked) and refused again
        # before an attempt can finish. One passing probe is therefore not
        # evidence the session is usable -- demand that it hold steady, or the
        # chain fires an attempt into a closing window and leaves XG holding a
        # half-built dialog every time the screen twitches.
        if _steady(4, 5.0):
            log("desktop is available again -- resuming")
            return True
        if time.time() - announced >= 1800:
            log(f"still waiting for an unlocked desktop "
                f"({(deadline - time.time()) / 3600:.1f} h left)")
            announced = time.time()
    log(f"gave up waiting for an unlocked desktop after {max_hours:.0f} h")
    return False


def tidy_xg() -> None:
    """Close anything left over the XG main window, so Batch Analyze can open."""
    try:
        import xg_batch_win as win
        win.dismiss_message()
        win.close_open_dialog()      # modal leftover from a killed attempt
        win.close_batch_dialog()
    except Exception as e:                # never let cleanup end the chain
        log(f"(cleanup skipped: {type(e).__name__}: {e})")


def wait_level(tag: str, started_after: float, quiet_minutes: float,
               stall_minutes: float, poll_seconds: float) -> bool:
    """Block until every shard of ``tag`` has been rewritten since
    ``started_after`` and nothing has changed for ``quiet_minutes``.

    Returns False if the level stalled -- no shard changed for ``stall_minutes``
    while shards were still outstanding, which is XG having stopped rather than
    finished, and is the case that must not hang a chain running overnight.
    """
    folder = FLAT / tag
    files = sorted(folder.glob("*.xg"))
    if not files:
        log(f"{tag}: no shards in {folder}"); return False
    seen = {f: f.stat().st_mtime for f in files}
    last_change = time.time()
    while True:
        changed = 0
        for f in files:
            try:
                m = f.stat().st_mtime
            except OSError:
                continue                  # XG rewrites in place; a brief miss is normal
            if m != seen[f]:
                seen[f] = m; changed += 1
        if changed:
            last_change = time.time()
        done = sum(1 for f in files if seen[f] > started_after)
        quiet = (time.time() - last_change) / 60
        log(f"{tag}: {done}/{len(files)} shards rewritten, quiet {quiet:.1f} min")
        if done == len(files) and quiet >= quiet_minutes:
            log(f"{tag}: complete"); return True
        if quiet >= stall_minutes:
            log(f"{tag}: STALLED -- {len(files) - done} shards outstanding and nothing "
                f"has changed for {quiet:.0f} min; abandoning the chain")
            return False
        time.sleep(poll_seconds)


def run(*args: str) -> tuple[int, str]:
    """Run an xg_folder_batch subcommand; return (exit code, combined output)."""
    cmd = [sys.executable, str(_SCRIPT_DIR / "xg_folder_batch.py"), *args]
    log("$ " + " ".join(cmd[1:]))
    p = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.stdout:
        print(p.stdout, end="", flush=True)
    return p.returncode, p.stdout or ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--running", choices=list(LEVELS), default=None,
                    help="a level XG is analysing right now: wait it out and score it first")
    ap.add_argument("--running-started", type=float, default=None,
                    help="epoch seconds --running began (default: 6 h ago)")
    ap.add_argument("--then", nargs="*", default=[], choices=list(LEVELS),
                    help="levels to analyse, in order, once the running one is done")
    ap.add_argument("--quiet-minutes", type=float, default=8.0)
    ap.add_argument("--stall-minutes", type=float, default=75.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--analyze-retries", type=int, default=2,
                    help="retries for a REAL failure; a locked desktop costs none")
    ap.add_argument("--desktop-wait-hours", type=float, default=24.0)
    args = ap.parse_args(argv)

    if args.running:
        t0 = args.running_started if args.running_started is not None else time.time() - 6 * 3600
        log(f"waiting out the level already running: {args.running}")
        if not wait_level(args.running, t0, args.quiet_minutes,
                          args.stall_minutes, args.poll_seconds):
            raise SystemExit(2)
        if run("score", "--level", args.running)[0] != 0:
            log(f"{args.running}: SCORING FAILED (the analysed shards are on disk; "
                f"re-run `score --level {args.running}`) -- continuing")

    for tag in args.then:
        attempt = 0
        lock_fails = 0
        while True:
            if not wait_for_desktop(args.desktop_wait_hours, args.poll_seconds):
                raise SystemExit(4)
            tidy_xg()
            started = time.time() - 5     # a shard may land before analyze() returns
            rc, out = run("analyze", "--level", tag)
            if rc == 0:
                break
            if any(m in out for m in _LOCKED_MARKERS) or not desktop_interactive():
                # A sleeping machine can pass the probe and refuse input again
                # inside the ~15 s an attempt needs, so this can repeat forever.
                # Back off rather than retrying every poll: an attempt calls
                # set_focus() on XG's window, and hammering that every 90 s for
                # hours would fight a person who IS at the keyboard.
                lock_fails += 1
                back = min(60 * 2 ** min(lock_fails, 5), 1800)
                log(f"{tag}: the desktop went away mid-attempt (not a retry); "
                    f"backing off {back // 60} min before probing again")
                time.sleep(back)
                continue
            attempt += 1
            log(f"{tag}: analyze failed (rc {rc}), attempt {attempt}")
            if attempt > args.analyze_retries:
                log(f"{tag}: giving up; abandoning the chain")
                raise SystemExit(3)
            time.sleep(60)
        if not wait_level(tag, started, args.quiet_minutes,
                          args.stall_minutes, args.poll_seconds):
            raise SystemExit(2)
        if run("score", "--level", tag)[0] != 0:
            log(f"{tag}: SCORING FAILED -- continuing")

    log("chain finished: " + ", ".join(filter(None, [args.running, *args.then])))


if __name__ == "__main__":
    main()
