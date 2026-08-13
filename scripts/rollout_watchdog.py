# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Watchdog wrapper: keep restarting aggregate_xg_pr.py until it exits clean.

The rollout script is fully resumable via its JSONL output, so if the
heavy rollout python process dies for any reason (Windows process cleanup,
session signal, transient bgbot_cpp issue), the watchdog restarts it and
the new run picks up where the previous left off — wasted compute is
limited to the in-flight rollout at death time.

Run via the standard Start-Process detach. Each restart appends a banner
to the same log so you can see when restarts happened.

Usage (from the bgsage repo root):
    python scripts/rollout_watchdog.py --rollout-threads 32

All args after the script name are forwarded to aggregate_xg_pr.py.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET = _HERE / "aggregate_xg_pr.py"
_BACKOFF_SEC = 30
_MAX_RESTARTS = 200   # safety cap; ~10s of restarts is enough for any reasonable run


def main(argv: list[str]) -> int:
    args = [sys.executable, "-u", str(_TARGET), *argv]
    for attempt in range(1, _MAX_RESTARTS + 1):
        banner = f"=== watchdog attempt {attempt} at {_dt.datetime.now().isoformat()} ==="
        print(banner, flush=True)
        try:
            result = subprocess.run(args)
            code = result.returncode
        except KeyboardInterrupt:
            print("=== watchdog: interrupted ===", flush=True)
            return 130
        print(f"=== watchdog: child exited with code {code} ===", flush=True)
        if code == 0:
            return 0
        time.sleep(_BACKOFF_SEC)
    print("=== watchdog: hit max restarts; giving up ===", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
