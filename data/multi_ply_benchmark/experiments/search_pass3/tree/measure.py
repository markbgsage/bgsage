# SPDX-License-Identifier: MPL-2.0
"""Run the paired tree benchmark with this pass's revision and work counters."""
from pathlib import Path
import json
import multiprocessing as mp
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import ROOT, BASE

runner = runpy.run_path(
    str(HERE.parents[1] / "search_pass2" / "tree" / "measure.py"), run_name="tree_screen"
)
counters = ("base_requests", "base_misses", "delta_requests", "delta_hits",
            "reserved4", "reserved5", "reserved6", "reserved7", "reserved8")


def loaded_worker(*args):
    runner["worker"].__globals__["COUNTERS"] = counters
    runner["worker"](*args)


def worker(conn, *args):
    if "BGSAGE_TREE_COLD" not in os.environ:
        loaded_worker(conn, *args)
        return
    # The outer runner measures only the analytics call in the inner process.
    ctx = mp.get_context("spawn")
    while (req := conn.recv()) is not None:
        outer, inner = ctx.Pipe()
        proc = ctx.Process(target=loaded_worker, args=(inner, *args))
        proc.start()
        inner.close()
        outer.send(req)
        conn.send(outer.recv())
        outer.send(None)
        outer.close()
        proc.join()
        if proc.exitcode:
            raise RuntimeError(proc.exitcode)


if __name__ == "__main__":
    cold = "--cold" in sys.argv
    if cold:
        sys.argv.remove("--cold")
        os.environ["BGSAGE_TREE_COLD"] = "1"
    runner["main"].__globals__["worker"] = worker
    runner["main"]()
    if cold:
        path = Path(sys.argv[sys.argv.index("--output") + 1]) / "measurements.json"
        data = json.loads(path.read_bytes())
        data["fresh_process_per_call"] = True
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
