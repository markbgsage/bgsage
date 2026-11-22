# SPDX-License-Identifier: MPL-2.0
"""Time public 3-ply calls and attribute wall time to their Python/native phases.

Run each experiment in a fresh process. Per-worker elapsed times may overlap.
The saved public baseline allows full output parity to be checked per position.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from benchmark_multi_ply import APIS, DATA, ROOT, file_identity, inside_repo, load_engine, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", required=True, type=Path)
    p.add_argument("--python-dir", type=Path, default=ROOT / "python")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--apis", nargs="+", choices=APIS, default=list(APIS))
    p.add_argument("--per-cell", type=int, choices=range(1, 11), default=10)
    p.add_argument("--variant", choices=("normal", "skip-cubeless", "reuse-workers",
                                         "skip-and-reuse", "split-by-roll", "no-post-clear"),
                   default="normal", help="Experiments on the original analyzer")
    args = p.parse_args()
    output = inside_repo(args.output)
    if output.exists():
        p.error("Output already exists")
    engine, extension = load_engine(args.extension, args.python_dir)
    import bgsage.analyzer as az
    azr = engine.BgBotAnalyzer(eval_level="3ply", parallel_threads=args.threads)
    inner = azr._analyzer._inner
    shared = None
    if args.variant in ("reuse-workers", "skip-and-reuse"):
        shared = ThreadPoolExecutor(max_workers=args.threads)
        class BorrowPool:
            def __init__(self, **unused): pass
            def __enter__(self): return shared
            def __exit__(self, *unused): pass
        az.ThreadPoolExecutor = BorrowPool
    if args.variant in ("skip-cubeless", "skip-and-reuse", "split-by-roll"):
        def generate(board, d1, d2, *unused, **kwargs):
            return [{"board": b} for b in az.bgbot_cpp.possible_moves(board, d1, d2)]
        inner.checker_play_analytics = generate
    if args.variant == "split-by-roll":
        class SerialMap:
            def __init__(self, **unused): pass
            def __enter__(self): return self
            def __exit__(self, *unused): pass
            def map(self, fn, items): return map(fn, items)
        az.ThreadPoolExecutor = SerialMap
        old_convert = az.bgbot_cpp.cubeful_probs_and_equity_nply
        def split(*args_, **kw):
            kw["n_threads"] = args.threads
            return old_convert(*args_, **kw)
        az.bgbot_cpp.cubeful_probs_and_equity_nply = split
    if args.variant == "no-post-clear":
        if args.apis != ["post_move_analytics"]:
            p.error("no-post-clear requires --apis post_move_analytics")
        class NoClear:
            def clear_cache(self): pass
        inner._strategy_nply = NoClear()
    expected = {}
    for line in (DATA / "baseline_stage11/results.jsonl").read_bytes().splitlines():
        record = json.loads(line)
        if record["level"] == "3ply":
            expected[record["api"], record["position_id"]] = record["analytics"]
    phase_times = defaultdict(float)
    phase_calls = Counter()
    def meter(obj, name, label):
        old = getattr(obj, name)
        def measured(*a, **kw):
            start = time.perf_counter()
            ret = old(*a, **kw)
            phase_times[label] += time.perf_counter() - start
            phase_calls[label] += 1
            return ret
        setattr(obj, name, measured)
    meter(inner, "checker_play_analytics", "initial_checker_pass")
    for name in ("possible_moves", "cube_decision_nply_unified", "cubeful_probs_nply",
                 "cubeful_probs_and_equity_nply"):
        meter(az.bgbot_cpp, name, name)
    rows = []
    for api in args.apis:
        ds = "checker_play" if api == "checker_play" else "cube_action"
        for pos in json.loads((DATA / f"{ds}.json").read_bytes())["positions"]:
            if int(pos["id"].split("-")[-1]) > args.per_cell:
                continue
            inputs = pos["post_move_inputs" if api == "post_move_analytics" else "inputs"]
            start = time.perf_counter()
            result = getattr(azr, api)(**inputs)
            elapsed = time.perf_counter() - start
            actual = asdict(result)
            rows.append({"api": api, "id": pos["id"], "elapsed": elapsed,
                         "exact": actual == expected[api, pos["id"]],
                         "analytics_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()})
    if shared:
        shared.shutdown()
    totals = {api: sum(row["elapsed"] for row in rows if row["api"] == api) for api in args.apis}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"variant": args.variant, "threads": args.threads,
                        "extension": file_identity(extension),
                        "analyzer": file_identity(Path(az.__file__)),
                        "public_seconds": totals, "phase_seconds": dict(phase_times),
                        "phase_calls": dict(phase_calls), "rows": rows})
    print(totals, dict(phase_times), "exact", sum(r["exact"] for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
