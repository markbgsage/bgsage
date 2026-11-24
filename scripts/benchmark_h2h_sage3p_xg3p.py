# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Replay the fixed 1,000-game Sage 3P vs XG 3-ply benchmark cohort."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from threading import Thread, local
from time import monotonic, thread_time

from bgsage import BgBotAnalyzer
from bgsage import h2h
from bgsage.simulation import GameConfig
from bgsage.xg_h2h import _QueuedPolicy, _session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "logs" / "sage-3p-vs-xg-3p-1000-replay"
SEED = 2026112342
COUNT = 1000
STREAMS = int(os.environ.get("BGSAGE_XG_GAME_STREAMS", "80"))
SAGE_WORKERS = int(os.environ.get("BGSAGE_XG_SAGE_WORKERS", "12"))
PROFILE = "Sage3P"
CONFIG = GameConfig(max_cube_value=64)
START = monotonic()


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def write_json(path, value):
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


class Oracle:
    def __init__(self, db):
        self.db = db
        self.stack = ExitStack()
        self.current = None
        self.info = None
        self.next_recycle = 125

    def start(self):
        self.current, info = self.stack.enter_context(
            _session("3ply", PROFILE, None, None, CONFIG.max_cube_value)
        )
        if self.info is not None and self.info != info:
            raise RuntimeError("XG executable, template or profile changed between sessions")
        self.info = info
        with sqlite3.connect(self.db) as connection:
            previous = connection.execute("select value from metadata where key = 'xg'").fetchone()
            if previous and json.loads(previous[0]) != json.loads(json.dumps(info)):
                raise RuntimeError("XG identity differs from the saved cohort")
            connection.execute("insert or ignore into metadata values ('xg', ?)", (json.dumps(info),))
        log(f"XG PID {self.current.process.pid} calibrated at checker and cube 3-ply with book enabled")

    def stop(self):
        try:
            self.stack.close()
        finally:
            self.current = None

    def analyze(self, requests):
        for attempt in range(3):
            try:
                if self.current is not None and self.current.rounds >= self.next_recycle:
                    self.stop()
                if self.current is None:
                    self.start()
                begin = monotonic()
                results = self.current.analyze(requests)
                seconds = monotonic() - begin
                with sqlite3.connect(self.db) as connection:
                    connection.execute("insert into batches (items, seconds) values (?, ?)",
                                       (len(requests), seconds))
                if self.current.rounds % 20 == 0:
                    log(f"XG PID {self.current.process.pid} finished batch {self.current.rounds}; "
                        f"{len(requests)} decisions in {seconds:.2f}s")
                return results
            except (OSError, ValueError, RuntimeError) as error:
                log(f"XG attempt {attempt + 1} failed: {type(error).__name__}: {error}")
                self.stop()
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")


class SagePool:
    def __init__(self, executor):
        self.executor = executor
        self.native = local()
        self.game = local()

    def _execute(self, operation, args, kwargs):
        if not hasattr(self.native, "player"):
            self.native.player = BgBotAnalyzer(eval_level="3ply", parallel_threads=1)
            self.native.scorer = h2h._LuckScorer(BgBotAnalyzer(eval_level="2ply", parallel_threads=1), CONFIG)
        target = self.native.scorer if operation == "score" else self.native.player
        before = thread_time()
        value = getattr(target, operation)(*args, **kwargs)
        return value, thread_time() - before

    def call(self, operation, *args, **kwargs):
        value, seconds = self.executor.submit(self._execute, operation, args, kwargs).result()
        self.game.native_seconds = getattr(self.game, "native_seconds", 0.0) + seconds
        return value

    def checker_play(self, *args, **kwargs):
        return self.call("checker_play", *args, **kwargs)

    def cube_action(self, *args, **kwargs):
        return self.call("cube_action", *args, **kwargs)

    def score(self, *args):
        return self.call("score", *args)

    def cpu_clock(self):
        return thread_time() + getattr(self.game, "native_seconds", 0.0)


class Runner(h2h._GameRunner):
    def __init__(self, sage, xg):
        self.players = (sage, xg)
        self.scorer = sage
        self.config = CONFIG
        self.beta = 1.0
        self.cpu_clock = sage.cpu_clock


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "result.json"
    if result_path.is_file():
        log(f"This cohort is already complete at {result_path}")
        return

    db = OUTPUT_DIR / "checkpoint.sqlite"
    seeds = list(h2h._game_seeds(SEED, COUNT))
    seed_set = set(seeds)
    manifest = {
        "schema_version": 1, "model": h2h.PRODUCTION_MODEL,
        "engine_sha256": sha256(Path(h2h.bgbot_cpp.__file__).read_bytes()).hexdigest(),
        "player_a": {"bot": "sage", "eval_level": "3ply"},
        "player_b": {"bot": "xg", "eval_level": "3ply"},
        "reference": "sage_2ply_nd", "beta": 1.0, "seed": SEED,
        "rules": {"jacoby": True, "beaver": False, "max_cube_value": CONFIG.max_cube_value,
                  "max_turns": CONFIG.max_turns},
        "game_count": COUNT, "xg_profile": PROFILE,
    }
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            create table if not exists metadata (key text primary key, value text not null);
            create table if not exists games (seed integer primary key, record text not null);
            create table if not exists batches (items integer not null, seconds real not null);
        """)
        prior = connection.execute("select value from metadata where key = 'manifest'").fetchone()
        if prior and json.loads(prior[0]) != manifest:
            raise RuntimeError("Manifest differs from the saved cohort")
        connection.execute("insert or ignore into metadata values ('manifest', ?)", (json.dumps(manifest),))
        finished = {seed for (seed,) in connection.execute("select seed from games")}
    if not finished <= seed_set:
        raise RuntimeError("Saved records include seeds outside the planned cohort")
    log(f"Starting {COUNT - len(finished)} pending games, {len(finished)} committed; "
        f"{STREAMS} game streams and {SAGE_WORKERS} Sage compute workers")

    if finished != seed_set:
        oracle = Oracle(db)
        xg = _QueuedPolicy(oracle)

        def serve():
            try:
                xg.serve()
            finally:
                oracle.stop()

        service = Thread(target=serve, name="xg-batch")
        service.start()
        try:
            with ThreadPoolExecutor(max_workers=SAGE_WORKERS, thread_name_prefix="Sage") as compute:
                sage = SagePool(compute)
                runner = Runner(sage, xg)

                def play(seed):
                    record = runner.play(seed)
                    with sqlite3.connect(db) as connection:
                        connection.execute("insert into games values (?, ?)", (seed, json.dumps(record)))
                    return record

                with ThreadPoolExecutor(max_workers=STREAMS, thread_name_prefix="Game") as streams:
                    h2h._parallel_games(streams, (seed for seed in seeds if seed not in finished), STREAMS, play=play)
        finally:
            xg.queue.put(None)
            service.join()

    with sqlite3.connect(db) as connection:
        records = {seed: json.loads(record) for seed, record in connection.execute("select * from games")}
        info = json.loads(connection.execute("select value from metadata where key = 'xg'").fetchone()[0])
        rounds, seconds, items = connection.execute("select count(*), sum(seconds), sum(items) from batches").fetchone()
    if records.keys() != seed_set:
        raise RuntimeError("Missing or unexpected completed game seeds")
    ordered = [records[seed] for seed in seeds]
    info.update(batch_count=rounds, batch_seconds=seconds, decision_count=items)
    report = dict(manifest, games=ordered, workers=STREAMS, sage_compute_workers=SAGE_WORKERS,
                  xg=info, wall_seconds=monotonic() - START,
                  cpu_timing_scope="H2H game threads and Sage engine workers",
                  summary=h2h._summarize(ordered, external_bot=True))
    write_json(result_path, report)
    log(f"Finished all {len(ordered)} games in this cohort and wrote result.json")


if __name__ == "__main__":
    main()
