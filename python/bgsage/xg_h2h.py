# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Optional XG game policy for the standard recorded simulator."""

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import csv
from hashlib import sha256
import os
from pathlib import Path
from queue import Empty, Queue
from random import Random
import subprocess
from tempfile import TemporaryDirectory
from threading import Lock, Thread, local
from time import perf_counter
from types import SimpleNamespace

from . import _xg_shard
from .board import STARTING_BOARD, check_game_over, flip_board, possible_moves


XG_LEVELS = {"1ply": 0, "2ply": 1, "3ply": 2, "4ply": 3,
             "truncated1": 1000, "truncated2": 1001, "truncated3": 1002}
_PROFILE_KEY = r"Software\GameSite 2000\eXtreme Gammon 2\Analzye"
_PROFILE_SETTINGS = ("WithProfile", "WithoutProfile", "Imported", "Computers")
_DEFAULT_EXE = Path(r"C:\Program Files (x86)\eXtreme Gammon 2\eXtremeGammon2.exe")
_DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "data/money_benchmark/xg/seed_2.xg"


def _profile_stages(definition):
    checker, cube = definition.split("::")
    name, *checker_levels = checker.split(":")
    kinds = {}
    for kind, fields in (("move", checker_levels), ("cube", cube.split(":"))):
        first, second = map(int, fields[:2])
        levels = (first,) if second == -1 else (first, second)
        if any(level not in XG_LEVELS.values() for level in levels):
            raise ValueError("Unsupported XG profile definition")
        kinds[kind] = levels
    return name, kinds


def _select_profile(slots, level, name):
    if sorted(slots) != list(range(1, len(slots) + 1)):
        raise ValueError("XG custom profiles must occupy contiguous slots")
    matches = []
    wanted = XG_LEVELS[level]
    for slot, definition in slots.items():
        if name and definition.split(":", 1)[0] != name:
            continue
        try:
            label, stages = _profile_stages(definition)
        except ValueError:
            if name:
                raise
            continue
        if all(max(kind) == wanted for kind in stages.values()):
            matches.append({"name": label, "index": slot + 5, "slot": slot,
                            "definition": definition, "stages": stages})
    if len(matches) != 1:
        raise ValueError(f"Choose one existing XG custom profile for {level} checker and cube play "
                         "with --xg-profile. Create that profile in XG if needed.")
    return matches[0]


def _check_level(answer, item, stages):
    configured = stages[item["kind"]]
    wanted = max(configured)
    actual = answer["level"]
    requested = answer["level_request"]
    if actual != wanted and requested not in (*configured, actual):
        raise ValueError("XG returned inconsistent requested and analyzed levels")
    if item["kind"] == "move" and actual in (998, 999):
        return
    depth = 2 if actual == 12 else actual
    if depth == wanted or 0 <= depth <= min(wanted, 3):
        return
    raise ValueError(f"XG returned level {actual} for a request at level {wanted}")


def _position(board, cube_value, cube_owner, max_cube):
    return {"pre": list(board), "cube_value": cube_value, "cube_owner": cube_owner,
            "max_cube_value": max_cube}


def _move_item(position, dice, legal):
    post = legal[0]
    half_moves = _xg_shard.derive_half_moves(position["pre"], post, *dice)
    if half_moves is None:
        raise ValueError("Could not encode a legal placeholder for XG")
    return dict(position, kind="move", dice=dice, post=post, half_moves=half_moves)


def _calibration_items(max_cube):
    rng = Random(29192026)
    items = []
    for index in range(28):
        board = list(STARTING_BOARD)
        for _ in range(rng.randrange(3, 22)):
            post = rng.choice(possible_moves(board, rng.randrange(1, 7), rng.randrange(1, 7)))
            if check_game_over(post):
                break
            board = flip_board(post)
        cube_value = 2 if index % 4 == 0 else 1
        owner = "player" if cube_value == 2 else "centered"
        position = _position(board, cube_value, owner, max_cube)
        if cube_value < max_cube:
            items.append(dict(position, kind="cube"))
        dice = (rng.randrange(1, 7), rng.randrange(1, 7))
        legal = possible_moves(board, *dice)
        if len(legal) > 1:
            items.append(_move_item(position, dice, legal))
    opening = _move_item(_position(STARTING_BOARD, 1, "centered", max_cube),
                         (6, 4), possible_moves(STARTING_BOARD, 6, 4))
    return items + [opening]


def _xg_is_running(executable):
    result = subprocess.run(["tasklist.exe", "/FI", f"IMAGENAME eq {executable.name}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, timeout=15, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return any(row and row[0].lower() == executable.name.lower()
               for row in csv.reader(result.stdout.splitlines()))


class _XgBatch:
    def __init__(self, process, windows, registry, profile, template, path, max_cube):
        self.process = process
        self.windows = windows
        self.registry = registry
        self.profile = profile
        self.template = template
        self.tpl = _xg_shard.templates(template)
        self.path = path
        self.max_cube = max_cube
        self.sentinels = []
        self.rounds = 0
        self.seconds = 0.0

    def _verify_profile(self):
        import winreg

        expected = {name: self.profile["index"] for name in _PROFILE_SETTINGS}
        expected.update(useOpeningBook=1)
        expected[str(self.profile["slot"])] = self.profile["definition"]
        if any(winreg.QueryValueEx(self.registry, key)[0] != value for key, value in expected.items()):
            raise RuntimeError("XG analysis profile or opening book changed during the run")

    def analyze(self, requests):
        items = requests + [item for item, _ in self.sentinels]
        self._verify_profile()
        _xg_shard.write_shard(items, self.path, self.template, tpl=self.tpl, max_cube=self.max_cube)
        start = perf_counter()
        self.windows.run_batch(self.process.pid, self.path, self.profile["name"])
        self.seconds += perf_counter() - start
        self.rounds += 1
        answers = _xg_shard.harvest(self.path, items, self.max_cube)
        self._verify_profile()
        for answer, item in zip(answers, items, strict=True):
            _check_level(answer, item, self.profile["stages"])
        for answer, (_, expected) in zip(answers[len(requests):], self.sentinels, strict=True):
            if answer["level"] != expected:
                raise RuntimeError("An XG batch calibration position changed level")
        return answers[:len(requests)]

    def calibrate(self):
        items = _calibration_items(self.max_cube)
        answers = self.analyze(items)
        for kind, levels in self.profile["stages"].items():
            wanted = max(levels)
            match = next((item for item, answer in zip(items, answers, strict=True)
                          if item["kind"] == kind and answer["level"] == wanted), None)
            if match is None:
                raise RuntimeError(f"XG calibration did not reach the requested {kind} level {wanted}")
            self.sentinels.append((match, wanted))
        if answers[-1]["level"] not in (998, 999):
            raise RuntimeError("XG calibration did not use the opening book")
        self.sentinels.append((items[-1], answers[-1]["level"]))


@contextmanager
def _session(level, profile_name, executable, template, max_cube):
    if os.name != "nt":
        raise ValueError("XG head-to-head requires the Windows desktop version of eXtreme Gammon 2")
    import winreg
    from . import _xg_windows as windows

    executable = Path(executable or _DEFAULT_EXE).resolve(strict=True)
    template = Path(template or _DEFAULT_TEMPLATE)
    if not template.is_file():
        raise FileNotFoundError("Supply an XG game containing checker and cube records with --xg-template")
    template = template.resolve()
    required = {_xg_shard.xf.TS_HEADER_MATCH, _xg_shard.xf.TS_HEADER_GAME,
                _xg_shard.xf.TS_CUBE, _xg_shard.xf.TS_MOVE, _xg_shard.xf.TS_FOOTER_GAME}
    if not required <= _xg_shard.templates(template).keys():
        raise ValueError("The XG template needs a complete game with checker and cube records")

    with windows.lease(), TemporaryDirectory(prefix="bgsage-xg-") as directory:
        if _xg_is_running(executable) or windows.find_main():
            raise RuntimeError("Close the open XG instance before starting a dedicated head-to-head run")
        access = winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _PROFILE_KEY, 0, access) as registry:
            slots = {}
            for slot in range(1, 6):
                try:
                    slots[slot] = winreg.QueryValueEx(registry, str(slot))[0]
                except FileNotFoundError:
                    pass
            profile = _select_profile(slots, level, profile_name)
            saved = {key: winreg.QueryValueEx(registry, key) for key in (*_PROFILE_SETTINGS, "useOpeningBook")}
            process = None
            try:
                for key in _PROFILE_SETTINGS:
                    winreg.SetValueEx(registry, key, 0, winreg.REG_DWORD, profile["index"])
                winreg.SetValueEx(registry, "useOpeningBook", 0, winreg.REG_DWORD, 1)
                process = subprocess.Popen([str(executable)], cwd=executable.parent)
                windows.ready(process)
                oracle = _XgBatch(process, windows, registry, profile, template,
                                  Path(directory) / "positions.xg", max_cube)
                oracle.calibrate()
                info = {"executable": str(executable), "executable_sha256": sha256(executable.read_bytes()).hexdigest(),
                        "template_sha256": sha256(template.read_bytes()).hexdigest(),
                        "profile": profile, "opening_book": True}
                yield oracle, info
            finally:
                try:
                    if process is not None:
                        windows.close(process)
                finally:
                    for key, (value, kind) in saved.items():
                        winreg.SetValueEx(registry, key, 0, kind, value)


class _QueuedPolicy:
    """Allow multiple game streams to share the one installed XG process."""

    def __init__(self, batch):
        self.batch = batch
        self.queue = Queue()
        self.lock = Lock()
        self.failure = None

    def _request(self, item):
        future = Future()
        with self.lock:
            if self.failure is not None:
                raise self.failure
            self.queue.put((item, future))
        return future.result()

    def checker_play(self, board, die1, die2, cube_value, cube_owner, *, max_cube_value, **rules):
        dice = (die1, die2)
        legal = possible_moves(board, *dice)
        if len(legal) == 1:
            chosen = legal[0]
        else:
            pos = _position(board, cube_value, cube_owner, max_cube_value)
            chosen = self._request(_move_item(pos, dice, legal))["board"]
        return SimpleNamespace(moves=[SimpleNamespace(board=chosen)])

    def cube_action(self, board, cube_value, cube_owner, *, max_cube_value, **rules):
        pos = _position(board, cube_value, cube_owner, max_cube_value)
        answer = self._request(dict(pos, kind="cube"))
        return SimpleNamespace(should_double=answer["should_double"], should_take=answer["should_take"])

    def serve(self):
        while (first := self.queue.get()) is not None:
            batch = [first]
            end = perf_counter() + 0.05
            while len(batch) < 128:
                try:
                    batch.append(self.queue.get(timeout=max(0, end - perf_counter())))
                except Empty:
                    break
            try:
                answers = self.batch.analyze([item for item, _ in batch])
                for (_, future), answer in zip(batch, answers, strict=True):
                    future.set_result(answer)
            except Exception as error:
                with self.lock:
                    self.failure = error
                    while not self.queue.empty():
                        pending = self.queue.get_nowait()
                        if pending is not None:
                            batch.append(pending)
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                return


def play_xg_games(seeds, workers, runner_factory, parallel_games, *,
                  level, profile, executable, template, max_cube):
    """Play standard games with XG pending requests coalesced into batches."""
    with _session(level, profile, executable, template, max_cube) as (batch, info):
        xg = _QueuedPolicy(batch)
        service = Thread(target=xg.serve, name="xg-batch", daemon=True)
        service.start()
        thread = local()

        def play(seed):
            if not hasattr(thread, "runner"):
                thread.runner = runner_factory(xg)
            return thread.runner.play(seed)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = parallel_games(pool, seeds, workers, play=play)
        finally:
            xg.queue.put(None)
            service.join()
        info.update(batch_count=batch.rounds, batch_seconds=batch.seconds)
        return records, info
