# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Run Sage-vs-Sage money games (optionally in parallel) and write each game
as a Backgammon Galaxy / XG-import compatible .txt transcript.

Public entry point: ``run_sage_vs_sage_games(initial_seed, n_games, level,
workers=6, out_dir=None)``.

Each game is a money game with Jacoby + Beaver on, both sides played by Sage
at the specified eval level. Per-game transcripts go to
``<out_dir>/seed_<N>.txt`` (one file per seed). Default ``out_dir`` is
``bgsage/logs/sage_vs_sage`` (resolved from the script location — fully
self-contained within the bgsage repo).

Usage:

    python scripts/run_sage_vs_sage_games.py 1 30 --level 3P
    python scripts/run_sage_vs_sage_games.py 1 200 --level 3P --workers 6

When ``workers > 1``, games run in parallel via ``ProcessPoolExecutor``. Each
worker pre-loads its own analyzer at ``parallel_threads=1`` so 6 workers x 1
thread don't oversubscribe the host CPU.

Levels accepted: ``1P``, ``2P``, ``3P``, ``4P`` (N-ply), and ``1T``, ``2T``,
``3T`` (XG Roller / Roller+ / Roller++ truncated rollouts).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# bgsage path setup — fully self-contained within the bgsage repo. Never
# reaches into a parent project: weights, build artifacts, and log outputs
# all live under bgsage/.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = bgsage repo root
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"   # = bgsage/python
_BUILD_DIR = _PROJECT_ROOT / "build"        # = bgsage/build

for _p in (_BGSAGE_PYTHON, _BUILD_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))


_VALID_LEVELS = {"1P", "2P", "3P", "4P", "1T", "2T", "3T"}
_LEVEL_ALIASES: dict[str, str] = {
    "1P": "1ply", "2P": "2ply", "3P": "3ply", "4P": "4ply",
    "1T": "truncated1", "2T": "truncated2", "3T": "truncated3",
}
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "logs" / "sage_vs_sage"


def _resolve_level(level: str) -> str:
    canonical = _LEVEL_ALIASES.get(level.upper())
    if canonical is not None:
        return canonical
    if level in _LEVEL_ALIASES.values():
        return level
    raise ValueError(
        f"Unknown level {level!r}. Expected one of: "
        f"{', '.join(sorted(_LEVEL_ALIASES.keys()))}"
    )


# ---------------------------------------------------------------------------
# Per-process analyzer cache
# ---------------------------------------------------------------------------

_analyzer_cache: dict[str, object] = {}


def _make_analyzer(level: str, parallel_threads: int = 0):
    """Return (and cache) a BgBotAnalyzer for the canonical level string."""
    canonical = _resolve_level(level)
    cached = _analyzer_cache.get(canonical)
    if cached is not None:
        return cached
    from bgsage import BgBotAnalyzer

    analyzer = BgBotAnalyzer(
        eval_level=canonical, cubeful=True, parallel_threads=parallel_threads,
    )
    _analyzer_cache[canonical] = analyzer
    return analyzer


# ---------------------------------------------------------------------------
# Game model
# ---------------------------------------------------------------------------


@dataclass
class _Event:
    kind: str            # "cube_offer" | "cube_response" | "dice_roll" | "checker_play"
    player: int          # 1 or 2
    dice: tuple[int, int] | None = None
    decision: str | None = None         # cube_offer: "double"/"no_double"; cube_response: "take"/"pass"
    move_board_mover: list[int] | None = None  # post-move board, mover's perspective


@dataclass
class _GameRecord:
    events: list[_Event] = field(default_factory=list)
    winner: int | None = None
    win_type: str | None = None         # "single" | "gammon" | "backgammon"
    cube_at_end: int | None = None
    final_score: int | None = None      # signed, from P1's perspective


@dataclass
class _State:
    board: list[int]
    cube_value: int
    cube_owner: str   # "centered" | "player" | "opponent" (relative to active)
    active: int       # 1 or 2


def _initial_state() -> _State:
    from bgsage import STARTING_BOARD

    return _State(
        board=list(STARTING_BOARD),
        cube_value=1,
        cube_owner="centered",
        active=1,
    )


def _flip_state(state: _State) -> None:
    """Hand the turn to the next player; rotate board + cube perspective."""
    from bgsage import flip_board

    state.board = list(flip_board(state.board))
    if state.cube_owner == "player":
        state.cube_owner = "opponent"
    elif state.cube_owner == "opponent":
        state.cube_owner = "player"
    state.active = 3 - state.active


def _can_offer_cube(state: _State) -> bool:
    return state.cube_owner in ("centered", "player")


_WIN_TYPE_BY_MULT = {1: "single", 2: "gammon", 3: "backgammon"}


def _check_game_over(board: list[int]) -> tuple[str, int] | None:
    """bgsage.check_game_over returns ±1/±2/±3 from the mover's perspective.

    Returns ``(win_type, multiplier)`` if the mover has won (positive code),
    or ``None`` otherwise. We never reach a "loss" state during simulation
    since the mover is the one that just played: any terminal position must
    be a mover win (their last move bore off the final checker).
    """
    from bgsage import check_game_over

    code = check_game_over(board)
    if code > 0:
        return _WIN_TYPE_BY_MULT.get(code, "single"), code
    return None


def _record_game_over(record: _GameRecord, state: _State, win_type: str, mult: int) -> None:
    record.winner = state.active
    record.win_type = win_type
    record.cube_at_end = state.cube_value
    points = state.cube_value * mult
    record.final_score = points if state.active == 1 else -points


def _next_step(state: _State, events: list[_Event]) -> tuple[str, int]:
    """Return ``(kind, player)`` describing what event must come next."""
    last_checker_idx = -1
    for i in range(len(events) - 1, -1, -1):
        if events[i].kind == "checker_play":
            last_checker_idx = i
            break
    turn_events = events[last_checker_idx + 1:]

    has_roll = any(e.kind == "dice_roll" for e in turn_events)
    if has_roll:
        return ("checker_play", state.active)

    has_response = any(e.kind == "cube_response" for e in turn_events)
    if has_response:
        return ("dice_roll", state.active)

    has_offer = any(e.kind == "cube_offer" for e in turn_events)
    if has_offer:
        offer = next(e for e in turn_events if e.kind == "cube_offer")
        if offer.decision == "double":
            return ("cube_response", 3 - state.active)
        return ("dice_roll", state.active)

    if _can_offer_cube(state):
        return ("cube_offer", state.active)
    return ("dice_roll", state.active)


def _play(level: str, rng: random.Random, parallel_threads: int = 0) -> _GameRecord:
    """Play one Sage-vs-Sage money game (Jacoby + Beaver on)."""
    from bgsage import possible_moves

    analyzer = _make_analyzer(level, parallel_threads=parallel_threads)
    state = _initial_state()
    record = _GameRecord()

    MAX_EVENTS = 4 * 600  # 600 turns * 4 events each is plenty.
    for _ in range(MAX_EVENTS):
        kind, player = _next_step(state, record.events)

        if kind == "cube_offer":
            cube = analyzer.cube_action(
                state.board,
                cube_value=state.cube_value,
                cube_owner=state.cube_owner,
                jacoby=True,
                beaver=True,
            )
            decision = "double" if cube.should_double else "no_double"
            record.events.append(_Event("cube_offer", player, decision=decision))
            continue

        if kind == "cube_response":
            cube = analyzer.cube_action(
                state.board,
                cube_value=state.cube_value,
                cube_owner=state.cube_owner,
                jacoby=True,
                beaver=True,
            )
            decision = "take" if cube.should_take else "pass"
            record.events.append(_Event("cube_response", player, decision=decision))
            if decision == "pass":
                _record_game_over(record, state, "single", 1)
                return record
            state.cube_value *= 2
            state.cube_owner = "opponent"
            continue

        if kind == "dice_roll":
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            record.events.append(_Event("dice_roll", player, dice=(d1, d2)))
            continue

        if kind == "checker_play":
            roll = next(e for e in reversed(record.events) if e.kind == "dice_roll")
            d1, d2 = roll.dice
            cands = possible_moves(state.board, d1, d2)
            if not cands:
                post = list(state.board)
            else:
                result = analyzer.checker_play(
                    state.board, d1, d2,
                    cube_value=state.cube_value,
                    cube_owner=state.cube_owner,
                )
                post = list(result.moves[0].board)
            record.events.append(_Event("checker_play", player, move_board_mover=post))
            state.board = post
            outcome = _check_game_over(state.board)
            if outcome is not None:
                win_type, mult = outcome
                _record_game_over(record, state, win_type, mult)
                return record
            _flip_state(state)
            continue

        raise RuntimeError(f"Unknown next-step kind: {kind!r}")

    raise RuntimeError(f"Game exceeded MAX_EVENTS={MAX_EVENTS} at level {level}")


# ---------------------------------------------------------------------------
# Event replay -> match-history dict (for text export)
# ---------------------------------------------------------------------------


def _iterate_turns(events: list[_Event]):
    """Walk events, yielding one dict per turn (eliding silent no_doubles)."""
    state = _initial_state()
    n = len(events)
    i = 0
    while i < n:
        active = state.active
        cube_action: str | None = None

        if events[i].kind == "cube_offer" and events[i].player == active:
            offer = events[i]
            i += 1
            if offer.decision == "double":
                if i >= n or events[i].kind != "cube_response" or events[i].player != 3 - active:
                    raise ValueError(f"Expected cube_response after cube_offer at index {i - 1}")
                response = events[i]
                i += 1
                if response.decision == "take":
                    cube_action = "double/take"
                    state.cube_value *= 2
                    state.cube_owner = "opponent"
                else:
                    cube_action = "double/pass"
                    yield {
                        "player": active,
                        "cube_action": cube_action,
                        "dice": None,
                        "pre_board_mover": list(state.board),
                        "post_board_mover": list(state.board),
                    }
                    return
            # else: no_double — silent

        if i >= n or events[i].kind != "dice_roll" or events[i].player != active:
            raise ValueError(f"Expected dice_roll for player {active} at index {i}")
        d1, d2 = events[i].dice
        i += 1

        if i >= n or events[i].kind != "checker_play" or events[i].player != active:
            raise ValueError(f"Expected checker_play for player {active} at index {i}")
        play = events[i]
        i += 1

        pre_board = list(state.board)
        post_board = list(play.move_board_mover)
        yield {
            "player": active,
            "cube_action": cube_action,
            "dice": [d1, d2],
            "pre_board_mover": pre_board,
            "post_board_mover": post_board,
        }

        state.board = post_board
        _flip_state(state)


def _result_field(record: _GameRecord) -> tuple[str, int]:
    if record.winner is None or record.win_type is None or record.cube_at_end is None:
        return "", 0
    side = "player1" if record.winner == 1 else "player2"
    mult = 1 if record.win_type == "single" else 2 if record.win_type == "gammon" else 3
    return f"{side}-win-{record.win_type}", record.cube_at_end * mult


def _record_to_history_dict(record: _GameRecord) -> dict:
    """Convert a _GameRecord into the dict shape that text_export expects."""
    from bgsage.text_export import compute_move_notation

    move_history = []
    for turn in _iterate_turns(record.events):
        entry: dict = {
            "player": "user" if turn["player"] == 1 else "bot",
            "cube_action": turn["cube_action"],
        }
        if turn["dice"] is not None:
            d1, d2 = turn["dice"]
            entry["dice"] = [d1, d2]
            entry["move"] = compute_move_notation(
                turn["pre_board_mover"], turn["post_board_mover"], d1, d2,
            )
        else:
            entry["dice"] = None
            entry["move"] = None
        move_history.append(entry)

    result_str, points = _result_field(record)
    return {
        "player1_name": "Sage",
        "player2_name": "Sage",
        "mode": "unlimited",
        "result": result_str,
        "result_points": points,
        "move_history": move_history,
    }


def _export_record(record: _GameRecord, path: Path) -> None:
    from bgsage.text_export import export_history_to_txt

    path.write_bytes(export_history_to_txt(_record_to_history_dict(record)))


# ---------------------------------------------------------------------------
# Worker process glue
# ---------------------------------------------------------------------------


def _worker_init(canonical: str) -> None:
    """Per-process: pre-build analyzer with parallel_threads=1 to avoid CPU
    oversubscription when many workers run side-by-side."""
    _make_analyzer(canonical, parallel_threads=1)


def _play_one(canonical: str, seed: int, out_dir_str: str) -> dict:
    """Worker entry point: play one game, write its .txt, return summary."""
    rng = random.Random(seed)
    record = _play(canonical, rng, parallel_threads=1)
    out_path = Path(out_dir_str) / f"seed_{seed}.txt"
    _export_record(record, out_path)
    return {
        "seed": seed,
        "winner": record.winner,
        "win_type": record.win_type,
        "final_score": record.final_score,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_sage_vs_sage_games(
    initial_seed: int,
    n_games: int,
    level: str,
    workers: int = 6,
    out_dir: Path | str | None = None,
) -> list[Path]:
    """Play ``n_games`` Sage-vs-Sage money games at the given eval level.

    Game ``i`` uses RNG seed ``initial_seed + i`` (i in ``0..n_games-1``).
    Returns the written transcript paths in seed order.

    With ``workers == 1`` games run serially in-process at
    ``parallel_threads=0`` (single game uses every core). With ``workers > 1``
    games are distributed across worker processes, each pinned to one thread.
    """
    if n_games <= 0:
        raise ValueError(f"n_games must be positive (got {n_games})")
    if workers <= 0:
        raise ValueError(f"workers must be positive (got {workers})")
    canonical = _resolve_level(level)

    out_path_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    out_path_dir.mkdir(parents=True, exist_ok=True)

    seeds = [initial_seed + i for i in range(n_games)]
    written: list[Path] = []

    if workers == 1:
        for s in seeds:
            rng = random.Random(s)
            record = _play(canonical, rng, parallel_threads=0)
            out_path = out_path_dir / f"seed_{s}.txt"
            _export_record(record, out_path)
            print(
                f"[seed {s}] level={level}: P{record.winner} "
                f"{record.win_type} ({record.final_score:+d}) -> {out_path}",
                flush=True,
            )
            written.append(out_path)
        return written

    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(canonical,),
    ) as ex:
        futures = {ex.submit(_play_one, canonical, s, str(out_path_dir)): s for s in seeds}
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            print(
                f"[{completed}/{n_games} done] seed={res['seed']} level={level}: "
                f"P{res['winner']} {res['win_type']} ({res['final_score']:+d}) "
                f"-> {res['out_path']}",
                flush=True,
            )
            written.append(Path(res["out_path"]))

    written.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial_seed", type=int, help="Seed for the first game")
    parser.add_argument("n_games", type=int, help="Number of games to run")
    parser.add_argument(
        "--level",
        default="3P",
        help=f"Eval level: one of {sorted(_VALID_LEVELS)}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel worker processes (default: 6)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    if args.level not in _VALID_LEVELS:
        parser.error(f"--level must be one of {sorted(_VALID_LEVELS)}")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    run_sage_vs_sage_games(
        args.initial_seed, args.n_games, args.level,
        workers=args.workers, out_dir=args.out_dir,
    )
