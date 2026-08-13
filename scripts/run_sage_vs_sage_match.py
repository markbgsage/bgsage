# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Run Sage-vs-Sage N-point matches (optionally in parallel) and write each
match as one Backgammon Galaxy / XG-import compatible .txt transcript.

Public entry point: ``run_sage_vs_sage_match(match_length, n_matches, level,
initial_seed=1, workers=6, out_dir=None)``.

Each match is a sequence of games played to ``match_length`` points by Sage on
both sides at the specified eval level. Match state (away counts, Crawford)
is threaded into ``analyzer.cube_action`` / ``analyzer.checker_play`` so the
bot makes match-aware decisions. Jacoby and beaver are auto-disabled by the
analyzer when match params are non-zero. The Crawford game suppresses all
cube offers.

Per-match transcripts go to ``<out_dir>/match_seed_<N>.txt``. Default
``out_dir`` is ``bgsage/logs/sage_vs_sage_match`` inside the bgsage repo
(deliberately separate from the parent project's ``logs/`` directory).

Usage:

    python bgsage/scripts/run_sage_vs_sage_match.py 5 70
    python bgsage/scripts/run_sage_vs_sage_match.py 7 30 --level 3P --workers 6

Match ``i`` uses RNG seed ``initial_seed + i`` (``i`` in ``0..n_matches-1``).
When ``workers > 1``, matches run in parallel via ``ProcessPoolExecutor``; each
worker pre-loads its own analyzer at ``parallel_threads=1`` so workers don't
oversubscribe the CPU.

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
# bgsage path setup
# ---------------------------------------------------------------------------
# Keep this script strictly inside the bgsage repo: resolve to bgsage's own
# python package and build directory, not the host project's. This mirrors
# the boundary rule in bgsage/CLAUDE.md.

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent            # bgsage/ (this repo's root)
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"
_BUILD_DIR = _PROJECT_ROOT / "build"

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
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "logs" / "sage_vs_sage_match"


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
# Per-game model (same shape as money-game simulator)
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
    final_score: int | None = None      # signed cube*mult from P1's perspective (uncapped)
    # Match context for transcript export — populated by the match driver.
    game_number: int = 0
    start_score1: int = 0
    start_score2: int = 0
    is_crawford: bool = False
    points_awarded: int = 0             # capped contribution to the match score


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
    from bgsage import flip_board

    state.board = list(flip_board(state.board))
    if state.cube_owner == "player":
        state.cube_owner = "opponent"
    elif state.cube_owner == "opponent":
        state.cube_owner = "player"
    state.active = 3 - state.active


def _can_offer_cube(state: _State, is_crawford: bool) -> bool:
    if is_crawford:
        return False
    return state.cube_owner in ("centered", "player")


_WIN_TYPE_BY_MULT = {1: "single", 2: "gammon", 3: "backgammon"}


def _check_game_over(board: list[int]) -> tuple[str, int] | None:
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


def _next_step(state: _State, events: list[_Event], is_crawford: bool) -> tuple[str, int]:
    """Return ``(kind, player)`` describing what event must come next.

    During Crawford the cube layer is collapsed: we skip directly from
    end-of-turn to ``dice_roll`` without ever asking for a cube_offer. Outside
    Crawford the logic is identical to the money-game simulator.
    """
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

    if _can_offer_cube(state, is_crawford):
        return ("cube_offer", state.active)
    return ("dice_roll", state.active)


def _away_counts_for_mover(
    active: int, score1: int, score2: int, match_length: int,
) -> tuple[int, int]:
    """Compute (away1, away2) from the active player's perspective.

    ``away1`` is what the player on roll needs; ``away2`` is what the
    opponent needs. Both clamped to >= 1 (match isn't yet decided when
    we ask).
    """
    if active == 1:
        mover_score, opp_score = score1, score2
    else:
        mover_score, opp_score = score2, score1
    away_mover = max(1, match_length - mover_score)
    away_opp = max(1, match_length - opp_score)
    return away_mover, away_opp


def _play_game(
    level: str, rng: random.Random,
    *, match_length: int, score1: int, score2: int, is_crawford: bool,
    parallel_threads: int = 0,
) -> _GameRecord:
    """Play one game of an N-point match.

    ``score1`` / ``score2`` are the running scores AT THE START of the game.
    They stay constant for the duration. ``is_crawford`` is fixed for the
    whole game (true only for the unique Crawford game in the match).
    """
    from bgsage import possible_moves

    analyzer = _make_analyzer(level, parallel_threads=parallel_threads)
    state = _initial_state()
    record = _GameRecord()

    MAX_EVENTS = 4 * 600
    for _ in range(MAX_EVENTS):
        kind, player = _next_step(state, record.events, is_crawford)

        away_mover, away_opp = _away_counts_for_mover(
            state.active, score1, score2, match_length,
        )

        if kind == "cube_offer":
            cube = analyzer.cube_action(
                state.board,
                cube_value=state.cube_value,
                cube_owner=state.cube_owner,
                away1=away_mover, away2=away_opp,
                is_crawford=is_crawford,
            )
            decision = "double" if cube.should_double else "no_double"
            record.events.append(_Event("cube_offer", player, decision=decision))
            continue

        if kind == "cube_response":
            cube = analyzer.cube_action(
                state.board,
                cube_value=state.cube_value,
                cube_owner=state.cube_owner,
                away1=away_mover, away2=away_opp,
                is_crawford=is_crawford,
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
                    away1=away_mover, away2=away_opp,
                    is_crawford=is_crawford,
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
# Event replay -> per-game move history dict (same as money-game)
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


def _result_field(record: _GameRecord, points_awarded: int) -> tuple[str, int]:
    """Build the (result_str, result_points) pair for text export.

    ``points_awarded`` is the CAPPED contribution to the match score for this
    game. Using the capped value here keeps text_export's running totals (and
    therefore the displayed per-game start scores) consistent with how XG and
    the Galaxy parser cap excess points at ``match_length``.
    """
    if record.winner is None or record.win_type is None:
        return "", 0
    side = "player1" if record.winner == 1 else "player2"
    return f"{side}-win-{record.win_type}", int(points_awarded)


def _game_record_to_history_dict(record: _GameRecord) -> dict:
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

    result_str, points = _result_field(record, record.points_awarded)
    return {
        "game_number": record.game_number,
        "player_score": record.start_score1,
        "opponent_score": record.start_score2,
        "result": result_str,
        "result_points": points,
        "move_history": move_history,
    }


def _match_to_history_dict(records: list[_GameRecord], match_length: int) -> dict:
    return {
        "player1_name": "Sage",
        "player2_name": "Sage",
        "mode": "match",
        "match_length": match_length,
        "match_game_histories": [_game_record_to_history_dict(r) for r in records],
    }


def _export_match(records: list[_GameRecord], match_length: int, path: Path) -> None:
    from bgsage.text_export import export_history_to_txt

    path.write_bytes(export_history_to_txt(_match_to_history_dict(records, match_length)))


# ---------------------------------------------------------------------------
# Match driver
# ---------------------------------------------------------------------------


def _is_crawford_game(
    score1: int, score2: int, match_length: int, crawford_done: bool,
) -> bool:
    if crawford_done:
        return False
    return score1 == match_length - 1 or score2 == match_length - 1


def _play_match(
    level: str, match_length: int, rng: random.Random,
    parallel_threads: int = 0,
) -> list[_GameRecord]:
    """Play one N-point match and return the per-game records.

    Scores are tracked capped at ``match_length`` — the simulation stops as
    soon as either player reaches the cap. Each game's ``points_awarded`` is
    the capped point contribution (so the .txt transcript's running totals
    stay consistent with how XG / Galaxy parsers cap excess points).
    """
    if match_length <= 0:
        raise ValueError(f"match_length must be positive (got {match_length})")

    score1 = 0
    score2 = 0
    crawford_done = False
    records: list[_GameRecord] = []
    game_number = 0
    MAX_GAMES = 200  # safety net; real matches finish well under this.

    while score1 < match_length and score2 < match_length:
        game_number += 1
        if game_number > MAX_GAMES:
            raise RuntimeError(
                f"Match exceeded MAX_GAMES={MAX_GAMES} at {match_length}pt"
            )

        is_crawford = _is_crawford_game(score1, score2, match_length, crawford_done)

        record = _play_game(
            level, rng,
            match_length=match_length, score1=score1, score2=score2,
            is_crawford=is_crawford, parallel_threads=parallel_threads,
        )
        record.game_number = game_number
        record.start_score1 = score1
        record.start_score2 = score2
        record.is_crawford = is_crawford

        # Apply scoring with match-length cap. The uncapped final_score is
        # already on the record; points_awarded is what actually counts toward
        # the match.
        cube_mult = 1 if record.win_type == "single" else (
            2 if record.win_type == "gammon" else 3
        )
        raw_points = (record.cube_at_end or 1) * cube_mult
        if record.winner == 1:
            new_score = min(score1 + raw_points, match_length)
            record.points_awarded = new_score - score1
            score1 = new_score
        elif record.winner == 2:
            new_score = min(score2 + raw_points, match_length)
            record.points_awarded = new_score - score2
            score2 = new_score
        else:
            record.points_awarded = 0

        if is_crawford:
            crawford_done = True

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Worker glue
# ---------------------------------------------------------------------------


def _worker_init(canonical: str) -> None:
    _make_analyzer(canonical, parallel_threads=1)


def _play_one_match(
    canonical: str, seed: int, match_length: int, out_dir_str: str,
) -> dict:
    rng = random.Random(seed)
    records = _play_match(canonical, match_length, rng, parallel_threads=1)
    out_path = Path(out_dir_str) / f"match_seed_{seed}.txt"
    _export_match(records, match_length, out_path)
    final_score1 = sum(r.points_awarded for r in records if r.winner == 1)
    final_score2 = sum(r.points_awarded for r in records if r.winner == 2)
    return {
        "seed": seed,
        "n_games": len(records),
        "final_score1": final_score1,
        "final_score2": final_score2,
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_sage_vs_sage_match(
    match_length: int,
    n_matches: int,
    level: str,
    initial_seed: int = 1,
    workers: int = 6,
    out_dir: Path | str | None = None,
) -> list[Path]:
    """Play ``n_matches`` Sage-vs-Sage matches and return the .txt paths.

    Match ``i`` uses RNG seed ``initial_seed + i`` (``i`` in
    ``0..n_matches-1``). With ``workers == 1`` matches run serially in-process
    at ``parallel_threads=0``; with ``workers > 1`` they're distributed
    across worker processes each pinned to one thread.
    """
    if match_length <= 0:
        raise ValueError(f"match_length must be positive (got {match_length})")
    if n_matches <= 0:
        raise ValueError(f"n_matches must be positive (got {n_matches})")
    if workers <= 0:
        raise ValueError(f"workers must be positive (got {workers})")
    canonical = _resolve_level(level)

    out_path_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    out_path_dir.mkdir(parents=True, exist_ok=True)

    seeds = [initial_seed + i for i in range(n_matches)]
    written: list[Path] = []

    if workers == 1:
        for s in seeds:
            rng = random.Random(s)
            records = _play_match(canonical, match_length, rng, parallel_threads=0)
            out_path = out_path_dir / f"match_seed_{s}.txt"
            _export_match(records, match_length, out_path)
            sc1 = sum(r.points_awarded for r in records if r.winner == 1)
            sc2 = sum(r.points_awarded for r in records if r.winner == 2)
            print(
                f"[seed {s}] level={level} match={match_length}pt: "
                f"P1 {sc1}-{sc2} P2 in {len(records)} games -> {out_path}",
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
        futures = {
            ex.submit(_play_one_match, canonical, s, match_length, str(out_path_dir)): s
            for s in seeds
        }
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            print(
                f"[{completed}/{n_matches} done] seed={res['seed']} "
                f"level={level} match={match_length}pt: "
                f"P1 {res['final_score1']}-{res['final_score2']} P2 in "
                f"{res['n_games']} games -> {res['out_path']}",
                flush=True,
            )
            written.append(Path(res["out_path"]))

    written.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "match_length", type=int, nargs="?", default=5,
        help="Match length in points (default: 5)",
    )
    parser.add_argument(
        "n_matches", type=int, nargs="?", default=70,
        help="Number of matches to simulate (default: 70)",
    )
    parser.add_argument(
        "--initial-seed", type=int, default=1,
        help="RNG seed for the first match (default: 1)",
    )
    parser.add_argument(
        "--level",
        default="3P",
        help=f"Eval level: one of {sorted(_VALID_LEVELS)}",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel worker processes (default: 6)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=f"Output directory (default: {_DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    if args.level not in _VALID_LEVELS:
        parser.error(f"--level must be one of {sorted(_VALID_LEVELS)}")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.match_length <= 0:
        parser.error("match_length must be positive")
    if args.n_matches <= 0:
        parser.error("n_matches must be positive")

    run_sage_vs_sage_match(
        args.match_length, args.n_matches, args.level,
        initial_seed=args.initial_seed, workers=args.workers,
        out_dir=args.out_dir,
    )
