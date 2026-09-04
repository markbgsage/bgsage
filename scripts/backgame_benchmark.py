#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Stage 11 backgame benchmarks: reference positions in, benchmark decisions out.

Stage 11 is aimed at back game play, so it needs a benchmark that is *made of*
back games. ``backgame_ref_positions/Positions for Mark/`` holds one subfolder
per back game type (``21 backgame``, ``52 backgame``, ...) plus a
``Positions XG gets wrong`` grab bag, each holding a handful of hand-curated XG
position (``.xgp``) files. This script turns those seeds into one benchmark per
subfolder.

Two steps, two subcommands:

``import``
    Read every ``.xgp`` in each subfolder and write the seed positions to
    ``backgame_ref_positions/benchmark/<subfolder> starting.txt`` — the 26-int
    checker list plus the cube (value + owner). Cube values above 2 are clamped
    to 2, so a seed is either a centred 1-cube or an owned 2-cube.

``generate``
    Play cubeful self-play games out of those seeds and record the
    *decisions* that arise while a back game of the subfolder's type is still on
    the board, to ``backgame_ref_positions/benchmark/<subfolder> benchmark.txt``.

How a game is played (``generate``):

* A game starts from one of the subfolder's seed positions. Seeds are consumed
  in a shuffled cycle — every seed is used once before any is used twice.
* Which side moves first is a coin flip; that side rolls and plays. The first
  half-move of a game is a checker play, never a cube action (the seed position
  is taken to be *after* the turn's cube decision).
* Both sides then play full money-game backgammon **cubeful, with Jacoby and
  beaver on**, at ``--level`` (default 3P): a cube decision at the start of
  every subsequent turn the side on roll has cube access for, then the roll
  and the checker play.
* The game stops when contact is broken (``is_race``), when someone bears off,
  or when a double is passed.

Which decisions are recorded:

* The position must still hold a back game of the subfolder's type — at least
  one player anchored (2+ checkers) on both of the named points in the
  opponent's home board. ``Positions XG gets wrong`` accepts *any* back game:
  anchors on 2 or more of the opponent's 1- through 5-points, the union of the
  ten named types.
* The decision must count for PR. A checker play counts when there are 2+ legal
  moves and a meaningful best-vs-worst equity spread; a cube position counts
  when the doubler's decision is not trivially obvious (an obvious no-double,
  too-good or hopeless position), or when a double is actually offered and the
  take/pass is live. Those rules are imported from ``benchmark_money.py`` so the
  two benchmarks agree on what a decision is.
* Duplicates are dropped: a decision is unique on kind + board + cube + dice.

Reproducibility: everything is a pure function of ``--seed`` (default 1) and the
model, including which seed position each game starts from, the dice, and the
rollout seed of every evaluation. **No RNG state is carried between
games** — game N draws everything it needs from ``seed:<folder>:game<N>`` — so
game N plays out identically whether it is the Nth game of an uninterrupted run
or the first game after a restart.

That is what makes this safe to kill. A run collecting 10,000 decisions takes
days, so being killed is not an edge case:

* After every game, the decisions are written (temp file + atomic rename) and
  then a ``<subfolder> benchmark.state.json`` sidecar records the number of
  **completed** games. A restart reloads the decisions, sets the game index from
  the sidecar and carries on.
* The interrupted game is replayed, not skipped, because the counter advances
  only once a game finishes. At most one game's work is ever lost.
* Every crash window converges: a half-written ``.txt`` is impossible (atomic
  rename), and a crash between the two writes leaves the ``.txt`` one game ahead
  of the sidecar, whose replay re-finds decisions that dedupe away.
* A lost or corrupt sidecar costs replayed games, never collected work — the
  ``.txt`` is always adopted on restart (only ``--restart`` discards it).
* A ``<subfolder> benchmark.lock`` held for the process's lifetime stops a
  relaunch from starting a second worker on a subfolder whose worker is still
  alive. The OS drops it when a process dies, so a kill leaves nothing to clean
  up.

Changing ``--count`` needs ``--restart``: the output would otherwise no longer
be the thing the seed reproduces.

Usage::

    py -3.14 scripts/backgame_benchmark.py import
    py -3.14 scripts/backgame_benchmark.py generate --count 5
    py -3.14 scripts/backgame_benchmark.py generate --count 1000
    py -3.14 scripts/backgame_benchmark.py generate --count 1000 --folder "21 backgame"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple, Optional

# ---------------------------------------------------------------------------
# bgsage path setup - self-contained within the bgsage repo. Never reaches into
# a parent project: weights, build artifacts and outputs all live under bgsage/.
# (Mirrors scripts/benchmark_money.py.)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = bgsage repo root
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"   # = bgsage/python
_BUILD_DIR = _PROJECT_ROOT / "build"        # = bgsage/build

for _p in (_BGSAGE_PYTHON, _BUILD_DIR, _SCRIPT_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

# The PR "does this count as a decision?" rules live in benchmark_money.py; import
# them rather than restating them, so the two benchmarks can never drift apart on
# what a decision is.
from benchmark_money import TRIVIAL_SPREAD, _is_trivial_cube  # noqa: E402

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

_REF_DIR = _PROJECT_ROOT / "backgame_ref_positions"
_SRC_DIR = _REF_DIR / "Positions for Mark"
_OUT_DIR = _REF_DIR / "benchmark"

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

#: Evaluation level both sides play at while simulating games. Recorded in
#: the output header and in the resume state, so a run can never silently
#: mix decisions generated at two different levels.
DEFAULT_LEVEL = "3ply"

#: Default number of benchmark decisions to collect per subfolder.
DEFAULT_COUNT = 1000

#: Default master RNG seed. Everything about a run derives from this.
DEFAULT_SEED = 1

#: Safety valve: give up on a subfolder after this many games without reaching
#: the target, rather than looping forever on seeds that never qualify.
DEFAULT_MAX_GAMES = 1000

#: Safety cap on half-moves in one simulated game.
MAX_HALF_MOVES = 500

#: Points, in the opponent's home board, that ``Positions XG gets wrong``
#: accepts an anchor on.
ANY_BACKGAME_POINTS = (1, 2, 3, 4, 5, 6)
#: How many of those anchors a position needs. One, not two: that grab bag holds
#: deep-anchor and ace-point games as well as true back games -- four of the
#: hand-transcribed Lamford positions are a single anchor on the opponent's
#: 5-point -- and requiring two anchors let games from those seeds run to
#: completion recording nothing. Note this is looser than "back game": with the
#: 6-point included, an ordinary advanced-anchor holding game qualifies too.
ANY_BACKGAME_MIN_ANCHORS = 1


# ---------------------------------------------------------------------------
# Positions and decisions
# ---------------------------------------------------------------------------


class StartPosition(NamedTuple):
    """A seed position, player-1 frame (positive checkers = player 1)."""
    cube_value: int
    cube_owner: str            # "centered" | "player" | "opponent", rel. player 1
    board: tuple[int, ...]     # 26 ints


class Decision(NamedTuple):
    """One benchmark decision, from the perspective of the player to act."""
    kind: str                          # "checker" | "cube"
    cube_value: int
    cube_owner: str                    # "centered" | "player" | "opponent"
    dice: Optional[tuple[int, int]]    # None for a cube decision
    board: tuple[int, ...]             # 26 ints


def _flip_owner(owner: str) -> str:
    """Cube owner as seen from the other side of the board."""
    if owner == "player":
        return "opponent"
    if owner == "opponent":
        return "player"
    return owner


def _clamp_cube(cube_value: int, cube_owner: str) -> tuple[int, str]:
    """Reduce a seed's cube to the two states the benchmark uses.

    A centred cube is a 1-cube; an owned cube is a 2-cube whatever level the
    reference position was saved at (the curated files include 4-cubes).
    """
    if cube_owner == "centered":
        return 1, "centered"
    return 2, cube_owner


# ---------------------------------------------------------------------------
# Back game filters
# ---------------------------------------------------------------------------
#
# Board indices: 1-24 are points, positive = player 1, negative = player 2;
# 0 is player 2's bar and 25 is player 1's. Player 1 bears off past point 1, so
# the opponent's n-point is index 25-n for player 1 and index n for player 2.
# An anchor is 2 or more checkers on a point.


def _anchored_p1(board, n: int) -> bool:
    return board[25 - n] >= 2


def _anchored_p2(board, n: int) -> bool:
    return board[n] <= -2


def _make_named_filter(points: tuple[int, int]) -> Callable[[tuple[int, ...]], bool]:
    """Either player anchored on both of the opponent's ``points``."""
    def matches(board) -> bool:
        return (all(_anchored_p1(board, n) for n in points)
                or all(_anchored_p2(board, n) for n in points))
    return matches


def _any_backgame(board) -> bool:
    """Either player anchored in the opponent's home board.

    Used by ``Positions XG gets wrong`` ONLY -- every ``NN backgame`` category
    tests its own two named points, so loosening this cannot affect them.
    """
    return (sum(1 for n in ANY_BACKGAME_POINTS
                if _anchored_p1(board, n)) >= ANY_BACKGAME_MIN_ANCHORS
            or sum(1 for n in ANY_BACKGAME_POINTS
                   if _anchored_p2(board, n)) >= ANY_BACKGAME_MIN_ANCHORS)


# --- Back-game FAMILY filters (2026-09-02) ---------------------------------
#
# Three more folders, each a region the anchor-pair folders never reach. Every
# predicate is written for the player-1 frame with player 1 as the side of
# interest and evaluated for both orientations. Bar convention: index 25 holds
# player 1's bar checkers and index 0 player 2's, BOTH as positive counts.


def _flip(board) -> tuple[int, ...]:
    # Points reverse and change sign; the two bars swap but stay positive
    # (index 0 = player 2's bar, 25 = player 1's, both stored as counts).
    # Until 2026-09-03 this negated the bars too, so a position was judged
    # in its flipped orientation with a bar checker on the wrong side.
    return (board[25],) + tuple(-board[25 - i] for i in range(1, 25)) + (board[0],)


def _p1_side(board, i: int) -> int:
    """Player 1's checkers at index i (bar = 25)."""
    return board[i] if 1 <= i <= 25 and board[i] > 0 else 0


def _p2_side(board, i: int) -> int:
    """Player 2's checkers at index i (bar = 0, stored positive)."""
    if i == 0:
        return board[0]
    return -board[i] if 1 <= i <= 24 and board[i] < 0 else 0


def _containment_p1_escaper(board) -> bool:
    """Player 1 is the escaper of a late containment game (K6-strict).

    Escaper has >= 6 checkers off, contact remains, and 1-3 of its checkers
    are trapped: outside its home board (bar included) with a container
    checker in front of them that is ALSO outside the escaper's home — a
    racer bearing in past a back-game anchor is not trapped. The container
    needs >= 4 checkers in the blocking region (two points' worth).
    """
    p1 = [_p1_side(board, i) for i in range(26)]
    p2 = [_p2_side(board, i) for i in range(26)]
    on_board = sum(p1)
    if 15 - on_board < 6 or on_board == 0:
        return False
    max_p1 = max(i for i in range(26) if p1[i])
    p2_any = [i for i in range(26) if p2[i]]
    if not p2_any or max_p1 <= min(p2_any):
        return False                                    # no contact
    outside = [i for i in range(7, 25) if p2[i]]
    if not outside:
        return False
    min_p2_out = min(outside)
    trapped = sum(p1[i] for i in range(7, 26) if i > min_p2_out)
    blockers = sum(p2[i] for i in range(7, 25) if i < max_p1)
    return 1 <= trapped <= 3 and blockers >= 4


def _containment(board) -> bool:
    return _containment_p1_escaper(board) or _containment_p1_escaper(_flip(board))


def _massive_p1_holder(board) -> bool:
    """Player 1 plays a massive back game: behind in pips with >= 3 anchors in
    the opponent's home, or >= 2 anchors and >= 7 checkers back (opponent's
    home board + bar)."""
    anchors = sum(1 for i in range(19, 25) if board[i] >= 2)
    if anchors < 2:
        return False
    back = sum(board[i] for i in range(19, 26) if board[i] > 0)
    if not (anchors >= 3 or back >= 7):
        return False
    pips_p1 = sum(board[i] * i for i in range(1, 26) if board[i] > 0)
    pips_p2 = sum(-board[i] * (25 - i) for i in range(1, 25) if board[i] < 0) + board[0] * 25
    return pips_p1 > pips_p2


def _massive(board) -> bool:
    # Disjoint from the two more specific families: a late containment game
    # (Lamford 01 holds three anchors) and a far-side prime (Snake) both read
    # as "many checkers back" but belong to their own folders.
    if _containment(board) or _snake(board):
        return False
    return _massive_p1_holder(board) or _massive_p1_holder(_flip(board))


def _snake_p1_prime(board) -> bool:
    """Player 1 contains a straggler behind a far-side prime ("snake").

    A run of >= 4 consecutive points, each held with >= 2 checkers, entirely
    on the opponent's half of the board (indices 13-24); the opponent has a
    straggler (on the bar or in player 1's home board) and >= 10 checkers
    already in its own home board — the crunched side of the Snake and
    Lamford ch. 41 shapes.
    """
    run = best = 0
    for i in range(13, 25):
        run = run + 1 if board[i] >= 2 else 0
        best = max(best, run)
    if best < 4:
        return False
    straggler = board[0] + sum(-board[i] for i in range(1, 7) if board[i] < 0)
    if straggler < 1:
        return False
    home = sum(-board[i] for i in range(19, 25) if board[i] < 0)
    return home >= 10


def _snake(board) -> bool:
    return _snake_p1_prime(board) or _snake_p1_prime(_flip(board))


FAMILY_FILTERS: dict[str, Callable[[tuple[int, ...]], bool]] = {
    "containment": _containment,
    "massive backgame": _massive,
    "snake": _snake,
}


def backgame_filter(folder_name: str) -> Callable[[tuple[int, ...]], bool]:
    """The "is this still a back game of this subfolder's type?" test."""
    m = re.match(r"^([1-6])([1-6])\s+backgame$", folder_name)
    if m:
        return _make_named_filter((int(m.group(1)), int(m.group(2))))
    if folder_name in FAMILY_FILTERS:
        return FAMILY_FILTERS[folder_name]
    return _any_backgame


# ---------------------------------------------------------------------------
# Reading XG position files
# ---------------------------------------------------------------------------


def read_xgp_position(path: Path) -> StartPosition:
    """Read one ``.xgp`` reference position: board (player-1 frame) + cube.

    A "to play" position carries its board on a MOVE record and a cube decision
    on a CUBE record; the dice are ignored either way (the benchmark rolls its
    own). XG stores the board in the player-1 frame and signs the bar cells by
    owner, and encodes the cube as 0 = centred, +N = player 1 owns 2^N,
    -N = player 2 owns 2^N.
    """
    from bgsage.xg_file import (
        TS_CUBE, TS_MOVE, XgArchive, iter_records, norm_bars,
        parse_cube_record, parse_move_record,
    )

    data = XgArchive.load(path).get("temp.xg")
    if data is None:
        raise ValueError(f"{path.name}: no temp.xg record stream")

    move = cube = None
    for off, rec_type in iter_records(data):
        if rec_type == TS_MOVE and move is None:
            move = parse_move_record(data, off)
        elif rec_type == TS_CUBE and cube is None:
            cube = parse_cube_record(data, off)

    if move is not None and all(1 <= d <= 6 for d in move["dice"][:2]):
        raw, encoded_cube = move["position_raw"], move["cube_a"]
    elif cube is not None:
        raw, encoded_cube = cube["position_raw"], cube["cube_b"]
    else:
        raise ValueError(f"{path.name}: no position record")

    board = norm_bars(raw)
    if len(board) != 26:
        raise ValueError(f"{path.name}: board is not 26 cells")

    if encoded_cube == 0:
        cube_value, cube_owner = 1, "centered"
    elif encoded_cube > 0:
        cube_value, cube_owner = 2 ** encoded_cube, "player"
    else:
        cube_value, cube_owner = 2 ** (-encoded_cube), "opponent"

    return StartPosition(cube_value, cube_owner, tuple(board))


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------


#: How long to keep retrying an atomic rename before giving up on one
#: checkpoint. On Windows a rename over a file another process has open fails
#: with WinError 5, and this repo lives under Dropbox, whose sync agent opens
#: files it is uploading (a virus scanner does the same). The lock is brief, so
#: retrying rides it out. Seen in the wild 2026-08-28: it killed a worker 34
#: games into a 10,000-decision run.
_REPLACE_RETRY_SECONDS = 30.0


def _atomic_write(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` via a temp file + rename. True if it landed.

    Returns False rather than raising when the rename keeps failing: a missed
    checkpoint only costs the games since the last good write (they replay on
    resume), which is never worth killing a run that is hours in.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    deadline = time.monotonic() + _REPLACE_RETRY_SECONDS
    delay = 0.1
    while True:
        try:
            tmp.replace(path)
            return True
        except PermissionError:
            if time.monotonic() >= deadline:
                print(f"  WARNING: could not replace {path.name} after "
                      f"{_REPLACE_RETRY_SECONDS:.0f}s (file locked by another "
                      f"process?) - continuing; the next game retries.",
                      flush=True)
                return False
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def _board_str(board) -> str:
    return " ".join(str(int(x)) for x in board)


def _data_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def start_file(folder_name: str) -> Path:
    return _OUT_DIR / f"{folder_name} starting.txt"


def manual_file(folder_name: str) -> Path:
    """Hand-entered seed positions for a category, merged in by ``import``.

    Some reference positions only exist as photographs or book diagrams, so they
    were transcribed by hand rather than saved as ``.xgp``. They live here, in
    the committed benchmark directory, precisely so that re-running ``import``
    -- which otherwise rebuilds ``starting.txt`` from the ``.xgp`` files alone,
    and would silently drop them -- keeps them. Same line format as
    ``starting.txt``.
    """
    return _OUT_DIR / f"{folder_name} manual.txt"


def benchmark_file(folder_name: str) -> Path:
    return _OUT_DIR / f"{folder_name} benchmark.txt"


def state_file(folder_name: str) -> Path:
    return _OUT_DIR / f"{folder_name} benchmark.state.json"


def write_start_positions(
    folder_name: str, entries: list[tuple[str, StartPosition]], notes: list[str],
) -> Path:
    """Write ``<folder> starting.txt``: one seed position per line."""
    out = [
        f"# {folder_name} - starting positions, imported from the XG reference files.",
        "# One position per line: cube_value cube_owner board(26 ints).",
        '# cube_owner is "centered" | "player" | "opponent", relative to player 1',
        "# (the positive checkers). Cube values above 2 are clamped to 2.",
    ]
    out += [f"# {n}" for n in notes]
    for source, pos in entries:
        out.append("#")
        out.append(f"# {source}")
        out.append(f"{pos.cube_value} {pos.cube_owner} {_board_str(pos.board)}")
    path = start_file(folder_name)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _read_positions_file(path: Path) -> list[StartPosition]:
    """Parse ``cube_value cube_owner board(26 ints)`` lines."""
    positions = []
    for line in _data_lines(path):
        fields = line.split()
        board = tuple(int(x) for x in fields[2:])
        if len(board) != 26:
            raise ValueError(f"{path.name}: expected 26 board ints, got {len(board)}")
        positions.append(StartPosition(int(fields[0]), fields[1], board))
    return positions


def read_start_positions(folder_name: str) -> list[StartPosition]:
    path = start_file(folder_name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run 'backgame_benchmark.py import' first")
    return _read_positions_file(path)


def _decision_line(d: Decision) -> str:
    die1, die2 = d.dice if d.dice else ("-", "-")
    return (f"{d.kind} {d.cube_value} {d.cube_owner} {die1} {die2} "
            f"{_board_str(d.board)}")


def _parse_decision_line(line: str) -> Decision:
    fields = line.split()
    dice = None if fields[3] == "-" else (int(fields[3]), int(fields[4]))
    board = tuple(int(x) for x in fields[5:])
    if len(board) != 26:
        raise ValueError(f"expected 26 board ints, got {len(board)}")
    return Decision(fields[0], int(fields[1]), fields[2], dice, board)


def write_decisions(
    folder_name: str, decisions: list[Decision], seed: int, model: str,
    level: str,
) -> bool:
    n_checker = sum(1 for d in decisions if d.kind == "checker")
    out = [
        f"# {folder_name} - cubeful benchmark decisions (money play, Jacoby + beaver on).",
        f"# {len(decisions)} decisions ({n_checker} checker, "
        f"{len(decisions) - n_checker} cube), from simulated {level} cubeful self-play",
        f"# out of \"{start_file(folder_name).name}\"; model={model}, seed={seed}.",
        "# One decision per line: kind cube_value cube_owner die1 die2 board(26 ints).",
        '#   kind        "checker" or "cube"',
        '#   cube_owner  "centered" | "player" | "opponent", relative to the player on roll',
        '#   die1 die2   "-" for a cube decision',
        "# The board is from the perspective of the player whose decision it is.",
    ]
    out += [_decision_line(d) for d in decisions]
    return _atomic_write(benchmark_file(folder_name), "\n".join(out) + "\n")


def read_decisions(folder_name: str) -> list[Decision]:
    path = benchmark_file(folder_name)
    if not path.exists():
        return []
    return [_parse_decision_line(line) for line in _data_lines(path)]


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def subfolders() -> list[Path]:
    if not _SRC_DIR.is_dir():
        raise FileNotFoundError(f"Reference positions not found: {_SRC_DIR}")
    return sorted(p for p in _SRC_DIR.iterdir() if p.is_dir())


def import_folder(folder: Path) -> tuple[int, list[str]]:
    """Import one subfolder's ``.xgp`` files. Returns (kept, notes)."""
    matches = backgame_filter(folder.name)
    seen: dict[tuple, str] = {}
    entries: list[tuple[str, StartPosition]] = []
    notes: list[str] = []

    for path in sorted(folder.glob("*.xgp")):
        pos = read_xgp_position(path)
        clamped_value, clamped_owner = _clamp_cube(pos.cube_value, pos.cube_owner)
        if clamped_value != pos.cube_value:
            notes.append(f"{path.name}: cube {pos.cube_value} clamped to {clamped_value}")
        pos = StartPosition(clamped_value, clamped_owner, pos.board)

        key = (pos.cube_value, pos.cube_owner, pos.board)
        if key in seen:
            notes.append(f"{path.name}: duplicate of {seen[key]} - skipped")
            continue
        if not matches(pos.board):
            notes.append(f"{path.name}: does NOT match the \"{folder.name}\" filter "
                         f"- check the filing")
        seen[key] = path.name
        entries.append((path.name, pos))

    # Hand-transcribed positions (photographs, book diagrams) merge in after the
    # .xgp files, deduped against them.
    manual = manual_file(folder.name)
    if manual.exists():
        for i, pos in enumerate(_read_positions_file(manual), start=1):
            clamped_value, clamped_owner = _clamp_cube(pos.cube_value, pos.cube_owner)
            pos = StartPosition(clamped_value, clamped_owner, pos.board)
            key = (pos.cube_value, pos.cube_owner, pos.board)
            label = f"{manual.name} #{i}"
            if key in seen:
                notes.append(f"{label}: duplicate of {seen[key]} - skipped")
                continue
            if not matches(pos.board):
                notes.append(f"{label}: does NOT match the \"{folder.name}\" filter")
            seen[key] = label
            entries.append((label, pos))

    write_start_positions(folder.name, entries, notes)
    return len(entries), notes


def cmd_import(args: argparse.Namespace) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = [f for f in subfolders() if args.folder in (None, f.name)]
    if not folders:
        raise SystemExit(f"No subfolder named {args.folder!r} under {_SRC_DIR}")

    total = 0
    all_notes = 0
    for folder in folders:
        kept, notes = import_folder(folder)
        total += kept
        all_notes += len(notes)
        print(f"{folder.name:32s} {kept:3d} positions -> {start_file(folder.name).name}")
        for note in notes:
            print(f"{'':32s}   ! {note}")
    print(f"\n{total} starting positions across {len(folders)} subfolders "
          f"({all_notes} note(s)) in {_OUT_DIR}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _start_for_game(starts: list[StartPosition], game_index: int, seed: int,
                    folder_name: str) -> StartPosition:
    """Seed position for game ``game_index``, stepping through a shuffled cycle.

    Every seed is used once before any is reused. A pure function of the game
    index so a resumed run picks up exactly where a fresh one would.
    """
    block, offset = divmod(game_index, len(starts))
    order = list(range(len(starts)))
    random.Random(f"{seed}:{folder_name}:cycle{block}").shuffle(order)
    return starts[order[offset]]


def _play_game(
    start: StartPosition,
    rng: random.Random,
    analyzer,
    matches: Callable[[tuple[int, ...]], bool],
    record: Callable[[Decision], bool],
) -> None:
    """Play one cubeful money game, feeding qualifying decisions to ``record``.

    ``record`` returns False when enough decisions have been collected, which
    ends the game immediately.
    """
    from bgsage import check_game_over, flip_board, is_race, possible_moves

    board = list(start.board)
    cube_value, cube_owner = start.cube_value, start.cube_owner

    # Coin flip for who moves first. The seed position is player-1 framed, so
    # giving player 2 the move flips the board and hands the cube over with it.
    if rng.random() < 0.5:
        board = flip_board(board)
        cube_owner = _flip_owner(cube_owner)

    for half_move in range(MAX_HALF_MOVES):
        if is_race(board):
            return

        board_key = tuple(board)
        on_backgame = matches(board_key)

        # --- Cube action. Skipped on the game's first half-move: the seed
        # position is taken to be after that turn's cube decision. ---
        if half_move > 0 and cube_owner in ("centered", "player"):
            analyzer.set_seed(rng.getrandbits(31))
            cube = analyzer.cube_action(
                board, cube_value=cube_value, cube_owner=cube_owner,
                jacoby=True, beaver=True,
            )
            nd, dt, dp = cube.equity_nd, cube.equity_dt, cube.equity_dp
            # The doubler's decision counts unless the position is trivial;
            # the receiver's take/pass exists only once a double is offered.
            has_double = not _is_trivial_cube(nd, dt, dp)
            has_take = bool(cube.should_double) and (
                abs(dt - dp) >= TRIVIAL_SPREAD or bool(getattr(cube, "is_beaver", False)))
            if on_backgame and (has_double or has_take):
                if not record(Decision("cube", cube_value, cube_owner, None, board_key)):
                    return
            if cube.should_double:
                if not cube.should_take:
                    return                      # doubled out
                cube_value *= 2
                cube_owner = "opponent"

        # --- Roll and play. ---
        die1, die2 = rng.randint(1, 6), rng.randint(1, 6)
        if possible_moves(board, die1, die2):
            analyzer.set_seed(rng.getrandbits(31))
            result = analyzer.checker_play(
                board, die1, die2, cube_value=cube_value, cube_owner=cube_owner,
                jacoby=True, beaver=True,
            )
            moves = result.moves
            # Counts as a decision: 2+ legal moves and a meaningful spread.
            if (on_backgame and len(moves) >= 2
                    and (moves[0].equity - moves[-1].equity) >= TRIVIAL_SPREAD):
                dice = tuple(sorted((die1, die2), reverse=True))
                if not record(Decision("checker", cube_value, cube_owner, dice, board_key)):
                    return
            board = list(moves[0].board)

        if check_game_over(board) != 0:
            return
        board = flip_board(board)
        cube_owner = _flip_owner(cube_owner)


class WorkerBusy(RuntimeError):
    """Another live process already owns this subfolder."""


class _FolderLock:
    """Exclusive per-subfolder lock, released by the OS when the process dies.

    Two workers on one subfolder would interleave their games into the same
    ``benchmark.txt`` and corrupt it. That is exactly the collision a relaunch
    invites: some workers are dead, some are still running, and the operator
    cannot tell which. An OS-level byte-range lock answers it exactly - a
    running worker holds it, a killed one does not, and there is no stale lock
    to clean up (which a pid file or a timestamp heuristic would leave behind).
    """

    def __init__(self, folder_name: str):
        self._path = _OUT_DIR / f"{folder_name} benchmark.lock"
        self._handle = None

    def __enter__(self) -> "_FolderLock":
        handle = open(self._path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise WorkerBusy(
                f"{self._path.stem}: another worker is already running "
                f"(lock held on {self._path.name})") from exc
        self._handle = handle
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass        # dying anyway; the OS drops the lock with the handle
        self._handle.close()
        self._handle = None


def _load_state(folder_name: str) -> dict:
    path = state_file(folder_name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(folder_name: str, state: dict) -> None:
    _atomic_write(state_file(folder_name), json.dumps(state, indent=2))


def generate_folder(
    folder_name: str, count: int, seed: int, model: str, level: str,
    max_games: int, restart: bool, analyzer,
) -> dict:
    """Collect ``count`` benchmark decisions for one subfolder."""
    starts = read_start_positions(folder_name)
    if not starts:
        raise SystemExit(f"{folder_name}: no starting positions")
    matches = backgame_filter(folder_name)

    state = {} if restart else _load_state(folder_name)
    if state and (state.get("seed") != seed or state.get("model") != model
                  or state.get("count") != count
                  or state.get("level") != level):
        raise SystemExit(
            f"{folder_name}: existing run used seed={state.get('seed')} "
            f"model={state.get('model')} level={state.get('level')} "
            f"count={state.get('count')}; asked for seed={seed} "
            f"model={model} level={level} count={count}. "
            f"Pass --restart to discard it and start fresh.")

    # Always adopt whatever is already on disk (unless restarting): the .txt is
    # the real output, and it must never be discarded because its state sidecar
    # went missing. A lost sidecar costs replayed games, not collected work -
    # replayed games re-find the same decisions and dedupe away.
    found: dict[Decision, None] = {}
    if not restart:
        for d in read_decisions(folder_name):
            found[d] = None
    # The seeds themselves are known-good positions of the family, so each is
    # a benchmark decision too — its cube decision, from the player-1 frame the
    # seed file uses. (2026-09-04: earlier benchmarks recorded only the
    # decisions that arose in play from a seed.)
    for st in starts:
        found.setdefault(Decision("cube", st.cube_value, st.cube_owner, None,
                                  tuple(st.board)), None)
    game_index = int(state.get("games", 0))

    def record(decision: Decision) -> bool:
        found.setdefault(decision, None)
        return len(found) < count

    started = time.perf_counter()
    # Timestamps let `status` report a rate and an ETA. They measure THIS
    # process, so a resumed run's ETA reflects the speed it is running at now
    # rather than an average across an interruption.
    started_at, found_at_start = time.time(), len(found)
    print(f"{folder_name}: {len(starts)} seeds, target {count} decisions"
          + (f" (resuming at game {game_index} with {len(found)} already found)"
             if state else ""))

    while len(found) < count and game_index < max_games:
        start = _start_for_game(starts, game_index, seed, folder_name)
        rng = random.Random(f"{seed}:{folder_name}:game{game_index}")
        before = len(found)
        _play_game(start, rng, analyzer, matches, record)
        game_index += 1

        # Only advance the checkpoint once the decisions are actually on disk.
        # A state file claiming games the .txt does not hold would make resume
        # skip past them, losing those decisions for good.
        if write_decisions(folder_name, list(found), seed, model, level):
            _save_state(folder_name, {
                "seed": seed, "model": model, "level": level, "count": count,
                "games": game_index, "found": len(found),
                "started_at": started_at, "found_at_start": found_at_start,
                "updated_at": time.time(),
            })
        elapsed = time.perf_counter() - started
        print(f"  game {game_index:4d}: +{len(found) - before:3d} -> "
              f"{len(found):5d}/{count}  ({elapsed / 60:.1f} min)", flush=True)

    n_checker = sum(1 for d in found if d.kind == "checker")
    summary = {
        "folder": folder_name,
        "decisions": len(found),
        "checker": n_checker,
        "cube": len(found) - n_checker,
        "games": game_index,
        "seconds": time.perf_counter() - started,
        "short": len(found) < count,
    }
    if summary["short"]:
        print(f"  ! stopped {len(found)}/{count} short after {game_index} games "
              f"(--max-games); raise --max-games or check the filter")
    return summary


def cmd_generate(args: argparse.Namespace) -> None:
    from bgsage import BgBotAnalyzer
    from bgsage.weights import WeightConfig

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    folders = [f.name for f in subfolders() if args.folder in (None, f.name)]
    if not folders:
        raise SystemExit(f"No subfolder named {args.folder!r} under {_SRC_DIR}")

    print(f"Model {args.model}, level {args.level} cubeful, seed {args.seed}, "
          f"threads {args.threads or 'auto'}, {len(folders)} subfolder(s)\n",
          flush=True)

    # Built on first use, AFTER the lock: relaunching the whole fleet to revive
    # one dead worker is the normal recovery, and the workers that are still
    # alive should cost a skipped process, not a model load each.
    analyzer = None

    summaries = []
    for name in folders:
        try:
            with _FolderLock(name):
                if analyzer is None:
                    weights = WeightConfig.from_model(args.model)
                    weights.validate()
                    analyzer = BgBotAnalyzer(
                        weights=weights, eval_level=args.level, cubeful=True,
                        parallel_threads=args.threads)
                summaries.append(generate_folder(
                    name, args.count, args.seed, args.model, args.level,
                    args.max_games, args.restart, analyzer))
        except WorkerBusy as exc:
            print(f"SKIPPED {exc}")
        print()

    if not summaries:
        return

    print(f"{'subfolder':32s} {'decisions':>9} {'checker':>8} {'cube':>6} "
          f"{'games':>6} {'minutes':>8}")
    for s in summaries:
        print(f"{s['folder']:32s} {s['decisions']:9d} {s['checker']:8d} "
              f"{s['cube']:6d} {s['games']:6d} {s['seconds'] / 60:8.1f}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


#: ``status`` keeps its own progress samples here, because a worker's state
#: sidecar records only a running total. A rate averaged over the whole run lags
#: badly once the real rate changes -- seen 2026-08-28, a category reporting
#: 9.6 decisions/min while it was actually managing 4. Samples also survive a
#: worker restart (they key off the category's cumulative total), which is
#: exactly when a since-process-start average is least meaningful.
_SAMPLES_FILE = _OUT_DIR / "status_samples.json"
#: History shorter than this is too noisy to quote a rate from; fall back to the
#: run average and say so.
_RATE_WINDOW_MIN = 300.0
#: Never measure over a window longer than this, or the lag comes straight back.
#: Long enough to ride out a single slow game (a game runs 1-3 min).
_RATE_WINDOW_MAX = 1800.0
_SAMPLES_MAX = 200


def _load_samples() -> dict:
    try:
        return json.loads(_SAMPLES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _window_rate(history: list, found: int, now: float) -> tuple[float | None, float]:
    """Decisions/min over the longest retained window, and that window's length.

    ``(None, 0)`` when there is not yet enough history to be worth quoting.
    """
    if not history:
        return None, 0.0
    then, then_found = history[0]
    span = now - then
    if span < _RATE_WINDOW_MIN or found < then_found:
        return None, 0.0
    return (found - then_found) / span * 60, span


def _format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "?"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def cmd_status(args: argparse.Namespace) -> None:
    """Report progress of a (possibly still running) generate run.

    Reads the per-subfolder state sidecars, so it is safe to run against a live
    run and cheap enough to poll on a timer. It does write one thing: its own
    ``status_samples.json``, the progress history the windowed rate is measured
    over. Nothing else reads that file, so a stale or deleted one only costs a
    fallback to the run average.
    """
    del args
    now = time.time()
    rows: list[tuple[str, str, str]] = []
    done_total = target_total = 0
    slowest_eta = 0.0
    unknown_eta = False
    samples = _load_samples()

    for folder in subfolders():
        state = _load_state(folder.name)
        if not state:
            rows.append((folder.name, "not started", ""))
            unknown_eta = True
            continue

        found, count = int(state.get("found", 0)), int(state.get("count", 0))
        done_total += found
        target_total += count
        remaining = max(0, count - found)

        # Rate over a recent window, falling back to the run average until
        # there is enough history. A restarted-from-scratch category (found
        # went backwards) drops its history rather than reporting nonsense.
        history = [s for s in samples.get(folder.name, [])
                   if now - s[0] <= _RATE_WINDOW_MAX and s[1] <= found]
        rate, window = _window_rate(history, found, now)
        if rate is None:
            elapsed = float(state.get("updated_at", 0)) - float(state.get("started_at", 0))
            gained = found - int(state.get("found_at_start", 0))
            rate = gained / elapsed * 60 if elapsed > 0 and gained > 0 else 0.0
            rate_text = f"{rate:5.2f}/min  avg  "
        else:
            rate_text = f"{rate:5.2f}/min {window / 60:3.0f}m "
        history.append([now, found])
        samples[folder.name] = history[-_SAMPLES_MAX:]

        if remaining == 0:
            eta = 0.0
        elif rate > 0:
            eta = remaining / rate * 60
        else:
            eta = float("inf")
            unknown_eta = True
        slowest_eta = max(slowest_eta, eta)

        # A worker still short of its target that has not finished a game in a
        # while is stalled or gone. A finished one is just quiet.
        silent = now - float(state.get("updated_at", 0))
        health = ("" if remaining == 0 or silent < 1800
                  else f"  SILENT {_format_duration(silent)}")

        pct = 100 * found / count if count else 0.0
        rows.append((
            folder.name,
            f"{found:6d}/{count} ({pct:5.1f}%)",
            f"{state.get('games', 0):5d} games  {rate_text} "
            f"eta {_format_duration(eta):>5s}{health}",
        ))

    _atomic_write(_SAMPLES_FILE, json.dumps(samples, separators=(",", ":")))

    overall = 100 * done_total / target_total if target_total else 0.0
    eta_text = "?" if unknown_eta and slowest_eta == float("inf") else \
        _format_duration(slowest_eta)
    print(f"[{time.strftime('%H:%M:%S')}] backgame benchmark: "
          f"{done_total:,}/{target_total:,} decisions ({overall:.1f}%), "
          f"slowest eta {eta_text}")
    for name, progress, detail in rows:
        print(f"  {name:26s} {progress:22s} {detail}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    from bgsage.weights import PRODUCTION_MODEL

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Import the XG reference positions")
    p_import.add_argument("--folder", default=None,
                          help="Only this subfolder (default: all)")
    p_import.set_defaults(func=cmd_import)

    p_gen = sub.add_parser("generate", help="Generate benchmark decisions")
    p_gen.add_argument("--count", type=int, default=DEFAULT_COUNT,
                       help=f"Decisions per subfolder (default: {DEFAULT_COUNT})")
    p_gen.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help=f"Master RNG seed (default: {DEFAULT_SEED})")
    p_gen.add_argument("--model", default=PRODUCTION_MODEL,
                       help=f"Model to play with (default: {PRODUCTION_MODEL})")
    p_gen.add_argument("--folder", default=None,
                       help="Only this subfolder (default: all)")
    p_gen.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES,
                       help=f"Give up after this many games (default: {DEFAULT_MAX_GAMES})")
    p_gen.add_argument("--level", default=DEFAULT_LEVEL,
                       help=f"Evaluation level both sides play at "
                            f"(default: {DEFAULT_LEVEL}). e.g. 3ply, truncated2.")
    p_gen.add_argument("--threads", type=int, default=0,
                       help="Evaluation threads (0 = every CPU). Evaluation "
                            "scales poorly with threads, so one process per "
                            "subfolder with a few threads each beats running "
                            "the subfolders one at a time on all of them.")
    p_gen.add_argument("--restart", action="store_true",
                       help="Discard any existing progress and start fresh")
    p_gen.set_defaults(func=cmd_generate)

    p_status = sub.add_parser("status", help="Report progress of a generate run")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
