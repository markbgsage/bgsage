#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Get eXtreme Gammon's decisions for the back game / containment folder
benchmarks, so XG can be scored on them exactly the way Sage is.

XG has no API. Instead every reference decision of a folder benchmark
(``backgame_ref_positions/benchmark/<folder> rollout.jsonl``, the files
``score_backgame_pr.py`` scores Sage against) is written as a ONE-DECISION GAME
inside a native ``.xg`` archive, a human runs XG's Batch Analyze over those files
(XG writes its analysis into each record in place), and ``score`` reads XG's #1
move / recommended cube action back out and scores it against the folder's
rollout reference with the formulas of ``score_backgame_pr.py``.

WHAT A HUMAN HAS TO DO
----------------------
1. ``py -3.14 scripts/export_folder_benchmark_xg.py generate``
   writes ``data/backgame_xg/<folder-slug>/bench_shard_NNN.xg`` (~200 games per
   shard, so XG can chew through a folder incrementally) plus a
   ``bench_shard_NNN.xg.sidecar.jsonl`` beside each shard mapping game index ->
   reference decision. ``--folders "21 backgame" snake`` restricts the export.
2. XG analyses a file IN PLACE, one analysis per record, so make one copy of the
   whole ``data/backgame_xg/`` tree per XG level you want scored, e.g.
   ``data/backgame_xg_xg2ply/``, ``..._xg3ply/``, ``..._xg4ply/``,
   ``..._xgroller/``, ``..._xgrollerplus/``, ``..._xgrollerpp/``. Copy the
   sidecars along (the harvester also falls back to the ones in
   ``data/backgame_xg/``).
3. In XG: File -> Batch Analyze -> point it at the per-level copy (one
   ``<folder-slug>`` subfolder at a time unless your XG build recurses into
   subfolders), choose the analysis level (2-ply, 3-ply, 4-ply, XG Roller,
   XG Roller+ or XG Roller++), tick "Save games after analyze", run. Repeat per
   level on its own copy of the tree.
4. ``py -3.14 scripts/export_folder_benchmark_xg.py score --xg-dir data/backgame_xg_xg3ply --level xg3ply``
   reports, per folder and pooled: PR, PR over rollout-graded picks only,
   checker PR, cube PR, decisions, blunders (error > 0.08), mismatches and
   unanalysed records, and writes ``data/backgame_xg_scores/xg_<level>_<folder-slug>.json``
   (those numbers plus a ``records`` list) and
   ``xg_<level>_<folder-slug>_picks.jsonl`` (``{key, kind, pick}`` per decision,
   the shape ``benchmark_pr_xg_levels_all.py`` writes for the money/match
   benchmarks) plus ``xg_<level>_summary.json``. ``--level`` is only a label
   for the output files; the report prints the eval levels XG actually stamped
   on the records, so a mislabelled copy is visible.

HOW A DECISION IS ENCODED
-------------------------
Record templates are cloned from a real XG-batch-analyzed game
(``data/money_benchmark/xg/seed_2.xg``: header "Sage" vs "Sage", unlimited
length, Jacoby + Beaver on -- every folder decision is an unlimited game with
both rules on). Each game is ``tsHeaderGame`` + one ``tsCube`` + one ``tsMove``
+ ``tsFooterGame``, exactly the layout the pasko position-eval shards used and
XG analysed (``score_xg_pasko_benchmark.py``). The mover is player 1, so the
mover-perspective reference board IS the player-1-frame board XG stores, and
the game header's initial position is that board.

* Cube state: ``CubeB`` (cube record) / ``CubeA`` (move record) carry the
  owner-signed exponent -- 0 centred, +n = 2^n owned by player 1 (the mover),
  -n owned by player 2. ``export_pasko_benchmark_xg.py`` verified XG honours
  these on every cube state when the records are explicit.
* Checker entry: the cube record has ``Double = -2`` (no cube decision
  recorded -- the value XG itself writes for the opening roll; XG skips such
  records, so no time goes into a cube analysis nobody reads), then a move
  record with the entry's dice playing the reference's best move. XG analyses
  the move record and stores its ranked candidates; #1 is its pick.
* Cube entry: the cube record has ``Double = 0`` ("no double" played -- XG
  analyses records with Double 0 or 1 and produces ND/DT/DP equities plus its
  recommendation in ``FlagDouble``), followed by a move record with a fixed
  continuation roll (the first of 3-1, 4-2, ... that has a legal move; a dance
  if none) playing the first legal move. XG needs a roll after a no-double to
  replay the game; nothing about the continuation is harvested.
* Only decisions the scorer grades are exported: cube entries the opponent owns
  (the mover has no decision) and cube entries whose rollout says both
  sub-decisions are trivial (``score_backgame_pr.cube_subdecisions``) are
  skipped and counted.
* Every written shard is re-parsed and checked: initial position, boards, dice,
  cube codes, the post-move board, that the stored half-moves replay the
  pre-move board into the post-move board (what XG does when it loads the
  game), and the footer.

Two things about this layout have NOT been confirmed against XG's own output and
should be checked on the first analysed shard (``score`` reports them as
unanalysed records if XG skipped them):

* the FIRST cube record of a game carrying a real decision (``Double = 0``) --
  every file XG has written for us starts with the opening roll's -2;
* ``Double = -2`` on a record where the mover owns the cube (checker entries
  with a player-owned cube).

HOW XG IS SCORED
----------------
The same formulas as ``score_backgame_pr.score_level``: a checker pick is
charged ``max(0, best_reference_equity - reference_equity_of_pick)``; a cube
pick is charged the doubler's error ``max(0, max(ND, min(DT, DP)) - actual)``
and/or the receiver's error ``max(0, actual - min(DT, DP))`` for whichever
sub-decisions the rollout makes live. XG's cube pick follows the convention of
``xg_batch_common.xg_level_picks``: double iff ``FlagDouble > 0`` (-1 = too
good, 0 = no double, 1/2 = double), take iff ``DT <= DP``. A checker pick that is
not in the reference's move list (the reference carries every legal move, so
this means an encoding problem, not a bad move) is counted as a MISMATCH and
charged the worst rollout-graded error in the list. As in the Sage scorer, the
full PR and the PR over rollout-graded picks only ("PR(RO)") bracket the truth:
picks outside the reference player's filter set carry a 1-/2-ply equity.

Usage::

    py -3.14 scripts/export_folder_benchmark_xg.py generate
    py -3.14 scripts/export_folder_benchmark_xg.py generate --folders "21 backgame" --limit 20
    py -3.14 scripts/export_folder_benchmark_xg.py score --xg-dir data/backgame_xg_xg3ply --level xg3ply
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = bgsage repo root
_BUILD_DIRS = (_PROJECT_ROOT / "build", _PROJECT_ROOT / "build_msvc")

# Final sys.path order: python, build, build_msvc (fallback), scripts.
for _p in (_SCRIPT_DIR, _BUILD_DIRS[1], _BUILD_DIRS[0], _PROJECT_ROOT / "python"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    for _d in _BUILD_DIRS:
        if _d.is_dir():
            os.add_dll_directory(str(_d))

from bgsage import possible_moves, xg_file as xf  # noqa: E402
from benchmark_money import BLUNDER_THRESHOLD, PR_MULTIPLIER, _scored  # noqa: E402
from export_pasko_benchmark_xg import (  # noqa: E402
    _build_move_rec, _templates, derive_half_moves, write_xg,
)
from score_backgame_pr import cube_subdecisions, load_reference  # noqa: E402

#: The thirteen folder benchmarks (names as ``score_backgame_pr --category`` takes them).
FOLDERS = ("21 backgame", "31 backgame", "32 backgame", "41 backgame", "42 backgame",
           "51 backgame", "52 backgame", "43 backgame", "53 backgame", "54 backgame",
           "containment", "snake", "massive backgame")

DEFAULT_TEMPLATE = _PROJECT_ROOT / "data" / "money_benchmark" / "xg" / "seed_2.xg"
DEFAULT_OUT_DIR = _PROJECT_ROOT / "data" / "backgame_xg"
DEFAULT_SCORES_DIR = _PROJECT_ROOT / "data" / "backgame_xg_scores"
DEFAULT_PER_FILE = 200
SHARD_NAME = "bench_shard_{:03d}.xg"
SIDECAR_SUFFIX = ".sidecar.jsonl"

#: Continuation rolls tried in order after a cube game's no-double; the first
#: with a legal move is played (its first legal move), else the game dances.
CONTINUATION_ROLLS = ((3, 1), (4, 2), (6, 1), (5, 3), (6, 5), (2, 1), (3, 2), (4, 1), (4, 3),
                      (5, 1), (5, 2), (5, 4), (6, 2), (6, 3), (6, 4),
                      (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6))

#: tsCube ``Double`` values (see xg_file.build_cube_record).
NO_CUBE_DECISION = -2
NO_DOUBLE = 0
#: EngineStructDoubleAction.FlagDouble of a record XG has never analysed.
UNANALYSED_FLAG_DOUBLE = -100
_DA_SIZE = 132
#: One-decision game footer: player 1 resigns a single point (100 = resign).
_FOOTER_TERMINATION = 101


def folder_slug(folder: str) -> str:
    return folder.replace(" ", "_")


def cube_code(cube_value: int, cube_owner: str) -> int:
    """XG cube code with the mover as player 1: 0 centred, +n / -n = 2^n owned
    by the mover / the opponent."""
    exp = int(cube_value).bit_length() - 1
    return {"centered": 0, "player": exp, "opponent": -exp}[cube_owner]


def xg_code_to_point(code: int) -> int:
    """XG move code -> bgsage point: -1 off -> 0, 24 bar -> 25, else code + 1."""
    return 0 if code == -1 else 25 if code == 24 else code + 1


def apply_half_moves(board, half_moves) -> list[int]:
    """Replay (from, to) half-moves on a mover-frame board the way XG rebuilds a
    position: a hit sends the blot to the opponent's bar (index 0), ``to == 0``
    bears off."""
    b = list(board)
    for f, t in half_moves:
        if b[f] <= 0:
            raise ValueError(f"half-move {f}->{t}: no mover checker on {f}")
        b[f] -= 1
        if t == 0:
            continue
        if b[t] == -1:
            b[t] = 0
            b[0] += 1
        elif b[t] < -1:
            raise ValueError(f"half-move {f}->{t}: point {t} is blocked")
        b[t] += 1
    return b


def continuation_move(board) -> tuple[int, int, list[int], list]:
    """``(die1, die2, post_board, half_moves)`` for the filler roll after a cube
    game's no-double."""
    for d1, d2 in CONTINUATION_ROLLS:
        cands = possible_moves(board, d1, d2)
        if cands:
            post = list(cands[0])
            return d1, d2, post, derive_half_moves(board, post, d1, d2)
    d1, d2 = CONTINUATION_ROLLS[0]
    return d1, d2, list(board), []          # fully blocked: a dance


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def plan_folder(folder: str, rows: list[dict]) -> tuple[list[dict], Counter]:
    """One-decision games for a folder's reference rows, in reference order,
    plus the counts of rows skipped."""
    games: list[dict] = []
    skipped: Counter = Counter()
    for entry in rows:
        board = list(entry["board"])
        base = {"key": entry["key"], "folder": folder, "kind": entry["kind"], "board": board,
                "cube_value": entry["cube_value"], "cube_owner": entry["cube_owner"],
                "cube_code": cube_code(entry["cube_value"], entry["cube_owner"])}
        if entry["kind"] == "checker":
            d1, d2 = entry["dice"]
            post = list(entry["moves"][0]["board"])
            if post not in possible_moves(board, d1, d2):
                raise ValueError(f"{folder} {entry['key']}: the reference's best move is not "
                                 f"a legal move for {d1}-{d2}")
            half = derive_half_moves(board, post, d1, d2)
            game = {**base, "double": NO_CUBE_DECISION, "dice": [d1, d2], "post": post,
                    "half_moves": half}
        else:
            if entry["cube_owner"] == "opponent":
                skipped["opponent_cube"] += 1
                continue
            has_double, has_take = cube_subdecisions(entry)
            if not (has_double or has_take):
                skipped["trivial_cube"] += 1
                continue
            d1, d2, post, half = continuation_move(board)
            if post == board:
                skipped["dance_continuation"] += 1     # still exported; counted for the log
            game = {**base, "double": NO_DOUBLE, "dice": [d1, d2], "post": post,
                    "half_moves": half, "has_double": has_double, "has_take": has_take}
        if apply_half_moves(board, game["half_moves"]) != game["post"]:
            raise ValueError(f"{folder} {entry['key']}: half-moves {game['half_moves']} do not "
                             f"replay the played move")
        games.append(game)
    return games, skipped


def sidecar_row(game_index: int, g: dict) -> dict:
    row = {"game_index": game_index, "key": g["key"], "folder": g["folder"], "kind": g["kind"],
           "board": g["board"], "dice": g["dice"], "cube_value": g["cube_value"],
           "cube_owner": g["cube_owner"], "cube_code": g["cube_code"], "double": g["double"],
           "played_board": g["post"]}
    if g["kind"] == "cube":
        row["has_double"] = g["has_double"]
        row["has_take"] = g["has_take"]
    return row


def build_cube_rec(tpl: dict, board, double: int, code: int) -> bytes:
    """A tsCube record at ``board`` (mover = player 1) with NO inherited analysis:
    XG's own never-analysed state is an all-zero EngineStructDoubleAction with
    FlagDouble = -100, which is what a fresh record must read as."""
    rec = bytearray(xf.build_cube_record(tpl[xf.TS_CUBE], board, actif=1, double=double,
                                         take=-1, cube_b=code))
    da = xf._CUBE_DOUBLE_ACTION
    rec[da:da + _DA_SIZE] = bytes(_DA_SIZE)
    struct.pack_into("<h", rec, da + xf._DA_FLAG_DOUBLE, UNANALYSED_FLAG_DOUBLE)
    return bytes(rec)


def build_shard_stream(tpl: dict, games: list[dict]) -> bytes:
    """The temp.xg record stream for one shard of one-decision games."""
    stream = bytearray(tpl[xf.TS_HEADER_MATCH])
    xf.set_header_timedelay_totals(stream, 0, 0, 0)
    for gi, g in enumerate(games):
        board = g["board"]
        # Player 1 is the mover, so the P1-frame initial position IS the entry
        # board. Player 2 has "won" every previous one-decision game (footer).
        stream += xf.build_game_header(tpl[xf.TS_HEADER_GAME], board, gi + 1, 0, gi)
        stream += build_cube_rec(tpl, board, g["double"], g["cube_code"])
        d1, d2 = g["dice"]
        stream += _build_move_rec(tpl, board, g["post"], d1, d2, g["half_moves"], 1,
                                  g["cube_code"])
        stream += xf.build_game_footer(tpl[xf.TS_FOOTER_GAME], 0, gi + 1, -1, 1,
                                       _FOOTER_TERMINATION)
    return bytes(stream)


def iter_games(tx: bytes) -> list[dict]:
    """The games of a temp.xg stream: header offset, game number and the offsets
    of the cube / move / footer records that follow each game header."""
    games: list[dict] = []
    cur: dict | None = None
    for off, rt in xf.iter_records(tx):
        if rt == xf.TS_HEADER_GAME:
            cur = {"header": off, "game_number": xf.parse_game_header(tx, off)["game_number"],
                   "cube": [], "move": [], "footer": None}
            games.append(cur)
        elif cur is None:
            continue
        elif rt == xf.TS_CUBE:
            cur["cube"].append(off)
        elif rt == xf.TS_MOVE:
            cur["move"].append(off)
        elif rt == xf.TS_FOOTER_GAME:
            cur["footer"] = off
    return games


def verify_shard(path: Path, games: list[dict]) -> None:
    """Re-load a written shard and check every game against what was intended."""
    arch = xf.XgArchive.load(path)
    tx = arch.get("temp.xg")
    if arch.get("temp.xgi") != xf.rebuild_xgi(tx):
        raise AssertionError(f"{path.name}: temp.xgi is not first+last record")
    recs = list(xf.iter_records(tx))
    if not recs or recs[0][1] != xf.TS_HEADER_MATCH:
        raise AssertionError(f"{path.name}: stream does not start with a match header")
    hdr = xf.parse_header(tx, recs[0][0])
    if not (hdr["jacoby"] and hdr["beaver"]):
        raise AssertionError(f"{path.name}: header must carry Jacoby + Beaver: {hdr}")
    parsed = iter_games(tx)
    if len(parsed) != len(games):
        raise AssertionError(f"{path.name}: {len(parsed)} games in stream, {len(games)} written")
    for gi, (g, pg) in enumerate(zip(games, parsed)):
        board, tag = list(g["board"]), f"{path.name} game {gi}"
        if pg["game_number"] != gi + 1:
            raise AssertionError(f"{tag}: game number {pg['game_number']}")
        posinit = xf.norm_bars(xf.read_position(tx, pg["header"] + xf._HG_POSINIT))
        if posinit != board:
            raise AssertionError(f"{tag}: initial position mismatch")
        if len(pg["cube"]) != 1 or len(pg["move"]) != 1 or pg["footer"] is None:
            raise AssertionError(f"{tag}: expected cube + move + footer, got "
                                 f"{len(pg['cube'])} cube / {len(pg['move'])} move records")
        c = xf.parse_cube_record(tx, pg["cube"][0])
        if (c["actif"], list(c["mover_board"]), c["double"], c["take"], c["cube_b"]) != \
                (1, board, g["double"], -1, g["cube_code"]):
            raise AssertionError(f"{tag}: cube record fields differ from intent")
        if c["analyze_c"] != -1 or c["flag_double"] != UNANALYSED_FLAG_DOUBLE:
            raise AssertionError(f"{tag}: cube record not in the never-analysed state")
        m_off = pg["move"][0]
        m = xf.parse_move_record(tx, m_off)
        if (m["actif"], list(m["mover_board"]), m["dice"], m["cube_a"], m["n_moves"]) != \
                (1, board, list(g["dice"]), g["cube_code"], 0):
            raise AssertionError(f"{tag}: move record fields differ from intent")
        post = xf.norm_bars(xf.read_position(tx, m_off + xf._MOVE_POSITION_END))
        if post != list(g["post"]):
            raise AssertionError(f"{tag}: post-move board mismatch")
        played = struct.unpack_from("<8i", tx, m_off + xf._MOVE_PLAYED)
        half = [(xg_code_to_point(played[2 * i]), xg_code_to_point(played[2 * i + 1]))
                for i in range(4) if played[2 * i] != -1]
        if apply_half_moves(board, half) != list(g["post"]):
            raise AssertionError(f"{tag}: stored half-moves {half} do not replay to the post board")
        s1, s2 = struct.unpack_from("<2i", tx, pg["footer"] + xf._FG_SCORE1)
        w, pts, term = struct.unpack_from("<3i", tx, pg["footer"] + xf._FG_WINNER)
        if (s1, s2, w, pts, term) != (0, gi + 1, -1, 1, _FOOTER_TERMINATION):
            raise AssertionError(f"{tag}: footer fields {(s1, s2, w, pts, term)}")


def _selected_folders(names) -> list[str]:
    if not names:
        return list(FOLDERS)
    bad = [n for n in names if n not in FOLDERS]
    if bad:
        raise SystemExit(f"Unknown folder(s) {bad}; choose from: {', '.join(FOLDERS)}")
    return list(names)


def cmd_generate(args) -> None:
    if not args.template.exists():
        raise SystemExit(f"Template .xg not found: {args.template}")
    tpl = _templates(args.template)
    for rt in (xf.TS_HEADER_MATCH, xf.TS_HEADER_GAME, xf.TS_CUBE, xf.TS_MOVE, xf.TS_FOOTER_GAME):
        if rt not in tpl:
            raise SystemExit(f"{args.template}: no record of type {rt} to clone")
    folders = _selected_folders(args.folders)
    per_file = max(1, args.per_file)
    print(f"Exporting {len(folders)} folder benchmark(s) to {args.out_dir} "
          f"(template {args.template.name}, <= {per_file} games per shard)", flush=True)

    rows_out = []
    started = time.perf_counter()
    for folder in folders:
        rows = load_reference(folder, args.limit)
        games, skipped = plan_folder(folder, rows)
        out_dir = args.out_dir / folder_slug(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in list(out_dir.glob("bench_shard_*.xg")) + \
                list(out_dir.glob("bench_shard_*" + SIDECAR_SUFFIX)):
            stale.unlink()
        n_shards = (len(games) + per_file - 1) // per_file
        for s in range(n_shards):
            chunk = games[s * per_file:(s + 1) * per_file]
            path = out_dir / SHARD_NAME.format(s)
            write_xg(args.template, build_shard_stream(tpl, chunk), path)
            with open(str(path) + SIDECAR_SUFFIX, "w", encoding="utf-8") as f:
                for gi, g in enumerate(chunk):
                    f.write(json.dumps(sidecar_row(gi, g), separators=(",", ":")) + "\n")
            verify_shard(path, chunk)
        n_checker = sum(1 for g in games if g["kind"] == "checker")
        row = {"folder": folder, "entries": len(rows), "games": len(games),
               "checker": n_checker, "cube": len(games) - n_checker, "shards": n_shards,
               "skipped_opponent_cube": skipped["opponent_cube"],
               "skipped_trivial_cube": skipped["trivial_cube"],
               "dance_continuations": skipped["dance_continuation"]}
        rows_out.append(row)
        print(f"  {folder:18s} {row['entries']:5d} entries -> {row['games']:5d} games "
              f"({row['checker']} checker, {row['cube']} cube) in {n_shards:2d} shard(s); "
              f"skipped {row['skipped_opponent_cube']} opponent-cube, "
              f"{row['skipped_trivial_cube']} trivial-cube"
              + (f"; {row['dance_continuations']} cube games dance" if row["dance_continuations"] else "")
              + " -- verified", flush=True)

    tot = {k: sum(r[k] for r in rows_out) for k in
           ("entries", "games", "checker", "cube", "shards", "skipped_opponent_cube",
            "skipped_trivial_cube")}
    print(f"\nTotal: {tot['entries']} entries -> {tot['games']} games ({tot['checker']} checker, "
          f"{tot['cube']} cube) in {tot['shards']} shards; skipped {tot['skipped_opponent_cube']} "
          f"opponent-cube + {tot['skipped_trivial_cube']} trivial-cube entries "
          f"[{time.perf_counter() - started:.0f}s]")
    print(f"\nNext: copy {args.out_dir} to one folder per XG level (e.g. "
          f"{args.out_dir.name}_xg3ply), XG -> File -> Batch Analyze each copy at that level "
          f"with 'Save games after analyze' ON, then:\n"
          f"  py -3.14 scripts/{Path(__file__).name} score --xg-dir <copy> --level <name>")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def extract_checker_pick(tx: bytes, move_off: int) -> tuple[list[int] | None, dict]:
    """XG's #1 move (mover frame) for a tsMove record, or None when unanalysed.

    XG stores its candidates already ranked, so ``moves[0]`` IS its pick -- never
    the argmax of the stored equities (see ``xg_batch_common.xg_level_picks``).
    """
    rec = xf.parse_move_record(tx, move_off)
    if rec["n_moves"] <= 0 or not rec["moves"]:
        return None, rec
    return list(rec["moves"][0]["board"]), rec


def extract_cube_pick(tx: bytes, cube_off: int) -> tuple[dict | None, dict]:
    """XG's cube recommendation for a tsCube record, or None when unanalysed."""
    rec = xf.parse_cube_record(tx, cube_off)
    if rec["analyze_c"] < 0 or rec["flag_double"] == UNANALYSED_FLAG_DOUBLE:
        return None, rec
    return {"should_double": rec["flag_double"] > 0,
            "should_take": rec["equity_dt"] <= rec["equity_dp"]}, rec


def score_checker(entry: dict, pick) -> dict:
    """``{error, rollout_graded, mismatch}`` for a checker pick, as score_backgame_pr
    charges it; a pick outside the reference list is a mismatch charged the
    worst rollout-graded error."""
    best = entry["moves"][0]["equity"]
    picked = next((m for m in entry["moves"] if list(m["board"]) == list(pick)), None)
    if picked is None:
        rolled = [m["equity"] for m in entry["moves"] if m["eval_level"] == "Rollout"] \
            or [m["equity"] for m in entry["moves"]]
        return {"error": max(0.0, best - min(rolled)), "rollout_graded": False, "mismatch": True}
    return {"error": max(0.0, best - picked["equity"]),
            "rollout_graded": picked["eval_level"] == "Rollout", "mismatch": False}


def score_cube(entry: dict, should_double: bool, should_take: bool) -> list[dict]:
    """The live sub-decisions of a cube entry scored exactly as score_backgame_pr does."""
    has_double, has_take = cube_subdecisions(entry)
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    scored = []
    if has_double:
        optimal = max(nd, min(dt, dp))
        actual = min(dt, dp) if should_double else nd
        scored.append(_scored("cube", "double", None, max(0.0, optimal - actual)))
    if has_take:
        optimal = min(dt, dp)
        actual = dt if should_take else dp
        scored.append(_scored("cube", "take", None, max(0.0, actual - optimal)))
    return scored


def new_tally() -> dict:
    return {"sum_checker": 0.0, "sum_cube": 0.0, "n_checker": 0, "n_cube": 0,
            "blunders_checker": 0, "blunders_cube": 0, "ro_sum": 0.0, "ro_n": 0,
            "filt_sum": 0.0, "mismatches": 0, "unanalysed_checker": 0, "unanalysed_cube": 0,
            "n_games": 0, "n_shards": 0, "xg_levels": Counter()}


def add_tally(total: dict, part: dict) -> None:
    for k, v in part.items():
        if isinstance(v, Counter):
            total[k].update(v)
        else:
            total[k] += v


def finish_tally(t: dict, level: str, folder: str) -> dict:
    n_total = t["n_checker"] + t["n_cube"]
    err_total = t["sum_checker"] + t["sum_cube"]

    def _pr(total: float, n: int) -> float:
        return (total / n * PR_MULTIPLIER) if n else 0.0

    return {
        "level": level,
        "folder": folder,
        "slug": folder_slug(folder) if folder != "ALL" else "ALL",
        "pr": _pr(err_total, n_total),
        # Cube sub-decisions are always reference-grade, so they count here.
        "pr_rollout_graded": _pr(t["ro_sum"] + t["sum_cube"], t["ro_n"] + t["n_cube"]),
        "checker_pr": _pr(t["sum_checker"], t["n_checker"]),
        "cube_pr": _pr(t["sum_cube"], t["n_cube"]),
        "n": n_total,
        "n_checker": t["n_checker"],
        "n_cube": t["n_cube"],
        "blunders": t["blunders_checker"] + t["blunders_cube"],
        "blunders_checker": t["blunders_checker"],
        "blunders_cube": t["blunders_cube"],
        "mean_error": err_total / n_total if n_total else 0.0,
        "rollout_grade_pct": 100 * t["ro_n"] / t["n_checker"] if t["n_checker"] else 0.0,
        "filter_graded_mass_pct": 100 * t["filt_sum"] / err_total if err_total else 0.0,
        "mismatches": t["mismatches"],
        "unanalysed": t["unanalysed_checker"] + t["unanalysed_cube"],
        "unanalysed_checker": t["unanalysed_checker"],
        "unanalysed_cube": t["unanalysed_cube"],
        "n_games": t["n_games"],
        "n_shards": t["n_shards"],
        "xg_levels": dict(sorted(t["xg_levels"].items())),
    }


def load_sidecar(path: Path, gen_dir: Path | None) -> list[dict]:
    """The shard's sidecar rows: beside the analysed file, else the generated tree's copy."""
    candidates = [Path(str(path) + SIDECAR_SUFFIX)]
    if gen_dir is not None:
        candidates.append(gen_dir / (path.name + SIDECAR_SUFFIX))
        if "__" in path.name:   # flat-layout name: <slug>__bench_shard_NNN.xg
            candidates.append(gen_dir / (path.name.split("__", 1)[1] + SIDECAR_SUFFIX))
    for p in candidates:
        if p.exists():
            return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    raise FileNotFoundError(f"{path}: no sidecar found at {' or '.join(map(str, candidates))}")


def harvest_folder(xg_dir: Path, gen_dir: Path, folder: str) -> tuple[dict, list[dict]] | None:
    """Score every analysed shard of one folder. Returns ``(tally, records)`` or
    None when the level directory has no shards for the folder."""
    slug = folder_slug(folder)
    files = sorted((xg_dir / slug).glob("bench_shard_*.xg"))
    if not files:
        # flat layout: one folder per XG level holding every family's shards as
        # <slug>__bench_shard_NNN.xg (so one Batch Analyze selection covers them all)
        files = sorted(xg_dir.glob(f"{slug}__bench_shard_*.xg"))
    if not files:
        return None
    ref = {r["key"]: r for r in load_reference(folder, None)}
    tally, records = new_tally(), []
    for path in files:
        sidecar = load_sidecar(path, gen_dir / slug)
        tx = xf.XgArchive.load(path).get("temp.xg")
        games = iter_games(tx)
        if len(games) != len(sidecar):
            raise ValueError(f"{path}: {len(games)} games in file, {len(sidecar)} sidecar rows")
        tally["n_shards"] += 1
        for sc, pg in zip(sidecar, games):
            tally["n_games"] += 1
            entry = ref.get(sc["key"])
            if entry is None:
                raise KeyError(f"{path} game {sc['game_index']}: key {sc['key']} is not in the "
                               f"{folder!r} reference any more")
            if len(pg["cube"]) != 1 or len(pg["move"]) != 1:
                raise ValueError(f"{path} game {sc['game_index']}: expected one cube + one move "
                                 f"record, found {len(pg['cube'])} + {len(pg['move'])}")
            tag = f"{path.name} game {sc['game_index']} ({sc['kind']} {sc['key']})"
            if sc["kind"] == "checker":
                pick, rec = extract_checker_pick(tx, pg["move"][0])
                if list(rec["mover_board"]) != list(sc["board"]) or \
                        sorted(rec["dice"]) != sorted(sc["dice"]):
                    raise ValueError(f"{tag}: move record does not match the sidecar position")
                if pick is None:
                    tally["unanalysed_checker"] += 1
                    continue
                tally["xg_levels"][xf.player_level_label(rec["analyze_m"])] += 1
                s = score_checker(entry, pick)
                err = s["error"]
                tally["sum_checker"] += err
                tally["n_checker"] += 1
                tally["blunders_checker"] += err > BLUNDER_THRESHOLD
                tally["mismatches"] += s["mismatch"]
                if s["rollout_graded"]:
                    tally["ro_sum"] += err
                    tally["ro_n"] += 1
                else:
                    tally["filt_sum"] += err
                records.append({"key": sc["key"], "kind": "checker", "pick": pick,
                                "scored": [_scored("checker", "checker", None, err)],
                                "rollout_graded": s["rollout_graded"], "mismatch": s["mismatch"]})
            else:
                pick, rec = extract_cube_pick(tx, pg["cube"][0])
                if list(rec["mover_board"]) != list(sc["board"]):
                    raise ValueError(f"{tag}: cube record does not match the sidecar position")
                if pick is None:
                    tally["unanalysed_cube"] += 1
                    continue
                tally["xg_levels"][xf.player_level_label(rec["level"])] += 1
                scored = score_cube(entry, pick["should_double"], pick["should_take"])
                for s in scored:
                    tally["sum_cube"] += s["error"]
                    tally["n_cube"] += 1
                    tally["blunders_cube"] += s["is_blunder"]
                records.append({"key": sc["key"], "kind": "cube", "pick": pick, "scored": scored,
                                "xg_equities": {"nd": rec["equity_nd"], "dt": rec["equity_dt"],
                                                "dp": rec["equity_dp"]}})
    return tally, records


def _print_table(results: list[dict]) -> None:
    print(f"\n{'folder':18s} {'PR':>7} {'PR(RO)':>7} {'checker':>8} {'cube':>7} {'n':>6} "
          f"{'blund':>6} {'RO-grade':>9} {'filt mass':>10} {'mism':>5} {'unanal':>7} {'games':>6}  xg levels")
    print("-" * 132)
    for r in results:
        levels = " ".join(f"{k}x{v}" for k, v in r["xg_levels"].items()) or "-"
        print(f"{r['folder']:18s} {r['pr']:7.2f} {r['pr_rollout_graded']:7.2f} "
              f"{r['checker_pr']:8.2f} {r['cube_pr']:7.2f} {r['n']:6d} {r['blunders']:6d} "
              f"{r['rollout_grade_pct']:8.1f}% {r['filter_graded_mass_pct']:9.1f}% "
              f"{r['mismatches']:5d} {r['unanalysed']:7d} {r['n_games']:6d}  {levels}")


def cmd_score(args) -> None:
    xg_dir = args.xg_dir
    if not xg_dir.is_dir():
        raise SystemExit(f"{xg_dir} is not a directory")
    folders = _selected_folders(args.folders)
    args.scores_dir.mkdir(parents=True, exist_ok=True)

    results, pooled, missing = [], new_tally(), []
    for folder in folders:
        out = harvest_folder(xg_dir, args.gen_dir, folder)
        if out is None:
            missing.append(folder)
            continue
        tally, records = out
        add_tally(pooled, tally)
        res = finish_tally(tally, args.level, folder)
        results.append(res)
        slug = folder_slug(folder)
        (args.scores_dir / f"xg_{args.level}_{slug}.json").write_text(
            json.dumps({**res, "records": records}, indent=1), encoding="utf-8")
        (args.scores_dir / f"xg_{args.level}_{slug}_picks.jsonl").write_text(
            "".join(json.dumps({"key": r["key"], "kind": r["kind"], "pick": r["pick"]},
                               separators=(",", ":")) + "\n" for r in records),
            encoding="utf-8")
    if not results:
        raise SystemExit(f"No bench_shard_*.xg found under {xg_dir}/<folder-slug>/ for "
                         f"{', '.join(folders)}")

    print(f"XG level {args.level!r} from {xg_dir}: {len(results)} folder(s), "
          f"{pooled['n_games']} games in {pooled['n_shards']} shards; blunder > {BLUNDER_THRESHOLD}, "
          f"PR = mean error x {PR_MULTIPLIER}")
    if missing:
        print(f"  no shards for: {', '.join(missing)}")
    all_res = finish_tally(pooled, args.level, "ALL")
    _print_table(results + [all_res])
    (args.scores_dir / f"xg_{args.level}_summary.json").write_text(
        json.dumps({"level": args.level, "xg_dir": str(xg_dir), "folders": results,
                    "pooled": all_res}, indent=1), encoding="utf-8")
    print(f"\nWrote xg_{args.level}_<folder-slug>.json / _picks.jsonl and "
          f"xg_{args.level}_summary.json to {args.scores_dir}")
    if pooled["unanalysed_checker"] + pooled["unanalysed_cube"]:
        print(f"NOTE: {pooled['unanalysed_checker']} checker + {pooled['unanalysed_cube']} cube "
              f"records carry no XG analysis (not yet batch-analysed, or XG skipped them).")
    if pooled["mismatches"]:
        print("WARNING: some XG picks were not in the reference move list (charged the worst "
              "rollout-graded error). The reference carries every legal move, so this points "
              "at a board-encoding problem, not at XG's play.")
    if all_res["n"]:
        print("RO-grade = share of checker picks whose reference equity is rollout-grade; "
              "PR(RO) scores only those (plus cube decisions); 'filt mass' is the share of the "
              "error carried by filter-graded picks. PR and PR(RO) bracket the truth.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the folder benchmarks as .xg shards for XG")
    g.add_argument("--folders", nargs="+", default=None,
                   help='Folder benchmarks to export (default: all thirteen), e.g. "21 backgame"')
    g.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                   help="Real XG-analysed .xg archive to clone record templates from")
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output tree (default: {DEFAULT_OUT_DIR})")
    g.add_argument("--per-file", type=int, default=DEFAULT_PER_FILE,
                   help=f"Games per shard (default {DEFAULT_PER_FILE})")
    g.add_argument("--limit", type=int, default=None,
                   help="Export only the first N reference decisions of each folder")
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="harvest XG-analysed shards and score XG's picks")
    s.add_argument("--xg-dir", type=Path, required=True,
                   help="Per-level copy of the generated tree that XG has analysed")
    s.add_argument("--level", required=True,
                   help="Label for the output files, e.g. xg3ply, xgroller, xgrollerpp")
    s.add_argument("--folders", nargs="+", default=None,
                   help="Folder benchmarks to score (default: every one with shards)")
    s.add_argument("--gen-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Generated tree, used for sidecars missing beside the analysed shards")
    s.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR,
                   help=f"Where the score files go (default: {DEFAULT_SCORES_DIR})")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
