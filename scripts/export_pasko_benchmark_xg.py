# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Export the Paskogammon benchmark games as native .xg files for XG Batch Analyze.

The benchmark's ``.txt`` transcripts (``data/pasko_money_benchmark/xg/seed_N.txt``)
cannot be imported by XG: the Jellyfish/Galaxy text format has no way to declare a
starting position, and XG's text import replays the move list from the STANDARD
backgammon start -- but Paskogammon starts player 2's checkers in a scattered
arrangement, so the transcripts only replay correctly from the Pasko start.

This script writes each benchmark game as a **native .xg archive** instead. The .xg
record format carries every position explicitly (each move record stores its own
pre/post boards, and the game header stores the initial position), so XG hosts the
Paskogammon start without needing to understand the variant -- the same mechanism
``score_xg_pasko_benchmark.py`` used to get XG to batch-analyze single Pasko
positions, validated end-to-end there.

Record encoding (verified against XG's own output: ``example.xg``, a pasko-start
game recorded inside XG, and the money benchmark's XG-batch-analyzed games):

  * one ``tsCube`` + ``tsMove`` record pair per half-turn (mover ``actif`` = +1/-1);
  * the cube record's ``Double`` field is -2 when no cube decision exists (the
    game-opening roll, or the opponent owns the cube), 0 for a real no-double
    decision, 1 for a double -- with ``Take`` = -1 / 1 (taken) / 0 (dropped);
  * ``CubeB`` (cube record) / ``CubeA`` (move record) hold the cube as an
    owner-signed exponent: 0 = centered 1-cube, +n / -n = 2^n owned by P1 / P2;
  * a double/pass emits the cube record only, then the game footer
    (``termination`` 0 = drop, 1/2/3 = single/gammon/backgammon win);
  * no match footer (matches XG's own .txt-import output).

Each game is **re-simulated from its seed** (the pass-1 self-play is deterministic)
and cross-checked byte-for-byte against the stored ``.txt`` transcript before
anything is written, so the .xg files are guaranteed to encode exactly the games the
benchmark captured. Every written archive is then re-parsed and its positions,
dice, and cube actions verified against the replay.

The match header (player names "Sage", unlimited length, Jacoby + Beaver on) and
all record templates are cloned from an XG-batch-analyzed money benchmark game.

Usage::

    python scripts/export_pasko_benchmark_xg.py                # all stage-1 seeds
    python scripts/export_pasko_benchmark_xg.py --seeds 1 2 3  # specific seeds

Then: XG -> File -> Batch Analyze -> point it at data/pasko_money_benchmark/xg_native/
("Save games after analyze" ON; XG updates the .xg files in place).
"""

from __future__ import annotations

import argparse
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"
_BUILD_DIR = _PROJECT_ROOT / "build"

for _p in (_SCRIPT_DIR, _BGSAGE_PYTHON, _BUILD_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import os  # noqa: E402

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

import benchmark_pasko as bp  # noqa: E402  (pasko paths + deterministic simulator)

#: Default record-template archive: an XG-batch-analyzed money benchmark game --
#: its match header already carries "Sage"/"Sage", unlimited length, Jacoby+Beaver.
DEFAULT_TEMPLATE = _PROJECT_ROOT / "data" / "money_benchmark" / "xg" / "seed_2.xg"

#: Output directory -- kept SEPARATE from the .txt dir so an XG Batch Analyze of
#: this folder never chews the (standard-start-replayed, i.e. wrong) transcripts.
DEFAULT_OUT_DIR = bp._DATA_DIR / "xg_native"

_TERMINATION = {"drop": 0, "single": 1, "gammon": 2, "backgammon": 3}


# ---------------------------------------------------------------------------
# Game replay: move_history (notation) -> per-turn boards + half-moves
# ---------------------------------------------------------------------------


def _cube_code(cube_value: int, owner: int) -> int:
    """XG cube code: 0 = centered 1-cube; +/-n = 2^n owned by P1 / P2."""
    exp = cube_value.bit_length() - 1
    return 0 if owner == 0 else owner * exp


def derive_half_moves(pre, post, d1, d2):
    """Exact per-die (from, to) half-move sequence (bgsage points) for pre -> post.

    XG replays half-moves, so each must use a single die. Same DFS as
    ``score_xg_pasko_benchmark.derive_half_moves``.
    """
    from bgsage import possible_single_die_moves

    target = tuple(post)
    orders = [[d1] * k for k in (4, 3, 2, 1)] if d1 == d2 else [[d1, d2], [d2, d1]]

    def dfs(board, dice_left):
        if tuple(board) == target:
            return []
        if not dice_left:
            return None
        for mv in possible_single_die_moves(board, dice_left[0]):
            sub = dfs(mv["board"], dice_left[1:])
            if sub is not None:
                return [(mv["from"], mv["to"])] + sub
        return dfs(board, dice_left[1:])

    best = None
    for order in orders:
        res = dfs(pre, order)
        if res is not None and (best is None or len(res) > len(best)):
            best = res
    return best or []


def _replay_turns(move_history: list[dict]) -> list[dict]:
    """Replay the recorded game; return one dict per half-turn with boards resolved.

    Each returned turn has: ``player`` (+1/-1), ``cube_action``, ``pre`` (mover-frame
    pre-roll board), and for rolled turns ``dice``, ``post`` (mover-frame post-move
    board), ``half_moves``. The post board is recovered by matching the stored
    notation against ``possible_moves`` -- the same function that generated the
    notation -- and the match is asserted unique.
    """
    from bgsage import flip_board, possible_moves
    from bgsage.text_export import compute_move_notation

    board = list(bp.PASKO_STARTING_BOARD)
    turns: list[dict] = []
    for entry in move_history:
        p = 1 if entry.get("player") == "user" else -1
        turn: dict = {"player": p, "cube_action": entry.get("cube_action"),
                      "pre": list(board)}
        dice = entry.get("dice")
        if dice is None:                      # double/pass: no roll this turn
            turns.append(turn)
            continue
        d1, d2 = int(dice[0]), int(dice[1])
        notation = entry.get("move") or ""
        cands = {tuple(c) for c in possible_moves(board, d1, d2)}
        if not cands:
            if notation:
                raise ValueError(f"turn {len(turns)}: no legal moves but notation {notation!r}")
            post = list(board)                # dance: board unchanged
        else:
            matches = {c for c in cands
                       if compute_move_notation(board, list(c), d1, d2) == notation}
            if len(matches) != 1:
                raise ValueError(
                    f"turn {len(turns)}: notation {notation!r} matched {len(matches)} "
                    f"of {len(cands)} candidates")
            post = list(matches.pop())
        turn.update({
            "dice": (d1, d2),
            "post": post,
            "half_moves": derive_half_moves(board, post, d1, d2) if post != board else [],
        })
        turns.append(turn)
        board = list(flip_board(post))        # next mover's frame
    return turns


# ---------------------------------------------------------------------------
# .xg assembly
# ---------------------------------------------------------------------------


def _templates(template_path: Path) -> dict:
    """One record of each type from a real .xg, to clone (as in score_xg_pasko_benchmark)."""
    from bgsage import xg_file as xf

    tempxg = xf.XgArchive.load(template_path).get("temp.xg")
    tpl = {}
    for off, rt in xf.iter_records(tempxg):
        tpl.setdefault(rt, tempxg[off:off + xf.TSAVEREC_SIZE])
    return tpl


def _build_cube_rec(tpl: dict, board_mover, actif: int,
                    double: int, take: int, cube_b: int) -> bytes:
    """A tsCube record with the double/take/cube fields set explicitly.

    ``build_cube_record`` clears the analysis fields but leaves Double/Take/CubeB
    at the template's values -- for full games those must reflect the actual cube
    history or XG mis-reconstructs the game state.
    """
    from bgsage import xg_file as xf

    rec = bytearray(xf.build_cube_record(tpl[xf.TS_CUBE], board_mover, actif=actif))
    struct.pack_into("<i", rec, xf._CUBE_DOUBLE, double)
    struct.pack_into("<i", rec, xf._CUBE_TAKE, take)
    struct.pack_into("<i", rec, xf._CUBE_B, cube_b)
    return bytes(rec)


def _build_move_rec(tpl: dict, pre_mover, post_mover, d1: int, d2: int,
                    half_moves, actif: int, cube_a: int) -> bytes:
    """A tsMove record with the cube-state field set (build_move_record leaves it)."""
    from bgsage import xg_file as xf

    rec = bytearray(xf.build_move_record(
        tpl[xf.TS_MOVE], pre_mover, post_mover, d1, d2, half_moves, actif=actif))
    struct.pack_into("<i", rec, xf._MOVE_CUBE_A, cube_a)
    return bytes(rec)


def build_game_stream(tpl: dict, turns: list[dict], result: dict) -> bytes:
    """Assemble the temp.xg record stream for one Paskogammon game.

    ``turns`` from :func:`_replay_turns`; ``result`` carries ``winner`` (+1/-1),
    ``win_type`` ("drop"/"single"/"gammon"/"backgammon"), ``points``.
    """
    from bgsage import xg_file as xf  # noqa: F811 (used throughout)

    stream = bytearray(tpl[xf.TS_HEADER_MATCH])
    xf.set_header_timedelay_totals(stream, 0, 0, 0)
    stream += xf.build_game_header(tpl[xf.TS_HEADER_GAME], list(bp.PASKO_STARTING_BOARD),
                                   game_number=1, score1=0, score2=0)

    cube_value, owner = 1, 0                  # owner: 0 centered, +1 / -1 absolute
    for i, t in enumerate(turns):
        p = t["player"]
        code_before = _cube_code(cube_value, owner)
        ca = t["cube_action"]

        if ca == "double/take":
            stream += _build_cube_rec(tpl, t["pre"], p, double=1, take=1, cube_b=code_before)
            cube_value, owner = cube_value * 2, -p
        elif ca == "double/pass":
            stream += _build_cube_rec(tpl, t["pre"], p, double=1, take=0, cube_b=code_before)
            break                             # game ends; footer follows
        else:
            # -2 = no cube decision exists: the opening roll, or no cube access.
            has_access = owner in (0, p)
            double = 0 if (i > 0 and has_access) else -2
            stream += _build_cube_rec(tpl, t["pre"], p, double=double, take=-1,
                                      cube_b=code_before)

        d1, d2 = t["dice"]
        stream += _build_move_rec(tpl, t["pre"], t["post"], d1, d2, t["half_moves"],
                                  actif=p, cube_a=_cube_code(cube_value, owner))

    winner = result["winner"]
    points = result["points"]
    score1 = points if winner == 1 else 0
    score2 = points if winner == -1 else 0
    stream += xf.build_game_footer(tpl[xf.TS_FOOTER_GAME], score1, score2, winner,
                                   points, _TERMINATION[result["win_type"]])
    return bytes(stream)


def write_xg(template_path: Path, stream: bytes, out_path: Path) -> None:
    from bgsage import xg_file as xf

    arch = xf.XgArchive.load(template_path)
    arch.set("temp.xg", stream)
    arch.set("temp.xgi", xf.rebuild_xgi(stream))
    arch.save(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _strip_timestamps(txt: bytes) -> bytes:
    """Drop the EventDate/EventTime header lines (stamped at write time, not game data)."""
    return b"\n".join(
        line for line in txt.split(b"\n")
        if not line.startswith((b'; [EventDate', b'; [EventTime'))
    )


def _verify_txt_roundtrip(seed: int, xg_record: dict) -> None:
    """The re-simulated game must reproduce the stored transcript byte-for-byte
    (up to the write-time EventDate/EventTime header stamps)."""
    from bgsage.text_export import export_history_to_txt

    txt_path = bp._XG_DIR / f"seed_{seed}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"{txt_path} missing -- run pass 1 first")
    regenerated = export_history_to_txt(xg_record)
    if _strip_timestamps(regenerated) != _strip_timestamps(txt_path.read_bytes()):
        raise RuntimeError(
            f"seed {seed}: re-simulated game differs from the stored transcript "
            f"({txt_path}) -- refusing to export a mismatched .xg")


def _verify_archive(out_path: Path, turns: list[dict], result: dict) -> None:
    """Re-load the written .xg and check records against the replay."""
    from bgsage import xg_file as xf

    tx = xf.XgArchive.load(out_path).get("temp.xg")
    recs = list(xf.iter_records(tx))
    assert recs[0][1] == xf.TS_HEADER_MATCH and recs[1][1] == xf.TS_HEADER_GAME
    posinit = xf.norm_bars(xf.read_position(tx, recs[1][0] + xf._HG_POSINIT))
    assert posinit == list(bp.PASKO_STARTING_BOARD), "posinit mismatch"
    assert recs[-1][1] == xf.TS_FOOTER_GAME, "missing game footer"

    body = recs[2:-1]
    ti = 0
    for off, rt in body:
        t = turns[ti]
        if rt == xf.TS_CUBE:
            c = xf.parse_cube_record(tx, off)
            assert c["actif"] == t["player"], f"turn {ti}: cube actif"
            assert list(c["mover_board"]) == list(t["pre"]), f"turn {ti}: cube board"
        elif rt == xf.TS_MOVE:
            m = xf.parse_move_record(tx, off)
            assert m["actif"] == t["player"], f"turn {ti}: move actif"
            assert list(m["mover_board"]) == list(t["pre"]), f"turn {ti}: move board"
            assert m["dice"] == list(t["dice"]), f"turn {ti}: dice"
            post = xf.norm_bars(xf.read_position(tx, off + xf._MOVE_POSITION_END))
            assert post == list(t["post"]), f"turn {ti}: post board"
            ti += 1
        else:
            raise AssertionError(f"unexpected record type {rt} mid-game")
    # A double/pass turn has a cube record but no move record.
    expected_moves = sum(1 for t in turns if "dice" in t)
    assert ti == expected_moves, f"replayed {ti} move records, expected {expected_moves}"

    s1, s2 = struct.unpack_from("<2i", tx, recs[-1][0] + xf._FG_SCORE1)
    w, pts, term = struct.unpack_from("<3i", tx, recs[-1][0] + xf._FG_WINNER)
    assert (s1, s2) == ((pts, 0) if w == 1 else (0, pts)), "footer scores"
    assert w == result["winner"] and pts == result["points"], "footer winner/points"
    assert term == _TERMINATION[result["win_type"]], "footer termination"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _result_from_xg_record(xg_record: dict) -> dict:
    """Winner/points/termination for the footer, from the simulator's game record."""
    res = xg_record.get("result") or ""
    points = int(xg_record.get("result_points") or 0)
    winner = 1 if res.startswith("player1-win") else -1
    win_type = res.rsplit("-", 1)[-1] if res else "single"
    # A double/pass game: the loser dropped -- XG terms it 0 (drop). Detect it from
    # the last history entry rather than the result string (which says "single").
    hist = xg_record.get("move_history") or []
    if hist and (hist[-1].get("cube_action") == "double/pass"):
        win_type = "drop"
    return {"winner": winner, "points": points, "win_type": win_type}


def export_seed(seed: int, tpl: dict, template_path: Path, out_dir: Path) -> dict:
    """Re-simulate one seed, validate, and write ``seed_<N>.xg``. Returns stats."""
    decisions, xg_record = bp._simulate_and_capture(seed, parallel_threads=1)
    del decisions
    _verify_txt_roundtrip(seed, xg_record)

    turns = _replay_turns(xg_record["move_history"])
    result = _result_from_xg_record(xg_record)
    stream = build_game_stream(tpl, turns, result)

    out_path = out_dir / f"seed_{seed}.xg"
    write_xg(template_path, stream, out_path)
    _verify_archive(out_path, turns, result)

    n_moves = sum(1 for t in turns if "dice" in t)
    n_cube_actions = sum(1 for t in turns if t["cube_action"] in ("double/take", "double/pass"))
    return {"seed": seed, "turns": len(turns), "moves": n_moves,
            "cube_actions": n_cube_actions, "result": result, "path": out_path}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds to export (default: every stage-1 seed on disk)")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help="Real .xg archive to clone record templates from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel export processes (re-simulation is the cost; "
                             "default 16, like pass-1). Pass 1 for serial.")
    args = parser.parse_args(argv)

    seeds = args.seeds
    if not seeds:
        seeds = sorted(int(p.stem.split("_")[1]) for p in bp._STAGE1_DIR.glob("seed_*.json"))
    if not seeds:
        raise SystemExit("No stage-1 games found -- run benchmark_pasko.py build --stages pass1")
    if not args.template.exists():
        raise SystemExit(f"Template .xg not found: {args.template}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tpl = _templates(args.template)

    print(f"Exporting {len(seeds)} Paskogammon games to {args.out_dir} "
          f"(template: {args.template.name}, workers={args.workers})", flush=True)

    def _report(st):
        r = st["result"]
        print(f"  seed_{st['seed']}.xg: {st['moves']} moves, {st['cube_actions']} cube actions, "
              f"P{'1' if r['winner'] == 1 else '2'} wins {r['points']} ({r['win_type']}) -- verified",
              flush=True)

    if args.workers <= 1:
        for seed in seeds:
            _report(export_seed(seed, tpl, args.template, args.out_dir))
    else:
        # Re-simulation (3-ply replay) is the cost; parallelize across seeds like pass-1.
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(export_seed, s, tpl, args.template, args.out_dir): s for s in seeds}
            for fut in as_completed(futs):
                _report(fut.result())
    print("\nAll games re-simulated, transcript-verified, written, and re-parse-verified.", flush=True)
    print("Next: XG -> Batch Analyze this folder ('Save games after analyze' ON).", flush=True)


if __name__ == "__main__":
    main()
