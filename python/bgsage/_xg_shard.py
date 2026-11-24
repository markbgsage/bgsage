# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Encode and verify positions for XG Batch Analyze.

These checks follow the audited lockstep runner's one-position game format.
"""

import math
import struct

from . import xg_file as xf
from .board import possible_moves, possible_single_die_moves


TOP_N = 5


def derive_half_moves(pre, post, d1, d2):
    """Per-die (from, to) sequence taking pre -> post. XG replays half-moves."""
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

    for order in orders:
        got = dfs(list(pre), order)
        if got is not None:
            return got
    return None


def cube_code(cube_value, cube_owner):
    """XG's CubeA/CubeB: sign(owner) * log2(value), 0 for a centred 1-cube.

    The mover is written as player +1, and cube_owner is in the mover's frame,
    so the mover owning it is positive and the opponent owning it negative.
    Verified against real match files (tests/3pt_match.xg, single_game.xg).
    """
    if (not isinstance(cube_value, int) or cube_value < 1
            or cube_value & (cube_value - 1)
            or cube_owner not in ("centered", "player", "opponent")
            or (cube_owner == "centered") != (cube_value == 1)):
        raise ValueError("invalid cube value/owner")
    n = cube_value.bit_length() - 1
    if n <= 0 or cube_owner == "centered":
        return 0
    return n if cube_owner == "player" else -n


def templates(path):
    tempxg = xf.XgArchive.load(path).get("temp.xg")
    tpl = {}
    for off, rt in xf.iter_records(tempxg):
        tpl.setdefault(rt, tempxg[off:off + xf.TSAVEREC_SIZE])
    return tpl


def write_shard(items, out_path, template, *, tpl=None, max_cube=64):
    """items: list of dicts.

    checker: {"kind":"move", "pre":board, "dice":(d1,d2), "post":board, "half_moves":[...]}
    cube:    {"kind":"cube", "pre":board}
    Returns the number of games written; record order == item order, which is
    how results are matched back (never by content key -- duplicate positions
    are common and would collide).
    """
    tpl = tpl or templates(template)
    stream = bytearray(tpl[xf.TS_HEADER_MATCH])
    # The template is an XG money-session export (MatchLength 99999, Jacoby on)
    # but has Beaver ON. The engine never allows a beaver and Sage is told
    # beaver=False, so XG must be told the same or its double/take equities
    # carry a beaver risk that does not exist in these games. Byte 102 is the
    # Beaver flag (Crawford 100, Jacoby 101, Beaver 102, AutoDouble 103).
    if max_cube < 2 or max_cube > 1024 or max_cube & (max_cube - 1):
        raise ValueError("XG max_cube must be a power of two in [2, 1024]")
    struct.pack_into('<i', stream, xf._HDR_MATCH_LENGTH, 99999)
    stream[100:104] = bytes((0, 1, 0, 0))
    struct.pack_into('<i', stream, 612, max_cube.bit_length() - 1)
    n = 0
    for i, it in enumerate(items):
        pre = it["pre"]
        if it.get("max_cube_value", max_cube) != max_cube or it.get("cube_value", 1) > max_cube:
            raise ValueError("shard item has incompatible cube limit")
        # The cube state is genuine game state and XG's cubeful equities depend
        # on it (Jacoby on the no-double line, who holds cube access), so every
        # record carries it. Items without it are analyzed at a centred 1-cube.
        cb = cube_code(it.get("cube_value", 1), it.get("cube_owner", "centered"))
        stream += xf.build_game_header(tpl[xf.TS_HEADER_GAME], pre, i + 1, 0, 0)
        if it["kind"] == "cube":
            # double=0 => "had cube access, did not double" => XG analyzes it.
            stream += xf.build_cube_record(tpl[xf.TS_CUBE], pre, actif=1, double=0,
                                           cube_b=cb)
        else:
            stream += xf.build_cube_record(tpl[xf.TS_CUBE], pre, actif=1, double=-2,
                                           cube_b=cb)
            d1, d2 = it["dice"]
            mv = bytearray(xf.build_move_record(tpl[xf.TS_MOVE], pre, it["post"],
                                                d1, d2, it["half_moves"], actif=1))
            struct.pack_into('<i', mv, xf._MOVE_CUBE_A, cb)
            stream += bytes(mv)
        stream += xf.build_game_footer(tpl[xf.TS_FOOTER_GAME], 0, i + 1, -1, 1, 101)
        n += 1
    # Templates may contain old analysis. Clear every cube's complete payload,
    # including the recommendation sentinel (build_cube_record resets indices).
    for off, rt in xf.iter_records(stream):
        if rt == xf.TS_CUBE:
            da = off + xf._CUBE_DOUBLE_ACTION
            stream[da:da + 132] = bytes(132)
            struct.pack_into('<i', stream, da + xf._DA_LEVEL, -1)
            struct.pack_into('<h', stream, da + xf._DA_FLAG_DOUBLE, -100)
    xf.set_header_timedelay_totals(stream, 0, 0, 0)
    arch = xf.XgArchive.load(template)
    arch.set("temp.xg", bytes(stream))
    arch.set("temp.xgi", xf.rebuild_xgi(bytes(stream)))
    arch.save(out_path)
    return n


#: How far outside [0, 1] a probability XG genuinely produced can sit. This is
#: a GARBAGE bound, not a precision bound. Float32 storage leaves a genuinely
#: zero value at up to -1.15e-5, and the variance reduction in XG's Roller
#: mini-rollouts pushes a rare outcome's estimate below zero: over 13,264 snake
#: probability rows the worst was -0.007, and one Roller+ double/take line gave
#: P(backgammon win) -0.012 -- which a 0.01 bound rejected, deadlocking a
#: 1,000-game cohort at game 163 (2026-09-21). Nothing observed exceeded 1.
#: Play never reads these values (answers come from the board and equity
#: fields), and a record that was never analysed, or was read at the wrong
#: offset, fails the win + loss = 1 test regardless.
PROB_SLACK = 0.25

#: Tolerance on win + loss = 1, the test that actually separates an analysed
#: record from one that is not (an unanalysed record sums to 0). Worst measured
#: deviation 9.7e-5, on a ranked Roller+ candidate.
PROB_SUM_TOL = 0.01


class IncompleteAnalysis(ValueError):
    """The shard is not a complete, matching response; retry the exact request."""


def validate_shard(tx, items, max_cube=64):
    def require(ok, message):
        if not ok:
            raise IncompleteAnalysis(message)

    require(len(tx) % xf.TSAVEREC_SIZE == 0, "truncated XG record")
    records = list(xf.iter_records(tx))
    require(bool(records) and records[0][1] == xf.TS_HEADER_MATCH, "missing match header")
    header = xf.parse_header(tx)
    require(header["match_length"] == 99999 and header["jacoby"] and not header["beaver"]
            and tx[100] == 0 and tx[103] == 0
            and struct.unpack_from('<i', tx, 612)[0] == max_cube.bit_length() - 1,
            "XG changed the game rules")
    cursor = 1

    def take(expected):
        nonlocal cursor
        require(cursor < len(records) and records[cursor][1] == expected,
                "missing, extra, or reordered decision record")
        off = records[cursor][0]
        cursor += 1
        return off

    def probabilities(ev):
        require(len(ev) == 7 and all(math.isfinite(v) for v in ev), "nonfinite evaluation")
        # Test what separates an analysed record from one that is not: an
        # unanalysed record is all zeros and fails the sum, a misread one is
        # out of range. Anything stricter judges XG's MODEL, and because
        # IncompleteAnalysis means "retry the exact request", a genuine answer
        # that fails it is re-driven for identical bytes forever. That
        # deadlocked Paskogammon cohorts three times: at game 101 on float32
        # noise (-1.15e-5), and at games 356 and 493 on nesting.
        require(all(-PROB_SLACK <= v <= 1 + PROB_SLACK for v in ev[:6])
                and abs(ev[2] + ev[3] - 1) < PROB_SUM_TOL,
                "missing or invalid probabilities")
        # Nesting -- backgammon <= gammon <= win on each side -- is NOT tested.
        # It is a property of XG's nets, not of a complete record: on snake
        # positions XG's own 3-ply ANSWER reported P(gammon win) 0.408 against
        # P(win) 0.378, and 2-ply candidates broke it by up to 0.145 (240-
        # decision Roller+ pilot, 2026-09-21), so no level or rank is safe.

    for i, it in enumerate(items):
        off = take(xf.TS_HEADER_GAME)
        h = xf.parse_game_header(tx, off)
        require(h["game_number"] == i + 1 and h["score1"] == h["score2"] == 0
                and not h["crawford"]
                and tuple(xf.norm_bars(xf.read_position(tx, off + xf._HG_POSINIT))) == tuple(it["pre"]),
                "game identity/initial position mismatch")
        cube_off = take(xf.TS_CUBE)
        cube = xf.parse_cube_record(tx, cube_off)
        code = cube_code(it.get("cube_value", 1), it.get("cube_owner", "centered"))
        require(cube["actif"] == 1 and tuple(cube["mover_board"]) == tuple(it["pre"])
                and cube["cube_b"] == code, "cube position/owner mismatch")
        if it["kind"] == "cube":
            # XG also writes 2 for a positive double recommendation and -1
            # for too-good/no-double. -100 is the unanalysed sentinel.
            require(cube["double"] == 0 and cube["flag_double"] in (-1, 0, 1, 2)
                    and cube["level"] >= 0 and cube["analyze_c"] >= 0,
                    "cube analysis is incomplete")
            da = cube_off + xf._CUBE_DOUBLE_ACTION
            wanted_owner = {"centered": 0, "player": 1, "opponent": -1}[it.get("cube_owner", "centered")]
            require(tuple(xf.norm_bars(xf.read_position(tx, da))) == tuple(it["pre"])
                    and struct.unpack_from('<5i', tx, da + 32) == (-1, -1, it.get("cube_value", 1), wanted_owner, 1)
                    and struct.unpack_from('<h', tx, da + 52)[0] == 0,
                    "cube evaluation inputs do not match unlimited game state")
            require(all(math.isfinite(cube[k]) for k in ("equity_nd", "equity_dt", "equity_dp"))
                    and abs(cube["equity_dp"] - 1.0) < 1e-4 and cube["is_beaver"] == 0,
                    "invalid cube equities/beaver recommendation")
            probabilities(cube["eval_nd"])
            probabilities(cube["eval_dt"])
        else:
            require(cube["double"] == -2, "invalid checker filler cube")
            move_off = take(xf.TS_MOVE)
            require(1 <= struct.unpack_from('<i', tx, move_off + xf._MOVE_DATAMOVES + xf._BM_NMOVES)[0] <= 32,
                    "invalid candidate count")
            move = xf.parse_move_record(tx, move_off)
            require(move["actif"] == 1 and tuple(move["mover_board"]) == tuple(it["pre"])
                    and tuple(move["dice"]) == tuple(it["dice"]) and move["cube_a"] == code,
                    "checker position/dice/owner mismatch")
            dm = move_off + xf._MOVE_DATAMOVES
            wanted_owner = {"centered": 0, "player": 1, "opponent": -1}[it.get("cube_owner", "centered")]
            require(tuple(xf.norm_bars(xf.read_position(tx, dm))) == tuple(it["pre"])
                    and tuple(struct.unpack_from('<2i', tx, dm + 28)) == tuple(it["dice"])
                    and struct.unpack_from('<6i', tx, dm + 40) == (-1, -1, it.get("cube_value", 1), wanted_owner, 0, 1),
                    "checker evaluation inputs do not match unlimited game state")
            require(move["analyze_m"] >= 0 and bool(move["moves"]), "checker analysis is incomplete")
            legal = {tuple(b) for b in possible_moves(it["pre"], *it["dice"])} or {tuple(it["pre"])}
            require(tuple(it["post"]) in legal, "illegal placeholder")
            for rank, row in enumerate(move["moves"]):
                require(row["level"] >= 0 and tuple(row["board"]) in legal, "illegal/unanalyzed candidate")
                # XG may leave probabilities on lower-ranked candidates
                # unevaluated. Keep the audited top-rank probability check.
                require(len(row["eval"]) == 7 and all(math.isfinite(v) for v in row["eval"]),
                        "nonfinite evaluation")
                if rank < TOP_N:
                    probabilities(row["eval"])
        take(xf.TS_FOOTER_GAME)
    require(cursor == len(records), "unexpected trailing games/records")


def harvest(path, items, max_cube=64):
    """Read validated XG decisions in request order."""
    tx = xf.XgArchive.load(path).get("temp.xg")
    validate_shard(tx, items, max_cube)
    answers = []
    for offset, record_type in xf.iter_records(tx):
        if record_type == xf.TS_MOVE:
            move = xf.parse_move_record(tx, offset)
            best = move["moves"][0]  # XG's own rank, including its move filters.
            requested, = struct.unpack_from("<i", tx, offset + xf._MOVE_DATAMOVES + xf._BM_LEVEL)
            answers.append({"board": best["board"], "level": best["level"],
                            "level_request": requested})
        elif record_type == xf.TS_CUBE:
            cube = xf.parse_cube_record(tx, offset)
            if cube["double"] == 0:
                answers.append({"should_double": cube["flag_double"] > 0,
                                "should_take": cube["equity_dt"] <= cube["equity_dp"],
                                "level": cube["level"], "level_request": cube["level_request"]})
    return answers
