# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Parse eXtreme Gammon (.xg) batch-analyzed game files and compute PR stats.

Used to benchmark Sage's evaluations against XG's: feed Sage-vs-Sage transcripts
(produced by ``scripts/run_sage_vs_sage_games.py``) into XG's Batch Analyze with
"Save Games after analyze" enabled, then run this module on the resulting .xg
files to extract per-game error totals and Performance Rating (PR).

PR = sum(equity errors) / decision count * 500.

Public API:
- ``parse_xg_game(xg_bytes)`` — parse a single-game .xg file into turn dicts.
- ``compute_game_pr_stats(turns)`` — sum errors + decisions across both players
  for one game and report PR.
- ``apply_decision_flags(turns)`` — annotate turns with ``is_cube_decision`` /
  ``is_checker_decision`` / ``cube_error`` / ``opp_cube_error`` / ``checker_error``
  using the same filters XG uses to count "real" decisions.

The module reads .xg binary files only; it does not call XG and does not
require XG to be running.
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .board import flip_board


# ---------------------------------------------------------------------------
# XG binary format constants
# ---------------------------------------------------------------------------

TSAVEREC_SIZE = 2560
RICH_HEADER_SIZE = 8232
MAGIC_RGMH = 0x484D4752  # 'RGMH'
MAGIC_DMLI = 0x494C4D44  # 'DMLI'

TS_HEADER_MATCH = 0
TS_HEADER_GAME = 1
TS_CUBE = 2
TS_MOVE = 3
TS_FOOTER_GAME = 4
TS_FOOTER_MATCH = 5

# XG comp_level (header) -> our label
_XG_LEVEL_MAP = {0: "1ply", 1: "2ply", 2: "3ply", 3: "4ply", 4: "5ply", 100: "rollout"}

# XG analyze_m / analyze_c (per-record) -> our label
_XG_ANALYZE_LEVEL_MAP = {
    1: "1ply", 2: "2ply", 3: "3ply", 4: "4ply", 5: "5ply",
    10: "rollout", 11: "rollout", 12: "rollout",
}


# ---------------------------------------------------------------------------
# Low-level binary helpers
# ---------------------------------------------------------------------------


def _xg_eval_to_probs(xg_eval: tuple | list) -> list[float]:
    """Convert XG 7-element eval to our 5-element probs.

    XG order: ``[p_bg_loss, p_gam_loss, p_loss, p_win, p_gam_win, p_bg_win, equity]``
    Ours:     ``[p_win, p_gam_win, p_bg_win, p_gam_loss, p_bg_loss]``
    """
    return [
        float(xg_eval[3]),  # p_win
        float(xg_eval[4]),  # p_gam_win
        float(xg_eval[5]),  # p_bg_win
        float(xg_eval[1]),  # p_gam_loss
        float(xg_eval[0]),  # p_bg_loss
    ]


def _read_position(data: bytes, offset: int) -> list[int]:
    """Read a 26-byte signed position from binary data."""
    return [struct.unpack_from('<b', data, offset + i)[0] for i in range(26)]


def _flip_board_to_user(board: list[int]) -> list[int]:
    """Flip an XG-perspective board to player-1 (user) perspective."""
    flipped = [0] * 26
    flipped[0] = board[25]
    flipped[25] = board[0]
    for i in range(1, 25):
        flipped[i] = -board[25 - i]
    return flipped


def _mover_to_user_board(board: list[int], actif: int) -> list[int]:
    """Convert a mover's-perspective board to user (player1) perspective.

    actif: 1 = player1 (user); 2 or any negative value = player2 (bot).
    """
    if actif == 1:
        return list(board)
    if actif < 0 or actif == 2:
        return _flip_board_to_user(board)
    return list(board)


def _active_player_from_actif(actif: int) -> str:
    """Normalise XG ActiveP into ``"user"`` / ``"bot"``.

    Supports both legacy exports (±1) and XG-native (1/2) encodings.
    """
    if actif == 2 or actif < 0:
        return "bot"
    return "user"


def _read_pascal_string(data: bytes, offset: int, max_len: int) -> str:
    """Read a Delphi short string[N]: 1 length byte + N char bytes."""
    length = data[offset]
    length = min(length, max_len)
    return data[offset + 1:offset + 1 + length].decode('ascii', errors='replace').strip()


def _read_short_unicode_string(data: bytes, offset: int) -> str:
    """Read a TShortUnicodeString (258 bytes = 128 wchars + null)."""
    raw = data[offset:offset + 256]
    try:
        text = raw.decode('utf-16-le')
        return text.split('\x00')[0].strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def _extract_game_data(xg_bytes: bytes) -> bytes:
    """Extract the temp.xg game data blob from the XG ZlibArchive container."""
    if len(xg_bytes) < RICH_HEADER_SIZE + 36:
        raise ValueError("File too small to be a valid XG file")

    magic = struct.unpack_from('<I', xg_bytes, 0)[0]
    if magic != MAGIC_RGMH:
        raise ValueError(f"Invalid XG file: bad magic 0x{magic:08X}, expected RGMH")

    # ArchiveRecord is the last 36 bytes
    arc_start = len(xg_bytes) - 36
    (_crc, file_count, _version, reg_size,
     data_size, comp_reg) = struct.unpack_from('<iiiiii', xg_bytes, arc_start)

    # Layout: [RichHeader][file_data][compressed_registry][ArchiveRecord]
    reg_end = arc_start
    reg_start = reg_end - reg_size
    data_start = reg_start - data_size

    if data_start < RICH_HEADER_SIZE:
        raise ValueError("Invalid archive layout")

    reg_compressed = xg_bytes[reg_start:reg_end]
    if comp_reg and reg_size > 0:
        index_data = zlib.decompress(reg_compressed)
    else:
        index_data = reg_compressed

    for i in range(file_count):
        rec_off = i * 532
        if rec_off + 532 > len(index_data):
            break
        rec = index_data[rec_off:rec_off + 532]
        name_len = rec[0]
        name = rec[1:1 + name_len].decode('ascii', errors='replace')
        # osize at +512 (unused), csize at +516, start at +520, compressed flag at +528
        csize = struct.unpack_from('<i', rec, 516)[0]
        start = struct.unpack_from('<i', rec, 520)[0]
        is_compressed = (rec[528] == 0)

        if name == 'temp.xg':
            file_data = xg_bytes[data_start + start:data_start + start + csize]
            if is_compressed and csize > 0:
                return zlib.decompress(file_data)
            return file_data

    raise ValueError("temp.xg not found in XG archive")


# ---------------------------------------------------------------------------
# Record parsers
# ---------------------------------------------------------------------------


def _delphi_date_to_iso(delphi_date: float) -> str:
    """Convert Delphi TDateTime (days since 12/30/1899) to ISO string."""
    try:
        from datetime import timedelta
        epoch = datetime(1899, 12, 30, tzinfo=timezone.utc)
        return (epoch + timedelta(seconds=delphi_date * 86400)).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _parse_header_match(rec: bytes) -> dict[str, Any]:
    splayer1 = _read_pascal_string(rec, 9, 40)
    splayer2 = _read_pascal_string(rec, 50, 40)

    match_length = struct.unpack_from('<i', rec, 92)[0]
    jacoby = bool(rec[101]) if len(rec) > 103 else True
    beaver = bool(rec[102]) if len(rec) > 103 else True
    delphi_date = struct.unpack_from('<d', rec, 128)[0]
    date_iso = _delphi_date_to_iso(delphi_date) if delphi_date > 0 else ""

    version = 0
    for off in range(136, min(600, len(rec) - 8)):
        if struct.unpack_from('<I', rec, off + 4)[0] == MAGIC_DMLI:
            version = struct.unpack_from('<i', rec, off)[0]
            break

    comp_level1 = 0
    if len(rec) > 280:
        comp_level1 = struct.unpack_from('<i', rec, 272)[0]

    # Optional unicode names (XG v24+); fall back to ASCII pascal strings.
    player1 = splayer1
    player2 = splayer2
    for base_off in [558, 560, 562]:
        if base_off + 5 * 258 > len(rec):
            continue
        p1 = _read_short_unicode_string(rec, base_off + 258)
        p2 = _read_short_unicode_string(rec, base_off + 258 * 2)
        if p1 and len(p1) > 1 and p1 != "!":
            player1 = p1
            player2 = p2 if (p2 and p2 != "!") else splayer2
            break

    return {
        "type": TS_HEADER_MATCH,
        "player1": player1 or "Player 1",
        "player2": player2 or "Player 2",
        "match_length": match_length,
        "date_iso": date_iso,
        "version": version,
        "comp_level": comp_level1,
        "jacoby": jacoby,
        "beaver": beaver,
    }


def _parse_header_game(rec: bytes) -> dict[str, Any]:
    score1 = struct.unpack_from('<i', rec, 12)[0]
    score2 = struct.unpack_from('<i', rec, 16)[0]
    crawford = rec[20]
    game_number = struct.unpack_from('<i', rec, 48)[0]
    return {
        "type": TS_HEADER_GAME,
        "score1": score1,
        "score2": score2,
        "crawford_apply": bool(crawford),
        "game_number": game_number,
    }


def _parse_engine_double_action(data: bytes) -> dict[str, Any] | None:
    """Parse the 132-byte EngineStructDoubleAction sub-record of a CUBE record."""
    if len(data) < 132:
        return None
    flag_double = struct.unpack_from('<h', data, 56)[0]
    xg_eval = struct.unpack_from('<7f', data, 60)
    eq_nd = struct.unpack_from('<f', data, 88)[0]
    eq_dt = struct.unpack_from('<f', data, 92)[0]
    eq_dp = struct.unpack_from('<f', data, 96)[0]

    if all(abs(v) < 1e-10 for v in xg_eval[:6]):
        return None

    # flag_double is XG's cube-action recommendation:
    #   0 = no double
    #   1 = double (take recommended)
    #   2 = double, pass (probably)
    #   -1 = too good to double (still NOT recommending double)
    #   -100 = no analysis (record skipped)
    # The naive bool(flag_double) misreads -1 as "double", which is wrong —
    # treat only positive values as a recommendation to double.
    return {
        "probs": _xg_eval_to_probs(xg_eval),
        "equity_nd": eq_nd,
        "equity_dt": eq_dt,
        "equity_dp": eq_dp,
        "should_double": flag_double > 0,
    }


def _parse_engine_best_move(
    data: bytes, actif: int
) -> dict[str, Any] | None:
    """Parse the 2184-byte EngineStructBestMove sub-record of a MOVE record."""
    if len(data) < 2184:
        return None

    n_moves = struct.unpack_from('<i', data, 64)[0]
    if n_moves <= 0:
        return None
    n_moves = min(n_moves, 32)

    positions = []
    for i in range(n_moves):
        pos = _read_position(data, 68 + i * 26)
        positions.append(_mover_to_user_board(pos, actif))

    evals = []
    for i in range(n_moves):
        eval_offset = 1284 + i * 28
        evals.append(struct.unpack_from('<7f', data, eval_offset))

    is_forced = (data[2180] == 1)
    return {
        "n_moves": n_moves,
        "positions": positions,
        "evals": evals,
        "is_forced": is_forced,
    }


def _parse_cube_record(rec: bytes) -> dict[str, Any]:
    actif = struct.unpack_from('<i', rec, 12)[0]
    doubled = struct.unpack_from('<i', rec, 16)[0]
    take = struct.unpack_from('<i', rec, 20)[0]
    cube_b = struct.unpack_from('<i', rec, 32)[0]
    double_action = _parse_engine_double_action(rec[64:64 + 132])
    err_cube = struct.unpack_from('<d', rec, 200)[0]
    analyze_c = struct.unpack_from('<i', rec, 232)[0]
    return {
        "type": TS_CUBE,
        "actif": actif,
        "doubled": doubled,
        "take": take,
        "cube_b": cube_b,
        "double_action": double_action,
        "err_cube": err_cube if err_cube > -999 else None,
        "analyze_c": analyze_c,
    }


def _parse_move_record(rec: bytes) -> dict[str, Any]:
    position_i = _read_position(rec, 9)
    position_end = _read_position(rec, 35)
    actif = struct.unpack_from('<i', rec, 64)[0]
    die1 = struct.unpack_from('<i', rec, 100)[0]
    die2 = struct.unpack_from('<i', rec, 104)[0]
    cube_a = struct.unpack_from('<i', rec, 108)[0]
    n_move_eval = struct.unpack_from('<i', rec, 120)[0]
    best_move = _parse_engine_best_move(rec[124:124 + 2184], actif)
    err_move = struct.unpack_from('<d', rec, 2312)[0]
    analyze_m = struct.unpack_from('<i', rec, 2472)[0]
    return {
        "type": TS_MOVE,
        "position_i": position_i,
        "position_end": position_end,
        "actif": actif,
        "die1": die1,
        "die2": die2,
        "cube_a": cube_a,
        "n_move_eval": n_move_eval,
        "best_move": best_move,
        "err_move": err_move if err_move > -999 else None,
        "analyze_m": analyze_m,
    }


def _parse_footer_game(rec: bytes) -> dict[str, Any]:
    return {
        "type": TS_FOOTER_GAME,
        "score1": struct.unpack_from('<i', rec, 12)[0],
        "score2": struct.unpack_from('<i', rec, 16)[0],
        "crawford_next": bool(rec[20]),
        "winner": struct.unpack_from('<i', rec, 24)[0],
        "points_won": struct.unpack_from('<i', rec, 28)[0],
        "termination": struct.unpack_from('<i', rec, 32)[0],
    }


def _parse_footer_match(rec: bytes) -> dict[str, Any]:
    return {
        "type": TS_FOOTER_MATCH,
        "score1": struct.unpack_from('<i', rec, 12)[0],
        "score2": struct.unpack_from('<i', rec, 16)[0],
        "winner": struct.unpack_from('<i', rec, 20)[0],
    }


def _parse_records(game_data: bytes) -> list[dict[str, Any]]:
    """Split the inflated game data into typed records."""
    records: list[dict[str, Any]] = []
    offset = 0
    while offset + TSAVEREC_SIZE <= len(game_data):
        rec = game_data[offset:offset + TSAVEREC_SIZE]
        entry_type = rec[8]

        if entry_type == TS_HEADER_MATCH:
            records.append(_parse_header_match(rec))
        elif entry_type == TS_HEADER_GAME:
            records.append(_parse_header_game(rec))
        elif entry_type == TS_CUBE:
            records.append(_parse_cube_record(rec))
        elif entry_type == TS_MOVE:
            records.append(_parse_move_record(rec))
        elif entry_type == TS_FOOTER_GAME:
            records.append(_parse_footer_game(rec))
        elif entry_type == TS_FOOTER_MATCH:
            records.append(_parse_footer_match(rec))
        # Unknown record types silently ignored.

        offset += TSAVEREC_SIZE

    return records


def _infer_analysis_level(records: list[dict]) -> str:
    """Infer the analysis level from per-record analyze_m / analyze_c values."""
    levels: Counter[int] = Counter()
    for r in records:
        if r.get("type") == TS_MOVE:
            am = r.get("analyze_m", -1)
            if am > 0:
                levels[am] += 1
        elif r.get("type") == TS_CUBE:
            ac = r.get("analyze_c", -1)
            if ac > 0:
                levels[ac] += 1
    if not levels:
        return "xg-2ply"
    most_common = levels.most_common(1)[0][0]
    base = _XG_ANALYZE_LEVEL_MAP.get(most_common, f"{most_common}ply")
    return f"xg-{base}"


def _group_records_by_game(records: list[dict]) -> list[list[dict]]:
    """Partition records into per-game lists (HEADER_GAME .. FOOTER_GAME)."""
    games: list[list[dict]] = []
    current_game: list[dict] = []
    in_game = False
    for rec in records:
        if rec["type"] == TS_HEADER_GAME:
            current_game = [rec]
            in_game = True
        elif rec["type"] == TS_FOOTER_GAME:
            if in_game:
                current_game.append(rec)
                games.append(current_game)
            current_game = []
            in_game = False
        elif in_game:
            current_game.append(rec)
    return games


# ---------------------------------------------------------------------------
# Turn assembly
# ---------------------------------------------------------------------------


def _build_cube_analysis(da: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a parsed EngineStructDoubleAction to the analyzer-style dict."""
    if not da:
        return None
    nd = da["equity_nd"]
    dt = da["equity_dt"]
    dp = da["equity_dp"]
    probs = da["probs"]
    should_double = da["should_double"]
    should_take = dt <= dp
    if should_double:
        optimal_action = "double/take" if should_take else "double/pass"
    else:
        optimal_action = "no double"
    optimal = max(nd, min(dt, dp)) if should_double else nd
    p_win, p_gw, p_bw, p_gl, p_bl = probs
    p_loss = 1.0 - p_win
    cubeless_eq = (p_win - p_loss) + (p_gw - p_gl) + (p_bw - p_bl)
    return {
        "probs": probs,
        "cubeless_equity": cubeless_eq,
        "cubeful_equity": nd,
        "equity_nd": nd,
        "equity_dt": dt,
        "equity_dp": dp,
        "should_double": should_double,
        "should_take": should_take,
        "optimal_action": optimal_action,
        "optimal_equity": optimal,
    }


def _build_checker_analysis(
    bm: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Convert a parsed EngineStructBestMove to a list of move dicts.

    XG stores moves in its own recommendation order (sorted by eval level
    first, then by cubeful equity within the same level), so positions[0]
    is XG's #1 recommendation. We trust that order — do NOT re-sort by the
    stored ``equity`` field, since the cubeful equity at offset 6 doesn't
    capture eval-level priority and naive sorting can promote a deeply-
    evaluated lower-cubeful move above XG's actual recommendation.
    """
    if not bm or bm["n_moves"] == 0:
        return None
    moves: list[dict[str, Any]] = []
    for i in range(bm["n_moves"]):
        xg_eval = bm["evals"][i]
        probs = _xg_eval_to_probs(xg_eval)
        equity = float(xg_eval[6])
        moves.append({
            "board": bm["positions"][i],
            "equity": equity,
            "probs": probs,
        })
    # Compute equity_diff vs XG's #1 (= moves[0]). Note this can be POSITIVE
    # for moves that have higher cubeful equity than XG's recommendation —
    # those are typically moves XG evaluated at a lower level and thus
    # didn't rank first despite the higher raw cubeful number.
    best_eq = moves[0]["equity"]
    for m in moves:
        m["equity_diff"] = m["equity"] - best_eq
    return moves


def _assemble_turns(
    game_records: list[dict],
    analysis_level: str,
) -> list[dict[str, Any]]:
    """Assemble cube + move records into per-turn dicts.

    XG emits separate CUBE and MOVE records; this function pairs them so each
    turn dict carries the cube action (if any) alongside the checker move.
    """
    turns: list[dict[str, Any]] = []
    turn_number = 0
    pending_cube: dict[str, Any] | None = None

    for rec in game_records:
        if rec["type"] == TS_CUBE:
            # Trailing cube + new cube => the prior was a double/pass.
            if pending_cube is not None and pending_cube["doubled"] == 1:
                turn_number += 1
                player = _active_player_from_actif(pending_cube["actif"])
                turns.append({
                    "turn_number": turn_number,
                    "player": player,
                    "board_before": None,
                    "board_after": None,
                    "dice": None,
                    "move": None,
                    "cube_action": "double/pass",
                    "cube_analysis": _build_cube_analysis(pending_cube["double_action"]),
                    "checker_analysis": None,
                    "analysis_level": analysis_level if pending_cube.get("analyze_c", 0) > 0 else None,
                    "checker_error": pending_cube.get("err_cube"),
                    "cube_error": pending_cube.get("err_cube"),
                    "opp_cube_error": None,
                })
            pending_cube = rec

        elif rec["type"] == TS_MOVE:
            turn_number += 1
            actif = rec["actif"]
            player = _active_player_from_actif(actif)

            board_before = list(rec["position_i"])
            board_after = _mover_to_user_board(rec["position_end"], actif)

            cube_action: str | None = None
            cube_analysis: dict[str, Any] | None = None
            cube_error = None
            if pending_cube is not None:
                if pending_cube["doubled"] == 1:
                    cube_action = "double/take" if pending_cube["take"] == 1 else "double/pass"
                elif pending_cube["doubled"] == 0:
                    cube_action = "no double"
                else:
                    cube_action = None
                if cube_action is not None:
                    cube_analysis = _build_cube_analysis(pending_cube["double_action"])
                    cube_error = pending_cube.get("err_cube")
                pending_cube = None

            checker_analysis = _build_checker_analysis(rec["best_move"])
            checker_error = rec.get("err_move")

            dice = None if (rec["die1"] == 0 and rec["die2"] == 0) else [rec["die1"], rec["die2"]]
            has_checker_analysis = rec.get("analyze_m", 0) > 0
            has_cube_analysis = cube_analysis is not None

            turns.append({
                "turn_number": turn_number,
                "player": player,
                "board_before": board_before,
                "board_after": board_after,
                "dice": dice,
                "move": None,
                "cube_action": cube_action,
                "cube_analysis": cube_analysis,
                "checker_analysis": checker_analysis,
                "analysis_level": analysis_level if (has_checker_analysis or has_cube_analysis) else None,
                "checker_error": checker_error,
                "cube_error": cube_error,
                "opp_cube_error": None,
            })

    # Trailing pending cube => game-ending double/pass.
    if pending_cube is not None and pending_cube["doubled"] == 1:
        turn_number += 1
        player = _active_player_from_actif(pending_cube["actif"])
        turns.append({
            "turn_number": turn_number,
            "player": player,
            "board_before": None,
            "board_after": None,
            "dice": None,
            "move": None,
            "cube_action": "double/pass",
            "cube_analysis": _build_cube_analysis(pending_cube["double_action"]),
            "checker_analysis": None,
            "analysis_level": analysis_level if pending_cube.get("analyze_c", 0) > 0 else None,
            "checker_error": None,
            "cube_error": pending_cube.get("err_cube"),
            "opp_cube_error": None,
        })

    # Fill in board_before for double/pass turns from the previous turn.
    for i, turn in enumerate(turns):
        if turn["board_before"] is None and i > 0:
            prev = turns[i - 1]
            if prev.get("board_after"):
                turn["board_before"] = list(prev["board_after"])

    # XG stores PosPlayed from a fixed (player-1) view; flip bot turns'
    # checker_analysis boards back to mover's perspective so checker-error
    # board matching against ``board_after`` works correctly.
    for turn in turns:
        if turn["player"] == "bot" and turn.get("checker_analysis"):
            for m in turn["checker_analysis"]:
                m["board"] = list(flip_board(m["board"]))

    return turns


# ---------------------------------------------------------------------------
# Decision filters (XG-style: skip "trivial" or forced decisions)
# ---------------------------------------------------------------------------


def _is_trivial_cube(analysis: dict[str, Any]) -> bool:
    """A cube position is trivial when the action is so obvious it shouldn't
    count as a real decision (XG's filter)."""
    nd = analysis.get("equity_nd")
    dt = analysis.get("equity_dt")
    dp = analysis.get("equity_dp")
    if nd is None or dt is None or dp is None:
        return False
    if abs(nd - dt) < 0.001:
        return True
    if nd - dt > 0.200:
        return True
    if nd - dp > 0.200:
        return True
    if nd < -0.900 and dt < -0.900:
        return True
    return False


def _is_trivial_take_pass(analysis: dict[str, Any]) -> bool:
    """A take/pass decision is trivial when DT ≈ DP (no real choice)."""
    dt = analysis.get("equity_dt")
    dp = analysis.get("equity_dp")
    if dt is None or dp is None:
        return False
    return abs(dt - dp) < 0.001


def _is_cube_decision(entry: dict[str, Any], had_cube_access: bool) -> bool:
    """Whether the cube_action on this turn counts as a PR decision."""
    if not had_cube_access:
        return False
    cube_action = entry.get("cube_action")
    if cube_action is None or cube_action == "resign":
        return False
    analysis = entry.get("cube_analysis")
    if analysis is None:
        return False
    if _is_trivial_cube(analysis):
        # Trivial position but a wrong choice still counts.
        if compute_cube_error(entry) < 0.001:
            return False
    return True


def _is_checker_decision(entry: dict[str, Any]) -> bool:
    """Whether the checker play on this turn counts as a PR decision."""
    checker_analysis = entry.get("checker_analysis")
    if not checker_analysis or len(checker_analysis) < 2:
        return False
    first = checker_analysis[0]
    last = checker_analysis[-1]
    best_eq = first.get("equity") if isinstance(first, dict) else getattr(first, "equity", None)
    worst_eq = last.get("equity") if isinstance(last, dict) else getattr(last, "equity", None)
    if best_eq is not None and worst_eq is not None:
        if abs(best_eq - worst_eq) < 0.001:
            return False
    return True


# ---------------------------------------------------------------------------
# Error computation
# ---------------------------------------------------------------------------


def compute_cube_error(entry: dict[str, Any]) -> float:
    """Equity lost on the doubler's no-double / double choice (>= 0)."""
    analysis = entry.get("cube_analysis")
    if not analysis:
        return 0.0
    cube_action = entry.get("cube_action")
    if cube_action is None or cube_action == "resign":
        return 0.0
    nd = analysis["equity_nd"]
    dt = analysis["equity_dt"]
    dp = analysis["equity_dp"]
    optimal = analysis.get("optimal_equity", max(nd, min(dt, dp)))
    doubled = cube_action in ("double/take", "double/pass")
    actual = min(dt, dp) if doubled else nd
    return max(0.0, optimal - actual)


def compute_opp_cube_error(entry: dict[str, Any]) -> float:
    """Equity lost on the responder's take/pass choice (>= 0)."""
    cube_action = entry.get("cube_action")
    if cube_action not in ("double/take", "double/pass"):
        return 0.0
    analysis = entry.get("cube_analysis")
    if not analysis:
        return 0.0
    dt = analysis["equity_dt"]
    dp = analysis["equity_dp"]
    optimal = min(dt, dp)
    actual = dt if cube_action == "double/take" else dp
    return max(0.0, actual - optimal)


def compute_checker_error(entry: dict[str, Any]) -> float:
    """Equity lost on the checker move (>= 0).

    Looks up ``board_after`` in ``checker_analysis`` and returns the equity gap
    vs the best move. Bot turns have ``board_after`` in user perspective; the
    analysis boards in turn dicts have already been flipped to mover's
    perspective during turn assembly, so we need to flip ``board_after`` for
    bot turns to match.
    """
    analysis = entry.get("checker_analysis")
    if not analysis or len(analysis) < 2:
        return 0.0
    played_board = entry.get("board_after")
    if played_board is None:
        return 0.0
    if entry.get("player") == "bot":
        played_board = flip_board(played_board)
    played_tuple = tuple(played_board)
    for move in analysis:
        if tuple(move.get("board", [])) == played_tuple:
            return abs(move.get("equity_diff", 0.0))
    return 0.0


# ---------------------------------------------------------------------------
# Decision-flag application + per-game aggregation
# ---------------------------------------------------------------------------


def apply_decision_flags(turns: list[dict[str, Any]]) -> None:
    """Annotate each turn with ``is_cube_decision`` / ``is_checker_decision``
    and fill in ``cube_error`` / ``opp_cube_error`` / ``checker_error``."""
    cube_owner = "centered"  # relative to the active player at each turn

    for entry in turns:
        player = entry["player"]
        had_cube_access = (
            cube_owner == "centered"
            or (cube_owner == "player" and player == "user")
            or (cube_owner == "opponent" and player == "bot")
        )

        entry["is_cube_decision"] = _is_cube_decision(entry, had_cube_access)
        if entry.get("cube_analysis") is not None and entry.get("cube_action") is not None:
            entry["cube_error"] = compute_cube_error(entry)
            entry["opp_cube_error"] = compute_opp_cube_error(entry)
        elif not had_cube_access or entry.get("cube_action") in (None, "resign"):
            entry["cube_error"] = 0.0
            entry["opp_cube_error"] = 0.0

        entry["is_checker_decision"] = _is_checker_decision(entry)
        entry["checker_error"] = (
            compute_checker_error(entry)
            if entry.get("checker_analysis") is not None
            else 0.0
        )

        # Update cube ownership for the next turn.
        if entry.get("cube_action") == "double/take":
            cube_owner = "player" if player == "bot" else "opponent"


def compute_game_pr_stats(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-player error totals + decision counts over one game.

    Calls ``apply_decision_flags`` first (idempotent in effect; rewrites flag
    fields). Returns:

    - ``user_err`` / ``user_dec``: P1 error total (cube + checker) and decision count
    - ``bot_err``  / ``bot_dec``:  P2 error total and decision count
    - ``total_err`` / ``total_dec``: sums across both players
    - ``pr``: ``(total_err / total_dec) * 500``, or ``float('nan')`` if no
      decisions were recorded
    """
    apply_decision_flags(turns)

    user_err = 0.0
    user_dec = 0
    bot_err = 0.0
    bot_dec = 0

    for entry in turns:
        player = entry.get("player")
        is_user = player == "user"

        if entry.get("is_cube_decision"):
            err = entry.get("cube_error", 0.0) or 0.0
            if is_user:
                user_dec += 1
                user_err += err
            else:
                bot_dec += 1
                bot_err += err

        # opp_cube_error belongs to the OTHER player (the responder).
        cube_action = entry.get("cube_action")
        analysis = entry.get("cube_analysis")
        if (
            cube_action in ("double/take", "double/pass")
            and analysis is not None
            and not _is_trivial_take_pass(analysis)
        ):
            opp_err = entry.get("opp_cube_error", 0.0) or 0.0
            if is_user:
                bot_dec += 1
                bot_err += opp_err
            else:
                user_dec += 1
                user_err += opp_err

        if entry.get("is_checker_decision"):
            err = entry.get("checker_error", 0.0) or 0.0
            if is_user:
                user_dec += 1
                user_err += err
            else:
                bot_dec += 1
                bot_err += err

    total_err = user_err + bot_err
    total_dec = user_dec + bot_dec
    pr = (total_err / total_dec * 500.0) if total_dec > 0 else float("nan")

    return {
        "user_err": user_err,
        "user_dec": user_dec,
        "bot_err": bot_err,
        "bot_dec": bot_dec,
        "total_err": total_err,
        "total_dec": total_dec,
        "pr": pr,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_xg_game(xg_bytes: bytes) -> list[dict[str, Any]]:
    """Parse a single-game .xg file and return a list of turn dicts.

    Raises ``ValueError`` if the file is not a valid XG archive or contains
    more than one game.
    """
    game_data = _extract_game_data(xg_bytes)
    records = _parse_records(game_data)
    games = _group_records_by_game(records)
    if len(games) != 1:
        raise ValueError(
            f"Expected exactly 1 game in .xg file, found {len(games)}"
        )
    return _assemble_turns(games[0], _infer_analysis_level(records))


def parse_xg_match(xg_bytes: bytes) -> dict[str, Any]:
    """Parse a multi-game (match) .xg file.

    Returns a dict with match-level metadata and a list of per-game records
    in play order::

        {
            "match_length": int,         # 0 if money game
            "player1": str,
            "player2": str,
            "jacoby": bool,
            "beaver": bool,
            "analysis_level": str,       # e.g. "xg-2ply"
            "games": [
                {
                    "game_number": int,
                    "score1_start": int,     # P1 score at game start
                    "score2_start": int,     # P2 score at game start
                    "crawford_apply": bool,  # True for the Crawford game
                    "turns": [...],          # output of _assemble_turns
                },
                ...
            ],
        }

    Single-game .xg files parse fine too (``games`` will be a list of one).
    Raises ``ValueError`` if the file isn't a valid XG archive.
    """
    game_data = _extract_game_data(xg_bytes)
    records = _parse_records(game_data)
    analysis_level = _infer_analysis_level(records)

    match_header: dict[str, Any] | None = None
    for r in records:
        if r.get("type") == TS_HEADER_MATCH:
            match_header = r
            break

    games_records = _group_records_by_game(records)
    if not games_records:
        raise ValueError("No games found in .xg file")

    games_out: list[dict[str, Any]] = []
    for game_recs in games_records:
        header = game_recs[0]
        turns = _assemble_turns(game_recs, analysis_level)
        games_out.append({
            "game_number": header.get("game_number", len(games_out) + 1),
            "score1_start": header.get("score1", 0),
            "score2_start": header.get("score2", 0),
            "crawford_apply": header.get("crawford_apply", False),
            "turns": turns,
        })

    # Normalize XG's sentinel for "money / unlimited" (99999) to 0 so
    # downstream code can treat ``match_length == 0`` as the universal
    # "no match state" signal.
    raw_match_length = (
        match_header.get("match_length", 0) if match_header else 0
    )
    match_length = 0 if raw_match_length >= 99999 else int(raw_match_length)

    return {
        "match_length": match_length,
        "player1": match_header.get("player1", "Player 1") if match_header else "Player 1",
        "player2": match_header.get("player2", "Player 2") if match_header else "Player 2",
        "jacoby": match_header.get("jacoby", True) if match_header else True,
        "beaver": match_header.get("beaver", True) if match_header else True,
        "analysis_level": analysis_level,
        "games": games_out,
    }
