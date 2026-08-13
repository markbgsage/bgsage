# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Plain-text export for backgammon games.

Two public functions:

- ``compute_move_notation(before, after, die1, die2)`` — return a string like
  ``"13/7 8/7"`` or ``"bar/20*"`` from the boards before/after the move
  (mover's perspective; positive checkers = mover) and the dice rolled.

- ``export_history_to_txt(record_dict)`` — render a game (or a match of games)
  as a Backgammon Galaxy / XG-import compatible UTF-8 text transcript.

The transcript format is the one accepted by eXtreme Gammon's
"Import Match Text" / "Batch Analyze" features, which lets external analysis
roundtrip through plain text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Move notation
# ---------------------------------------------------------------------------


def compute_move_notation(
    before: list[int], after: list[int], die1: int, die2: int
) -> str:
    """Compute backgammon move notation from before/after board arrays.

    Board is from the mover's perspective (positive checkers are the mover's).
    Returns notation like ``"13/7 8/7"`` or ``"bar/20*"`` or ``"6/off(2)"``.
    """
    hit_points: set[int] = set()
    for i in range(1, 25):
        if before[i] < 0 and (after[i] >= 0 or after[i] > before[i]):
            hit_points.add(i)

    from_pts: list[int] = []
    to_pts: list[int] = []

    bar_diff = after[25] - before[25]
    if bar_diff < 0:
        from_pts.extend([25] * (-bar_diff))

    for i in range(1, 25):
        wb = before[i] if before[i] > 0 else 0
        wa = after[i] if after[i] > 0 else 0
        if before[i] < 0 and after[i] > 0:
            wa = after[i]
            wb = 0
        elif before[i] > 0 and after[i] < 0:
            wb = before[i]
            wa = 0
        diff = wa - wb
        if diff > 0:
            to_pts.extend([i] * diff)
        elif diff < 0:
            from_pts.extend([i] * (-diff))

    on_board_before = before[25] + sum(v for v in before[1:25] if v > 0)
    on_board_after = after[25] + sum(v for v in after[1:25] if v > 0)
    borne_off = on_board_before - on_board_after
    to_pts.extend([0] * borne_off)

    from_pts.sort(reverse=True)
    to_pts.sort(reverse=True)

    dice = [die1, die1, die1, die1] if die1 == die2 else [die1, die2]
    moves: list[tuple[int, int, bool]] = []
    used_from = [False] * len(from_pts)
    used_to = [False] * len(to_pts)
    used_die = [False] * len(dice)

    for di, d in enumerate(dice):
        if used_die[di]:
            continue
        for fi, f in enumerate(from_pts):
            if used_from[fi]:
                continue
            expected = (25 - d) if f == 25 else (f - d)
            for ti, t in enumerate(to_pts):
                if used_to[ti]:
                    continue
                if t == expected or (expected <= 0 and t == 0):
                    is_hit = t in hit_points
                    if is_hit:
                        hit_points.discard(t)
                    moves.append((f, t, is_hit))
                    used_from[fi] = used_to[ti] = used_die[di] = True
                    break
                if used_die[di]:
                    break

    for fi, f in enumerate(from_pts):
        if used_from[fi]:
            continue
        for ti, t in enumerate(to_pts):
            if used_to[ti]:
                continue
            is_hit = t in hit_points
            if is_hit:
                hit_points.discard(t)
            moves.append((f, t, is_hit))
            used_from[fi] = used_to[ti] = True
            break

    # Split moves that span multiple dice through intermediate hit points.
    # e.g. "24/20" with 2-2 should be "24/22* 22/20" if point 22 had a blot.
    if die1 == die2:
        die = die1
        for mi in range(len(moves) - 1, -1, -1):
            f, t, h = moves[mi]
            dist = (25 - t) if f == 25 else (f - t)
            if dist <= die or dist % die != 0:
                continue
            n_dice = dist // die
            hit_mids: list[int] = []
            for i in range(1, n_dice):
                mid = (25 - i * die) if f == 25 else (f - i * die)
                if 1 <= mid <= 24 and mid in hit_points:
                    hit_mids.append(mid)
            if not hit_mids:
                continue
            for hm in hit_mids:
                hit_points.discard(hm)
            sub_moves: list[tuple[int, int, bool]] = []
            prev = f
            for hm in hit_mids:
                sub_moves.append((prev, hm, True))
                prev = hm
            sub_moves.append((prev, t, h))
            moves[mi:mi + 1] = sub_moves
    else:
        for mi in range(len(moves) - 1, -1, -1):
            f, t, h = moves[mi]
            dist = (25 - t) if f == 25 else (f - t)
            if dist != die1 + die2:
                continue
            for d1, d2 in [(die1, die2), (die2, die1)]:
                mid = (25 - d1) if f == 25 else (f - d1)
                if 1 <= mid <= 24 and mid in hit_points:
                    hit_points.discard(mid)
                    moves[mi:mi + 1] = [(f, mid, True), (mid, t, h)]
                    break

    moves.sort(key=lambda m: (-m[0], -m[1]))

    combined: list[tuple[int, int, bool, int]] = []
    for f, t, h in moves:
        if combined and combined[-1][:3] == (f, t, h):
            combined[-1] = (f, t, h, combined[-1][3] + 1)
        else:
            combined.append((f, t, h, 1))

    parts = []
    for f, t, h, count in combined:
        fs = "bar" if f == 25 else str(f)
        ts = "off" if t == 0 else str(t)
        hs = "*" if h else ""
        ms = f"{fs}/{ts}{hs}"
        parts.append(f"{ms}({count})" if count > 1 else ms)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Match-text export
# ---------------------------------------------------------------------------


def _parse_iso(iso_str: str | None) -> datetime:
    if not iso_str:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(iso_str)
    except Exception:
        return datetime.now(timezone.utc)


def _winner_side(result: str | None) -> str | None:
    if not result:
        return None
    s = result.lower()
    if s.startswith("player1-win"):
        return "user"
    if s.startswith("player2-win"):
        return "bot"
    if s.startswith("win"):
        return "user"
    if s.startswith("loss"):
        return "bot"
    return None


def _point_token(token: str) -> str:
    t = token.strip().lower()
    if t in {"bar", "b"}:
        return "25"
    if t in {"off", "o"}:
        return "0"
    if t.isdigit():
        return str(int(t))
    return token.strip()


def _display_move_notation(move: str | None) -> str:
    """Render stored notation with text-export conventions.

    - Strip hit markers ('*')
    - Convert bar to 25 and off to 0
    - Expand repeated moves like ``"8/2(2)"`` -> ``"8/2 8/2"``
    """
    if not move or not isinstance(move, str):
        return ""

    parts: list[str] = []
    for raw in move.split():
        token = raw.strip()
        if not token:
            continue

        count = 1
        if "(" in token and token.endswith(")"):
            base, count_part = token.rsplit("(", 1)
            token = base
            if count_part[:-1].isdigit():
                count = max(1, int(count_part[:-1]))

        if "/" not in token:
            continue

        frm, to = token.split("/", 1)
        frm_txt = _point_token(frm)
        to_txt = _point_token(to.rstrip("*"))
        rendered = f"{frm_txt}/{to_txt}"
        for _ in range(count):
            parts.append(rendered)

    return " ".join(parts)


def _move_action_text(turn: dict[str, Any]) -> str | None:
    dice = turn.get("dice")
    if not isinstance(dice, list) or len(dice) != 2:
        return None
    try:
        d1 = int(dice[0])
        d2 = int(dice[1])
    except Exception:
        return None
    move_txt = _display_move_notation(turn.get("move"))
    if move_txt:
        return f"{d1}{d2}: {move_txt}"
    return f"{d1}{d2}:"


def _turns_to_actions(turns: list[dict[str, Any]]) -> list[tuple[str, str]]:
    actions: list[tuple[str, str]] = []
    cube_value = 1

    for turn in turns:
        player = turn.get("player", "user")
        other = "bot" if player == "user" else "user"
        cube_action = (turn.get("cube_action") or "").strip().lower()

        if cube_action == "double/take":
            new_cube = cube_value * 2
            actions.append((player, f"Doubles => {new_cube}"))
            actions.append((other, "Takes"))
            cube_value = new_cube
        elif cube_action == "double/pass":
            new_cube = cube_value * 2
            actions.append((player, f"Doubles => {new_cube}"))
            actions.append((other, "Drops"))
        elif cube_action == "resign":
            actions.append((player, "Resigns"))

        move_action = _move_action_text(turn)
        if move_action:
            actions.append((player, move_action))

    return actions


def _actions_to_rows(
    actions: list[tuple[str, str]],
) -> list[tuple[str | None, str | None]]:
    rows: list[tuple[str | None, str | None]] = []
    left: str | None = None
    right: str | None = None

    for side, text in actions:
        if side == "user":
            if left is not None:
                rows.append((left, right))
                left, right = None, None
            if right is not None:
                rows.append((left, right))
                left, right = None, None
            left = text
            if right is not None:
                rows.append((left, right))
                left, right = None, None
            continue

        # side == "bot"
        if right is not None:
            rows.append((left, right))
            left, right = None, None
        right = text
        if left is not None:
            rows.append((left, right))
            left, right = None, None
        else:
            rows.append((left, right))
            left, right = None, None

    if left is not None or right is not None:
        rows.append((left, right))
    return rows


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def export_history_to_txt(record_dict: dict[str, Any]) -> bytes:
    """Export a match-history dict to plain-text match notation (UTF-8 bytes).

    The input dict shape mirrors a single MatchHistory record:

    - ``player1_name`` / ``player2_name``: display names (strings).
    - ``mode``: ``"unlimited"`` for money games, ``"match"`` for match play.
    - ``match_length``: target points (only used when ``mode == "match"``).
    - ``started_at``: ISO-8601 timestamp string (optional).
    - ``history_id``: optional unique identifier.

    For unlimited mode (single game): ``move_history``, ``result``,
    ``result_points``.

    For match mode: ``match_game_histories`` is a list of game dicts each
    with ``game_number``, ``player_score``, ``opponent_score``,
    ``move_history``, ``result``, ``result_points``.

    Each entry in a ``move_history`` list should have:
    - ``player``: ``"user"`` (player 1) or ``"bot"`` (player 2)
    - ``dice``: ``[d1, d2]`` or ``None``
    - ``move``: notation string from ``compute_move_notation`` (or ``None``)
    - ``cube_action``: one of ``"double/take"`` / ``"double/pass"`` /
      ``"no double"`` / ``"resign"`` / ``None``
    """
    p1 = (record_dict.get("player1_name") or "").strip() or "You"
    p2 = (record_dict.get("player2_name") or "").strip() or "Sage Bot"
    mode = (record_dict.get("mode") or "unlimited").strip().lower()
    started = _parse_iso(record_dict.get("started_at"))
    history_id = (record_dict.get("history_id") or "").strip()

    lines: list[str] = [
        '; [Site "BackgammonSage"]',
    ]
    if history_id:
        lines.append(f'; [Match ID "{history_id}"]')
    lines.extend([
        f'; [Player 1 "{p1}"]',
        f'; [Player 2 "{p2}"]',
        f'; [EventDate "{started.strftime("%Y.%m.%d")}"]',
        f'; [EventTime "{started.strftime("%H.%M")}"]',
        '; [Variation "Backgammon"]',
        '; [Unrated "Off"]',
        '; [Crawford "On"]',
        '; [CubeLimit "1024"]',
        "",
    ])

    if mode == "match":
        match_length = max(1, _coerce_int(record_dict.get("match_length"), 0))
    else:
        match_length = 0
    lines.append(f"{match_length} point match")
    lines.append("")

    games_raw = record_dict.get("match_game_histories") or []
    if mode == "match" and games_raw:
        games = sorted(
            [g for g in games_raw if isinstance(g, dict)],
            key=lambda g: _coerce_int(g.get("game_number"), 0),
        )
    else:
        games = [{
            "game_number": 1,
            "player_score": 0,
            "opponent_score": 0,
            "result": record_dict.get("result"),
            "result_points": record_dict.get("result_points", 0),
            "move_history": record_dict.get("move_history") or [],
        }]

    running_user = 0
    running_bot = 0
    last_game_idx = len(games) - 1

    for idx, game in enumerate(games):
        game_number = _coerce_int(game.get("game_number"), idx + 1) or (idx + 1)
        start_user = running_user
        start_bot = running_bot

        turns = game.get("move_history") or []
        if not isinstance(turns, list):
            turns = []

        actions = _turns_to_actions(turns)
        rows = _actions_to_rows(actions)

        result = game.get("result")
        points = abs(_coerce_int(game.get("result_points"), 0))
        winner = _winner_side(result)
        if winner == "user":
            running_user += points
        elif winner == "bot":
            running_bot += points

        lines.append(f" Game {game_number}")
        score_left = f"{p1} : {start_user}"
        score_right = f"{p2} : {start_bot}"

        left_width = max(34, len(score_left) + 2)
        for row_left, _ in rows:
            if row_left:
                left_width = max(left_width, len(row_left) + 6)

        lines.append(f" {score_left.ljust(left_width)}{score_right}")

        for row_no, (row_left, row_right) in enumerate(rows, start=1):
            left_txt = row_left or ""
            right_txt = row_right or ""
            lines.append(f"{row_no:>3}) {left_txt.ljust(left_width)}{right_txt}")

        if winner:
            win_text = f"Wins {max(1, points)} point"
            if mode == "match" and idx == last_game_idx:
                win_text += " and the match"
            if winner == "user":
                lines.append(f"      {win_text}")
            else:
                lines.append(f"      {''.ljust(left_width)}{win_text}")

        if idx != last_game_idx:
            lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    return text.encode("utf-8")
