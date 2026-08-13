# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Aggregate per-game PR stats across all .xg files in a folder, re-analyze
every XG-flagged Sage decision at 2T (truncated2 / XGRoller+), then roll out
the remaining 2T-vs-XG disagreements to produce a rollout-truth "new PR".

Pipeline
========

1. **PR aggregation.** For each .xg file in the folder, parse turns via
   ``bgsage.xg_compare.parse_xg_game``, apply XG-style decision filters,
   then sum errors and decisions across BOTH players. Reports per-game total
   PR plus across-games mean/std/SEM and the aggregate PR from summed
   errors/decisions.

2. **2T re-analysis + dispute detection.** For every decision where Sage's
   3-ply play differs from XG's recommendation (i.e. XG flagged it as a
   Sage error), the position is re-analyzed at 2T. **2T is used only to
   identify what Sage would have played at 2T level** ("Sage's smart
   pick" = 2T's #1). The error magnitude is then computed using **XG's
   own equities** — for cube decisions, XG's nd/dt/dp; for checker
   decisions, Sage's 2T pick is looked up in XG's checker_analysis and
   its XG equity is used. If 2T agrees with XG (same pick), the error is
   zero and no Dispute is emitted. ``sage_choice`` stores Sage's 2T pick.
   For checker decisions, both Sage's played move and XG's #1 are
   explicitly evaluated at 2T quality (the TINY filter inside
   ``checker_play`` may otherwise leave them at only 1-ply, biasing the
   moves[0] determination). The 2T evaluations are cached in
   ``sage_2T_cache.jsonl`` and appended incrementally; on every use the
   cached entry's quality for the important boards is re-checked and
   upgraded if necessary.

3. **Rollout truth.** For each disputed decision whose XG-measured Sage
   error exceeds ``--threshold`` (default 0.005), run a full Sage rollout
   (1,296 trials, no truncation, 3-ply checker + 3-ply cube throughout) to
   establish the true equity of each candidate. Compute Sage's and XG's
   error to the rollout-best as non-negative numbers, then the signed
   "net Sage error" contribution ``sage_err - xg_err`` per decision:
   negative when Sage was right and XG was wrong, positive when XG was
   right and Sage was wrong, signed when both were wrong. Disputes below
   the threshold are assumed to be cases where Sage was wrong and XG was
   right, and contribute ``+xg_measured_error`` to net Sage error without
   actually running a rollout.

   Sign of the resulting net PR: positive means XG is on net the better
   player; negative means Sage is the better player.

4. **Output.** Append each rollout result as a JSON line to
   ``rollout_disputes.jsonl`` in the same folder (resumable: existing
   entries are skipped). Then print: per-case tallies, net Sage error,
   the user's literal "new PR = net Sage error * 500", and a
   per-decision-normalized variant for comparison against the original PR.

PR = sum(equity errors) / decision count * 500.

Usage:

    python scripts/aggregate_xg_pr.py [folder] [--pattern '*.xg']
    python scripts/aggregate_xg_pr.py --skip-rollouts
    python scripts/aggregate_xg_pr.py --limit 5    # test on a few disputes

The default folder is ``bgsage/logs/sage_vs_sage`` (resolved from the script
location — fully self-contained within the bgsage repo), matching where
``run_sage_vs_sage_games.py`` writes its .txt files. After producing those
files, run XG's Batch Analyze with "Save Games after analyze" checked; XG
writes one .xg file per .txt next to it. Then run this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
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


from bgsage.board import flip_board  # noqa: E402
from bgsage.xg_compare import (  # noqa: E402
    _is_trivial_take_pass,
    compute_game_pr_stats,
    parse_xg_game,
)

_DEFAULT_DIR = _PROJECT_ROOT / "logs" / "sage_vs_sage"
_DEFAULT_ROLLOUT_FILENAME = "rollout_disputes.jsonl"

# Rollout config: full rollout, no truncation, 3-ply throughout.
_ROLLOUT_N_TRIALS = 1296
_ROLLOUT_TRUNCATION_DEPTH = 0
_ROLLOUT_DECISION_PLY = 3

# Re-analysis level used when Sage's 3-ply played move differs from XG's
# recommendation. The result becomes Sage's "effective" choice, and the
# measured error driving threshold + PR is the re-eval-perspective equity
# gap between that choice and XG's. A separate JSONL cache keyed by
# (file, turn, kind) lets us resume. Cache filename derives from the level
# so 2T and 3T runs don't collide.
_DEFAULT_RE_EVAL_LEVEL = "truncated3"
_RE_EVAL_LEVEL_CHOICES = ["truncated2", "truncated3"]

# Mutable module-level config: set from CLI in main() before any analyzer
# call. The two_t_* function/variable names are kept for compatibility but
# now refer to whatever level was selected via --re-eval-level.
_TWO_T_EVAL_LEVEL = _DEFAULT_RE_EVAL_LEVEL

# Optional overrides for the re-eval truncated rollout (set from CLI in main()).
# When either is not None, the re-analyzer is built as eval_level="rollout"
# replicating the named level's base params (below) with these fields overridden,
# e.g. "3T but truncation_depth=7, ultra_late_threshold=9999".
_RE_EVAL_TRUNC_DEPTH: int | None = None
_RE_EVAL_ULTRA_LATE: int | None = None

# Base params for the named truncated levels (mirror analyzer.py's eval_level
# resolution); used only when an override above is active. prefilter_threshold
# 0.15 matches what BgBotAnalyzer auto-applies for truncated2/truncated3.
_NAMED_TRUNC_PARAMS = {
    "truncated2": dict(n_trials=360, truncation_depth=7, decision_ply=2,
                       late_ply=1, late_threshold=2, ultra_late_threshold=2,
                       prefilter_threshold=0.15),
    "truncated3": dict(n_trials=360, truncation_depth=7, decision_ply=3,
                       late_ply=2, late_threshold=2, ultra_late_threshold=9999,
                       prefilter_threshold=0.15),
}


def _re_eval_cache_filename(level: str) -> str:
    """Map an eval level to its default cache filename.

    truncated2 -> sage_2T_cache.jsonl
    truncated3 -> sage_3T_cache.jsonl
    """
    short = {"truncated2": "2T", "truncated3": "3T"}.get(level, level)
    return f"sage_{short}_cache.jsonl"


# ---------------------------------------------------------------------------
# 2T cache infrastructure
# ---------------------------------------------------------------------------


_two_t_analyzer = None  # lazy-built


def _get_two_t_analyzer(n_threads: int = 0):
    """Build (once) and return the re-analysis analyzer.

    Normally the named level (``_TWO_T_EVAL_LEVEL``). If a truncation-depth or
    ultra-late override is set, build an equivalent ``eval_level="rollout"`` from
    the named level's base params plus the override(s).
    """
    global _two_t_analyzer
    if _two_t_analyzer is None:
        from bgsage import BgBotAnalyzer
        if _RE_EVAL_TRUNC_DEPTH is None and _RE_EVAL_ULTRA_LATE is None:
            _two_t_analyzer = BgBotAnalyzer(
                eval_level=_TWO_T_EVAL_LEVEL, cubeful=True,
                parallel_threads=n_threads,
            )
        else:
            params = dict(_NAMED_TRUNC_PARAMS[_TWO_T_EVAL_LEVEL])
            if _RE_EVAL_TRUNC_DEPTH is not None:
                params["truncation_depth"] = _RE_EVAL_TRUNC_DEPTH
            if _RE_EVAL_ULTRA_LATE is not None:
                params["ultra_late_threshold"] = _RE_EVAL_ULTRA_LATE
            print(f"  [re-eval] custom rollout from {_TWO_T_EVAL_LEVEL}: {params}",
                  flush=True)
            _two_t_analyzer = BgBotAnalyzer(
                eval_level="rollout", cubeful=True,
                parallel_threads=n_threads, **params,
            )
    return _two_t_analyzer


def _two_t_cache_key(file: str, turn_number: int, kind: str) -> str:
    return f"{file}#{turn_number}#{kind}"


def _load_two_t_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[_two_t_cache_key(rec["file"], rec["turn_number"], rec["kind"])] = rec
    return cache


def _write_two_t_cache_entry(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def _evaluate_single_board_at_2t(
    analyzer, post_move_board: list[int], pre_move_board: list[int],
    cube_value: int, cube_owner: str,
) -> float:
    """Evaluate a single post-move board at the analyzer's full rollout
    cubeful quality, bypassing the TINY filter that ``checker_play`` applies.

    Works for any analyzer whose underlying engine is a rollout strategy —
    truncated2 (2T) or the full rollout. The returned equity reflects the
    analyzer's own configuration (trials, truncation, decision_ply, etc.).
    Used when a specific candidate (e.g. the move Sage actually played) was
    filtered out of the standard checker_play result and only has a 1-ply
    equity but we need its full-quality equity for a fair comparison.
    """
    from bgsage.analyzer import resolve_owner
    inner = analyzer._analyzer
    if hasattr(inner, "_inner"):
        inner = inner._inner
    r = inner._rollout_strategy.cubeful_evaluate_board(
        list(post_move_board), list(pre_move_board),
        cube_value=cube_value, owner=resolve_owner(cube_owner),
        jacoby=True, beaver=True,
    )
    return float(r["cubeful_equity"])


def _ensure_boards_at_2t(
    entry: dict, pre_move_board: list[int],
    target_boards,                       # iterable of post-move boards
    cube_value: int, cube_owner: str, n_threads: int,
) -> tuple[dict, bool]:
    """Ensure every board in ``target_boards`` is present in ``entry['moves']``
    at full 2T (``'Rollout'``) cubeful quality. Any board that's missing or
    only at 1-ply quality is explicitly evaluated via the rollout strategy's
    ``cubeful_evaluate_board`` and inserted; the moves list is re-sorted by
    equity descending so moves[0] is always the true 2T best (which may end
    up being a board the TINY filter originally excluded).

    Returns ``(entry, was_updated)``. The caller should persist when updated.
    """
    moves = list(entry.get("moves") or [])
    analyzer = None
    updated = False
    for tb in target_boards:
        if tb is None:
            continue
        tb_norm = _normalize_bar(list(tb))
        tb_tuple = tuple(tb_norm)
        existing = next(
            (m for m in moves if tuple(m["board"]) == tb_tuple), None,
        )
        if existing is not None and existing.get("eval_level") == "Rollout":
            continue  # already at 2T quality
        if analyzer is None:
            analyzer = _get_two_t_analyzer(n_threads)
        eq_2t = _evaluate_single_board_at_2t(
            analyzer, tb_norm, pre_move_board, cube_value, cube_owner,
        )
        moves = [m for m in moves if tuple(m["board"]) != tb_tuple]
        moves.append({"board": list(tb_norm), "equity": eq_2t,
                      "eval_level": "Rollout"})
        updated = True
    if updated:
        moves.sort(key=lambda m: -m["equity"])
        return {**entry, "moves": moves}, True
    return entry, False


def _get_or_compute_two_t_checker(
    cache: dict, cache_path: Path, file: str, turn_number: int,
    board: list[int], dice: list[int],
    cube_value: int, cube_owner: str, n_threads: int,
    important_boards=(),
) -> dict:
    """Get or compute the 2T ``checker_play`` result for a position. Any
    board passed in ``important_boards`` (e.g. Sage's actual played move
    and XG's #1 recommendation) is guaranteed to be present in the moves
    list at full 2T quality — the TINY filter inside ``checker_play`` can
    silently drop candidates to 1-ply, which would skew the moves[0]
    "2T best" determination and the 2T equity gap calculation. After any
    such upgrade, the moves list is re-sorted so moves[0] is the true 2T
    best.
    """
    key = _two_t_cache_key(file, turn_number, "checker")
    entry = cache.get(key)
    if entry is None:
        analyzer = _get_two_t_analyzer(n_threads)
        result = analyzer.checker_play(
            board, dice[0], dice[1],
            cube_value=cube_value, cube_owner=cube_owner,
            jacoby=True, beaver=True,
        )
        entry = {
            "file": file, "turn_number": turn_number, "kind": "checker",
            "moves": [
                {"board": _normalize_bar(list(m.board)),
                 "equity": float(m.equity),
                 "eval_level": m.eval_level}
                for m in result.moves
            ],
        }
        entry, _ = _ensure_boards_at_2t(
            entry, board, important_boards,
            cube_value, cube_owner, n_threads,
        )
        cache[key] = entry
        _write_two_t_cache_entry(cache_path, entry)
        return entry
    updated, was_updated = _ensure_boards_at_2t(
        entry, board, important_boards,
        cube_value, cube_owner, n_threads,
    )
    if was_updated:
        cache[key] = updated
        _write_two_t_cache_entry(cache_path, updated)
        return updated
    return entry


def _get_or_compute_two_t_cube(
    cache: dict, cache_path: Path, file: str, turn_number: int,
    board: list[int], cube_value: int, cube_owner: str, n_threads: int,
) -> dict:
    key = _two_t_cache_key(file, turn_number, "cube")
    if key in cache:
        return cache[key]
    analyzer = _get_two_t_analyzer(n_threads)
    result = analyzer.cube_action(
        board, cube_value=cube_value, cube_owner=cube_owner,
        jacoby=True, beaver=True,
    )
    entry = {
        "file": file, "turn_number": turn_number, "kind": "cube",
        "equity_nd": float(result.equity_nd),
        "equity_dt": float(result.equity_dt),
        "equity_dp": float(result.equity_dp),
    }
    cache[key] = entry
    _write_two_t_cache_entry(cache_path, entry)
    return entry


# ---------------------------------------------------------------------------
# Existing aggregation helpers
# ---------------------------------------------------------------------------


def _seed_sort_key(p: Path) -> tuple[int, str]:
    """Sort by trailing integer in the stem (``seed_<N>``), else by name."""
    parts = p.stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), p.name)
    return (10**9, p.name)


# ---------------------------------------------------------------------------
# Dispute detection
# ---------------------------------------------------------------------------


@dataclass
class Dispute:
    """One decision where Sage's chosen action differs from XG's recommendation.

    ``board`` and ``cube_owner`` are always in the active turn-player's
    perspective (which equals the doubler for cube disputes and the mover
    for checker disputes). For responder disputes the responder is the
    opposite player from ``turn_player``; ``board`` stays in the doubler's
    perspective so the cube_action rollout is identical to the doubler's.
    """

    file: str
    turn_number: int
    turn_player: str            # active player of the turn ("user" | "bot")
    deciding_player: str        # who actually makes this decision
    decision_type: str          # "doubler" | "responder" | "checker"
    board: list[int]            # turn_player's POV
    dice: list[int] | None      # [d1, d2] for checker, None for cube
    cube_value: int
    cube_owner: str             # turn_player's POV
    sage_choice: dict           # type-specific (see below)
    xg_choice: dict
    xg_measured_error: float    # XG's own equity-loss for Sage's choice (>= 0)


def _flip_owner(owner: str) -> str:
    return {"player": "opponent", "opponent": "player", "centered": "centered"}[owner]


def _owner_relative_to(user_perspective: str, player: str) -> str:
    """xg_compare stores cube_owner from the user's POV; convert to the
    given active player's POV."""
    return user_perspective if player == "user" else _flip_owner(user_perspective)


def _normalize_bar(board: list[int]) -> list[int]:
    """Convert XG sign-coupled bar to bgsage's positive-count convention.

    bgsage convention: indices 0 (P2 bar) and 25 (P1 bar) hold absolute
    counts, always ``>= 0`` (positive whether the bar belongs to P1 or P2).
    XG stores the bar sign-coupled to the owner — an opp's checker on bar
    shows as ``-1`` at index 0. Taking ``abs()`` at both bar indices
    normalizes XG-parsed boards into the convention ``bgbot_cpp.possible_moves``
    produces in its output, so tuple comparisons match.
    """
    return [abs(board[0])] + list(board[1:25]) + [abs(board[25])]


def _board_for_analysis(turn: dict) -> list[int] | None:
    """Return ``board_before`` in the active player's perspective.

    xg_compare stores ``board_before`` in user (player-1) POV for every turn
    (raw XG ``position_i`` for turns with a checker move; backfill from the
    previous turn's ``board_after`` for double/pass turns — both end up in
    user POV). For bot turns, flip to bot POV so the resulting board is in
    the active player's perspective, which is what the rollout/analyzer
    expects. Also normalises bar sign to bgsage convention.
    """
    board = turn.get("board_before")
    if board is None:
        return None
    if turn.get("player") == "bot":
        board = list(flip_board(board))
    return _normalize_bar(board)


def _sage_played_board(turn: dict) -> list[int] | None:
    """Return Sage's played post-move board in mover's perspective.

    ``board_after`` is in user POV; bot turns need a flip — matches the
    convention used by ``compute_checker_error``. Also normalises bar
    sign to bgsage convention.
    """
    played = turn.get("board_after")
    if played is None:
        return None
    if turn.get("player") == "bot":
        played = list(flip_board(played))
    return _normalize_bar(played)


def _find_disputes_in_game(
    file_name: str, turns: list[dict],
    two_t_cache: dict, two_t_cache_path: Path, two_t_threads: int,
    progress_cb=None,
) -> list[Dispute]:
    """Walk turns and emit a Dispute for each decision where Sage-3p differs
    from XG **AND** 2T re-analysis still differs from XG.

    For every decision where the Sage 3-ply played choice does not match XG's
    recommendation (i.e. XG flagged it as a Sage error), the position is
    re-analyzed at 2T (truncated2 / XGRoller+):

      - If 2T's choice matches XG's, the disagreement vanishes — no dispute.
      - If 2T's choice still differs from XG's, emit a Dispute whose
        ``sage_choice`` is **2T's recommendation** and whose
        ``xg_measured_error`` is the 2T-measured equity gap between 2T's
        choice and XG's choice. This drives threshold and PR contributions.

    The 2T analyses are cached in ``two_t_cache`` (loaded from disk, appended
    incrementally). Assumes ``compute_game_pr_stats`` has already run.
    """
    cube_owner_user = "centered"
    cube_value = 1
    disputes: list[Dispute] = []

    for turn in turns:
        player = turn.get("player")
        if player not in ("user", "bot"):
            continue
        cube_owner_mover = _owner_relative_to(cube_owner_user, player)
        cube_action = turn.get("cube_action")
        cube_analysis = turn.get("cube_analysis")

        # Mid-turn cube state — reflects the state AFTER any cube action this
        # turn (double/take). The cube doubler and responder decisions are made
        # at the PRE-double state, but the checker move is played at the
        # POST-double state. If the active player doubled and the opp took,
        # the cube has already doubled and changed ownership by the time the
        # mover rolls and plays. (double/pass ends the game with no move; no
        # double leaves the state unchanged.)
        mid_cube_value = cube_value
        mid_cube_owner_user = cube_owner_user
        if cube_action == "double/take":
            mid_cube_value = cube_value * 2
            mid_cube_owner_user = "player" if player == "bot" else "opponent"
        mid_cube_owner_mover = _owner_relative_to(mid_cube_owner_user, player)

        # Cube 2T cache is shared between doubler and responder dispute checks
        # on the same turn — compute once.
        cube_2t = None
        def _ensure_cube_2t():
            nonlocal cube_2t
            if cube_2t is not None:
                return cube_2t
            board_2t = _board_for_analysis(turn)
            if board_2t is None:
                return None
            if progress_cb:
                progress_cb(file_name, turn["turn_number"], "cube")
            cube_2t = _get_or_compute_two_t_cube(
                two_t_cache, two_t_cache_path,
                file_name, turn["turn_number"], board_2t,
                cube_value, cube_owner_mover, two_t_threads,
            )
            return cube_2t

        # 1) Doubler dispute: Sage-3p flagged by XG → re-analyze at 2T.
        # Framework: 2T determines what Sage's "smart pick" is; the error
        # is then the XG-equity gap between XG's #1 and Sage's 2T pick.
        # If 2T agrees with XG, no dispute (error = 0). If 2T disagrees,
        # the error uses XG's equities (not 2T's), consistent with how
        # XG itself measures errors. ``sage_choice`` stores Sage's 2T
        # recommended action.
        if turn.get("is_cube_decision") and cube_analysis is not None:
            sage_3p_doubled = cube_action in ("double/take", "double/pass")
            xg_doubled = bool(cube_analysis.get("should_double", False))
            if sage_3p_doubled != xg_doubled:
                entry = _ensure_cube_2t()
                if entry is not None:
                    nd_2t = entry["equity_nd"]
                    dt_2t = entry["equity_dt"]
                    dp_2t = entry["equity_dp"]
                    two_t_doubled = min(dt_2t, dp_2t) > nd_2t
                    if two_t_doubled != xg_doubled:
                        # Compute error using XG's own equities for both
                        # XG's pick and Sage's 2T pick.
                        nd_xg = cube_analysis["equity_nd"]
                        dt_xg = cube_analysis["equity_dt"]
                        dp_xg = cube_analysis["equity_dp"]
                        eq_xg_best = max(nd_xg, min(dt_xg, dp_xg))
                        eq_2t_pick_per_xg = min(dt_xg, dp_xg) if two_t_doubled else nd_xg
                        measured = max(0.0, eq_xg_best - eq_2t_pick_per_xg)
                        if measured > 0.0:
                            board = _board_for_analysis(turn)
                            if board is not None:
                                disputes.append(Dispute(
                                    file=file_name,
                                    turn_number=turn["turn_number"],
                                    turn_player=player,
                                    deciding_player=player,
                                    decision_type="doubler",
                                    board=board,
                                    dice=None,
                                    cube_value=cube_value,
                                    cube_owner=cube_owner_mover,
                                    sage_choice={"action": "double" if two_t_doubled else "no_double"},
                                    xg_choice={"action": "double" if xg_doubled else "no_double"},
                                    xg_measured_error=measured,
                                ))

        # 2) Responder dispute: same framework as doubler.
        if (
            cube_action in ("double/take", "double/pass")
            and cube_analysis is not None
            and not _is_trivial_take_pass(cube_analysis)
        ):
            sage_3p_took = cube_action == "double/take"
            xg_took = bool(cube_analysis.get("should_take", True))
            if sage_3p_took != xg_took:
                entry = _ensure_cube_2t()
                if entry is not None:
                    dt_2t = entry["equity_dt"]
                    dp_2t = entry["equity_dp"]
                    two_t_took = dt_2t <= dp_2t
                    if two_t_took != xg_took:
                        # Responder error in doubler-equity space using XG's
                        # equities: the doubler-equity Sage's 2T pick gives
                        # XG, minus the best (minimum) per XG.
                        dt_xg = cube_analysis["equity_dt"]
                        dp_xg = cube_analysis["equity_dp"]
                        best_doubler_eq_xg = min(dt_xg, dp_xg)
                        eq_2t_pick_per_xg = dt_xg if two_t_took else dp_xg
                        measured = max(0.0, eq_2t_pick_per_xg - best_doubler_eq_xg)
                        if measured > 0.0:
                            board = _board_for_analysis(turn)
                            if board is not None:
                                responder = "bot" if player == "user" else "user"
                                disputes.append(Dispute(
                                    file=file_name,
                                    turn_number=turn["turn_number"],
                                    turn_player=player,
                                    deciding_player=responder,
                                    decision_type="responder",
                                    board=board,
                                    dice=None,
                                    cube_value=cube_value,
                                    cube_owner=cube_owner_mover,
                                    sage_choice={"action": "take" if two_t_took else "pass"},
                                    xg_choice={"action": "take" if xg_took else "pass"},
                                    xg_measured_error=measured,
                                ))

        # 3) Checker dispute: Sage-3p flagged → re-analyze at 2T.
        # NB: the checker move happens AFTER any cube action on this turn,
        # so we use the mid-turn cube state. Both Sage's played AND XG's
        # #1 are passed as ``important_boards`` so the 2T cache helper
        # guarantees both are at 2T quality — needed so moves[0] is the
        # true 2T-best (the TINY filter inside checker_play would
        # otherwise drop them to 1-ply).
        #
        # Framework: 2T determines Sage's "smart pick" (= moves[0] after
        # upgrade). The error uses XG's equities — we find Sage's 2T pick
        # in XG's checker_analysis and read its XG cubeful equity, then
        # subtract from XG's #1's equity. ``sage_choice`` stores Sage's
        # 2T pick (the move we attribute to Sage at the 2T level).
        if turn.get("is_checker_decision"):
            checker_analysis = turn.get("checker_analysis") or []
            dice = turn.get("dice")
            if checker_analysis and dice:
                xg_best_board = _normalize_bar(list(checker_analysis[0]["board"]))
                sage_3p_board = _sage_played_board(turn)
                if (
                    sage_3p_board is not None
                    and tuple(sage_3p_board) != tuple(xg_best_board)
                ):
                    board = _board_for_analysis(turn)
                    if board is not None:
                        d1, d2 = dice
                        if progress_cb:
                            progress_cb(file_name, turn["turn_number"], "checker")
                        entry = _get_or_compute_two_t_checker(
                            two_t_cache, two_t_cache_path,
                            file_name, turn["turn_number"],
                            board, [int(d1), int(d2)],
                            mid_cube_value, mid_cube_owner_mover, two_t_threads,
                            important_boards=(sage_3p_board, xg_best_board),
                        )
                        moves_2t = entry.get("moves") or []
                        if moves_2t:
                            two_t_best_board = list(moves_2t[0]["board"])
                            two_t_best_tuple = tuple(two_t_best_board)
                            xg_tuple = tuple(xg_best_board)
                            if two_t_best_tuple != xg_tuple:
                                # 2T disagrees with XG → find Sage's 2T pick
                                # in XG's analysis to read its XG equity, then
                                # compute error as XG's equity gap.
                                xg_best_eq = float(checker_analysis[0]["equity"])
                                xg_eq_of_2t_pick = None
                                for xg_m in checker_analysis:
                                    if tuple(_normalize_bar(list(xg_m["board"]))) == two_t_best_tuple:
                                        xg_eq_of_2t_pick = float(xg_m["equity"])
                                        break
                                if xg_eq_of_2t_pick is not None:
                                    measured = max(0.0, xg_best_eq - xg_eq_of_2t_pick)
                                    if measured > 0.0:
                                        disputes.append(Dispute(
                                            file=file_name,
                                            turn_number=turn["turn_number"],
                                            turn_player=player,
                                            deciding_player=player,
                                            decision_type="checker",
                                            board=board,
                                            dice=[int(d1), int(d2)],
                                            cube_value=mid_cube_value,
                                            cube_owner=mid_cube_owner_mover,
                                            sage_choice={"board": list(two_t_best_board)},
                                            xg_choice={"board": xg_best_board},
                                            xg_measured_error=measured,
                                        ))

        # Update cube state for the NEXT turn.
        if cube_action == "double/take":
            cube_owner_user = "player" if player == "bot" else "opponent"
            cube_value *= 2

    return disputes


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------


def _build_rollout_analyzer(seed: int = 42, n_threads: int = 0,
                            n_trials: int = _ROLLOUT_N_TRIALS):
    """Full rollout: n_trials (default 1,296), no truncation, 3-ply throughout."""
    from bgsage import BgBotAnalyzer
    import bgbot_cpp
    return BgBotAnalyzer(
        eval_level="rollout",
        cubeful=True,
        n_trials=n_trials,
        truncation_depth=_ROLLOUT_TRUNCATION_DEPTH,
        decision_ply=1,  # overridden by checker/cube below
        checker=bgbot_cpp.TrialEvalConfig(ply=_ROLLOUT_DECISION_PLY),
        cube=bgbot_cpp.TrialEvalConfig(ply=_ROLLOUT_DECISION_PLY),
        ultra_late_threshold=9999,  # no ply drop-down
        seed=seed,
        parallel_threads=n_threads,
    )


def _classify_case(sage_err: float, xg_err: float, tol: float = 1e-6) -> str:
    """sage_best / xg_best / both_wrong (both_best shouldn't happen on a real
    dispute, but report it if it slips through due to floating-point noise)."""
    sage_zero = sage_err < tol
    xg_zero = xg_err < tol
    if sage_zero and xg_zero:
        return "both_best"
    if sage_zero:
        return "sage_best"
    if xg_zero:
        return "xg_best"
    return "both_wrong"


def _rollout_quality_equity(
    analyzer, moves, target_board, pre_move_board,
    cube_value: int, cube_owner: str,
) -> tuple[float | None, str | None]:
    """Return ``(equity, eval_level)`` for ``target_board`` at the analyzer's
    full rollout quality, evaluating it explicitly if the standard
    ``checker_play`` returned only a 1-ply equity (because the TINY filter
    excluded it from the rollouts).

    Returns ``(None, None)`` if ``target_board`` isn't in the candidate list
    at all (shouldn't happen for legal played moves).
    """
    target_tuple = tuple(target_board)
    for m in moves:
        if tuple(m.board) == target_tuple:
            if m.eval_level == "Rollout":
                return float(m.equity), "Rollout"
            eq = _evaluate_single_board_at_2t(
                analyzer, target_board, pre_move_board,
                cube_value, cube_owner,
            )
            return eq, "Rollout (upgraded)"
    return None, None


def _rollout_checker(analyzer, dispute: Dispute) -> dict:
    d1, d2 = dispute.dice
    result = analyzer.checker_play(
        dispute.board, d1, d2,
        cube_value=dispute.cube_value,
        cube_owner=dispute.cube_owner,
        jacoby=True, beaver=True,
    )
    moves = list(result.moves)
    if not moves:
        return {"error": "no_legal_moves"}

    sage_board = dispute.sage_choice["board"]
    xg_board = dispute.xg_choice["board"]

    # Ensure Sage's played move and XG's pick are both at rollout quality.
    # If checker_play's TINY filter excluded either, evaluate them explicitly
    # via the rollout strategy's cubeful_evaluate_board.
    sage_eq, sage_eval = _rollout_quality_equity(
        analyzer, moves, sage_board, dispute.board,
        dispute.cube_value, dispute.cube_owner,
    )
    xg_eq, xg_eval = _rollout_quality_equity(
        analyzer, moves, xg_board, dispute.board,
        dispute.cube_value, dispute.cube_owner,
    )

    # Build the canonical set of rollout-quality equities to determine the
    # true best. Start with every rollout-survivor's equity, then layer in
    # the (possibly upgraded) sage and xg equities — Sage's or XG's pick
    # might actually be the rollout-best once properly evaluated.
    rollout_eqs: dict[tuple, float] = {}
    for m in moves:
        if m.eval_level == "Rollout":
            rollout_eqs[tuple(m.board)] = float(m.equity)
    if sage_eq is not None:
        rollout_eqs[tuple(sage_board)] = sage_eq
    if xg_eq is not None:
        rollout_eqs[tuple(xg_board)] = xg_eq

    best_tuple = max(rollout_eqs, key=lambda t: rollout_eqs[t])
    best_eq = rollout_eqs[best_tuple]

    out: dict = {
        "best_board": list(best_tuple),
        "best_equity": float(best_eq),
        "best_eval_level": "Rollout",
        "sage_equity": sage_eq,
        "sage_eval_level": sage_eval,
        "xg_equity": xg_eq,
        "xg_eval_level": xg_eval,
    }

    missing = []
    if sage_eq is None:
        missing.append("sage_move_not_found")
    if xg_eq is None:
        missing.append("xg_move_not_found")
    if missing:
        out["error"] = ";".join(missing)
        return out

    sage_err = max(0.0, best_eq - sage_eq)
    xg_err = max(0.0, best_eq - xg_eq)
    out["sage_error"] = float(sage_err)
    out["xg_error"] = float(xg_err)
    out["net_contribution"] = float(sage_err - xg_err)
    out["case"] = _classify_case(sage_err, xg_err)
    return out


def _shared_cube_result(analyzer, dispute: Dispute, cube_cache: dict):
    """Run cube_action once per (file, turn) and cache. A doubler dispute and
    a responder dispute on the same turn share the same rollout."""
    key = (dispute.file, dispute.turn_number)
    if key in cube_cache:
        return cube_cache[key]
    result = analyzer.cube_action(
        dispute.board,
        cube_value=dispute.cube_value,
        cube_owner=dispute.cube_owner,
        jacoby=True, beaver=True,
    )
    cube_cache[key] = result
    return result


def _rollout_cube_doubler(analyzer, dispute: Dispute, cube_cache: dict) -> dict:
    result = _shared_cube_result(analyzer, dispute, cube_cache)
    nd = float(result.equity_nd)
    dt = float(result.equity_dt)
    dp = float(result.equity_dp)
    eq_double = min(dt, dp)
    best_eq = max(nd, eq_double)
    best_action = "double" if eq_double > nd else "no_double"

    sage_action = dispute.sage_choice["action"]
    xg_action = dispute.xg_choice["action"]
    sage_eq = eq_double if sage_action == "double" else nd
    xg_eq = eq_double if xg_action == "double" else nd
    sage_err = max(0.0, best_eq - sage_eq)
    xg_err = max(0.0, best_eq - xg_eq)

    return {
        "equity_nd": nd,
        "equity_dt": dt,
        "equity_dp": dp,
        "best_action": best_action,
        "best_equity": best_eq,
        "sage_equity": sage_eq,
        "xg_equity": xg_eq,
        "sage_error": float(sage_err),
        "xg_error": float(xg_err),
        "net_contribution": float(sage_err - xg_err),
        "case": _classify_case(sage_err, xg_err),
    }


def _rollout_cube_responder(analyzer, dispute: Dispute, cube_cache: dict) -> dict:
    """Responder error is computed in doubler-equity space (matches the
    convention of ``compute_opp_cube_error``): each responder choice has an
    equity gift to the doubler; the responder's best is whatever minimises
    that gift; their error is the excess."""
    result = _shared_cube_result(analyzer, dispute, cube_cache)
    nd = float(result.equity_nd)
    dt = float(result.equity_dt)
    dp = float(result.equity_dp)
    best_doubler_eq = min(dt, dp)
    best_action = "take" if dt <= dp else "pass"

    sage_action = dispute.sage_choice["action"]
    xg_action = dispute.xg_choice["action"]
    sage_doubler_eq = dt if sage_action == "take" else dp
    xg_doubler_eq = dt if xg_action == "take" else dp
    sage_err = max(0.0, sage_doubler_eq - best_doubler_eq)
    xg_err = max(0.0, xg_doubler_eq - best_doubler_eq)

    return {
        "equity_nd": nd,
        "equity_dt": dt,
        "equity_dp": dp,
        "best_action": best_action,
        "best_doubler_equity": best_doubler_eq,
        "sage_doubler_equity": sage_doubler_eq,
        "xg_doubler_equity": xg_doubler_eq,
        "sage_error": float(sage_err),
        "xg_error": float(xg_err),
        "net_contribution": float(sage_err - xg_err),
        "case": _classify_case(sage_err, xg_err),
    }


def _run_rollout_for_dispute(analyzer, dispute: Dispute, cube_cache: dict) -> dict:
    if dispute.decision_type == "checker":
        return _rollout_checker(analyzer, dispute)
    if dispute.decision_type == "doubler":
        return _rollout_cube_doubler(analyzer, dispute, cube_cache)
    if dispute.decision_type == "responder":
        return _rollout_cube_responder(analyzer, dispute, cube_cache)
    return {"error": f"unknown_decision_type:{dispute.decision_type}"}


def _below_threshold_stub(dispute: Dispute, threshold: float) -> dict:
    """When XG-measured Sage error <= threshold, skip the rollout and assume
    Sage was wrong / XG was right. Contributes +xg_measured_error to net."""
    err = float(dispute.xg_measured_error)
    return {
        "skipped_below_threshold": True,
        "threshold": float(threshold),
        "sage_error": err,
        "xg_error": 0.0,
        "net_contribution": err,
        "case": "below_threshold",
    }


# ---------------------------------------------------------------------------
# JSONL output + resume
# ---------------------------------------------------------------------------


def _dispute_key(d: Dispute) -> str:
    return f"{d.file}#{d.turn_number}#{d.decision_type}"


def _record_key(rec: dict) -> str:
    return f"{rec.get('file')}#{rec.get('turn_number')}#{rec.get('decision_type')}"


def _load_done_rollouts(path: Path, threshold: float = 0.0) -> dict[str, dict]:
    """Read existing JSONL and return ``{key: record}`` for resumed runs.

    Records without a ``net_contribution`` (i.e. errored rollouts) are
    treated as NOT done so they get retried on resume.

    Skipped records whose ``xg_measured_error`` exceeds the current
    ``threshold`` are also treated as NOT done — at this (lower) threshold
    they should be rolled out, not skipped. This lets the user switch
    between thresholds without manually pruning the JSONL. Real rollout
    results are always reused regardless of threshold (rollout truth beats
    an assumed-Sage-wrong skip).
    """
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = rec.get("rollout") or {}
            if "net_contribution" not in r:
                continue
            if r.get("skipped_below_threshold"):
                xg_err = float(rec.get("xg_measured_error", 0.0))
                if xg_err > threshold:
                    continue   # stale skip; re-evaluate at lower threshold
            done[_record_key(rec)] = rec
    return done


def _write_rollout_record(
    path: Path, dispute: Dispute, rollout_result: dict
) -> dict:
    """Append one record and flush, so the file is always consistent."""
    rec = {
        "file": dispute.file,
        "turn_number": dispute.turn_number,
        "turn_player": dispute.turn_player,
        "deciding_player": dispute.deciding_player,
        "decision_type": dispute.decision_type,
        "board": list(dispute.board),
        "dice": dispute.dice,
        "cube_value": dispute.cube_value,
        "cube_owner": dispute.cube_owner,
        "sage_choice": dispute.sage_choice,
        "xg_choice": dispute.xg_choice,
        "xg_measured_error": float(dispute.xg_measured_error),
        "rollout": rollout_result,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=_DEFAULT_DIR,
        help=f"Folder to scan for .xg files (default: {_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--pattern",
        default="*.xg",
        help="Glob pattern within folder (default: *.xg)",
    )
    parser.add_argument(
        "--skip-rollouts",
        action="store_true",
        help="Stop after the original PR aggregation; do no rollouts.",
    )
    parser.add_argument(
        "--rollout-file",
        type=Path,
        default=None,
        help=(
            "Output JSONL for rollout results "
            f"(default: <folder>/{_DEFAULT_ROLLOUT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=42,
        help="Rollout RNG seed (default: 42).",
    )
    parser.add_argument(
        "--rollout-threads",
        type=int,
        default=0,
        help="Threads per rollout (default: 0 = auto-detect cores).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_ROLLOUT_N_TRIALS,
        help=(
            f"Trials per rollout (default: {_ROLLOUT_N_TRIALS}; keep a multiple "
            "of 36 for VR stratification, e.g. 5184 = 4x1296)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only roll out the first N pending disputes (handy for testing).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help=(
            "Skip the rollout when XG's measured Sage error is <= this value "
            "(default: 0.005). Skipped disputes are assumed to be cases where "
            "Sage was wrong and XG was right, contributing +xg_measured_error "
            "to net Sage error."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report how many rollouts would be run at the current threshold "
            "and stop (no rollouts, no JSONL writes). Re-analysis still "
            "runs so the dispute count reflects post-re-eval reduction."
        ),
    )
    parser.add_argument(
        "--re-eval-level",
        choices=_RE_EVAL_LEVEL_CHOICES,
        default=_DEFAULT_RE_EVAL_LEVEL,
        help=(
            f"Eval level used to re-evaluate positions where Sage 3-ply "
            f"disagrees with XG (default: {_DEFAULT_RE_EVAL_LEVEL})."
        ),
    )
    parser.add_argument(
        "--re-eval-trunc-depth",
        type=int,
        default=None,
        help=(
            "Override the re-eval rollout's truncation_depth (default: the named "
            "level's: 7 for both 3T and 2T). Builds eval_level='rollout' from the "
            "named level's base params with this override."
        ),
    )
    parser.add_argument(
        "--re-eval-ultra-late",
        type=int,
        default=None,
        help=(
            "Override the re-eval rollout's ultra_late_threshold (default: the "
            "named level's: 2 for 1T/2T, 9999 for 3T). E.g. 9999 keeps the configured late ply instead of "
            "dropping to 1-ply."
        ),
    )
    parser.add_argument(
        "--two-t-cache-file",
        type=Path,
        default=None,
        help=(
            "Path to the re-analysis cache JSONL (default: "
            "<folder>/sage_<level>_cache.jsonl, e.g. sage_3T_cache.jsonl)."
        ),
    )
    parser.add_argument(
        "--two-t-threads",
        type=int,
        default=0,
        help="Threads for the re-analyzer (default: 0 = auto-detect cores).",
    )
    args = parser.parse_args()

    # Apply re-eval level + optional overrides before any analyzer is built.
    global _TWO_T_EVAL_LEVEL, _RE_EVAL_TRUNC_DEPTH, _RE_EVAL_ULTRA_LATE
    _TWO_T_EVAL_LEVEL = args.re_eval_level
    _RE_EVAL_TRUNC_DEPTH = args.re_eval_trunc_depth
    _RE_EVAL_ULTRA_LATE = args.re_eval_ultra_late

    xg_files = sorted(args.folder.glob(args.pattern), key=_seed_sort_key)
    if not xg_files:
        print(f"No .xg files found in {args.folder} matching {args.pattern!r}")
        return

    print(f"Scanning {len(xg_files)} .xg files in {args.folder}\n")
    header = (
        f"{'file':<16} "
        f"{'P1 err':>9} {'P1 dec':>7} "
        f"{'P2 err':>9} {'P2 dec':>7} "
        f"{'tot err':>9} {'tot dec':>7} "
        f"{'PR':>7}"
    )
    print(header)
    print("-" * len(header))

    per_game_pr: list[float] = []
    sum_err = 0.0
    sum_dec = 0
    all_disputes: list[Dispute] = []

    # Set up re-analysis cache (incrementally appended on disk).
    two_t_cache_path = (
        args.two_t_cache_file
        or (args.folder / _re_eval_cache_filename(_TWO_T_EVAL_LEVEL))
    )
    two_t_cache_path.parent.mkdir(parents=True, exist_ok=True)
    two_t_cache = _load_two_t_cache(two_t_cache_path)
    print(f"\nRe-analysis level: {_TWO_T_EVAL_LEVEL}")
    print(f"Re-analysis cache: {two_t_cache_path}")
    print(f"  pre-loaded entries: {len(two_t_cache)}")

    two_t_misses = {"checker": 0, "cube": 0}
    def _two_t_progress(file, turn_number, kind):
        # Hook fired before potentially-missing 2T computation. We rely on
        # the cache dict membership check inside the helpers to decide hits
        # vs misses; this callback just gives us a turn-level progress mark.
        key = _two_t_cache_key(file, turn_number, kind)
        if key not in two_t_cache:
            two_t_misses[kind] += 1

    for xg_path in xg_files:
        turns = parse_xg_game(xg_path.read_bytes())
        s = compute_game_pr_stats(turns)  # also populates decision flags
        per_game_pr.append(s["pr"])
        sum_err += s["total_err"]
        sum_dec += s["total_dec"]
        pr_text = f"{s['pr']:.2f}" if not math.isnan(s["pr"]) else "  nan"
        print(
            f"{xg_path.name:<16} "
            f"{s['user_err']:>9.4f} {s['user_dec']:>7} "
            f"{s['bot_err']:>9.4f} {s['bot_dec']:>7} "
            f"{s['total_err']:>9.4f} {s['total_dec']:>7} "
            f"{pr_text:>7}"
        )
        all_disputes.extend(_find_disputes_in_game(
            xg_path.name, turns,
            two_t_cache, two_t_cache_path, args.two_t_threads,
            progress_cb=_two_t_progress,
        ))

    print(
        f"\n{_TWO_T_EVAL_LEVEL} re-analysis: {two_t_misses['checker']} checker + "
        f"{two_t_misses['cube']} cube positions computed this run "
        f"(cache size now {len(two_t_cache)})."
    )

    n = len(per_game_pr)
    valid_pr = [p for p in per_game_pr if not math.isnan(p)]
    mean_pr = statistics.mean(valid_pr) if valid_pr else float("nan")
    std_pr = statistics.stdev(valid_pr) if len(valid_pr) > 1 else 0.0
    sem_pr = std_pr / math.sqrt(len(valid_pr)) if valid_pr else 0.0
    agg_pr = (sum_err / sum_dec * 500.0) if sum_dec > 0 else float("nan")

    print("-" * len(header))
    print()
    print(f"Games:                {n}")
    print(f"Per-game PR mean:     {mean_pr:.3f}")
    print(f"Per-game PR std dev:  {std_pr:.3f}")
    print(f"Per-game PR SEM:      {sem_pr:.3f}")
    print()
    print(f"Total errors summed:  {sum_err:.4f}")
    print(f"Total decisions:      {sum_dec}")
    print(f"Aggregate PR:         {agg_pr:.3f}")

    # -------------------------------------------------------------------
    # Rollout phase
    # -------------------------------------------------------------------
    by_type = Counter(d.decision_type for d in all_disputes)
    print()
    print(f"Disputed decisions:   {len(all_disputes)}")
    for k in ("doubler", "responder", "checker"):
        if by_type.get(k):
            print(f"  {k:>10}: {by_type[k]}")

    if args.skip_rollouts:
        print("\n--skip-rollouts: stopping before rollout phase.")
        return

    if not all_disputes:
        print("\nNo disputes; nothing to roll out.")
        return

    rollout_file = args.rollout_file or (args.folder / _DEFAULT_ROLLOUT_FILENAME)
    done = _load_done_rollouts(rollout_file, threshold=args.threshold)
    todo = [d for d in all_disputes if _dispute_key(d) not in done]
    print(f"\nRollouts: {len(done)} already done, {len(todo)} remaining.")
    print(f"Output file: {rollout_file}")

    if args.limit is not None and args.limit >= 0:
        todo = todo[: args.limit]
        print(f"--limit applied: rolling out only the first {len(todo)}.")

    if todo:
        n_to_roll = sum(1 for d in todo if d.xg_measured_error > args.threshold)
        n_to_skip = len(todo) - n_to_roll
        print(
            f"Threshold {args.threshold:.4f}: "
            f"{n_to_roll} will be rolled out, {n_to_skip} skipped (assumed Sage wrong)."
        )
        if args.dry_run:
            print(f"\n--dry-run: would roll out {n_to_roll} disputes. Stopping.")
            return
        rollout_file.parent.mkdir(parents=True, exist_ok=True)
        analyzer = None
        if n_to_roll > 0:
            print(f"Building rollout analyzer ({args.n_trials} trials, full play-out, 3-ply throughout)...")
            analyzer = _build_rollout_analyzer(
                seed=args.rollout_seed, n_threads=args.rollout_threads,
                n_trials=args.n_trials,
            )
        cube_cache: dict = {}
        for i, dispute in enumerate(todo, 1):
            tag = (
                f"[{i}/{len(todo)}] {dispute.file} turn {dispute.turn_number} "
                f"{dispute.decision_type} ({dispute.deciding_player})"
            )
            below = dispute.xg_measured_error <= args.threshold
            if below:
                print(
                    f"{tag}: skipped (xg_err={dispute.xg_measured_error:.4f} "
                    f"<= {args.threshold:.4f})",
                    flush=True,
                )
                rollout_result = _below_threshold_stub(dispute, args.threshold)
            else:
                print(
                    f"{tag}: rolling out (xg_err={dispute.xg_measured_error:.4f})...",
                    flush=True,
                )
                try:
                    rollout_result = _run_rollout_for_dispute(
                        analyzer, dispute, cube_cache,
                    )
                except Exception as e:
                    rollout_result = {
                        "error": f"exception: {type(e).__name__}: {e}",
                    }
            rec = _write_rollout_record(rollout_file, dispute, rollout_result)
            done[_dispute_key(dispute)] = rec

            r = rec["rollout"]
            if "net_contribution" in r:
                print(
                    f"    sage_err={r.get('sage_error', 0):.4f}  "
                    f"xg_err={r.get('xg_error', 0):.4f}  "
                    f"net={r.get('net_contribution', 0):+.4f}  "
                    f"case={r.get('case', '?')}",
                    flush=True,
                )
            else:
                print(f"    error: {r.get('error', 'unknown')}", flush=True)

    # -------------------------------------------------------------------
    # Summary across rolled-out disputes only.
    #
    # The "rollout is the ground truth" framing: for each dispute that was
    # actually rolled out, we measure both Sage's and XG's chosen moves
    # against the rollout's best, using rollout-derived equities (NOT XG's
    # estimates). Cases skipped below the threshold are excluded — they
    # carry no rollout information.
    # -------------------------------------------------------------------
    cases: Counter = Counter()
    sage_errs: list[float] = []
    xg_errs: list[float] = []
    pending = 0
    skipped_no_info = 0
    errored: list[tuple[str, str]] = []

    for d in all_disputes:
        rec = done.get(_dispute_key(d))
        if rec is None:
            pending += 1
            continue
        r = rec.get("rollout") or {}
        if r.get("skipped_below_threshold"):
            skipped_no_info += 1
            continue
        if "net_contribution" not in r:
            errored.append((_dispute_key(d), str(r.get("error", "missing"))))
            continue
        cases[r.get("case", "?")] += 1
        sage_errs.append(float(r.get("sage_error", 0.0)))
        xg_errs.append(float(r.get("xg_error", 0.0)))

    n = len(sage_errs)
    print()
    print(f"Rolled-out disputes: {n}    "
          f"(skipped {skipped_no_info} below xg_err <= {args.threshold:.4f}, "
          f"pending {pending}, errored {len(errored)})")
    if errored:
        print(f"  Errored (first 5): {errored[:5]}")

    if n == 0:
        print("\nNo completed rollouts — nothing to summarize.")
        return

    def pct(c: int) -> str:
        return f"{c}/{n} ({100.0 * c / n:.1f}%)"

    print()
    print("Decision outcomes vs rollout best (using rollout equities):")
    print(f"  Sage matched rollout best:    {pct(cases.get('sage_best', 0))}")
    print(f"  XG matched rollout best:      {pct(cases.get('xg_best', 0))}")
    print(f"  Both matched (tied):          {pct(cases.get('both_best', 0))}")
    print(f"  Neither matched:              {pct(cases.get('both_wrong', 0))}")

    avg_sage = sum(sage_errs) / n
    avg_xg = sum(xg_errs) / n
    print()
    print("Average rollout-equity error (per disputed decision):")
    print(f"  Sage: {avg_sage:.4f}    (sum {sum(sage_errs):.4f} over {n} disputes)")
    print(f"  XG:   {avg_xg:.4f}    (sum {sum(xg_errs):.4f} over {n} disputes)")
    print(f"  Diff (Sage - XG):    {(avg_sage - avg_xg):+.4f}")
    print("  Sign: positive = XG is the better player; negative = Sage is better.")


if __name__ == "__main__":
    main()
