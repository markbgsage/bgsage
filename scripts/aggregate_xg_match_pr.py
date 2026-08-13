# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Match-play analog of ``aggregate_xg_pr.py``.

Each input .xg file is a single match containing multiple games. For every
game in every match this script:

1. Aggregates per-game (and across-match) PR using XG's own equities — XG
   already encoded match equity into its equity_nd/dt/dp/checker numbers,
   so ``compute_game_pr_stats`` works unchanged.

2. Re-evaluates Sage's flagged decisions at 3T (configurable via
   ``--re-eval-level``) with match state passed through (away counts,
   Crawford). 2T disagreement with XG drives ``Dispute`` emission; the
   measured error uses XG's own equities (consistent with the money-game
   pipeline).

3. Rolls out each remaining disputed decision (1,296 trials, no truncation,
   3-ply throughout) with match state threaded through so MWC scoring is
   used. Net Sage error per dispute is rollout-best minus chosen-equity,
   signed (sage_err - xg_err) so positive means XG is the better player.

Outputs:
  - Per-game PR table per match
  - Per-match aggregate PR + per-game PR mean/std
  - Across-matches aggregate PR
  - Dispute counts by type
  - Rollout-based net Sage error and "new PR"

Cache files:
  - ``sage_<level>_cache.jsonl`` — re-eval cache keyed by
    (file, game_number, turn_number, kind). Match state changes between
    games inside a match, so including game_number is mandatory.
  - ``rollout_disputes.jsonl`` — JSONL of rolled-out (or threshold-skipped)
    dispute records; keyed by (file, game_number, turn_number, decision_type)
    so resumes work cleanly.

Default folder is ``bgsage/logs/sage_vs_sage_match`` (inside the bgsage repo,
matching where ``run_sage_vs_sage_match.py`` writes its .txt files). After
running XG's Batch Analyze on those files with "Save Games after analyze"
checked, run this script.

Usage:

    python bgsage/scripts/aggregate_xg_match_pr.py
    python bgsage/scripts/aggregate_xg_match_pr.py --re-eval-level truncated2
    python bgsage/scripts/aggregate_xg_match_pr.py --skip-rollouts
    python bgsage/scripts/aggregate_xg_match_pr.py --limit 5
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
# bgsage path setup — strictly inside the bgsage repo per CLAUDE.md
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent           # bgsage/
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


from bgsage.board import flip_board  # noqa: E402
from bgsage.xg_compare import (  # noqa: E402
    _is_trivial_take_pass,
    compute_game_pr_stats,
    parse_xg_match,
)

_DEFAULT_DIR = _PROJECT_ROOT / "logs" / "sage_vs_sage_match"
_DEFAULT_ROLLOUT_FILENAME = "rollout_disputes.jsonl"

# Rollout config matches the money-game aggregator: full rollout, no
# truncation, 3-ply throughout.
_ROLLOUT_N_TRIALS = 1296
_ROLLOUT_TRUNCATION_DEPTH = 0
_ROLLOUT_DECISION_PLY = 3

_DEFAULT_RE_EVAL_LEVEL = "truncated3"
_RE_EVAL_LEVEL_CHOICES = ["truncated2", "truncated3"]

_RE_EVAL_LEVEL = _DEFAULT_RE_EVAL_LEVEL


def _re_eval_cache_filename(level: str) -> str:
    short = {"truncated2": "2T", "truncated3": "3T"}.get(level, level)
    return f"sage_{short}_cache.jsonl"


# ---------------------------------------------------------------------------
# Re-analyzer cache
# ---------------------------------------------------------------------------


_re_eval_analyzer = None  # lazy-built


def _get_re_eval_analyzer(n_threads: int = 0):
    global _re_eval_analyzer
    if _re_eval_analyzer is None:
        from bgsage import BgBotAnalyzer
        _re_eval_analyzer = BgBotAnalyzer(
            eval_level=_RE_EVAL_LEVEL, cubeful=True,
            parallel_threads=n_threads,
        )
    return _re_eval_analyzer


def _re_eval_cache_key(file: str, game_number: int, turn_number: int, kind: str) -> str:
    return f"{file}#{game_number}#{turn_number}#{kind}"


def _load_re_eval_cache(path: Path) -> dict[str, dict]:
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
            cache[_re_eval_cache_key(
                rec["file"], rec.get("game_number", 0),
                rec["turn_number"], rec["kind"],
            )] = rec
    return cache


def _write_re_eval_cache_entry(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def _evaluate_single_board_at_re_eval(
    analyzer, post_move_board: list[int], pre_move_board: list[int],
    cube_value: int, cube_owner: str,
    away1: int, away2: int, is_crawford: bool,
) -> float:
    """Evaluate one post-move board at the analyzer's full cubeful quality.

    For match play (``away1 > 0`` or ``away2 > 0``) jacoby and beaver are
    auto-disabled — matches the convention BgBotAnalyzer uses. The C++
    binding takes those flags directly, so we explicitly suppress them.
    """
    from bgsage.analyzer import resolve_owner
    is_match = away1 > 0 or away2 > 0
    jacoby = not is_match
    beaver = not is_match
    inner = analyzer._analyzer
    if hasattr(inner, "_inner"):
        inner = inner._inner
    r = inner._rollout_strategy.cubeful_evaluate_board(
        list(post_move_board), list(pre_move_board),
        cube_value=cube_value, owner=resolve_owner(cube_owner),
        away1=away1, away2=away2, is_crawford=is_crawford,
        jacoby=jacoby, beaver=beaver,
    )
    return float(r["cubeful_equity"])


def _ensure_boards_at_re_eval(
    entry: dict, pre_move_board: list[int],
    target_boards,
    cube_value: int, cube_owner: str,
    away1: int, away2: int, is_crawford: bool,
    n_threads: int,
) -> tuple[dict, bool]:
    """Upgrade any board in ``target_boards`` to full Rollout-level cubeful
    equity inside the cached ``entry``. Returns (entry, was_updated)."""
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
            continue
        if analyzer is None:
            analyzer = _get_re_eval_analyzer(n_threads)
        eq = _evaluate_single_board_at_re_eval(
            analyzer, tb_norm, pre_move_board, cube_value, cube_owner,
            away1, away2, is_crawford,
        )
        moves = [m for m in moves if tuple(m["board"]) != tb_tuple]
        moves.append({"board": list(tb_norm), "equity": eq,
                      "eval_level": "Rollout"})
        updated = True
    if updated:
        moves.sort(key=lambda m: -m["equity"])
        return {**entry, "moves": moves}, True
    return entry, False


def _get_or_compute_re_eval_checker(
    cache: dict, cache_path: Path,
    file: str, game_number: int, turn_number: int,
    board: list[int], dice: list[int],
    cube_value: int, cube_owner: str,
    away1: int, away2: int, is_crawford: bool,
    n_threads: int,
    important_boards=(),
) -> dict:
    """Get-or-compute the re-eval checker_play result and guarantee the
    important boards (Sage's played + XG's #1) are at Rollout quality so
    moves[0] reflects the true re-eval best."""
    key = _re_eval_cache_key(file, game_number, turn_number, "checker")
    entry = cache.get(key)
    if entry is None:
        analyzer = _get_re_eval_analyzer(n_threads)
        result = analyzer.checker_play(
            board, dice[0], dice[1],
            cube_value=cube_value, cube_owner=cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
        )
        entry = {
            "file": file, "game_number": game_number,
            "turn_number": turn_number, "kind": "checker",
            "away1": away1, "away2": away2, "is_crawford": is_crawford,
            "moves": [
                {"board": _normalize_bar(list(m.board)),
                 "equity": float(m.equity),
                 "eval_level": m.eval_level}
                for m in result.moves
            ],
        }
        entry, _ = _ensure_boards_at_re_eval(
            entry, board, important_boards,
            cube_value, cube_owner, away1, away2, is_crawford,
            n_threads,
        )
        cache[key] = entry
        _write_re_eval_cache_entry(cache_path, entry)
        return entry
    updated, was_updated = _ensure_boards_at_re_eval(
        entry, board, important_boards,
        cube_value, cube_owner, away1, away2, is_crawford,
        n_threads,
    )
    if was_updated:
        cache[key] = updated
        _write_re_eval_cache_entry(cache_path, updated)
        return updated
    return entry


def _get_or_compute_re_eval_cube(
    cache: dict, cache_path: Path,
    file: str, game_number: int, turn_number: int,
    board: list[int], cube_value: int, cube_owner: str,
    away1: int, away2: int, is_crawford: bool,
    n_threads: int,
) -> dict:
    key = _re_eval_cache_key(file, game_number, turn_number, "cube")
    if key in cache:
        return cache[key]
    analyzer = _get_re_eval_analyzer(n_threads)
    result = analyzer.cube_action(
        board, cube_value=cube_value, cube_owner=cube_owner,
        away1=away1, away2=away2, is_crawford=is_crawford,
    )
    entry = {
        "file": file, "game_number": game_number,
        "turn_number": turn_number, "kind": "cube",
        "away1": away1, "away2": away2, "is_crawford": is_crawford,
        "equity_nd": float(result.equity_nd),
        "equity_dt": float(result.equity_dt),
        "equity_dp": float(result.equity_dp),
    }
    cache[key] = entry
    _write_re_eval_cache_entry(cache_path, entry)
    return entry


# ---------------------------------------------------------------------------
# File-listing helpers
# ---------------------------------------------------------------------------


def _seed_sort_key(p: Path) -> tuple[int, str]:
    parts = p.stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (int(parts[1]), p.name)
    return (10**9, p.name)


# ---------------------------------------------------------------------------
# Dispute model
# ---------------------------------------------------------------------------


@dataclass
class Dispute:
    """Same shape as the money-game ``Dispute`` plus match context fields.

    ``away1``/``away2`` are stored in the active turn-player's perspective —
    that's the perspective the board is in, and the perspective the analyzer
    and rollout expect.
    """

    file: str
    game_number: int
    turn_number: int
    turn_player: str
    deciding_player: str
    decision_type: str          # "doubler" | "responder" | "checker"
    board: list[int]
    dice: list[int] | None
    cube_value: int
    cube_owner: str
    away1: int
    away2: int
    is_crawford: bool
    sage_choice: dict
    xg_choice: dict
    xg_measured_error: float


def _flip_owner(owner: str) -> str:
    return {"player": "opponent", "opponent": "player", "centered": "centered"}[owner]


def _owner_relative_to(user_perspective: str, player: str) -> str:
    return user_perspective if player == "user" else _flip_owner(user_perspective)


def _normalize_bar(board: list[int]) -> list[int]:
    return [abs(board[0])] + list(board[1:25]) + [abs(board[25])]


def _board_for_analysis(turn: dict) -> list[int] | None:
    board = turn.get("board_before")
    if board is None:
        return None
    if turn.get("player") == "bot":
        board = list(flip_board(board))
    return _normalize_bar(board)


def _sage_played_board(turn: dict) -> list[int] | None:
    played = turn.get("board_after")
    if played is None:
        return None
    if turn.get("player") == "bot":
        played = list(flip_board(played))
    return _normalize_bar(played)


def _away_counts_for_active(
    player: str, score1: int, score2: int, match_length: int,
) -> tuple[int, int]:
    """Return (away_mover, away_opp) given match length, current scores, and
    the active player's identity ('user' = P1, 'bot' = P2)."""
    if match_length <= 0:
        return (0, 0)  # money game
    if player == "user":
        mover_score, opp_score = score1, score2
    else:
        mover_score, opp_score = score2, score1
    return (max(1, match_length - mover_score), max(1, match_length - opp_score))


def _find_disputes_in_game(
    file_name: str, game_number: int, turns: list[dict],
    score1_start: int, score2_start: int, match_length: int, is_crawford: bool,
    re_eval_cache: dict, re_eval_cache_path: Path, re_eval_threads: int,
    progress_cb=None,
) -> list[Dispute]:
    """Walk turns in one match-game and emit Disputes.

    Match state (``away1``/``away2``/``is_crawford``) is constant within a
    game — passed in from the caller. Cube state evolves turn-by-turn the
    same way the money-game version tracks it.
    """
    cube_owner_user = "centered"
    cube_value = 1
    disputes: list[Dispute] = []

    for turn in turns:
        player = turn.get("player")
        if player not in ("user", "bot"):
            continue

        away_mover, away_opp = _away_counts_for_active(
            player, score1_start, score2_start, match_length,
        )

        cube_owner_mover = _owner_relative_to(cube_owner_user, player)
        cube_action = turn.get("cube_action")
        cube_analysis = turn.get("cube_analysis")

        # Mid-turn cube state for checker disputes (same convention as the
        # money-game aggregator).
        mid_cube_value = cube_value
        mid_cube_owner_user = cube_owner_user
        if cube_action == "double/take":
            mid_cube_value = cube_value * 2
            mid_cube_owner_user = "player" if player == "bot" else "opponent"
        mid_cube_owner_mover = _owner_relative_to(mid_cube_owner_user, player)

        cube_re_eval = None
        def _ensure_cube_re_eval():
            nonlocal cube_re_eval
            if cube_re_eval is not None:
                return cube_re_eval
            board_re_eval = _board_for_analysis(turn)
            if board_re_eval is None:
                return None
            if progress_cb:
                progress_cb(file_name, game_number, turn["turn_number"], "cube")
            cube_re_eval = _get_or_compute_re_eval_cube(
                re_eval_cache, re_eval_cache_path,
                file_name, game_number, turn["turn_number"], board_re_eval,
                cube_value, cube_owner_mover,
                away_mover, away_opp, is_crawford,
                re_eval_threads,
            )
            return cube_re_eval

        # 1) Doubler dispute
        if turn.get("is_cube_decision") and cube_analysis is not None:
            sage_3p_doubled = cube_action in ("double/take", "double/pass")
            xg_doubled = bool(cube_analysis.get("should_double", False))
            if sage_3p_doubled != xg_doubled:
                entry = _ensure_cube_re_eval()
                if entry is not None:
                    nd_re = entry["equity_nd"]
                    dt_re = entry["equity_dt"]
                    dp_re = entry["equity_dp"]
                    re_eval_doubled = min(dt_re, dp_re) > nd_re
                    if re_eval_doubled != xg_doubled:
                        nd_xg = cube_analysis["equity_nd"]
                        dt_xg = cube_analysis["equity_dt"]
                        dp_xg = cube_analysis["equity_dp"]
                        eq_xg_best = max(nd_xg, min(dt_xg, dp_xg))
                        eq_re_pick_per_xg = (
                            min(dt_xg, dp_xg) if re_eval_doubled else nd_xg
                        )
                        measured = max(0.0, eq_xg_best - eq_re_pick_per_xg)
                        if measured > 0.0:
                            board = _board_for_analysis(turn)
                            if board is not None:
                                disputes.append(Dispute(
                                    file=file_name,
                                    game_number=game_number,
                                    turn_number=turn["turn_number"],
                                    turn_player=player,
                                    deciding_player=player,
                                    decision_type="doubler",
                                    board=board,
                                    dice=None,
                                    cube_value=cube_value,
                                    cube_owner=cube_owner_mover,
                                    away1=away_mover, away2=away_opp,
                                    is_crawford=is_crawford,
                                    sage_choice={"action": "double" if re_eval_doubled else "no_double"},
                                    xg_choice={"action": "double" if xg_doubled else "no_double"},
                                    xg_measured_error=measured,
                                ))

        # 2) Responder dispute
        if (
            cube_action in ("double/take", "double/pass")
            and cube_analysis is not None
            and not _is_trivial_take_pass(cube_analysis)
        ):
            sage_3p_took = cube_action == "double/take"
            xg_took = bool(cube_analysis.get("should_take", True))
            if sage_3p_took != xg_took:
                entry = _ensure_cube_re_eval()
                if entry is not None:
                    dt_re = entry["equity_dt"]
                    dp_re = entry["equity_dp"]
                    re_eval_took = dt_re <= dp_re
                    if re_eval_took != xg_took:
                        dt_xg = cube_analysis["equity_dt"]
                        dp_xg = cube_analysis["equity_dp"]
                        best_doubler_eq_xg = min(dt_xg, dp_xg)
                        eq_re_pick_per_xg = dt_xg if re_eval_took else dp_xg
                        measured = max(0.0, eq_re_pick_per_xg - best_doubler_eq_xg)
                        if measured > 0.0:
                            board = _board_for_analysis(turn)
                            if board is not None:
                                responder = "bot" if player == "user" else "user"
                                disputes.append(Dispute(
                                    file=file_name,
                                    game_number=game_number,
                                    turn_number=turn["turn_number"],
                                    turn_player=player,
                                    deciding_player=responder,
                                    decision_type="responder",
                                    board=board,
                                    dice=None,
                                    cube_value=cube_value,
                                    cube_owner=cube_owner_mover,
                                    away1=away_mover, away2=away_opp,
                                    is_crawford=is_crawford,
                                    sage_choice={"action": "take" if re_eval_took else "pass"},
                                    xg_choice={"action": "take" if xg_took else "pass"},
                                    xg_measured_error=measured,
                                ))

        # 3) Checker dispute — uses mid-turn cube state but the same
        # per-game match state (away counts / Crawford don't change mid-game).
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
                            progress_cb(file_name, game_number, turn["turn_number"], "checker")
                        entry = _get_or_compute_re_eval_checker(
                            re_eval_cache, re_eval_cache_path,
                            file_name, game_number, turn["turn_number"],
                            board, [int(d1), int(d2)],
                            mid_cube_value, mid_cube_owner_mover,
                            away_mover, away_opp, is_crawford,
                            re_eval_threads,
                            important_boards=(sage_3p_board, xg_best_board),
                        )
                        moves_re = entry.get("moves") or []
                        if moves_re:
                            re_best_board = list(moves_re[0]["board"])
                            re_best_tuple = tuple(re_best_board)
                            xg_tuple = tuple(xg_best_board)
                            if re_best_tuple != xg_tuple:
                                xg_best_eq = float(checker_analysis[0]["equity"])
                                xg_eq_of_re_pick = None
                                for xg_m in checker_analysis:
                                    if tuple(_normalize_bar(list(xg_m["board"]))) == re_best_tuple:
                                        xg_eq_of_re_pick = float(xg_m["equity"])
                                        break
                                if xg_eq_of_re_pick is not None:
                                    measured = max(0.0, xg_best_eq - xg_eq_of_re_pick)
                                    if measured > 0.0:
                                        disputes.append(Dispute(
                                            file=file_name,
                                            game_number=game_number,
                                            turn_number=turn["turn_number"],
                                            turn_player=player,
                                            deciding_player=player,
                                            decision_type="checker",
                                            board=board,
                                            dice=[int(d1), int(d2)],
                                            cube_value=mid_cube_value,
                                            cube_owner=mid_cube_owner_mover,
                                            away1=away_mover, away2=away_opp,
                                            is_crawford=is_crawford,
                                            sage_choice={"board": list(re_best_board)},
                                            xg_choice={"board": xg_best_board},
                                            xg_measured_error=measured,
                                        ))

        if cube_action == "double/take":
            cube_owner_user = "player" if player == "bot" else "opponent"
            cube_value *= 2

    return disputes


# ---------------------------------------------------------------------------
# Rollout runner
# ---------------------------------------------------------------------------


def _build_rollout_analyzer(seed: int = 42, n_threads: int = 0,
                            n_trials: int = _ROLLOUT_N_TRIALS):
    from bgsage import BgBotAnalyzer
    import bgbot_cpp
    return BgBotAnalyzer(
        eval_level="rollout",
        cubeful=True,
        n_trials=n_trials,
        truncation_depth=_ROLLOUT_TRUNCATION_DEPTH,
        decision_ply=1,
        checker=bgbot_cpp.TrialEvalConfig(ply=_ROLLOUT_DECISION_PLY),
        cube=bgbot_cpp.TrialEvalConfig(ply=_ROLLOUT_DECISION_PLY),
        ultra_late_threshold=9999,
        seed=seed,
        parallel_threads=n_threads,
    )


def _classify_case(sage_err: float, xg_err: float, tol: float = 1e-6) -> str:
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
    away1: int, away2: int, is_crawford: bool,
) -> tuple[float | None, str | None]:
    target_tuple = tuple(target_board)
    for m in moves:
        if tuple(m.board) == target_tuple:
            if m.eval_level == "Rollout":
                return float(m.equity), "Rollout"
            eq = _evaluate_single_board_at_re_eval(
                analyzer, target_board, pre_move_board,
                cube_value, cube_owner,
                away1, away2, is_crawford,
            )
            return eq, "Rollout (upgraded)"
    return None, None


def _rollout_checker(analyzer, dispute: Dispute) -> dict:
    d1, d2 = dispute.dice
    result = analyzer.checker_play(
        dispute.board, d1, d2,
        cube_value=dispute.cube_value,
        cube_owner=dispute.cube_owner,
        away1=dispute.away1, away2=dispute.away2,
        is_crawford=dispute.is_crawford,
    )
    moves = list(result.moves)
    if not moves:
        return {"error": "no_legal_moves"}

    sage_board = dispute.sage_choice["board"]
    xg_board = dispute.xg_choice["board"]

    sage_eq, sage_eval = _rollout_quality_equity(
        analyzer, moves, sage_board, dispute.board,
        dispute.cube_value, dispute.cube_owner,
        dispute.away1, dispute.away2, dispute.is_crawford,
    )
    xg_eq, xg_eval = _rollout_quality_equity(
        analyzer, moves, xg_board, dispute.board,
        dispute.cube_value, dispute.cube_owner,
        dispute.away1, dispute.away2, dispute.is_crawford,
    )

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
    """One cube_action per (file, game, turn). A doubler dispute and a
    responder dispute on the same turn share the rollout."""
    key = (dispute.file, dispute.game_number, dispute.turn_number)
    if key in cube_cache:
        return cube_cache[key]
    result = analyzer.cube_action(
        dispute.board,
        cube_value=dispute.cube_value,
        cube_owner=dispute.cube_owner,
        away1=dispute.away1, away2=dispute.away2,
        is_crawford=dispute.is_crawford,
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
# JSONL I/O
# ---------------------------------------------------------------------------


def _dispute_key(d: Dispute) -> str:
    return f"{d.file}#{d.game_number}#{d.turn_number}#{d.decision_type}"


def _dispute_summary_row(d: Dispute) -> tuple[str, str]:
    """Render Sage's and XG's choices as short human-readable strings.

    Cube disputes use the action labels ("double" / "no_double" / "take" /
    "pass"). Checker disputes use ``compute_move_notation`` to render the
    pre-board -> post-board difference as a notation string like ``13/10 8/5``.
    """
    if d.decision_type == "checker":
        from bgsage.text_export import compute_move_notation
        d1, d2 = d.dice
        sage_move = compute_move_notation(
            list(d.board), list(d.sage_choice["board"]), d1, d2,
        )
        xg_move = compute_move_notation(
            list(d.board), list(d.xg_choice["board"]), d1, d2,
        )
        return sage_move, xg_move
    return d.sage_choice.get("action", "?"), d.xg_choice.get("action", "?")


_DEFAULT_THRESHOLD_BREAKDOWN: tuple[float, ...] = (
    0.005, 0.01, 0.02, 0.03, 0.05, 0.10,
)


def _print_threshold_breakdown(
    disputes: list[Dispute],
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLD_BREAKDOWN,
) -> None:
    """Print how many disputes would be rolled out vs skipped at each
    threshold, so the user can pick a rollout-budget tradeoff. The skipped
    disputes are assumed to be cases where Sage was wrong (and they
    contribute ``+xg_measured_error`` to net Sage error without an actual
    rollout)."""
    if not disputes:
        return
    print()
    print("Threshold breakdown (skipped disputes assumed Sage-wrong):")
    print(f"  {'threshold':>10} {'roll out':>10} {'skip':>6}")
    n = len(disputes)
    for t in thresholds:
        n_roll = sum(1 for d in disputes if d.xg_measured_error > t)
        n_skip = n - n_roll
        print(f"  {t:>10.4f} {n_roll:>10} {n_skip:>6}")


def _print_top_disputes(disputes: list[Dispute], n: int) -> None:
    """Print the top-N disputes by XG-measured Sage error, sorted descending.

    Includes match context (away counts + Crawford) so a reader can see why
    a position matters. Decisions tied on error follow insertion order.
    """
    if not disputes or n <= 0:
        return
    ranked = sorted(disputes, key=lambda d: -d.xg_measured_error)[:n]
    print()
    print(f"Top {len(ranked)} disputes by XG-measured Sage error")
    print("=" * 60)
    header = (
        f"{'#':<3} {'file':<22} {'g.t':>6} {'type':<9} "
        f"{'decider':<7} {'sage choice':<22} {'xg choice':<22} "
        f"{'err':>7} {'away':>6} craw"
    )
    print(header)
    print("-" * len(header))
    for i, d in enumerate(ranked, 1):
        sage_txt, xg_txt = _dispute_summary_row(d)
        gt = f"{d.game_number}.{d.turn_number}"
        away = f"{d.away1}-{d.away2}" if d.away1 or d.away2 else "money"
        craw = "Y" if d.is_crawford else ""
        print(
            f"{i:<3} {d.file:<22} {gt:>6} {d.decision_type:<9} "
            f"{d.deciding_player:<7} {sage_txt:<22} {xg_txt:<22} "
            f"{d.xg_measured_error:>7.4f} {away:>6} {craw}"
        )


def _record_key(rec: dict) -> str:
    return (
        f"{rec.get('file')}#{rec.get('game_number', 0)}#"
        f"{rec.get('turn_number')}#{rec.get('decision_type')}"
    )


def _load_done_rollouts(path: Path, threshold: float = 0.0) -> dict[str, dict]:
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
                    continue
            done[_record_key(rec)] = rec
    return done


def _write_rollout_record(
    path: Path, dispute: Dispute, rollout_result: dict
) -> dict:
    rec = {
        "file": dispute.file,
        "game_number": dispute.game_number,
        "turn_number": dispute.turn_number,
        "turn_player": dispute.turn_player,
        "deciding_player": dispute.deciding_player,
        "decision_type": dispute.decision_type,
        "board": list(dispute.board),
        "dice": dispute.dice,
        "cube_value": dispute.cube_value,
        "cube_owner": dispute.cube_owner,
        "away1": dispute.away1,
        "away2": dispute.away2,
        "is_crawford": dispute.is_crawford,
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
    # Line-buffer stdout so per-file PR rows and dispute progress show up in
    # real time when the script's output is piped to a log file. Default
    # full-buffering on redirected stdout otherwise hides progress until
    # process exit.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=_DEFAULT_DIR,
        help=f"Folder to scan for .xg files (default: {_DEFAULT_DIR})",
    )
    parser.add_argument(
        "--pattern", default="*.xg",
        help="Glob pattern within folder (default: *.xg)",
    )
    parser.add_argument(
        "--skip-rollouts", action="store_true",
        help="Stop after the PR aggregation; do no rollouts.",
    )
    parser.add_argument(
        "--rollout-file", type=Path, default=None,
        help=(
            "Output JSONL for rollout results "
            f"(default: <folder>/{_DEFAULT_ROLLOUT_FILENAME})"
        ),
    )
    parser.add_argument(
        "--rollout-seed", type=int, default=42,
        help="Rollout RNG seed (default: 42).",
    )
    parser.add_argument(
        "--rollout-threads", type=int, default=0,
        help="Threads per rollout (default: 0 = auto-detect).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=_ROLLOUT_N_TRIALS,
        help=(
            f"Trials per rollout (default: {_ROLLOUT_N_TRIALS}; keep a multiple "
            "of 36 for VR stratification, e.g. 5184 = 4x1296)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only roll out the first N pending disputes.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.005,
        help=(
            "Skip the rollout when XG's measured Sage error is <= this value "
            "(default: 0.005). Skipped disputes contribute +xg_measured_error "
            "to net Sage error without an actual rollout."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report rollout counts without running them or writing JSONL.",
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
        "--re-eval-cache-file", type=Path, default=None,
        help=(
            "Re-analysis cache JSONL "
            "(default: <folder>/sage_<level>_cache.jsonl)."
        ),
    )
    parser.add_argument(
        "--re-eval-threads", type=int, default=0,
        help="Threads for the re-analyzer (default: 0 = auto-detect).",
    )
    parser.add_argument(
        "--top-disputes", type=int, default=0,
        help=(
            "After PR aggregation, print a table of the top N disputes by "
            "XG-measured Sage error (highest first). Useful with "
            "--skip-rollouts for a quick look at the biggest disagreements."
        ),
    )
    args = parser.parse_args()

    global _RE_EVAL_LEVEL
    _RE_EVAL_LEVEL = args.re_eval_level

    xg_files = sorted(args.folder.glob(args.pattern), key=_seed_sort_key)
    if not xg_files:
        print(f"No .xg files found in {args.folder} matching {args.pattern!r}")
        return

    print(f"Scanning {len(xg_files)} .xg match files in {args.folder}\n")
    print(
        f"{'file':<22} {'game':>4} {'pr':>7} "
        f"{'tot err':>9} {'tot dec':>7} "
        f"{'sc1':>4} {'sc2':>4} {'craw':>5}"
    )
    print("-" * 70)

    re_eval_cache_path = (
        args.re_eval_cache_file
        or (args.folder / _re_eval_cache_filename(_RE_EVAL_LEVEL))
    )
    re_eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
    re_eval_cache = _load_re_eval_cache(re_eval_cache_path)
    print(f"\nRe-analysis level: {_RE_EVAL_LEVEL}")
    print(f"Re-analysis cache: {re_eval_cache_path}")
    print(f"  pre-loaded entries: {len(re_eval_cache)}\n")

    re_eval_misses = {"checker": 0, "cube": 0}
    def _re_eval_progress(file, game, turn_number, kind):
        key = _re_eval_cache_key(file, game, turn_number, kind)
        if key not in re_eval_cache:
            re_eval_misses[kind] += 1

    per_match_pr: list[float] = []   # one entry per .xg file
    per_game_pr: list[float] = []    # flat across all games
    sum_err_all = 0.0
    sum_dec_all = 0
    all_disputes: list[Dispute] = []

    for xg_path in xg_files:
        match_data = parse_xg_match(xg_path.read_bytes())
        match_length = int(match_data.get("match_length", 0))

        match_sum_err = 0.0
        match_sum_dec = 0

        for game in match_data["games"]:
            turns = game["turns"]
            s = compute_game_pr_stats(turns)  # populates flags & errors
            per_game_pr.append(s["pr"])
            match_sum_err += s["total_err"]
            match_sum_dec += s["total_dec"]
            pr_text = f"{s['pr']:.2f}" if not math.isnan(s["pr"]) else "  nan"
            craw_text = "Y" if game["crawford_apply"] else "-"
            print(
                f"{xg_path.name:<22} {game['game_number']:>4} {pr_text:>7} "
                f"{s['total_err']:>9.4f} {s['total_dec']:>7} "
                f"{game['score1_start']:>4} {game['score2_start']:>4} "
                f"{craw_text:>5}"
            )
            all_disputes.extend(_find_disputes_in_game(
                xg_path.name, game["game_number"], turns,
                game["score1_start"], game["score2_start"],
                match_length, game["crawford_apply"],
                re_eval_cache, re_eval_cache_path, args.re_eval_threads,
                progress_cb=_re_eval_progress,
            ))

        sum_err_all += match_sum_err
        sum_dec_all += match_sum_dec
        match_pr = (
            (match_sum_err / match_sum_dec * 500.0)
            if match_sum_dec > 0 else float("nan")
        )
        per_match_pr.append(match_pr)
        print(
            f"  -> match aggregate: err={match_sum_err:.4f} "
            f"dec={match_sum_dec} pr={match_pr:.3f}"
        )

    print(
        f"\n{_RE_EVAL_LEVEL} re-analysis: {re_eval_misses['checker']} checker "
        f"+ {re_eval_misses['cube']} cube positions computed this run "
        f"(cache size now {len(re_eval_cache)})."
    )

    print()
    print("=" * 60)
    print("Aggregate stats")
    print("=" * 60)

    n_matches = len(per_match_pr)
    n_games = len(per_game_pr)
    valid_match_pr = [p for p in per_match_pr if not math.isnan(p)]
    valid_game_pr = [p for p in per_game_pr if not math.isnan(p)]

    if valid_match_pr:
        mean_match = statistics.mean(valid_match_pr)
        std_match = statistics.stdev(valid_match_pr) if len(valid_match_pr) > 1 else 0.0
        sem_match = std_match / math.sqrt(len(valid_match_pr))
    else:
        mean_match = std_match = sem_match = float("nan")

    if valid_game_pr:
        mean_game = statistics.mean(valid_game_pr)
        std_game = statistics.stdev(valid_game_pr) if len(valid_game_pr) > 1 else 0.0
    else:
        mean_game = std_game = float("nan")

    agg_pr = (sum_err_all / sum_dec_all * 500.0) if sum_dec_all > 0 else float("nan")

    print(f"Matches:                  {n_matches}")
    print(f"Games:                    {n_games}")
    print(f"Per-match PR mean:        {mean_match:.3f}")
    print(f"Per-match PR std dev:     {std_match:.3f}")
    print(f"Per-match PR SEM:         {sem_match:.3f}")
    print(f"Per-game  PR mean:        {mean_game:.3f}")
    print(f"Per-game  PR std dev:     {std_game:.3f}")
    print(f"Total errors summed:      {sum_err_all:.4f}")
    print(f"Total decisions:          {sum_dec_all}")
    print(f"Aggregate PR (across all decisions): {agg_pr:.3f}")

    by_type = Counter(d.decision_type for d in all_disputes)
    print()
    print(f"Disputed decisions:   {len(all_disputes)}")
    for k in ("doubler", "responder", "checker"):
        if by_type.get(k):
            print(f"  {k:>10}: {by_type[k]}")

    _print_threshold_breakdown(all_disputes)

    if args.top_disputes > 0:
        _print_top_disputes(all_disputes, args.top_disputes)

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
            f"{n_to_roll} will be rolled out, {n_to_skip} skipped "
            f"(assumed Sage wrong)."
        )
        if args.dry_run:
            print(f"\n--dry-run: would roll out {n_to_roll} disputes. Stopping.")
            return
        rollout_file.parent.mkdir(parents=True, exist_ok=True)
        analyzer = None
        if n_to_roll > 0:
            print(
                f"Building rollout analyzer "
                f"({args.n_trials} trials, full play-out, 3-ply throughout)..."
            )
            analyzer = _build_rollout_analyzer(
                seed=args.rollout_seed, n_threads=args.rollout_threads,
                n_trials=args.n_trials,
            )
        cube_cache: dict = {}
        for i, dispute in enumerate(todo, 1):
            tag = (
                f"[{i}/{len(todo)}] {dispute.file} g{dispute.game_number} "
                f"t{dispute.turn_number} {dispute.decision_type} "
                f"({dispute.deciding_player})"
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
                    f"{tag}: rolling out "
                    f"(xg_err={dispute.xg_measured_error:.4f}, "
                    f"away1={dispute.away1} away2={dispute.away2} "
                    f"craw={dispute.is_crawford})...",
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

    # ----------------------------------------------------------------
    # Summary across rolled-out disputes
    # ----------------------------------------------------------------
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
