# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Match-play "PR against a rollout" benchmark.

The match-play analog of :mod:`benchmark_money`. It builds a production-quality
reference data set of backgammon **match** decisions (checker plays and cube
actions) with rollout-grade analytics, and scores an arbitrary bot against it
with the same custom Performance Rating (PR) calculation.

Everything mirrors the money-game pipeline -- three adaptive-precision passes
(3-ply for clear decisions, 3T for closer ones, full rollout for the closest),
the same gap thresholds, the same blunder threshold, the same resumable on-disk
layout, and the same distributed export modes -- with one essential difference:
**match state is threaded through every decision.** Each decision carries the
``away1`` / ``away2`` / ``is_crawford`` context (in the mover's perspective), the
analyzer is asked with those match params (so cube decisions use the MET and
Jacoby/beaver are auto-disabled), and the decision key folds them in so the same
board at different match scores never collides.

The unit of simulation is a whole **match** to ``match_length`` points (not a
single game): games are played until one side reaches the target, with the
Crawford game suppressing all cube offers and post-Crawford resuming them. Each
match is written as a single XG-import ``.txt`` transcript (one file per match),
exactly like ``run_sage_vs_sage_match.py`` -- so the same transcripts can be fed
to XG's Batch Analyze and scored by :mod:`benchmark_pr_xg_match`.

Both the **match length** and the **number of matches** are parameters. Results
for an ``L``-point benchmark live under ``data/match_benchmark/{L}pt/`` so several
match lengths can coexist.

Two public entry points:

  * :func:`build_benchmark_data` - simulate Sage-3P vs Sage-3P matches and emit
    the reference data set (three adaptive-precision passes). Crash-safe and
    resumable.

  * :func:`benchmark_pr` - score a :class:`BenchmarkBot` against the data set and
    return total / checker / cube PR plus a per-game-plan breakdown.

Reuse: the match-agnostic machinery (scoring formulas, aggregation, the
refinement-pass driver, the analyzer factories, JSONL helpers, gap/tier logic)
is imported from :mod:`benchmark_money` so the two stay byte-for-byte consistent;
only the match-aware pieces (simulation+capture, decision key, re-eval calls,
the bot interface, job export, dataset assembly) are defined here.

Usage::

    # Build the first (3P) pass for 130 five-point matches (one .txt per match):
    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 \
        --stages pass1 --write-txt

    # Later passes (locally, in serial -- a fresh clone can reproduce these):
    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass2
    python scripts/benchmark_match.py build --match-length 5 --n-matches 130 --stages pass3

    # Score the production model at 3-ply against the built data set:
    python scripts/benchmark_match.py score --match-length 5 --level 3ply --n-threads 16
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# bgsage path setup - self-contained within the bgsage repo (mirrors
# benchmark_money.py). Importing benchmark_money also runs its identical path
# setup; doing it here too keeps this module importable on its own (e.g. when
# the parent-repo Parallelizor scripts add bgsage/scripts to sys.path).
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = bgsage repo root
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"   # = bgsage/python
_BUILD_DIR = _PROJECT_ROOT / "build"        # = bgsage/build

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

# benchmark_money supplies all the match-agnostic machinery. Its bgsage imports
# are deferred, so importing it here does NOT load the engine.
import benchmark_money as bm  # noqa: E402

# Re-export the shared knobs so callers/readers see them on this module too.
TRIVIAL_SPREAD = bm.TRIVIAL_SPREAD
THREE_T_GAP = bm.THREE_T_GAP
ROLLOUT_GAP = bm.ROLLOUT_GAP
BLUNDER_THRESHOLD = bm.BLUNDER_THRESHOLD
PR_MULTIPLIER = bm.PR_MULTIPLIER
ROLLOUT_N_TRIALS = bm.ROLLOUT_N_TRIALS
ROLLOUT_TARGET_SE = bm.ROLLOUT_TARGET_SE
ROLLOUT_MAX_BATCHES = bm.ROLLOUT_MAX_BATCHES
GAME_PLANS = bm.GAME_PLANS
TIER_3P = bm.TIER_3P
TIER_3T = bm.TIER_3T
TIER_ROLLOUT = bm.TIER_ROLLOUT

_log = bm._log
_fmt_dur = bm._fmt_dur

#: Safety net for the per-match game loop (real matches finish well under this).
_MAX_GAMES = 200


# ===========================================================================
# Output locations (parameterized by match length)
# ===========================================================================


class _Paths:
    """All on-disk locations for an ``L``-point match benchmark.

    Everything lives under ``bgsage/data/match_benchmark/{L}pt/`` so multiple
    match lengths coexist. Mirrors the money benchmark's directory layout.
    """

    def __init__(self, match_length: int):
        self.match_length = int(match_length)
        self.data_dir = _PROJECT_ROOT / "data" / "match_benchmark" / f"{self.match_length}pt"
        self.build = self.data_dir / "build"
        self.stage1 = self.build / "stage1"                 # per-match decisions (3P)
        self.stage2 = self.build / "stage2_3t.jsonl"        # 3T re-evals, keyed by decision hash
        self.stage3 = self.build / "stage3_rollout.jsonl"   # rollouts, keyed by decision hash
        self.xg = self.data_dir / "xg"                      # XG-import .txt transcripts (one per match)
        self.scores = self.data_dir / "scores"             # per-bot scoring caches
        self.dataset = self.data_dir / "benchmark.json"
        self.rollout_jobs = self.data_dir / "rollout_jobs.jsonl"
        self.threet_jobs = self.data_dir / "threet_jobs.jsonl"


# ===========================================================================
# Decision keys and small helpers (match-aware)
# ===========================================================================


def make_decision_key(
    kind: str, board: list[int], dice: Optional[tuple[int, int] | list[int]],
    cube_value: int, cube_owner: str, away1: int, away2: int, is_crawford: bool,
) -> str:
    """Stable hash identifying a match decision position.

    Identical to the money key but with the match context (``away1``, ``away2``,
    ``is_crawford``) folded in, so the same board/dice/cube at different match
    scores -- which have genuinely different reference analytics -- never share a
    key. Dice are sorted (move generation is die-order independent).
    """
    dice_part = None if dice is None else sorted(int(d) for d in dice)
    payload = json.dumps(
        [kind, [int(x) for x in board], dice_part, int(cube_value), str(cube_owner),
         int(away1), int(away2), bool(is_crawford)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _away_counts_for_mover(
    active: int, score1: int, score2: int, match_length: int,
) -> tuple[int, int]:
    """``(away_mover, away_opp)`` from the active player's perspective.

    ``away_mover`` is what the player on roll needs; ``away_opp`` what the
    opponent needs. Both clamped to >= 1 (match isn't decided when we ask).
    Matches ``run_sage_vs_sage_match._away_counts_for_mover``.
    """
    if active == 1:
        mover_score, opp_score = score1, score2
    else:
        mover_score, opp_score = score2, score1
    return (max(1, match_length - mover_score), max(1, match_length - opp_score))


# ===========================================================================
# Bot interface (match-aware)
# ===========================================================================


CheckerCandidate = bm.CheckerCandidate
CubeAssessment = bm.CubeAssessment


class BenchmarkBot(ABC):
    """Interface a bot must implement to be scored by :func:`benchmark_pr`.

    Same as the money-game interface but every method receives the match context
    (``away1`` / ``away2`` in the mover's perspective, plus ``is_crawford``).
    Implementations must be thread-safe if scored with ``n_threads > 1``.
    """

    @abstractmethod
    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> list[CheckerCandidate]:
        """Return candidate plays ranked best-first (the bot's choice is index 0)."""

    @abstractmethod
    def cube_action(
        self, board: list[int], cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> CubeAssessment:
        """Return the bot's cube assessment for the player on roll."""


class SageBot(BenchmarkBot):
    """The Open Sage production engine as a match-aware :class:`BenchmarkBot`.

    Args mirror :class:`benchmark_money.SageBot`. Jacoby/beaver are never passed
    (the analyzer auto-disables them for match play); the match state is passed
    per call instead.
    """

    def __init__(
        self,
        eval_level: str = "3ply",
        weights: Any = None,
        model: Optional[str] = None,
        parallel_threads: int = 0,
    ):
        from bgsage import BgBotAnalyzer
        from bgsage.weights import WeightConfig

        if weights is None and model is not None:
            weights = WeightConfig.from_model(model)
        self.eval_level = eval_level
        self._analyzer = BgBotAnalyzer(
            weights=weights,
            eval_level=eval_level,
            cubeful=True,
            parallel_threads=parallel_threads,
        )

    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> list[CheckerCandidate]:
        res = self._analyzer.checker_play(
            board, die1, die2, cube_value=cube_value, cube_owner=cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
        )
        return [CheckerCandidate(board=list(m.board), equity=m.equity) for m in res.moves]

    def cube_action(
        self, board: list[int], cube_value: int, cube_owner: str,
        away1: int, away2: int, is_crawford: bool,
    ) -> CubeAssessment:
        c = self._analyzer.cube_action(
            board, cube_value=cube_value, cube_owner=cube_owner,
            away1=away1, away2=away2, is_crawford=is_crawford,
        )
        return CubeAssessment(should_double=c.should_double, should_take=c.should_take)


# ===========================================================================
# Pass 1 - match self-play with analytics capture (+ optional XG export)
# ===========================================================================


def _is_crawford_game(score1: int, score2: int, match_length: int, crawford_done: bool) -> bool:
    if crawford_done:
        return False
    return score1 == match_length - 1 or score2 == match_length - 1


def _capture_one_game(
    analyzer, rng: random.Random, match_length: int,
    score1: int, score2: int, is_crawford: bool, seed: int, game_number: int,
) -> tuple[list[dict], list[dict], Optional[int], Optional[str], int]:
    """Play one game of an N-point match at 3P, capturing each decision's analytics.

    Returns ``(decisions, move_history, winner, win_type, cube_at_end)``.
    ``score1``/``score2`` are constant for the game's duration; ``is_crawford``
    is fixed (true only for the unique Crawford game). All decisions carry the
    mover-perspective match context (``away1``/``away2``/``is_crawford``).
    """
    from bgsage import (
        STARTING_BOARD, check_game_over, classify_game_plan, possible_moves,
    )
    from bgsage.text_export import compute_move_notation

    state = bm._SimState(board=list(STARTING_BOARD), cube_value=1, cube_owner="centered", active=1)
    decisions: list[dict] = []
    move_history: list[dict] = []

    winner: Optional[int] = None
    win_type: Optional[str] = None
    cube_at_end = 1

    for turn in range(bm._MAX_TURNS):
        away_mover, away_opp = _away_counts_for_mover(state.active, score1, score2, match_length)
        cube_action_str: Optional[str] = None

        # --- Cube decision (only with cube access and outside Crawford) ---
        if (not is_crawford) and state.cube_owner in ("centered", "player"):
            cube = analyzer.cube_action(
                state.board, cube_value=state.cube_value, cube_owner=state.cube_owner,
                away1=away_mover, away2=away_opp, is_crawford=is_crawford,
            )
            nd, dt, dp = cube.equity_nd, cube.equity_dt, cube.equity_dp
            # Doubler's decision counts unless the position is trivial (XG-style).
            has_double = not bm._is_trivial_cube(nd, dt, dp)
            # The receiver's take/pass exists only when a double is offered, and
            # counts unless degenerate (DT ~= DP). No beaver in match play.
            has_take = bool(cube.should_double) and abs(dt - dp) >= TRIVIAL_SPREAD
            if has_double or has_take:
                decisions.append({
                    "kind": "cube",
                    "has_double": has_double,
                    "has_take": has_take,
                    "board": list(state.board),
                    "dice": None,
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                    "away1": away_mover,
                    "away2": away_opp,
                    "is_crawford": is_crawford,
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "key": make_decision_key(
                        "cube", state.board, None, state.cube_value, state.cube_owner,
                        away_mover, away_opp, is_crawford),
                    "seed": seed,
                    "game_number": game_number,
                    "turn": turn,
                    "player": state.active,
                    **bm._cube_fields_from_result(cube),
                })

            if cube.should_double:
                if cube.should_take:
                    cube_action_str = "double/take"
                    state.cube_value *= 2
                    state.cube_owner = "opponent"
                else:
                    cube_action_str = "double/pass"
                    move_history.append({
                        "player": "user" if state.active == 1 else "bot",
                        "cube_action": cube_action_str, "dice": None, "move": None,
                    })
                    winner, win_type, cube_at_end = state.active, "single", state.cube_value
                    break

        # --- Dice roll + checker play ---
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        cands = possible_moves(state.board, d1, d2)
        if not cands:
            post = list(state.board)
        else:
            result = analyzer.checker_play(
                state.board, d1, d2, cube_value=state.cube_value, cube_owner=state.cube_owner,
                away1=away_mover, away2=away_opp, is_crawford=is_crawford,
            )
            post = list(result.moves[0].board)
            moves = bm._move_list_from_result(result)
            # Counts as a decision: >= 2 legal moves and a meaningful best/worst spread.
            if len(moves) >= 2 and (moves[0]["equity"] - moves[-1]["equity"]) >= TRIVIAL_SPREAD:
                decisions.append({
                    "kind": "checker",
                    "board": list(state.board),
                    "dice": [d1, d2],
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                    "away1": away_mover,
                    "away2": away_opp,
                    "is_crawford": is_crawford,
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "moves": moves,
                    "key": make_decision_key(
                        "checker", state.board, (d1, d2), state.cube_value, state.cube_owner,
                        away_mover, away_opp, is_crawford),
                    "seed": seed,
                    "game_number": game_number,
                    "turn": turn,
                    "player": state.active,
                })

        move_history.append({
            "player": "user" if state.active == 1 else "bot",
            "cube_action": cube_action_str,
            "dice": [d1, d2],
            "move": compute_move_notation(state.board, post, d1, d2),
        })

        state.board = post
        code = check_game_over(state.board)
        if code > 0:
            winner = state.active
            win_type = {1: "single", 2: "gammon", 3: "backgammon"}.get(code, "single")
            cube_at_end = state.cube_value
            break
        bm._flip_sim(state)
    else:
        raise RuntimeError(f"game seed={seed} g{game_number} exceeded {bm._MAX_TURNS} turns")

    return decisions, move_history, winner, win_type, cube_at_end


def _simulate_match_and_capture(seed: int, match_length: int, parallel_threads: int) -> tuple[list[dict], dict]:
    """Play one Sage-3P vs Sage-3P match; capture every decision's 3P analytics.

    Returns ``(decisions, match_record)`` where ``decisions`` is the flat list of
    benchmark decision dicts (all games) and ``match_record`` is the match-history
    dict ``bgsage.text_export.export_history_to_txt`` consumes (``mode="match"``,
    one game per entry in ``match_game_histories``).
    """
    analyzer = bm._get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)

    score1 = score2 = 0
    crawford_done = False
    all_decisions: list[dict] = []
    game_histories: list[dict] = []
    game_number = 0

    while score1 < match_length and score2 < match_length:
        game_number += 1
        if game_number > _MAX_GAMES:
            raise RuntimeError(f"match seed={seed} exceeded MAX_GAMES={_MAX_GAMES} at {match_length}pt")

        is_crawford = _is_crawford_game(score1, score2, match_length, crawford_done)
        start1, start2 = score1, score2

        decisions, move_history, winner, win_type, cube_at_end = _capture_one_game(
            analyzer, rng, match_length, start1, start2, is_crawford, seed, game_number)
        all_decisions.extend(decisions)

        # Apply scoring with the match-length cap (so transcript running totals
        # stay consistent with how XG / Galaxy parsers cap excess points).
        mult = {"single": 1, "gammon": 2, "backgammon": 3}.get(win_type or "single", 1)
        raw_points = (cube_at_end or 1) * mult
        if winner == 1:
            new_score = min(start1 + raw_points, match_length)
            points_awarded = new_score - start1
            score1 = new_score
        elif winner == 2:
            new_score = min(start2 + raw_points, match_length)
            points_awarded = new_score - start2
            score2 = new_score
        else:
            points_awarded = 0

        if winner is None:
            result_str, result_points = "", 0
        else:
            side = "player1" if winner == 1 else "player2"
            result_str, result_points = f"{side}-win-{win_type}", int(points_awarded)

        game_histories.append({
            "game_number": game_number,
            "player_score": start1,
            "opponent_score": start2,
            "result": result_str,
            "result_points": result_points,
            "move_history": move_history,
        })

        if is_crawford:
            crawford_done = True

    match_record = {
        "player1_name": "Sage",
        "player2_name": "Sage",
        "mode": "match",
        "match_length": match_length,
        "match_game_histories": game_histories,
    }
    return all_decisions, match_record


def _build_stage1_match(seed: int, match_length: int, write_txt: bool, parallel_threads: int) -> dict:
    """Worker: play one match, persist its decisions (and optional XG .txt), atomically."""
    paths = _Paths(match_length)
    decisions, match_record = _simulate_match_and_capture(seed, match_length, parallel_threads)

    out_path = paths.stage1 / f"match_seed_{seed}.json"
    tmp_path = paths.stage1 / f"match_seed_{seed}.json.tmp"
    tmp_path.write_text(
        json.dumps({"seed": seed, "match_length": match_length, "decisions": decisions},
                   separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)

    if write_txt:
        from bgsage.text_export import export_history_to_txt

        (paths.xg / f"match_seed_{seed}.txt").write_bytes(export_history_to_txt(match_record))

    n_games = len(match_record["match_game_histories"])
    n_checker = sum(1 for d in decisions if d["kind"] == "checker")
    n_cube = len(decisions) - n_checker
    return {"seed": seed, "n_games": n_games, "n_checker": n_checker, "n_cube": n_cube}


def _run_pass1(paths: _Paths, n_matches: int, match_length: int, seed: int,
               write_txt: bool, workers: int) -> None:
    """Simulate ``n_matches`` matches at 3P, resuming over already-completed seeds."""
    paths.stage1.mkdir(parents=True, exist_ok=True)
    if write_txt:
        paths.xg.mkdir(parents=True, exist_ok=True)

    seeds = [seed + i for i in range(n_matches)]
    todo = [s for s in seeds if not (paths.stage1 / f"match_seed_{s}.json").exists()]
    done = len(seeds) - len(todo)
    _log(f"Pass 1/3 ({match_length}pt 3P self-play): {len(seeds)} matches, {done} already done, "
         f"{len(todo)} to play (workers={workers}).")
    if not todo:
        return

    total = len(todo)
    start = time.perf_counter()
    completed = 0

    def _report(res):
        nonlocal completed
        completed += 1
        rate = completed / max(1e-9, time.perf_counter() - start)
        eta = (total - completed) / rate if rate > 0 else 0
        _log(f"  [{completed}/{total}] seed={res['seed']}: {res['n_games']} games, "
             f"{res['n_checker']} checker + {res['n_cube']} cube positions  "
             f"({rate:.2f} matches/s, ETA {_fmt_dur(eta)})")

    if workers == 1:
        for s in todo:
            _report(_build_stage1_match(s, match_length, write_txt, parallel_threads=0))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_stage1_match, s, match_length, write_txt, 1): s for s in todo}
            for fut in as_completed(futures):
                _report(fut.result())
    _log(f"Pass 1/3 ({match_length}pt 3P self-play): complete -- {total} matches in "
         f"{_fmt_dur(time.perf_counter() - start)}.")


def _load_stage1_decisions(paths: _Paths) -> list[dict]:
    """Load every captured decision from all per-match stage-1 files."""
    out: list[dict] = []
    for path in sorted(paths.stage1.glob("match_seed_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.extend(data["decisions"])
    return out


# ===========================================================================
# Passes 2 & 3 - adaptive-precision re-evaluation (3T, then rollout)
# ===========================================================================


def _reeval_decision(analyzer, decision: dict, progress_callback=None) -> dict:
    """Re-evaluate one decision with ``analyzer`` (match state threaded through)."""
    a1, a2, cr = decision["away1"], decision["away2"], decision["is_crawford"]
    if decision["kind"] == "checker":
        d1, d2 = decision["dice"]
        result = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
            away1=a1, away2=a2, is_crawford=cr,
            progress_callback=progress_callback,
        )
        return {"key": decision["key"], "kind": "checker", "moves": bm._move_list_from_result(result)}
    cube = analyzer.cube_action(
        decision["board"], cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
        away1=a1, away2=a2, is_crawford=cr,
        progress_callback=progress_callback,
    )
    return {"key": decision["key"], "kind": "cube", **bm._cube_fields_from_result(cube)}


def _reeval_decision_rollout(analyzer, decision: dict, progress_callback=None) -> dict:
    """Roll out one decision to ROLLOUT_TARGET_SE (match state threaded through).

    Same lockstep / target-SE logic as ``benchmark_money._reeval_decision_rollout``;
    the only difference is the match params passed to every analyzer call.
    """
    a1, a2, cr = decision["away1"], decision["away2"], decision["is_crawford"]

    if decision["kind"] == "cube":
        analyzer.set_seed(bm._ROLLOUT_BASE_SEED)
        cube = analyzer.cube_action(
            decision["board"], cube_value=decision["cube_value"],
            cube_owner=decision["cube_owner"], away1=a1, away2=a2, is_crawford=cr,
            progress_callback=progress_callback,
        )
        rec = {"key": decision["key"], "kind": "cube", **bm._cube_fields_from_result(cube)}
        rec["rollout_se"] = cube.equity_nd_se
        return rec

    # Checker: lockstep over batches, gating on the best move's aggregate SE.
    d1, d2 = decision["dice"]
    acc: dict = {}
    n_batches = 0
    best_se = float("inf")
    agg: list = []
    for b in range(ROLLOUT_MAX_BATCHES):
        analyzer.set_seed((bm._ROLLOUT_BASE_SEED + b * bm._SEED_STEP) & 0xFFFFFFFF)
        res = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
            away1=a1, away2=a2, is_crawford=cr,
            progress_callback=progress_callback,
        )
        for m in res.moves:
            key = tuple(m.board)
            a = acc.get(key)
            if a is None:
                a = {"eq": 0.0, "cl": 0.0, "p": [0.0] * 5, "se2": 0.0, "n": 0,
                     "lvl": m.eval_level}
                acc[key] = a
            a["eq"] += m.equity
            a["cl"] += m.cubeless_equity
            pl = m.probs.to_list()
            for k in range(5):
                a["p"][k] += pl[k]
            se = m.std_error or 0.0
            a["se2"] += se * se
            a["n"] += 1
        n_batches = b + 1
        agg = []
        for bkey, a in acc.items():
            n = a["n"]
            agg.append({
                "board": list(bkey),
                "equity": a["eq"] / n,
                "cubeless_equity": a["cl"] / n,
                "probs": [v / n for v in a["p"]],
                "eval_level": a["lvl"],
                "std_error": (a["se2"] ** 0.5) / n,
            })
        agg.sort(key=lambda x: -x["equity"])
        best_se = agg[0]["std_error"]
        if ROLLOUT_TARGET_SE <= 0 or best_se <= ROLLOUT_TARGET_SE:
            break
    return {"key": decision["key"], "kind": "checker", "moves": agg,
            "rollout_batches": n_batches, "rollout_se": best_se}


# ===========================================================================
# Job export (for the distributed Parallelizor passes)
# ===========================================================================


def _export_threet_jobs(paths: _Paths, unique: dict[str, dict],
                        min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions Pass 2 would re-evaluate at 3T, with all match inputs.

    The match analog of ``benchmark_money._export_threet_jobs``: each job carries
    the board/dice/cube state AND the match context (``away1``/``away2``/
    ``is_crawford``) so a distributed worker reproduces the exact 3T re-eval.
    Skips positions already in the stage-2 file (resumable).
    """
    done = bm._load_jsonl_by_key(paths.stage2)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or bm._gap_for(dec) >= THREE_T_GAP or not bm._seed_in_range(dec, min_seed, max_seed):
            continue
        jobs.append({
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
            "away1": dec["away1"], "away2": dec["away2"], "is_crawford": dec["is_crawford"],
            "game_plan": dec.get("game_plan"),
            "level": "truncated3",
        })
    paths.threet_jobs.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.threet_jobs.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(paths.threet_jobs)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} 3T jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {paths.threet_jobs} -- no 3T evals computed.")
    return paths.threet_jobs


def _export_rollout_jobs(paths: _Paths, unique: dict[str, dict],
                         min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions Pass 3 would roll out, with all match + rollout inputs.

    The match analog of ``benchmark_money._export_rollout_jobs``. Skips positions
    already in the stage-3 file (resumable).
    """
    done = bm._load_jsonl_by_key(paths.stage3)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or bm._gap_for(dec) >= ROLLOUT_GAP or not bm._seed_in_range(dec, min_seed, max_seed):
            continue
        jobs.append({
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
            "away1": dec["away1"], "away2": dec["away2"], "is_crawford": dec["is_crawford"],
            "jacoby": False, "beaver": False,
            "game_plan": dec.get("game_plan"),
            "rollout": {
                "n_trials": ROLLOUT_N_TRIALS,
                "truncation_depth": 0,
                "decision_ply": 3,
                "target_se": ROLLOUT_TARGET_SE,
                "max_batches": ROLLOUT_MAX_BATCHES,
                "gate": "equity_nd_se" if dec["kind"] == "cube" else "best_move_equity_se",
            },
        })
    paths.rollout_jobs.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.rollout_jobs.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(paths.rollout_jobs)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} rollout jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {paths.rollout_jobs} -- no rollouts computed.")
    return paths.rollout_jobs


# ===========================================================================
# Final assembly
# ===========================================================================


def _assemble_dataset(paths: _Paths, unique: dict[str, dict],
                      n_matches: int, match_length: int, seed: int) -> dict:
    """Write the final benchmark JSON from the most-refined analytics per decision."""
    decisions = sorted(
        unique.values(),
        key=lambda d: (d.get("seed", 0), d.get("game_number", 0), d.get("turn", 0), d["kind"]),
    )

    tier_counts = {TIER_3P: 0, TIER_3T: 0, TIER_ROLLOUT: 0}
    n_checker = n_cube_entries = n_double = n_take = 0
    for d in decisions:
        tier_counts[d["tier"]] += 1
        if d["kind"] == "checker":
            n_checker += 1
        else:
            n_cube_entries += 1
            n_double += int(d.get("has_double", False))
            n_take += int(d.get("has_take", False))
    n_cube_decisions = n_double + n_take
    n_decisions = n_checker + n_cube_decisions

    meta = {
        "mode": "match",
        "match_length": match_length,
        "n_matches": n_matches,
        "seed": seed,
        "n_decisions": n_decisions,           # total scoreable decisions
        "n_positions": len(decisions),        # dataset entries (checker + cube positions)
        "n_checker": n_checker,
        "n_cube_decisions": n_cube_decisions,
        "n_cube_double": n_double,
        "n_cube_take": n_take,
        "n_cube_entries": n_cube_entries,
        "tier_counts": tier_counts,           # per dataset entry (position)
        "three_t_gap": THREE_T_GAP,
        "rollout_gap": ROLLOUT_GAP,
        "rollout_n_trials": ROLLOUT_N_TRIALS,
        "trivial_spread": TRIVIAL_SPREAD,
        "blunder_threshold": BLUNDER_THRESHOLD,
        "pr_multiplier": PR_MULTIPLIER,
    }
    paths.dataset.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.dataset.with_suffix(paths.dataset.suffix + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "decisions": decisions}), encoding="utf-8")
    tmp.replace(paths.dataset)
    # gzip sibling (read transparently by bm._read_dataset on a fresh clone).
    Path(f"{paths.dataset}.gz").write_bytes(
        gzip.compress(paths.dataset.read_bytes(), compresslevel=9))
    _log(f"Wrote {len(decisions)} positions ({n_decisions} decisions) to {paths.dataset} (+ .gz)")
    _log(f"  checker={n_checker}  cube: {n_cube_decisions} decisions "
         f"(double={n_double}, take={n_take}) over {n_cube_entries} positions  tiers={tier_counts}")
    return meta


def build_benchmark_data(
    match_length: int,
    n_matches: int,
    seed: int = 1,
    n_threads: int = 0,
    write_txt: bool = True,
    workers: int = 6,
    rollout_mode: str = "compute",
    threet_mode: str = "compute",
    export_min_seed: int | None = None,
    export_max_seed: int | None = None,
    stages: tuple = ("pass1", "pass2", "pass3"),
) -> dict:
    """Build the match-game benchmark data set (three adaptive-precision passes).

    Pass 1: simulate ``n_matches`` Sage-3P vs Sage-3P ``match_length``-point
    matches, capturing 3-ply analytics for every real decision (optionally writing
    one XG-import ``.txt`` per match). Pass 2: re-evaluate decisions whose 3-ply
    best-vs-2nd-best gap < 0.05 at 3T. Pass 3: roll out (1,296 paths, VR, 3-ply
    checker + cube) decisions whose best-available gap < 0.02. Match state is
    threaded through every evaluation.

    Crash-safe and resumable; ``stages`` / ``rollout_mode`` / ``threet_mode``
    behave exactly as in :func:`benchmark_money.build_benchmark_data`. Returns the
    dataset metadata dict.
    """
    if match_length <= 0:
        raise ValueError(f"match_length must be positive (got {match_length})")
    if n_matches <= 0:
        raise ValueError(f"n_matches must be positive (got {n_matches})")

    paths = _Paths(match_length)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    stages = set(stages)
    _log(f"=== build_benchmark_data (match): match_length={match_length} n_matches={n_matches} "
         f"seed={seed} n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} threet_mode={threet_mode} stages={sorted(stages)} ===")

    # Pass 1 - self-play + 3P capture.
    if "pass1" in stages:
        _run_pass1(paths, n_matches, match_length, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions(paths)
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = bm._unique_by_key(all_decisions)
    _log(f"Captured {len(all_decisions)} decisions ({len(unique)} unique positions).")

    # Pass 2 - 3T refinement (compute or export).
    if "pass2" in stages:
        if threet_mode == "export":
            for key, refined in bm._load_jsonl_by_key(paths.stage2).items():
                if key in unique:
                    unique[key] = bm._apply_refined(unique[key], refined, TIER_3T)
            _export_threet_jobs(paths, unique, export_min_seed, export_max_seed)
        else:
            analyzer_3t = bm._make_3t_analyzer(n_threads)
            unique = bm._run_refinement_pass(
                unique, analyzer_3t, paths.stage2, THREE_T_GAP, TIER_3T,
                "Pass 2/3 (3T refine)", reeval_fn=_reeval_decision)
    else:
        done2 = bm._load_jsonl_by_key(paths.stage2)
        for key, refined in done2.items():
            if key in unique:
                unique[key] = bm._apply_refined(unique[key], refined, TIER_3T)
        pending2 = sum(1 for key, dec in unique.items()
                       if key not in done2 and bm._gap_for(dec) < THREE_T_GAP)
        _log(f"Pass 2 (3T) not run: {pending2} positions want 3T "
             f"(3P gap < {THREE_T_GAP}); {len(done2)} already done. Run with --stages pass2.")

    # Pass 3 - rollout (compute or export).
    if "pass3" in stages:
        if rollout_mode == "export":
            for key, refined in bm._load_jsonl_by_key(paths.stage3).items():
                if key in unique:
                    unique[key] = bm._apply_refined(unique[key], refined, TIER_ROLLOUT)
            _export_rollout_jobs(paths, unique, export_min_seed, export_max_seed)
        else:
            analyzer_ro = bm._make_rollout_analyzer(n_threads)
            unique = bm._run_refinement_pass(
                unique, analyzer_ro, paths.stage3, ROLLOUT_GAP, TIER_ROLLOUT,
                "Pass 3/3 (full rollout to SE)", reeval_fn=_reeval_decision_rollout)
    else:
        done3 = bm._load_jsonl_by_key(paths.stage3)
        for key, refined in done3.items():
            if key in unique:
                unique[key] = bm._apply_refined(unique[key], refined, TIER_ROLLOUT)
        pending3 = sum(1 for key, dec in unique.items()
                       if key not in done3 and bm._gap_for(dec) < ROLLOUT_GAP)
        _log(f"Pass 3 (rollout) not run: {pending3} positions want rollout "
             f"(current-tier gap < {ROLLOUT_GAP}); {len(done3)} already done. "
             f"Run with --stages pass3.")

    return _assemble_dataset(paths, unique, n_matches, match_length, seed)


# ===========================================================================
# Scoring - benchmark_pr (match-aware bot calls)
# ===========================================================================


def _score_checker(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    """Scored checker decision (bot's chosen play vs the reference best)."""
    cands = bot.checker_play(
        entry["board"], entry["dice"][0], entry["dice"][1],
        entry["cube_value"], entry["cube_owner"],
        entry["away1"], entry["away2"], entry["is_crawford"],
    )
    if not cands:
        return None
    chosen = tuple(cands[0].board)
    ref_by_board = {tuple(m["board"]): m["equity"] for m in entry["moves"]}
    chosen_eq = ref_by_board.get(chosen)
    if chosen_eq is None:
        return None  # bot's move not among the reference's legal moves
    best_eq = entry["moves"][0]["equity"]
    return bm._scored("checker", "checker", entry.get("game_plan"), max(0.0, best_eq - chosen_eq))


def _score_cube(bot: BenchmarkBot, entry: dict) -> list[dict]:
    """Scored cube sub-decisions (doubler and/or receiver), standard PR formulas."""
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    plan = entry.get("game_plan")
    a = bot.cube_action(
        entry["board"], entry["cube_value"], entry["cube_owner"],
        entry["away1"], entry["away2"], entry["is_crawford"],
    )
    out: list[dict] = []
    if entry.get("has_double"):
        optimal = max(nd, min(dt, dp))
        actual = min(dt, dp) if a.should_double else nd
        out.append(bm._scored("cube", "double", plan, max(0.0, optimal - actual)))
    if entry.get("has_take"):
        optimal = min(dt, dp)
        actual = dt if a.should_take else dp
        out.append(bm._scored("cube", "take", plan, max(0.0, actual - optimal)))
    return out


def _score_entry(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    """Score one position; return ``{key, scored:[...]}`` or None on a checker mismatch."""
    if entry["kind"] == "checker":
        s = _score_checker(bot, entry)
        if s is None:
            return None
        scored = [s]
    else:
        scored = _score_cube(bot, entry)
    return {"key": entry["key"], "scored": scored}


def benchmark_pr(
    bot: BenchmarkBot,
    match_length: int,
    dataset_path: Path | str | None = None,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the match benchmark data set; return the PR breakdown.

    Mirrors :func:`benchmark_money.benchmark_pr` exactly (resumable per-decision
    cache, skip-coarse-reference logic, aggregation) but asks the bot with match
    state and reads/writes under ``data/match_benchmark/{L}pt/``.
    """
    paths = _Paths(match_length)
    dataset_path = Path(dataset_path) if dataset_path is not None else paths.dataset
    data = bm._read_dataset(dataset_path)
    all_entries = data["decisions"]
    if max_seed is not None:
        n_all = len(all_entries)
        all_entries = [e for e in all_entries if e.get("seed", 0) <= max_seed]
        _log(f"Limiting to the first {max_seed} matches (seed <= {max_seed}): "
             f"{len(all_entries)} of {n_all} decisions.")

    # Skip decisions whose reference is below the precision their closeness requires.
    entries = []
    skipped_missing = {"rollout": 0, "3t": 0}
    for e in all_entries:
        miss = bm._missing_tier(e)
        if miss is None:
            entries.append(e)
        else:
            skipped_missing[miss] += 1
    n_skipped = skipped_missing["rollout"] + skipped_missing["3t"]

    if cache_path is None:
        paths.scores.mkdir(parents=True, exist_ok=True)
        cache_path = paths.scores / f"{label}.jsonl"
    else:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = bm._load_jsonl_by_key(cache_path)
    todo = [e for e in entries if e["key"] not in cached]
    if progress:
        _log(f"Scoring {label}: {len(entries)} positions, {len(cached)} cached, "
             f"{len(todo)} to score (n_threads={n_threads}).")

    lock = threading.Lock()
    mismatches = 0
    f = cache_path.open("a", encoding="utf-8")
    try:
        done = 0
        total = len(todo)
        start = time.perf_counter()
        last_print = [start]

        def _work(entry: dict):
            nonlocal mismatches, done
            result = _score_entry(bot, entry)
            with lock:
                if result is None:
                    mismatches += 1
                else:
                    cached[result["key"]] = result
                    f.write(json.dumps(result, separators=(",", ":")) + "\n")
                    f.flush()
                done += 1
                now = time.perf_counter()
                if progress and (done == total or now - last_print[0] >= bm._PROGRESS_EVERY_S):
                    last_print[0] = now
                    rate = done / max(1e-9, now - start)
                    eta = (total - done) / rate if rate > 0 else 0
                    _log(f"  [{done}/{total}] scored ({rate:.1f}/s, ETA {_fmt_dur(eta)})")

        if total:
            if n_threads <= 1:
                for e in todo:
                    _work(e)
            else:
                with ThreadPoolExecutor(max_workers=n_threads) as ex:
                    list(ex.map(_work, todo))
    finally:
        f.close()

    if mismatches:
        _log(f"WARNING: {mismatches} decisions skipped - the bot's chosen checker play "
             f"was not a legal reference move (not counted toward PR).")
    if n_skipped:
        _log(f"NOTE: skipped {n_skipped} positions with a missing higher-precision "
             f"reference ({skipped_missing['rollout']} missing rollout, "
             f"{skipped_missing['3t']} missing 3T) -- not scored.")

    scoreable_keys = {e["key"] for e in entries}
    result = bm._aggregate(r for k, r in cached.items() if k in scoreable_keys)
    result["mismatches"] = mismatches
    result["skipped"] = n_skipped
    result["skipped_missing_rollout"] = skipped_missing["rollout"]
    result["skipped_missing_3t"] = skipped_missing["3t"]
    if progress:
        _log(f"{label}: total PR={result['total_pr']:.2f} "
             f"(checker={result['checker_pr']:.2f}, cube={result['cube_pr']:.2f}) "
             f"over {result['n_decisions']} decisions")
    return result


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the match benchmark data set")
    p_build.add_argument("--match-length", type=int, required=True, help="Match length in points")
    p_build.add_argument("--n-matches", type=int, default=130, help="Matches to simulate")
    p_build.add_argument("--seed", type=int, default=1, help="First match seed")
    p_build.add_argument("--n-threads", type=int, default=0,
                         help="Threads for the 3T and rollout passes (0=auto)")
    p_build.add_argument("--workers", type=int, default=6,
                         help="Parallel processes for the pass-1 self-play")
    p_build.add_argument("--write-txt", action=argparse.BooleanOptionalAction, default=True,
                         help="Write an XG-import .txt per match to xg/ (default: on)")
    p_build.add_argument("--rollout-mode", choices=["compute", "export"], default="compute",
                         help="compute: run Pass-3 rollouts in-process; "
                              "export: write rollout_jobs.jsonl instead (no rollouts)")
    p_build.add_argument("--threet-mode", choices=["compute", "export"], default="compute",
                         help="compute: run the Pass-2 3T re-evals in-process; "
                              "export: write threet_jobs.jsonl instead (no 3T evals)")
    p_build.add_argument("--stages", default="pass1,pass2,pass3",
                         help="comma-separated passes to run this invocation "
                              "(pass1=self-play+3P, pass2=3T, pass3=rollout)")
    p_build.add_argument("--min-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed >= this")
    p_build.add_argument("--max-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed <= this")

    p_score = sub.add_parser("score", help="Score the Sage engine against the data set")
    p_score.add_argument("--match-length", type=int, required=True, help="Match length in points")
    p_score.add_argument("--level", default="3ply",
                         help="SageBot eval level (1ply..4ply, truncated1/2/3, rollout)")
    p_score.add_argument("--model", default=None, help="Model name (default: production)")
    p_score.add_argument("--n-threads", type=int, default=1,
                         help="Threads scoring decisions concurrently")
    p_score.add_argument("--label", default=None, help="Resume-cache label")
    p_score.add_argument("--max-seed", type=int, default=None,
                         help="Only score the first N matches (decisions with seed <= MAX_SEED)")

    args = parser.parse_args(argv)

    if args.command == "build":
        build_benchmark_data(
            match_length=args.match_length, n_matches=args.n_matches, seed=args.seed,
            n_threads=args.n_threads, write_txt=args.write_txt, workers=args.workers,
            rollout_mode=args.rollout_mode, threet_mode=args.threet_mode,
            export_min_seed=args.min_seed, export_max_seed=args.max_seed,
            stages=tuple(s.strip() for s in args.stages.split(",") if s.strip()),
        )
    elif args.command == "score":
        bot = SageBot(eval_level=args.level, model=args.model, parallel_threads=1)
        label = args.label or f"sage_{args.level}" + (f"_{args.model}" if args.model else "")
        result = benchmark_pr(bot, match_length=args.match_length, n_threads=args.n_threads,
                              label=label, max_seed=args.max_seed)
        bm._print_report(result)


if __name__ == "__main__":
    _main()
