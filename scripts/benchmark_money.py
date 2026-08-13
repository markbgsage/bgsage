# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Money-game "PR against a rollout" benchmark.

This module builds a production-quality reference data set of backgammon money-game
decisions (checker plays and cube actions) with rollout-grade analytics, and scores
an arbitrary bot against it with a custom Performance Rating (PR) calculation.

The idea: a PR is the average equity error per decision x 500 (the XG convention).
Normally you score a bot against a far stronger reference. Here the reference is a
cached data set of decisions whose analytics are computed at adaptive precision:

  * **3-ply** for clear decisions (the next-best move is far away, so the precise
    error of an alternative barely matters and most bots get these right anyway),
  * **3T** (XG Roller++ truncated rollout) for closer decisions,
  * **full rollout** (1,296 paths, variance reduction, 3-ply checker + cube
    decisions) for the closest decisions, where errors actually show up.

Once built, the data set is just data: scoring needs nothing Sage-specific. A bot is
any subclass of :class:`BenchmarkBot`; :class:`SageBot` wraps the Open Sage engine,
and a developer can drop their own engine behind the same two-method interface.

Two public entry points:

  * :func:`build_benchmark_data` - simulate Sage-3P vs Sage-3P games and emit the
    reference data set (three adaptive-precision passes). Crash-safe and resumable:
    every stage persists results as they arrive and a re-run skips ahead to the
    first uncomputed work.

  * :func:`benchmark_pr` - score a :class:`BenchmarkBot` against the data set and
    return total / checker / cube PR plus a per-game-plan breakdown. Also crash-safe
    and resumable: per-decision results are cached to disk as they are computed.

Relationship to the older ``generate_benchmark_pr.py`` / ``score_benchmark_pr.py``
(``data/benchmark_pr/``): that earlier system is **checker-only**, rolls out every
candidate at a **single** 2-ply precision, is **Sage-specific**, and has no pluggable
bot. This module is a separate, more general system (checker **and** cube, adaptive
precision, pluggable bot, resumable) and writes to its own ``data/money_benchmark/``
directory; it does not read or modify the older one.

Usage::

    # Build (does NOT run by import; call explicitly or via the CLI):
    python scripts/benchmark_money.py build --n-games 200 --n-threads 16 --write-txt

    # Score the production model at 3-ply against the built data set:
    python scripts/benchmark_money.py score --level 3ply --n-threads 16

    # In code:
    from benchmark_money import build_benchmark_data, benchmark_pr, SageBot
    build_benchmark_data(n_games=200, n_threads=16)
    result = benchmark_pr(SageBot(eval_level="3ply", parallel_threads=1), n_threads=16)
    print(result["total_pr"], result["checker_pr"], result["cube_pr"])
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# bgsage path setup - self-contained within the bgsage repo. Never reaches into
# a parent project: weights, build artifacts and outputs all live under bgsage/.
# (Mirrors scripts/run_sage_vs_sage_games.py.)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent          # = bgsage repo root
_BGSAGE_PYTHON = _PROJECT_ROOT / "python"   # = bgsage/python
_BUILD_DIR = _PROJECT_ROOT / "build"        # = bgsage/build

for _p in (_BGSAGE_PYTHON, _BUILD_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))


# ---------------------------------------------------------------------------
# Output locations (all under bgsage/data/money_benchmark/)
# ---------------------------------------------------------------------------

_DATA_DIR = _PROJECT_ROOT / "data" / "money_benchmark"
_BUILD_SUBDIR = _DATA_DIR / "build"
_STAGE1_DIR = _BUILD_SUBDIR / "stage1"            # per-seed game decisions (3P)
_STAGE2_FILE = _BUILD_SUBDIR / "stage2_3t.jsonl"  # 3T re-evals, keyed by decision hash
_STAGE3_FILE = _BUILD_SUBDIR / "stage3_rollout.jsonl"  # rollouts, keyed by decision hash
_XG_DIR = _DATA_DIR / "xg"                        # XG-import .txt transcripts
_SCORES_DIR = _DATA_DIR / "scores"                # per-bot scoring caches
DEFAULT_DATASET = _DATA_DIR / "benchmark.json"


def _read_dataset(dataset_path) -> dict:
    """Load the assembled benchmark, transparently reading a gzip ``.gz`` sibling.

    The full ``benchmark.json`` is ~100 MB (over GitHub's 100 MB file-size limit), so
    the repo ships ``benchmark.json.gz`` instead. A local uncompressed ``benchmark.json``
    (e.g. freshly rebuilt) is preferred when present; otherwise the ``.gz`` is read -- so
    a fresh clone can score a bot against the saved dataset with no rebuild.
    """
    p = Path(dataset_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    gz = Path(f"{p}.gz")
    if gz.exists():
        return json.loads(gzip.decompress(gz.read_bytes()).decode("utf-8"))
    raise FileNotFoundError(f"Benchmark dataset not found: {p} (or {gz})")


# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

#: A checker play counts as a decision only when there are >= 2 legal moves AND the
#: best/worst reference-equity spread is at least this. A cube position counts only
#: when it is non-trivial (see ``_is_trivial_cube``). Matches the app's PR filters.
TRIVIAL_SPREAD = 0.001

#: Pass 2: re-evaluate a decision at 3T when its 3-ply best-vs-2nd-best gap < this.
THREE_T_GAP = 0.05
#: Pass 3: roll a decision out when its best-available best-vs-2nd-best gap < this.
ROLLOUT_GAP = 0.02

#: An equity error above this counts as a blunder (XG convention; matches the app).
BLUNDER_THRESHOLD = 0.08

#: PR = mean equity error per decision x this.
PR_MULTIPLIER = 500

#: Full-rollout reference config (matches scripts/rollout_positions_to_cache.py):
#: 1,296 trials, play to completion, 3-ply checker + cube decisions throughout, VR on.
ROLLOUT_N_TRIALS = 1296

#: Target standard error for rollouts. Each rollout runs repeated 1,296-path
#: batches (different seeds, shared caches) until the gating equity's SE drops
#: below this -- the ND-equity SE for cube decisions (done in C++), or the best
#: move's equity SE for checker decisions (Python lockstep over set_seed). The
#: 95% band is ~0.005 (1.96 x 0.00255).
ROLLOUT_TARGET_SE = 0.00255
#: Safety cap on the number of 1,296-path batches per rolled-out decision.
ROLLOUT_MAX_BATCHES = 16
#: Base RNG seed for rollout batches; batch b uses (base + b*step) mod 2^32.
_ROLLOUT_BASE_SEED = 42
_SEED_STEP = 0x9E3779B9

#: Canonical game-plan labels (must match bgsage.classify_game_plan output).
GAME_PLANS = ["purerace", "racing", "attacking", "priming", "anchoring"]

#: Tiers, weakest to strongest. The final entry uses the strongest tier reached.
TIER_3P = "3P"
TIER_3T = "3T"
TIER_ROLLOUT = "rollout"
_TIER_RANK = {TIER_3P: 0, TIER_3T: 1, TIER_ROLLOUT: 2}

#: Self-play guard rail (4 events/turn * generous turn cap).
_MAX_TURNS = 600

#: Progress heartbeat interval (seconds) for the long passes, and the per-item
#: duration above which an item is always logged on its own line.
_PROGRESS_EVERY_S = 15.0
_SLOW_ITEM_S = 2.0


# ===========================================================================
# Bot interface
# ===========================================================================


@dataclass
class CheckerCandidate:
    """One candidate checker play returned by a bot.

    ``board`` is the 26-element post-move board in the **mover's perspective**
    (positive = mover) - the same representation ``bgsage.possible_moves`` returns,
    so it matches the reference move list exactly. ``equity`` is the bot's own
    cubeful equity for ranking; the benchmark only uses ``equity`` to pick the bot's
    move (the candidate ranked first), and scores that move against the *reference*
    equities, never the bot's.
    """

    board: list[int]
    equity: float


@dataclass
class CubeAssessment:
    """A bot's cube assessment of a position (player-on-roll perspective).

    Both fields are scored, like a regular PR: ``should_double`` is the doubler's
    decision, and ``should_take`` is the receiver's take/pass decision (the receiver
    decision is scored on positions where a double was offered). The benchmark asks
    the same bot for both - it scores both sides against the cached reference.
    """

    should_double: bool
    should_take: bool


class BenchmarkBot(ABC):
    """Interface a bot must implement to be scored by :func:`benchmark_pr`.

    Both methods receive boards/cube state in the player-on-roll (mover's)
    perspective and money-game cube semantics. Implementations must be **thread-safe**
    for concurrent calls if scored with ``n_threads > 1`` (``SageBot`` is); otherwise
    score with ``n_threads=1``.
    """

    @abstractmethod
    def checker_play(
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str
    ) -> list[CheckerCandidate]:
        """Return candidate plays ranked best-first (the bot's choice is index 0).

        Each candidate's ``board`` is a legal 26-element post-move board in the
        mover's perspective. The list need not be exhaustive, but the bot's chosen
        move (index 0) must be a legal post-move board so it can be matched against
        the reference.
        """

    @abstractmethod
    def cube_action(self, board: list[int], cube_value: int, cube_owner: str) -> CubeAssessment:
        """Return the bot's cube assessment for the player on roll."""


class SageBot(BenchmarkBot):
    """The Open Sage production engine as a :class:`BenchmarkBot`.

    Args:
        eval_level: any ``BgBotAnalyzer`` level (``"1ply"``..``"4ply"``,
            ``"truncated1/2/3"``, ``"rollout"``).
        weights: optional ``WeightConfig`` to score a non-production model.
        model: optional model name (resolved via ``WeightConfig.from_model``); ignored
            if ``weights`` is given.
        parallel_threads: analyzer's internal thread count. When scoring with
            ``benchmark_pr(n_threads=N)``, prefer ``parallel_threads=1`` and let the
            benchmark parallelize across positions instead.
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
        self, board: list[int], die1: int, die2: int, cube_value: int, cube_owner: str
    ) -> list[CheckerCandidate]:
        res = self._analyzer.checker_play(
            board, die1, die2, cube_value=cube_value, cube_owner=cube_owner,
            jacoby=True, beaver=True,
        )
        return [CheckerCandidate(board=list(m.board), equity=m.equity) for m in res.moves]

    def cube_action(self, board: list[int], cube_value: int, cube_owner: str) -> CubeAssessment:
        c = self._analyzer.cube_action(
            board, cube_value=cube_value, cube_owner=cube_owner, jacoby=True, beaver=True,
        )
        return CubeAssessment(should_double=c.should_double, should_take=c.should_take)


# ===========================================================================
# Decision keys and small helpers
# ===========================================================================


def make_decision_key(
    kind: str, board: list[int], dice: Optional[tuple[int, int] | list[int]],
    cube_value: int, cube_owner: str,
) -> str:
    """Stable hash identifying a decision position.

    Dice are sorted (move generation is die-order independent), so the two die
    orders of the same roll collapse to one key - matching the analytics-cache
    convention. ``kind`` is folded in so a checker and a cube decision at the same
    board never collide.
    """
    dice_part = None if dice is None else sorted(int(d) for d in dice)
    payload = json.dumps(
        [kind, [int(x) for x in board], dice_part, int(cube_value), str(cube_owner)],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _is_trivial_cube(nd: float, dt: float, dp: float) -> bool:
    """Whether a cube position is too obvious to count as a decision.

    Mirrors backend ``_is_trivial_cube`` (money game): no meaningful difference,
    obvious no-double, obvious too-good, or hopeless. Equities are doubler's
    perspective; ``dp`` is +1.0 in money play.
    """
    if abs(nd - dt) < TRIVIAL_SPREAD:
        return True
    if nd - dt > 0.200:
        return True
    if nd - dp > 0.200:
        return True
    if nd < -0.900 and dt < -0.900:
        return True
    return False


def _cube_gap(nd: float, dt: float, dp: float) -> float:
    """Closeness of the doubler's decision: |No-Double vs best double outcome|."""
    return abs(nd - min(dt, dp))


def _checker_gap(moves: list[dict]) -> float:
    """Best-vs-2nd-best reference equity gap for a stored checker move list."""
    if len(moves) < 2:
        return float("inf")
    return moves[0]["equity"] - moves[1]["equity"]


def _take_gap(dt: float, dp: float) -> float:
    """Closeness of the receiver's take/pass decision: |Double-Take vs Double-Pass|."""
    return abs(dt - dp)


def _gap_for(entry: dict) -> float:
    """Escalation gap: a decision is refined when this drops below a threshold.

    For a cube position carrying both a doubler and a receiver sub-decision, the
    position is refined when *either* sub-decision is close (the min of their gaps),
    so the shared analytics reach the precision the closest side needs.
    """
    if entry["kind"] == "checker":
        return _checker_gap(entry["moves"])
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    gaps = []
    if entry.get("has_double"):
        gaps.append(_cube_gap(nd, dt, dp))
    if entry.get("has_take"):
        gaps.append(_take_gap(dt, dp))
    return min(gaps) if gaps else float("inf")


def _move_list_from_result(result) -> list[dict]:
    """Serialize a CheckerPlayResult's ranked moves to plain dicts (best-first)."""
    return [
        {
            "board": list(m.board),
            "equity": m.equity,
            "cubeless_equity": m.cubeless_equity,
            "probs": list(m.probs.to_list()),
            "eval_level": m.eval_level,
        }
        for m in result.moves
    ]


def _cube_fields_from_result(cube) -> dict:
    """Serialize a CubeActionResult's reference fields to plain dicts."""
    return {
        "equity_nd": cube.equity_nd,
        "equity_dt": cube.equity_dt,
        "equity_dp": cube.equity_dp,
        "probs": list(cube.probs.to_list()),
        "should_double_ref": cube.should_double,
        "should_take_ref": cube.should_take,
        "is_beaver": getattr(cube, "is_beaver", False),
        "optimal_action": cube.optimal_action,
        "eval_level": cube.eval_level,
    }


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration like ``2h05m``, ``7m30s``, ``42s``."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ===========================================================================
# Pass 1 - self-play simulation with analytics capture (+ optional XG export)
# ===========================================================================


@dataclass
class _SimState:
    board: list[int]
    cube_value: int
    cube_owner: str   # "centered" | "player" | "opponent" (relative to active)
    active: int       # 1 or 2


def _flip_sim(state: _SimState) -> None:
    from bgsage import flip_board

    state.board = list(flip_board(state.board))
    if state.cube_owner == "player":
        state.cube_owner = "opponent"
    elif state.cube_owner == "opponent":
        state.cube_owner = "player"
    state.active = 3 - state.active


_analyzer_cache: dict[str, Any] = {}


def _get_analyzer(eval_level: str, parallel_threads: int):
    """Cache a BgBotAnalyzer per (level, threads) within a process."""
    key = f"{eval_level}:{parallel_threads}"
    cached = _analyzer_cache.get(key)
    if cached is not None:
        return cached
    from bgsage import BgBotAnalyzer

    a = BgBotAnalyzer(eval_level=eval_level, cubeful=True, parallel_threads=parallel_threads)
    _analyzer_cache[key] = a
    return a


def _simulate_and_capture(seed: int, parallel_threads: int) -> tuple[list[dict], dict]:
    """Play one Sage-3P vs Sage-3P money game; capture every decision's 3P analytics.

    Returns ``(decisions, xg_record)`` where ``decisions`` is the list of benchmark
    decision dicts (with 3P analytics) and ``xg_record`` is the match-history dict
    ``bgsage.text_export.export_history_to_txt`` consumes.
    """
    from bgsage import (
        STARTING_BOARD, check_game_over, classify_game_plan, possible_moves,
    )
    from bgsage.text_export import compute_move_notation

    analyzer = _get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)

    state = _SimState(board=list(STARTING_BOARD), cube_value=1, cube_owner="centered", active=1)
    decisions: list[dict] = []
    move_history: list[dict] = []

    winner: Optional[int] = None
    win_type: Optional[str] = None
    cube_at_end = 1

    def _record_result(active_won: int, mult: int, cube_val: int):
        nonlocal winner, win_type, cube_at_end
        winner = active_won
        win_type = {1: "single", 2: "gammon", 3: "backgammon"}.get(mult, "single")
        cube_at_end = cube_val

    for turn in range(_MAX_TURNS):
        cube_action_str: Optional[str] = None

        # --- Cube decision (only if the active player has cube access) ---
        if state.cube_owner in ("centered", "player"):
            cube = analyzer.cube_action(
                state.board, cube_value=state.cube_value, cube_owner=state.cube_owner,
                jacoby=True, beaver=True,
            )
            nd, dt, dp = cube.equity_nd, cube.equity_dt, cube.equity_dp
            is_beaver = bool(getattr(cube, "is_beaver", False))
            # Doubler's decision counts unless the position is trivial (XG-style).
            has_double = not _is_trivial_cube(nd, dt, dp)
            # The receiver's take/pass decision exists only when a double is actually
            # offered (should_double). It counts unless it is degenerate (DT ~= DP and
            # no beaver option) - mirrors the app's responder rule.
            has_take = bool(cube.should_double) and (
                abs(dt - dp) >= TRIVIAL_SPREAD or is_beaver)
            if has_double or has_take:
                decisions.append({
                    "kind": "cube",
                    "has_double": has_double,
                    "has_take": has_take,
                    "board": list(state.board),
                    "dice": None,
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "key": make_decision_key(
                        "cube", state.board, None, state.cube_value, state.cube_owner),
                    "seed": seed,
                    "turn": turn,
                    "player": state.active,
                    **_cube_fields_from_result(cube),
                })

            if cube.should_double:
                # Opponent's take/pass (from the doubler's-perspective evaluation).
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
                    _record_result(state.active, 1, state.cube_value)
                    break

        # --- Dice roll + checker play ---
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        cands = possible_moves(state.board, d1, d2)
        if not cands:
            post = list(state.board)
        else:
            result = analyzer.checker_play(
                state.board, d1, d2, cube_value=state.cube_value, cube_owner=state.cube_owner,
            )
            post = list(result.moves[0].board)
            moves = _move_list_from_result(result)
            # Counts as a decision: >= 2 legal moves and a meaningful best/worst spread.
            if len(moves) >= 2 and (moves[0]["equity"] - moves[-1]["equity"]) >= TRIVIAL_SPREAD:
                decisions.append({
                    "kind": "checker",
                    "board": list(state.board),
                    "dice": [d1, d2],
                    "cube_value": state.cube_value,
                    "cube_owner": state.cube_owner,
                    "game_plan": classify_game_plan(state.board),
                    "tier": TIER_3P,
                    "moves": moves,
                    "key": make_decision_key(
                        "checker", state.board, (d1, d2), state.cube_value, state.cube_owner),
                    "seed": seed,
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
            _record_result(state.active, code, state.cube_value)
            break
        _flip_sim(state)
    else:
        raise RuntimeError(f"game seed={seed} exceeded {_MAX_TURNS} turns")

    # Build the XG-export record.
    if winner is None:
        result_str, points = "", 0
    else:
        side = "player1" if winner == 1 else "player2"
        mult = {"single": 1, "gammon": 2, "backgammon": 3}[win_type]
        result_str, points = f"{side}-win-{win_type}", cube_at_end * mult
    xg_record = {
        "player1_name": "Sage",
        "player2_name": "Sage",
        "mode": "unlimited",
        "result": result_str,
        "result_points": points,
        "move_history": move_history,
    }
    return decisions, xg_record


def _build_stage1_game(seed: int, write_txt: bool, parallel_threads: int) -> dict:
    """Worker: play one game, persist its decisions (and optional XG .txt), atomically.

    Writes ``stage1/seed_<N>.json`` via a temp-then-rename so a crash mid-write never
    leaves a half-written file that resume would mistake for "done".
    """
    decisions, xg_record = _simulate_and_capture(seed, parallel_threads)

    out_path = _STAGE1_DIR / f"seed_{seed}.json"
    tmp_path = _STAGE1_DIR / f"seed_{seed}.json.tmp"
    tmp_path.write_text(
        json.dumps({"seed": seed, "decisions": decisions}, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(out_path)

    if write_txt:
        from bgsage.text_export import export_history_to_txt

        (_XG_DIR / f"seed_{seed}.txt").write_bytes(export_history_to_txt(xg_record))

    n_checker = sum(1 for d in decisions if d["kind"] == "checker")
    n_cube = len(decisions) - n_checker
    return {"seed": seed, "n_checker": n_checker, "n_cube": n_cube}


def _run_pass1(n_games: int, seed: int, write_txt: bool, workers: int) -> None:
    """Simulate ``n_games`` games at 3P, resuming over any already-completed seeds."""
    _STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    if write_txt:
        _XG_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [seed + i for i in range(n_games)]
    todo = [s for s in seeds if not (_STAGE1_DIR / f"seed_{s}.json").exists()]
    done = len(seeds) - len(todo)
    _log(f"Pass 1/3 (3P self-play): {len(seeds)} games, {done} already done, "
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
        _log(f"  [{completed}/{total}] seed={res['seed']}: "
             f"{res['n_checker']} checker + {res['n_cube']} cube positions  "
             f"({rate:.2f} games/s, ETA {_fmt_dur(eta)})")

    if workers == 1:
        for s in todo:
            _report(_build_stage1_game(s, write_txt, parallel_threads=0))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_stage1_game, s, write_txt, 1): s for s in todo}
            for fut in as_completed(futures):
                _report(fut.result())
    _log(f"Pass 1/3 (3P self-play): complete --{total} games in "
         f"{_fmt_dur(time.perf_counter() - start)}.")


def _load_stage1_decisions() -> list[dict]:
    """Load every captured decision from all per-seed stage-1 files."""
    out: list[dict] = []
    for path in sorted(_STAGE1_DIR.glob("seed_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.extend(data["decisions"])
    return out


def _unique_by_key(decisions: list[dict]) -> dict[str, dict]:
    """Collapse repeated positions to one decision each (first occurrence wins)."""
    by_key: dict[str, dict] = {}
    for d in decisions:
        by_key.setdefault(d["key"], d)
    return by_key


# ===========================================================================
# Passes 2 & 3 - adaptive-precision re-evaluation (3T, then rollout)
# ===========================================================================


def _load_jsonl_by_key(path: Path) -> dict[str, dict]:
    """Load an incremental ``{key: refined-analytics}`` JSONL, if present."""
    out: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def _reeval_decision(analyzer, decision: dict, progress_callback=None) -> dict:
    """Re-evaluate one decision with ``analyzer``; return refined analytics (+ key).

    ``progress_callback(completed, total, ...)`` is forwarded to the engine so a long
    rollout can report intra-position trial progress.
    """
    if decision["kind"] == "checker":
        d1, d2 = decision["dice"]
        result = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
            progress_callback=progress_callback,
        )
        return {"key": decision["key"], "kind": "checker", "moves": _move_list_from_result(result)}
    cube = analyzer.cube_action(
        decision["board"], cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
        progress_callback=progress_callback,
    )
    return {"key": decision["key"], "kind": "cube", **_cube_fields_from_result(cube)}


def _reeval_decision_rollout(analyzer, decision: dict, progress_callback=None) -> dict:
    """Roll out one decision to ROLLOUT_TARGET_SE, sharing caches across batches.

    Cube: a single ``cube_action`` call -- the C++ rollout batches internally
    (``n_trials`` per batch, prefilled Move0/Move1 + SharedPosCache reused) until
    the ND-equity SE < target.

    Checker: a Python lockstep loop. Every batch reseeds the one rollout strategy
    (SharedPosCache stays warm) and rolls out ALL moves together via a single
    ``checker_play`` call -- so within a batch every move is simulated on the SAME
    dice sequence (common random numbers, which is what makes move-to-move equity
    comparisons low-variance), and across batches every move accumulates the SAME
    set of seeds. The loop stops only when the BEST move's aggregate SE drops below
    the target (or ROLLOUT_MAX_BATCHES). Crucially, **no move stops early**: a move
    whose own SE is already under target keeps getting batches until the best move
    converges, so all moves are compared on an identical set of simulation paths.
    Aggregation pools the independent batches: equity = mean of per-batch equities,
    SE = sqrt(sum se_b^2) / n_batches.
    """
    if decision["kind"] == "cube":
        analyzer.set_seed(_ROLLOUT_BASE_SEED)
        cube = analyzer.cube_action(
            decision["board"], cube_value=decision["cube_value"],
            cube_owner=decision["cube_owner"], progress_callback=progress_callback,
        )
        rec = {"key": decision["key"], "kind": "cube", **_cube_fields_from_result(cube)}
        rec["rollout_se"] = cube.equity_nd_se
        return rec

    # Checker: lockstep over batches, gating on the best move's aggregate SE.
    d1, d2 = decision["dice"]
    acc: dict = {}
    n_batches = 0
    best_se = float("inf")
    for b in range(ROLLOUT_MAX_BATCHES):
        analyzer.set_seed((_ROLLOUT_BASE_SEED + b * _SEED_STEP) & 0xFFFFFFFF)
        res = analyzer.checker_play(
            decision["board"], d1, d2,
            cube_value=decision["cube_value"], cube_owner=decision["cube_owner"],
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
        # Gate on the BEST move's SE only. Every move keeps getting batches (above
        # loop rolls out all of them) until this fires -- no per-move early stop.
        if ROLLOUT_TARGET_SE <= 0 or best_se <= ROLLOUT_TARGET_SE:
            break
    return {"key": decision["key"], "kind": "checker", "moves": agg,
            "rollout_batches": n_batches, "rollout_se": best_se}


def _apply_refined(entry: dict, refined: dict, tier: str) -> dict:
    """Return a copy of ``entry`` with refined analytics merged in and tier bumped."""
    merged = dict(entry)
    if entry["kind"] == "checker":
        merged["moves"] = refined["moves"]
    else:
        for k in ("equity_nd", "equity_dt", "equity_dp", "probs",
                  "should_double_ref", "should_take_ref", "is_beaver",
                  "optimal_action", "eval_level"):
            if k in refined:
                merged[k] = refined[k]
    merged["tier"] = tier
    return merged


def _run_refinement_pass(
    unique: dict[str, dict], analyzer, out_file: Path, gap_threshold: float, tier: str,
    pass_label: str, reeval_fn=_reeval_decision,
) -> dict[str, dict]:
    """Re-evaluate every decision whose current gap < ``gap_threshold`` at ``tier``.

    Persists each refined result to ``out_file`` (JSONL, keyed by decision hash) the
    instant it is computed, so a crash resumes without recomputation. ``unique`` maps
    key -> decision dict at its current-best tier; returns the same map updated with
    the refined analytics for the decisions that were re-evaluated.

    Progress is reported incrementally: a line per completed position (always for slow
    rollouts), plus an intra-position trial heartbeat at least every ~15s so a long
    rollout never looks stalled.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = _load_jsonl_by_key(out_file)

    # Apply already-computed refinements first (resume), then pick what's left to do.
    for key, refined in done.items():
        if key in unique:
            unique[key] = _apply_refined(unique[key], refined, tier)

    todo = [
        key for key, dec in unique.items()
        if key not in done and _gap_for(dec) < gap_threshold
    ]
    _log(f"{pass_label}: {len(done)} already refined, {len(todo)} to compute "
         f"(gap < {gap_threshold}).")
    if not todo:
        return unique

    total = len(todo)
    start = time.perf_counter()
    last_print = [start]
    cur_label = [""]

    def _heartbeat(completed, total_trials, *_):
        # Intra-position progress; throttled, so only slow rollouts actually print.
        now = time.perf_counter()
        if total_trials and now - last_print[0] >= _PROGRESS_EVERY_S:
            last_print[0] = now
            pct = 100 * completed // total_trials
            print(f"      {cur_label[0]}: {completed}/{total_trials} trials ({pct}%)",
                  flush=True)

    with out_file.open("a", encoding="utf-8") as f:
        for i, key in enumerate(todo, 1):
            dec = unique[key]
            cur_label[0] = f"[{i}/{total}] {dec['kind']} {key[:8]}"
            t0 = time.perf_counter()
            refined = reeval_fn(analyzer, dec, progress_callback=_heartbeat)
            item_dt = time.perf_counter() - t0
            f.write(json.dumps(refined, separators=(",", ":")) + "\n")
            f.flush()
            unique[key] = _apply_refined(dec, refined, tier)
            now = time.perf_counter()
            if item_dt >= _SLOW_ITEM_S or now - last_print[0] >= _PROGRESS_EVERY_S or i == total:
                last_print[0] = now
                rate = i / max(1e-9, now - start)
                eta = (total - i) / rate if rate > 0 else 0
                _log(f"  [{i}/{total}] {tier} {dec['kind']} {key[:8]} done in "
                     f"{item_dt:.1f}s  ({rate:.2f}/s, ETA {_fmt_dur(eta)})")
    _log(f"{pass_label}: complete --{total} refined in {_fmt_dur(time.perf_counter() - start)}.")
    return unique


def _make_3t_analyzer(n_threads: int):
    from bgsage import BgBotAnalyzer

    return BgBotAnalyzer(eval_level="truncated3", cubeful=True, parallel_threads=n_threads)


def _make_rollout_analyzer(n_threads: int):
    from bgsage import BgBotAnalyzer, TrialEvalConfig

    return BgBotAnalyzer(
        eval_level="rollout",
        cubeful=True,
        parallel_threads=n_threads,
        n_trials=ROLLOUT_N_TRIALS,
        truncation_depth=0,
        decision_ply=3,
        late_ply=3,
        late_threshold=20,
        ultra_late_threshold=9999,
        checker=TrialEvalConfig(ply=3),
        cube=TrialEvalConfig(ply=3),
        seed=_ROLLOUT_BASE_SEED,
        target_se=ROLLOUT_TARGET_SE,
        max_batches=ROLLOUT_MAX_BATCHES,
    )


# ===========================================================================
# Final assembly
# ===========================================================================


def _assemble_dataset(unique: dict[str, dict], dataset_path: Path, n_games: int, seed: int) -> dict:
    """Write the final benchmark JSON from the most-refined analytics per decision."""
    decisions = sorted(unique.values(), key=lambda d: (d.get("seed", 0), d.get("turn", 0), d["kind"]))

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
        "n_games": n_games,
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
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    tmp.write_text(json.dumps({"meta": meta, "decisions": decisions}), encoding="utf-8")
    tmp.replace(dataset_path)
    # Also emit a gzip sibling: the full JSON exceeds GitHub's 100 MB file limit, so the
    # repo ships benchmark.json.gz (read transparently by _read_dataset on a fresh clone).
    Path(f"{dataset_path}.gz").write_bytes(gzip.compress(dataset_path.read_bytes(), compresslevel=9))
    _log(f"Wrote {len(decisions)} positions ({n_decisions} decisions) to {dataset_path} (+ .gz)")
    _log(f"  checker={n_checker}  cube: {n_cube_decisions} decisions "
         f"(double={n_double}, take={n_take}) over {n_cube_entries} positions  tiers={tier_counts}")
    return meta


#: Export-mode output: the list of positions that need a rollout (Pass 3).
_ROLLOUT_JOBS_FILE = _DATA_DIR / "rollout_jobs.jsonl"


def _seed_in_range(dec: dict, min_seed: int | None, max_seed: int | None) -> bool:
    """Whether a decision's seed falls in the optional [min_seed, max_seed] export window."""
    s = dec.get("seed", 0)
    return (min_seed is None or s >= min_seed) and (max_seed is None or s <= max_seed)


def _export_rollout_jobs(unique: dict[str, dict],
                         min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions that Pass 3 would roll out, with all rollout inputs.

    Used by ``rollout_mode="export"``: instead of running the (expensive) rollouts
    in-process, dump one self-contained job per position so the rollouts can be run
    elsewhere (e.g. distributed workers). Each job has the board/dice/cube/match
    inputs plus the rollout config (target SE, batch size, gating quantity). A
    worker rolls the position out to ``target_se`` and writes a record
    ``{key, kind, moves|<cube fields>}`` (the shape ``_reeval_decision_rollout``
    returns) into ``stage3_rollout.jsonl``; re-running the build then ingests them.

    Skips positions already present in the stage-3 file (resumable).
    """
    done = _load_jsonl_by_key(_STAGE3_FILE)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or _gap_for(dec) >= ROLLOUT_GAP or not _seed_in_range(dec, min_seed, max_seed):
            continue
        jobs.append({
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
            "away1": 0, "away2": 0, "is_crawford": False,
            "jacoby": True, "beaver": True,
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
    _ROLLOUT_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _ROLLOUT_JOBS_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(_ROLLOUT_JOBS_FILE)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} rollout jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {_ROLLOUT_JOBS_FILE} -- no rollouts computed.")
    return _ROLLOUT_JOBS_FILE


#: Export-mode output: the list of positions that need a 3T re-eval (Pass 2).
_THREET_JOBS_FILE = _DATA_DIR / "threet_jobs.jsonl"


def _export_threet_jobs(unique: dict[str, dict],
                        min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions that Pass 2 would re-evaluate at 3T, with all inputs.

    The 3T analogue of ``_export_rollout_jobs`` (used by ``threet_mode="export"``):
    instead of running the 3T re-evals in-process, dump one self-contained job per
    position needing 3T so they can be computed elsewhere (distributed workers). A
    worker re-evaluates the position at ``truncated3`` and writes a record
    ``{key, kind, moves|<cube fields>}`` (the shape ``_reeval_decision`` returns) into
    ``stage2_3t.jsonl``; re-running the build then ingests them. Skips positions
    already present in the stage-2 file (resumable).
    """
    done = _load_jsonl_by_key(_STAGE2_FILE)
    jobs: list[dict] = []
    for key, dec in unique.items():
        if key in done or _gap_for(dec) >= THREE_T_GAP or not _seed_in_range(dec, min_seed, max_seed):
            continue
        jobs.append({
            "key": key,
            "kind": dec["kind"],
            "board": dec["board"],
            "dice": dec.get("dice"),
            "cube_value": dec["cube_value"],
            "cube_owner": dec["cube_owner"],
            "game_plan": dec.get("game_plan"),
            "level": "truncated3",
        })
    _THREET_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _THREET_JOBS_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    tmp.replace(_THREET_JOBS_FILE)
    n_ck = sum(1 for j in jobs if j["kind"] == "checker")
    _log(f"Export: wrote {len(jobs)} 3T jobs ({n_ck} checker, {len(jobs) - n_ck} cube) "
         f"to {_THREET_JOBS_FILE} -- no 3T evals computed.")
    return _THREET_JOBS_FILE


def build_benchmark_data(
    n_games: int,
    seed: int = 1,
    n_threads: int = 0,
    write_txt: bool = True,
    workers: int = 6,
    dataset_path: Path | str = DEFAULT_DATASET,
    rollout_mode: str = "compute",
    threet_mode: str = "compute",
    export_min_seed: int | None = None,
    export_max_seed: int | None = None,
    stages: tuple = ("pass1", "pass2", "pass3"),
) -> dict:
    """Build the money-game benchmark data set (three adaptive-precision passes).

    Pass 1: simulate ``n_games`` Sage-3P vs Sage-3P money games, capturing 3-ply
    analytics for every real decision (optionally writing an XG-import ``.txt`` per
    game). Pass 2: re-evaluate decisions whose 3-ply best-vs-2nd-best gap < 0.05 at
    3T. Pass 3: roll out (1,296 paths, VR, 3-ply checker + cube) decisions whose
    best-available gap < 0.02. The final entry for each decision uses the strongest
    tier reached.

    Every stage persists its results as they are produced; a re-run skips ahead to
    the first uncomputed work. ``n_threads`` is the thread count for the 3T and
    rollout passes (each evaluation parallelizes internally). ``workers`` is the
    number of parallel **processes** for the cheap pass-1 self-play.

    ``rollout_mode``: ``"compute"`` (default) runs Pass 3 in-process. ``"export"``
    skips the rollouts and instead writes the positions that need rolling out to
    ``rollout_jobs.jsonl`` (so they can be computed elsewhere), then assembles a
    provisional dataset using the best tier reached so far (3P/3T). Once the
    external rollout results are written into ``stage3_rollout.jsonl``, re-running
    in ``"compute"`` mode ingests them and finalizes the dataset.

    ``stages``: which passes to run this invocation, e.g. ``("pass1",)`` to only
    simulate + capture 3P, ``("pass2",)`` to only run the 3T pass later, or
    ``("pass3",)`` for only the rollout pass. Skipped passes still apply any results
    already on disk, so the assembled dataset always reflects the best tier reached.
    Each pass is independently resumable, so the passes can be run separately.

    Returns the dataset metadata dict.
    """
    dataset_path = Path(dataset_path)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stages = set(stages)
    _log(f"=== build_benchmark_data: n_games={n_games} seed={seed} "
         f"n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} stages={sorted(stages)} ===")

    # Pass 1 - self-play + 3P capture (runs only if requested; the rest of the
    # pipeline still loads whatever stage-1 files are already on disk).
    if "pass1" in stages:
        _run_pass1(n_games, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions()
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = _unique_by_key(all_decisions)
    _log(f"Captured {len(all_decisions)} decisions ({len(unique)} unique positions).")

    # Pass 2 - 3T refinement. Run it, or (when skipped) just apply any 3T results
    # already on disk so the gap/tier reflect prior work.
    if "pass2" in stages:
        if threet_mode == "export":
            for key, refined in _load_jsonl_by_key(_STAGE2_FILE).items():
                if key in unique:
                    unique[key] = _apply_refined(unique[key], refined, TIER_3T)
            _export_threet_jobs(unique, export_min_seed, export_max_seed)
        else:
            analyzer_3t = _make_3t_analyzer(n_threads)
            unique = _run_refinement_pass(
                unique, analyzer_3t, _STAGE2_FILE, THREE_T_GAP, TIER_3T, "Pass 2/3 (3T refine)")
    else:
        done2 = _load_jsonl_by_key(_STAGE2_FILE)
        for key, refined in done2.items():
            if key in unique:
                unique[key] = _apply_refined(unique[key], refined, TIER_3T)
        pending2 = sum(1 for key, dec in unique.items()
                       if key not in done2 and _gap_for(dec) < THREE_T_GAP)
        _log(f"Pass 2 (3T) not run: {pending2} positions want 3T "
             f"(3P gap < {THREE_T_GAP}); {len(done2)} already done. Run with --stages pass2.")

    # Pass 3 - rollout (compute or export). When skipped, apply any rollout results
    # already on disk.
    if "pass3" in stages:
        if rollout_mode == "export":
            for key, refined in _load_jsonl_by_key(_STAGE3_FILE).items():
                if key in unique:
                    unique[key] = _apply_refined(unique[key], refined, TIER_ROLLOUT)
            _export_rollout_jobs(unique, export_min_seed, export_max_seed)
        else:
            # Each rolled-out decision runs to ROLLOUT_TARGET_SE -- cube decisions
            # batch internally in C++; checker decisions use the Python lockstep.
            analyzer_ro = _make_rollout_analyzer(n_threads)
            unique = _run_refinement_pass(
                unique, analyzer_ro, _STAGE3_FILE, ROLLOUT_GAP, TIER_ROLLOUT,
                "Pass 3/3 (full rollout to SE)", reeval_fn=_reeval_decision_rollout)
    else:
        done3 = _load_jsonl_by_key(_STAGE3_FILE)
        for key, refined in done3.items():
            if key in unique:
                unique[key] = _apply_refined(unique[key], refined, TIER_ROLLOUT)
        pending3 = sum(1 for key, dec in unique.items()
                       if key not in done3 and _gap_for(dec) < ROLLOUT_GAP)
        _log(f"Pass 3 (rollout) not run: {pending3} positions want rollout "
             f"(current-tier gap < {ROLLOUT_GAP}); {len(done3)} already done. "
             f"Run with --stages pass3.")

    # Assemble the dataset at whatever tiers are now available.
    return _assemble_dataset(unique, dataset_path, n_games, seed)


# ===========================================================================
# Scoring - benchmark_pr
# ===========================================================================


def _scored(bucket: str, sub: str, plan: Optional[str], err: float) -> dict:
    return {"bucket": bucket, "sub": sub, "game_plan": plan, "error": err,
            "is_blunder": err > BLUNDER_THRESHOLD}


def _score_checker(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    """Scored checker decision (bot's chosen play vs the reference best). None on mismatch."""
    cands = bot.checker_play(
        entry["board"], entry["dice"][0], entry["dice"][1],
        entry["cube_value"], entry["cube_owner"],
    )
    if not cands:
        return None
    chosen = tuple(cands[0].board)
    ref_by_board = {tuple(m["board"]): m["equity"] for m in entry["moves"]}
    chosen_eq = ref_by_board.get(chosen)
    if chosen_eq is None:
        return None  # bot's move not among the reference's legal moves (contract issue)
    best_eq = entry["moves"][0]["equity"]
    return _scored("checker", "checker", entry.get("game_plan"), max(0.0, best_eq - chosen_eq))


def _score_cube(bot: BenchmarkBot, entry: dict) -> list[dict]:
    """Scored cube sub-decisions (doubler and/or receiver), per the standard PR formulas.

    One ``cube_action`` call yields both sides. The doubler's error is
    ``max(0, max(ND, min(DT,DP)) - actual)`` (actual = min(DT,DP) if it doubles else
    ND). The receiver's error is ``max(0, actual - min(DT,DP))`` (actual = DT if it
    takes else DP), both from the doubler's perspective - matching backend
    ``compute_cube_error`` / ``compute_opp_cube_error``.
    """
    nd, dt, dp = entry["equity_nd"], entry["equity_dt"], entry["equity_dp"]
    plan = entry.get("game_plan")
    a = bot.cube_action(entry["board"], entry["cube_value"], entry["cube_owner"])
    out: list[dict] = []
    if entry.get("has_double"):
        optimal = max(nd, min(dt, dp))
        actual = min(dt, dp) if a.should_double else nd
        out.append(_scored("cube", "double", plan, max(0.0, optimal - actual)))
    if entry.get("has_take"):
        optimal = min(dt, dp)
        actual = dt if a.should_take else dp
        out.append(_scored("cube", "take", plan, max(0.0, actual - optimal)))
    return out


def _score_entry(bot: BenchmarkBot, entry: dict) -> Optional[dict]:
    """Score one position; return ``{key, scored:[...]}`` or None on a checker mismatch.

    A checker position yields one scored decision; a cube position yields one or two
    (doubler and/or receiver). Each scored item is bucketed as ``checker`` or ``cube``.
    """
    if entry["kind"] == "checker":
        s = _score_checker(bot, entry)
        if s is None:
            return None
        scored = [s]
    else:
        scored = _score_cube(bot, entry)
    return {"key": entry["key"], "scored": scored}


def _aggregate(entry_results: Iterable[dict]) -> dict:
    """Aggregate scored positions into the PR breakdown (each sub-decision counts once)."""
    sums = {"checker": 0.0, "cube": 0.0}
    counts = {"checker": 0, "cube": 0}
    blunders = {"checker": 0, "cube": 0}
    by_plan = {p: {"error": 0.0, "n": 0, "checker_n": 0, "cube_n": 0, "blunders": 0}
               for p in GAME_PLANS}

    for er in entry_results:
        for s in er.get("scored", []):
            bucket = s["bucket"]
            err = s["error"]
            sums[bucket] += err
            counts[bucket] += 1
            if s["is_blunder"]:
                blunders[bucket] += 1
            plan = s.get("game_plan")
            if plan in by_plan:
                b = by_plan[plan]
                b["error"] += err
                b["n"] += 1
                b[f"{bucket}_n"] += 1
                if s["is_blunder"]:
                    b["blunders"] += 1

    def pr(total_err: float, n: int) -> float:
        return (total_err / n * PR_MULTIPLIER) if n > 0 else 0.0

    n_checker, n_cube = counts["checker"], counts["cube"]
    n_total = n_checker + n_cube
    err_total = sums["checker"] + sums["cube"]

    plan_out = {}
    for p, b in by_plan.items():
        plan_out[p] = {
            "pr": pr(b["error"], b["n"]),
            "n": b["n"],
            "checker_n": b["checker_n"],
            "cube_n": b["cube_n"],
            "error": b["error"],
            "blunders": b["blunders"],
        }

    return {
        "total_pr": pr(err_total, n_total),
        "checker_pr": pr(sums["checker"], n_checker),
        "cube_pr": pr(sums["cube"], n_cube),
        "n_decisions": n_total,
        "n_checker": n_checker,
        "n_cube": n_cube,
        "total_error": err_total,
        "checker_error": sums["checker"],
        "cube_error": sums["cube"],
        "blunders": {
            "total": blunders["checker"] + blunders["cube"],
            "checker": blunders["checker"],
            "cube": blunders["cube"],
        },
        "by_game_plan": plan_out,
    }


def _missing_tier(entry: dict) -> Optional[str]:
    """Whether a decision lacks the precision its closeness requires.

    A decision whose best-vs-2nd-best gap is below a refinement threshold should
    carry analytics at that tier; if it doesn't (e.g. its rollout hasn't been
    computed yet), its stored reference is too coarse to score against fairly.
    Returns the missing tier (``"rollout"`` or ``"3t"``), or ``None`` if complete.
    """
    gap = _gap_for(entry)
    tier = entry.get("tier")
    if gap < ROLLOUT_GAP and tier != TIER_ROLLOUT:
        return "rollout"
    if gap < THREE_T_GAP and tier == TIER_3P:
        return "3t"
    return None


def benchmark_pr(
    bot: BenchmarkBot,
    dataset_path: Path | str = DEFAULT_DATASET,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the benchmark data set; return the PR breakdown.

    Every decision in the data set is assumed to be a valid decision (the build
    already applied the forced-move / triviality filters). For each decision the bot
    is asked for its play (checker) or cube assessment, the equity error is computed
    against the cached reference analytics, and the decision is bucketed by the
    mover's game plan.

    Crash-safe and resumable: each decision's result is appended to a per-bot cache
    (``scores/<label>.jsonl``, keyed by decision hash) as it is scored; a re-run loads
    the cache, skips decisions already scored, computes only the remainder, and then
    aggregates the PR from the full cache.

    Args:
        bot: the bot to score. Must be thread-safe if ``n_threads > 1``.
        dataset_path: path to a data set built by :func:`build_benchmark_data`.
        n_threads: number of threads scoring decisions concurrently.
        label: names the resume cache (use distinct labels for distinct bots).
        cache_path: override the cache location (defaults to ``scores/<label>.jsonl``).
        progress: log progress periodically.
        max_seed: if set, only score decisions from the first ``max_seed`` games
            (those with ``seed <= max_seed``); decisions with a larger seed are skipped.
            Lets you benchmark the first N games while the full set is still building.

    Returns:
        Dict with ``total_pr``, ``checker_pr``, ``cube_pr``, decision/blunder counts,
        and ``by_game_plan`` (per-plan PR over checker + cube decisions combined).
    """
    dataset_path = Path(dataset_path)
    data = _read_dataset(dataset_path)
    all_entries = data["decisions"]
    if max_seed is not None:
        n_all = len(all_entries)
        all_entries = [e for e in all_entries if e.get("seed", 0) <= max_seed]
        _log(f"Limiting to the first {max_seed} games (seed <= {max_seed}): "
             f"{len(all_entries)} of {n_all} decisions.")

    # Skip decisions whose reference is below the precision their closeness
    # requires (e.g. a rollout-eligible decision whose rollout isn't done yet).
    # Scoring those against a coarser 3T/3P reference would be unfair.
    entries = []
    skipped_missing = {"rollout": 0, "3t": 0}
    for e in all_entries:
        miss = _missing_tier(e)
        if miss is None:
            entries.append(e)
        else:
            skipped_missing[miss] += 1
    n_skipped = skipped_missing["rollout"] + skipped_missing["3t"]

    if cache_path is None:
        _SCORES_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _SCORES_DIR / f"{label}.jsonl"
    else:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_jsonl_by_key(cache_path)
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
                if progress and (done == total or now - last_print[0] >= _PROGRESS_EVERY_S):
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
    result = _aggregate(r for k, r in cached.items() if k in scoreable_keys)
    result["mismatches"] = mismatches
    result["skipped"] = n_skipped
    result["skipped_missing_rollout"] = skipped_missing["rollout"]
    result["skipped_missing_3t"] = skipped_missing["3t"]
    if progress:
        _log(f"{label}: total PR={result['total_pr']:.2f} "
             f"(checker={result['checker_pr']:.2f}, cube={result['cube_pr']:.2f}) "
             f"over {result['n_decisions']} decisions")
    return result


def _print_report(result: dict) -> None:
    print()
    print(f"{'='*60}")
    print(f"Total PR:   {result['total_pr']:7.2f}   ({result['n_decisions']} decisions)")
    print(f"Checker PR: {result['checker_pr']:7.2f}   ({result['n_checker']} decisions)")
    print(f"Cube PR:    {result['cube_pr']:7.2f}   ({result['n_cube']} decisions)")
    print(f"Blunders:   {result['blunders']['total']} "
          f"(checker {result['blunders']['checker']}, cube {result['blunders']['cube']})")
    if result.get("skipped"):
        print(f"Skipped:    {result['skipped']} positions (missing reference: "
              f"{result.get('skipped_missing_rollout', 0)} rollout, "
              f"{result.get('skipped_missing_3t', 0)} 3T)")
    if result.get("mismatches"):
        print(f"Mismatches: {result['mismatches']} (bot move not a legal reference move)")
    print(f"{'-'*60}")
    print(f"{'Game plan':>10}  {'PR':>7}  {'N':>6}  {'chk':>6}  {'cube':>5}  {'blnd':>5}")
    for plan in GAME_PLANS:
        b = result["by_game_plan"][plan]
        print(f"{plan:>10}  {b['pr']:7.2f}  {b['n']:6d}  {b['checker_n']:6d}  "
              f"{b['cube_n']:5d}  {b['blunders']:5d}")
    print(f"{'='*60}")


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the benchmark data set")
    p_build.add_argument("--n-games", type=int, default=200, help="Games to simulate")
    p_build.add_argument("--seed", type=int, default=1, help="First game seed")
    p_build.add_argument("--n-threads", type=int, default=0,
                         help="Threads for the 3T and rollout passes (0=auto)")
    p_build.add_argument("--workers", type=int, default=6,
                         help="Parallel processes for the pass-1 self-play")
    p_build.add_argument("--write-txt", action=argparse.BooleanOptionalAction, default=True,
                         help="Write an XG-import .txt per game to xg/ (default: on; "
                              "use --no-write-txt to disable)")
    p_build.add_argument("--rollout-mode", choices=["compute", "export"], default="compute",
                         help="compute: run Pass-3 rollouts in-process; "
                              "export: write rollout_jobs.jsonl instead (no rollouts)")
    p_build.add_argument("--threet-mode", choices=["compute", "export"], default="compute",
                         help="compute: run the Pass-2 3T re-evals in-process; "
                              "export: write threet_jobs.jsonl instead (no 3T evals)")
    p_build.add_argument("--stages", default="pass1,pass2,pass3",
                         help="comma-separated passes to run this invocation "
                              "(pass1=self-play+3P, pass2=3T, pass3=rollout). "
                              "e.g. 'pass1' to only simulate games")
    p_build.add_argument("--min-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed >= this")
    p_build.add_argument("--max-seed", type=int, default=None,
                         help="In export modes, only emit jobs for positions with seed <= this")
    p_build.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                         help="Output dataset path")

    p_score = sub.add_parser("score", help="Score the Sage engine against the data set")
    p_score.add_argument("--level", default="3ply",
                         help="SageBot eval level (1ply..4ply, truncated1/2/3, rollout)")
    p_score.add_argument("--model", default=None, help="Model name (default: production)")
    p_score.add_argument("--n-threads", type=int, default=1,
                         help="Threads scoring decisions concurrently")
    p_score.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p_score.add_argument("--label", default=None, help="Resume-cache label")
    p_score.add_argument("--max-seed", type=int, default=None,
                         help="Only score the first N games (decisions with seed <= MAX_SEED)")

    args = parser.parse_args(argv)

    if args.command == "build":
        build_benchmark_data(
            n_games=args.n_games, seed=args.seed, n_threads=args.n_threads,
            write_txt=args.write_txt, workers=args.workers, dataset_path=args.dataset,
            rollout_mode=args.rollout_mode, threet_mode=args.threet_mode,
            export_min_seed=args.min_seed, export_max_seed=args.max_seed,
            stages=tuple(s.strip() for s in args.stages.split(",") if s.strip()),
        )
    elif args.command == "score":
        # Internal threads = 1 so benchmark_pr's n_threads parallelizes across positions.
        bot = SageBot(eval_level=args.level, model=args.model, parallel_threads=1)
        label = args.label or f"sage_{args.level}" + (f"_{args.model}" if args.model else "")
        result = benchmark_pr(bot, dataset_path=args.dataset, n_threads=args.n_threads,
                              label=label, max_seed=args.max_seed)
        _print_report(result)


if __name__ == "__main__":
    _main()
