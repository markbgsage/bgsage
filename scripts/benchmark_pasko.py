# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Paskogammon "PR against a rollout" benchmark.

The Paskogammon analog of :mod:`benchmark_money`. It builds a production-quality
reference data set of **Paskogammon** money-game decisions (checker plays and cube
actions) with rollout-grade analytics, and scores an arbitrary bot against it with
the same custom Performance Rating (PR) calculation.

Everything mirrors the money-game pipeline -- three adaptive-precision passes
(3-ply for clear decisions, 3T for decisions whose best-vs-2nd-best gap < 0.05,
full rollout for those whose gap < 0.02), the same blunder threshold, the same
resumable on-disk layout, and the same distributed export modes -- with one
essential difference: **every game starts from the Paskogammon opening position.**

Paskogammon rules (as played in the app; unlimited games only, Jacoby + Beaver on):

  * Player 1 -- the side with the standard backgammon layout -- ALWAYS moves
    first. Player 2's fifteen checkers start in the scattered Paskogammon
    arrangement, which compensates for moving second.
  * The opening roll is a normal full roll by Player 1: two dice, **doubles
    allowed** (there is no roll-one-die-each opening ritual).

After the opening position, play is standard backgammon, so all downstream
machinery (move generation, game plans, cube logic, 3T / rollout refinement,
scoring) is reused from :mod:`benchmark_money` unchanged. The decision key is the
money key (board + dice + cube state); results live in their own
``data/pasko_money_benchmark/`` directory so they can never mix with the standard
money benchmark.

Each game is exported as an XG-import ``.txt`` transcript (one file per game),
exactly like the money benchmark. CAVEAT: the transcript format has no way to
express a non-standard starting position -- the move list only replays correctly
from the Paskogammon start, so an importer that assumes the standard backgammon
start (e.g. XG's text import) will not reconstruct these games as-is.

Two public entry points (same shapes as :mod:`benchmark_money`):

  * :func:`build_benchmark_data` - simulate Sage-3P vs Sage-3P Paskogammon games
    and emit the reference data set (three adaptive-precision passes). Crash-safe
    and resumable.

  * :func:`benchmark_pr` - score a :class:`benchmark_money.BenchmarkBot` against
    the data set and return total / checker / cube PR plus a per-game-plan
    breakdown. The bot interface is the money one (no match state).

Distributed passes (from the PARENT repo, via Parallelizor):

    # Pass 1 locally, then export the 3T jobs:
    python bgsage/scripts/benchmark_pasko.py build --n-games 10 --stages pass1
    python bgsage/scripts/benchmark_pasko.py build --n-games 10 --stages pass2 --threet-mode export
    python scripts/threet_pasko_benchmark.py --vms 20            # distributed 3T
    # Ingest 3T results, then export the rollout jobs:
    python bgsage/scripts/benchmark_pasko.py build --n-games 10 --stages pass2,pass3 --rollout-mode export
    python scripts/rollout_pasko_benchmark.py --workers 100      # distributed rollouts
    # Final ingest + assembly:
    python bgsage/scripts/benchmark_pasko.py build --n-games 10 --stages pass2,pass3

Usage::

    # Build everything locally (small runs only -- rollouts are expensive):
    python scripts/benchmark_pasko.py build --n-games 10 --n-threads 16 --write-txt

    # Score the production model at 3-ply against the built data set:
    python scripts/benchmark_pasko.py score --level 3ply --n-threads 16
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

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

# benchmark_money supplies all the variant-agnostic machinery. Its bgsage imports
# are deferred, so importing it here does NOT load the engine.
import benchmark_money as bm  # noqa: E402

# Re-export the shared knobs / types so callers/readers see them on this module too.
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

#: The bot interface is exactly the money one -- Paskogammon decisions carry the
#: same (board, dice, cube) context, so any money BenchmarkBot scores unchanged.
BenchmarkBot = bm.BenchmarkBot
SageBot = bm.SageBot
CheckerCandidate = bm.CheckerCandidate
CubeAssessment = bm.CubeAssessment
make_decision_key = bm.make_decision_key

_log = bm._log
_fmt_dur = bm._fmt_dur


# ---------------------------------------------------------------------------
# Paskogammon opening position
# ---------------------------------------------------------------------------

#: Paskogammon starting board from the first mover's perspective (positive = the
#: player about to roll, who has the standard backgammon layout; negative = the
#: second player's scattered Paskogammon arrangement). Identical to the app's
#: ``backend/game_store.py::PASKO_BOARD`` and ``generate_pasko_data.py::PASKO_START``.
PASKO_STARTING_BOARD: list[int] = [
    0, -2, -2, 0, 0, -1, 5, -1, 3, 0, 0, 0, -2,   # [0]=opp bar, points 1-12
    5, 0, 0, 0, -2, -1, -3, -1, 0, 0, 0, 2,       # points 13-24
    0,                                            # [25]=player bar
]


# ---------------------------------------------------------------------------
# Output locations (all under bgsage/data/pasko_money_benchmark/)
# ---------------------------------------------------------------------------

_DATA_DIR = _PROJECT_ROOT / "data" / "pasko_money_benchmark"
_BUILD_SUBDIR = _DATA_DIR / "build"
_STAGE1_DIR = _BUILD_SUBDIR / "stage1"            # per-seed game decisions (3P)
_STAGE2_FILE = _BUILD_SUBDIR / "stage2_3t.jsonl"  # 3T re-evals, keyed by decision hash
_STAGE3_FILE = _BUILD_SUBDIR / "stage3_rollout.jsonl"  # rollouts, keyed by decision hash
_XG_DIR = _DATA_DIR / "xg"                        # XG-import .txt transcripts
_SCORES_DIR = _DATA_DIR / "scores"                # per-bot scoring caches
DEFAULT_DATASET = _DATA_DIR / "benchmark.json"

#: Export-mode outputs (consumed by the parent repo's Parallelizor scripts).
_THREET_JOBS_FILE = _DATA_DIR / "threet_jobs.jsonl"
_ROLLOUT_JOBS_FILE = _DATA_DIR / "rollout_jobs.jsonl"


# ===========================================================================
# Pass 1 - Paskogammon self-play simulation with analytics capture
# ===========================================================================


def _simulate_and_capture(seed: int, parallel_threads: int) -> tuple[list[dict], dict]:
    """Play one Sage-3P vs Sage-3P Paskogammon game; capture every decision's 3P analytics.

    Identical to ``benchmark_money._simulate_and_capture`` except the game starts
    from ``PASKO_STARTING_BOARD``. Player 1 (the standard-layout side) is on roll
    first, and the opening roll is drawn uniformly like every other roll --
    doubles allowed -- which is exactly the Paskogammon opening rule.

    Returns ``(decisions, xg_record)`` where ``decisions`` is the list of benchmark
    decision dicts (with 3P analytics) and ``xg_record`` is the match-history dict
    ``bgsage.text_export.export_history_to_txt`` consumes.
    """
    from bgsage import check_game_over, classify_game_plan, possible_moves
    from bgsage.text_export import compute_move_notation

    analyzer = bm._get_analyzer("3ply", parallel_threads)
    rng = random.Random(seed)

    state = bm._SimState(
        board=list(PASKO_STARTING_BOARD), cube_value=1, cube_owner="centered", active=1)
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

    for turn in range(bm._MAX_TURNS):
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
            has_double = not bm._is_trivial_cube(nd, dt, dp)
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
                    **bm._cube_fields_from_result(cube),
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

        # --- Dice roll + checker play (uniform roll; doubles allowed on turn 0) ---
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        cands = possible_moves(state.board, d1, d2)
        if not cands:
            post = list(state.board)
        else:
            result = analyzer.checker_play(
                state.board, d1, d2, cube_value=state.cube_value, cube_owner=state.cube_owner,
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
        bm._flip_sim(state)
    else:
        raise RuntimeError(f"game seed={seed} exceeded {bm._MAX_TURNS} turns")

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
    """Worker: play one game, persist its decisions (and optional XG .txt), atomically."""
    decisions, xg_record = _simulate_and_capture(seed, parallel_threads)

    out_path = _STAGE1_DIR / f"seed_{seed}.json"
    tmp_path = _STAGE1_DIR / f"seed_{seed}.json.tmp"
    tmp_path.write_text(
        json.dumps({"seed": seed, "variant": "paskogammon", "decisions": decisions},
                   separators=(",", ":")),
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
    """Simulate ``n_games`` Paskogammon games at 3P, resuming over completed seeds."""
    _STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    if write_txt:
        _XG_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [seed + i for i in range(n_games)]
    todo = [s for s in seeds if not (_STAGE1_DIR / f"seed_{s}.json").exists()]
    done = len(seeds) - len(todo)
    _log(f"Pass 1/3 (Paskogammon 3P self-play): {len(seeds)} games, {done} already done, "
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
    _log(f"Pass 1/3 (Paskogammon 3P self-play): complete --{total} games in "
         f"{_fmt_dur(time.perf_counter() - start)}.")


def _load_stage1_decisions() -> list[dict]:
    """Load every captured decision from all per-seed stage-1 files."""
    out: list[dict] = []
    for path in sorted(_STAGE1_DIR.glob("seed_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.extend(data["decisions"])
    return out


# ===========================================================================
# Job export (for the distributed Parallelizor passes)
# ===========================================================================


def _export_threet_jobs(unique: dict[str, dict],
                        min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions Pass 2 would re-evaluate at 3T, with all inputs.

    Same job shape as ``benchmark_money._export_threet_jobs`` (money semantics --
    no match context, Jacoby/beaver implied by the analyzer defaults), consumed by
    the parent repo's ``scripts/threet_pasko_benchmark.py``. Skips positions
    already in the stage-2 file (resumable).
    """
    done = bm._load_jsonl_by_key(_STAGE2_FILE)
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


def _export_rollout_jobs(unique: dict[str, dict],
                         min_seed: int | None = None, max_seed: int | None = None) -> Path:
    """Write the positions Pass 3 would roll out, with all rollout inputs.

    Same job shape as ``benchmark_money._export_rollout_jobs``, consumed by the
    parent repo's ``scripts/rollout_pasko_benchmark.py``. Skips positions already
    in the stage-3 file (resumable).
    """
    done = bm._load_jsonl_by_key(_STAGE3_FILE)
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
        "variant": "paskogammon",
        "starting_board": list(PASKO_STARTING_BOARD),
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
    # gzip sibling (read transparently by bm._read_dataset on a fresh clone).
    Path(f"{dataset_path}.gz").write_bytes(gzip.compress(dataset_path.read_bytes(), compresslevel=9))
    _log(f"Wrote {len(decisions)} positions ({n_decisions} decisions) to {dataset_path} (+ .gz)")
    _log(f"  checker={n_checker}  cube: {n_cube_decisions} decisions "
         f"(double={n_double}, take={n_take}) over {n_cube_entries} positions  tiers={tier_counts}")
    return meta


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
    """Build the Paskogammon benchmark data set (three adaptive-precision passes).

    Pass 1: simulate ``n_games`` Sage-3P vs Sage-3P Paskogammon games, capturing
    3-ply analytics for every real decision (optionally writing an XG-import
    ``.txt`` per game). Pass 2: re-evaluate decisions whose 3-ply best-vs-2nd-best
    gap < 0.05 at 3T. Pass 3: roll out (1,296 paths, VR, 3-ply checker + cube)
    decisions whose best-available gap < 0.02.

    Crash-safe and resumable; ``stages`` / ``rollout_mode`` / ``threet_mode``
    behave exactly as in :func:`benchmark_money.build_benchmark_data`. Returns the
    dataset metadata dict.
    """
    dataset_path = Path(dataset_path)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    stages = set(stages)
    _log(f"=== build_benchmark_data (paskogammon): n_games={n_games} seed={seed} "
         f"n_threads={n_threads} workers={workers} write_txt={write_txt} "
         f"rollout_mode={rollout_mode} threet_mode={threet_mode} stages={sorted(stages)} ===")

    # Pass 1 - self-play + 3P capture.
    if "pass1" in stages:
        _run_pass1(n_games, seed, write_txt, workers)
    all_decisions = _load_stage1_decisions()
    if not all_decisions:
        _log("No stage-1 decisions on disk -- run the 'pass1' stage first.")
        return {}
    unique = bm._unique_by_key(all_decisions)
    _log(f"Captured {len(all_decisions)} decisions ({len(unique)} unique positions).")

    # Pass 2 - 3T refinement (compute or export).
    if "pass2" in stages:
        if threet_mode == "export":
            for key, refined in bm._load_jsonl_by_key(_STAGE2_FILE).items():
                if key in unique:
                    unique[key] = bm._apply_refined(unique[key], refined, TIER_3T)
            _export_threet_jobs(unique, export_min_seed, export_max_seed)
        else:
            analyzer_3t = bm._make_3t_analyzer(n_threads)
            unique = bm._run_refinement_pass(
                unique, analyzer_3t, _STAGE2_FILE, THREE_T_GAP, TIER_3T, "Pass 2/3 (3T refine)")
    else:
        done2 = bm._load_jsonl_by_key(_STAGE2_FILE)
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
            for key, refined in bm._load_jsonl_by_key(_STAGE3_FILE).items():
                if key in unique:
                    unique[key] = bm._apply_refined(unique[key], refined, TIER_ROLLOUT)
            _export_rollout_jobs(unique, export_min_seed, export_max_seed)
        else:
            analyzer_ro = bm._make_rollout_analyzer(n_threads)
            unique = bm._run_refinement_pass(
                unique, analyzer_ro, _STAGE3_FILE, ROLLOUT_GAP, TIER_ROLLOUT,
                "Pass 3/3 (full rollout to SE)", reeval_fn=bm._reeval_decision_rollout)
    else:
        done3 = bm._load_jsonl_by_key(_STAGE3_FILE)
        for key, refined in done3.items():
            if key in unique:
                unique[key] = bm._apply_refined(unique[key], refined, TIER_ROLLOUT)
        pending3 = sum(1 for key, dec in unique.items()
                       if key not in done3 and bm._gap_for(dec) < ROLLOUT_GAP)
        _log(f"Pass 3 (rollout) not run: {pending3} positions want rollout "
             f"(current-tier gap < {ROLLOUT_GAP}); {len(done3)} already done. "
             f"Run with --stages pass3.")

    return _assemble_dataset(unique, dataset_path, n_games, seed)


# ===========================================================================
# Scoring - benchmark_pr (money bot interface, pasko paths)
# ===========================================================================


def benchmark_pr(
    bot: BenchmarkBot,
    dataset_path: Path | str | None = None,
    n_threads: int = 1,
    label: str = "bot",
    cache_path: Path | str | None = None,
    progress: bool = True,
    max_seed: int | None = None,
) -> dict:
    """Score ``bot`` against the Paskogammon benchmark; return the PR breakdown.

    Thin wrapper over :func:`benchmark_money.benchmark_pr` -- the decisions carry
    money semantics, so the scoring machinery is reused verbatim; only the dataset
    and per-bot resume-cache locations differ.
    """
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET
    if cache_path is None:
        _SCORES_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _SCORES_DIR / f"{label}.jsonl"
    return bm.benchmark_pr(
        bot, dataset_path=dataset_path, n_threads=n_threads, label=label,
        cache_path=cache_path, progress=progress, max_seed=max_seed,
    )


# ===========================================================================
# CLI
# ===========================================================================


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the Paskogammon benchmark data set")
    p_build.add_argument("--n-games", type=int, default=100, help="Games to simulate")
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
        bm._print_report(result)


if __name__ == "__main__":
    _main()
