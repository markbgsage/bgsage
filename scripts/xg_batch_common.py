# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Shared plumbing for the XG Batch Rollout pipeline (mark + harvest scripts).

The pipeline gets XG reference numbers for benchmark positions without clicking
through XG thousands of times:

1. ``xg_mark_rollouts.py`` copies the Roller++-batch-analyzed ``.xg`` game files
   and injects batch-rollout marks (``TimeDelayMove`` / ``TimeDelayCube``) on
   exactly the benchmark decisions that have Sage rollout references, writing
   staged files to ``data/.../xg_batch/rollout/``.
2. The user queues that folder once in XG (Batch Rollout dialog) and lets it run.
   XG writes rollout results back into the staged files.
3. ``xg_harvest_results.py`` parses the staged files and appends per-decision
   results to a key-indexed cache in ``data/.../xg_results/`` — separate from
   all Sage score/reference data.

The harvest also has a ``rollerpp`` mode that reads the *existing* Roller++
batch-analysis files directly (no new XG run) to cache XG Roller++ evaluations
for 3T-tier positions.

Matching follows benchmark_pr_xg(_match).py: decisions are located inside a
game file by content — ``(game_number, kind, mover-board, sorted-dice)`` — not
by turn index.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BUILD_DIR = _PROJECT_ROOT / "build"
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "python", _BUILD_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if sys.platform == "win32":
    _cuda_x64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
    if os.path.isdir(_cuda_x64):
        os.add_dll_directory(_cuda_x64)
    if _BUILD_DIR.is_dir():
        os.add_dll_directory(str(_BUILD_DIR))

from bgsage import xg_file  # noqa: E402
from bgsage.board import invert_probs  # noqa: E402


# ---------------------------------------------------------------------------
# Benchmark layout
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkPaths:
    """Where a benchmark keeps its dataset, XG base files, staging and caches."""
    name: str                 # "money" | "match"
    dataset: Path
    xg_base_dir: Path         # Roller++-analyzed .xg files (the mark bases)
    base_pattern: str         # e.g. "seed_{seed}_pp.xg"
    staged_name: str          # e.g. "seed_{seed}.xg" (name used in xg_batch/)
    batch_dir: Path           # staging root (subdir per set)
    results_dir: Path         # xg_results cache dir
    is_match: bool = False

    def base_file(self, seed: int) -> Path:
        return self.xg_base_dir / self.base_pattern.format(seed=seed)

    def xg_level_file(self, level: str, seed: int) -> Path:
        """Batch-analyze .xg file for an XG level ('roller'/'rollerplus'/'rollerpp').

        Money: xg/seed_<N>_{roller,p,pp}.xg. Match: xg_snapshots/{roller,roller_p,
        roller_pp}/match_seed_<N>.xg. 'rollerpp' is the mark base file.
        """
        if self.is_match:
            sub = {"roller": "roller", "rollerplus": "roller_p",
                   "rollerpp": "roller_pp"}[level]
            return self.xg_base_dir.parent / sub / f"match_seed_{seed}.xg"
        suf = {"roller": "_roller", "rollerplus": "_p", "rollerpp": "_pp"}[level]
        return self.xg_base_dir / f"seed_{seed}{suf}.xg"

    def staged_file(self, batch_set: str, seed: int) -> Path:
        return self.batch_dir / batch_set / self.staged_name.format(seed=seed)

    def cache_file(self, batch_set: str) -> Path:
        return self.results_dir / f"{batch_set}.jsonl"


def money_paths() -> BenchmarkPaths:
    data = _PROJECT_ROOT / "data" / "money_benchmark"
    return BenchmarkPaths(
        name="money",
        dataset=data / "benchmark.json",
        xg_base_dir=data / "xg",
        base_pattern="seed_{seed}_pp.xg",
        staged_name="seed_{seed}.xg",
        batch_dir=data / "xg_batch",
        results_dir=data / "xg_results",
    )


def match_paths(length: int = 5) -> BenchmarkPaths:
    data = _PROJECT_ROOT / "data" / "match_benchmark" / f"{length}pt"
    return BenchmarkPaths(
        name="match",
        dataset=data / "benchmark.json",
        xg_base_dir=data / "xg_snapshots" / "roller_pp",
        base_pattern="match_seed_{seed}.xg",
        staged_name="match_seed_{seed}.xg",
        batch_dir=data / "xg_batch",
        results_dir=data / "xg_results",
        is_match=True,
    )


def paths_for(benchmark: str, match_length: int = 5) -> BenchmarkPaths:
    if benchmark == "money":
        return money_paths()
    if benchmark == "match":
        return match_paths(match_length)
    raise ValueError(f"unknown benchmark {benchmark!r}")


# ---------------------------------------------------------------------------
# Dataset + cache IO
# ---------------------------------------------------------------------------


def load_decisions(paths: BenchmarkPaths, tiers: set[str]) -> list[dict]:
    """Benchmark decisions whose tier is in ``tiers``, in dataset order."""
    data = json.loads(paths.dataset.read_text(encoding="utf-8"))
    return [d for d in data["decisions"] if d.get("tier") in tiers]


def load_cache(path: Path) -> dict[str, dict]:
    """Key-indexed contents of a results cache jsonl (empty if absent)."""
    out: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    out[rec["key"]] = rec
    return out


def append_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Record index (content-based decision lookup inside one .xg game file)
# ---------------------------------------------------------------------------


@dataclass
class RecordRef:
    offset: int
    kind: str                 # "checker" | "cube"
    game_number: int


class GameFileIndex:
    """Index all decision records of a temp.xg stream by content.

    Lookup key: ``(game_number, kind, mover_board_tuple, dice_key)`` where
    ``dice_key`` is the sorted dice tuple for checker records and ``None`` for
    cube records. For money files (single game) ``find`` may be called with
    ``game_number=None`` and falls back to a game-agnostic lookup.
    """

    def __init__(self, tempxg: bytes | bytearray):
        self.header_offset: int | None = None
        self._by_full: dict[tuple, int] = {}
        self._by_content: dict[tuple, int] = {}
        game_number = 0
        for off, rtype in xg_file.iter_records(tempxg):
            if rtype == xg_file.TS_HEADER_MATCH:
                if self.header_offset is None:
                    self.header_offset = off
            elif rtype == xg_file.TS_HEADER_GAME:
                game_number = xg_file.parse_game_header(tempxg, off)["game_number"]
            elif rtype == xg_file.TS_MOVE:
                rec = xg_file.parse_move_record(tempxg, off)
                key = ("checker", rec["mover_board"], tuple(sorted(rec["dice"])))
                self._add(game_number, key, off)
            elif rtype == xg_file.TS_CUBE:
                rec = xg_file.parse_cube_record(tempxg, off)
                key = ("cube", rec["mover_board"], None)
                self._add(game_number, key, off)

    def _add(self, game_number: int, key: tuple, off: int) -> None:
        self._by_full.setdefault((game_number,) + key, off)
        self._by_content.setdefault(key, off)

    def find(self, kind: str, board, dice, game_number: int | None = None) -> int | None:
        dice_key = tuple(sorted(dice)) if kind == "checker" else None
        key = (kind, tuple(board), dice_key)
        if game_number is not None:
            return self._by_full.get((game_number,) + key)
        return self._by_content.get(key)


def find_decision_record(index: GameFileIndex, paths: BenchmarkPaths,
                         decision: dict) -> int | None:
    gn = decision.get("game_number") if paths.is_match else None
    return index.find(decision["kind"], decision["board"],
                      decision.get("dice") or (), game_number=gn)


# ---------------------------------------------------------------------------
# Checker-move flag policy
# ---------------------------------------------------------------------------


@dataclass
class FlagPlan:
    indices: list[int] = field(default_factory=list)   # stored-move indices to flag
    sage_present: int = 0                              # Sage candidates found in XG's list
    sage_missing: int = 0                              # Sage candidates XG didn't store
    sage_best_missing: bool = False                    # Sage's #1 not in XG's list
    extra_present: int = 0                             # level picks found in XG's list
    extra_missing: int = 0                             # level picks XG didn't store

    @property
    def bits(self) -> int:
        b = 0
        for i in self.indices:
            b |= 1 << i
        return b


def plan_checker_flags(decision: dict, move_rec: dict, threshold: float = 0.02,
                       min_moves: int = 2, max_moves: int = 4,
                       xg_top: int = 0, extra_boards=None) -> FlagPlan:
    """Choose which of XG's stored moves to mark for rollout.

    Policy: roll every move whose Sage reference equity is within ``threshold``
    of the best move (the "close" alternatives, where a rollout is most
    informative), bounded to [``min_moves``, ``max_moves``] by rank. Sage's
    dataset moves are already sorted best-first, so the within-threshold set is
    the contiguous top of the list.

    ``extra_boards`` (a set of 26-int board tuples) are each evaluation level's
    own chosen move — flag them too so every level's pick has a rollout equity
    and nothing is scored as a mismatch. Optionally also flag XG's own top
    ``xg_top`` stored moves (default 0 — off).

    A chosen move can only be rolled if XG stored it; misses are tracked
    (``sage_missing`` / ``sage_best_missing`` / ``extra_missing``) and reported
    by the mark script.
    """
    plan = FlagPlan()
    stored = {tuple(m["board"]): i for i, m in enumerate(move_rec["moves"])}

    sage_moves = decision.get("moves", [])
    if not sage_moves:
        return plan
    best_eq = sage_moves[0]["equity"]
    n_within = sum(1 for m in sage_moves if best_eq - m["equity"] < threshold)
    n_take = max(min_moves, min(n_within, max_moves))
    n_take = min(n_take, len(sage_moves))

    chosen: set[int] = set()
    for rank in range(n_take):
        b = tuple(sage_moves[rank]["board"])
        idx = stored.get(b)
        if idx is None:
            plan.sage_missing += 1
            if rank == 0:
                plan.sage_best_missing = True
            continue
        plan.sage_present += 1
        chosen.add(idx)
    for i in range(min(xg_top, move_rec["n_moves"])):
        chosen.add(i)
    for b in (extra_boards or ()):
        idx = stored.get(tuple(b))
        if idx is None:
            plan.extra_missing += 1
        else:
            plan.extra_present += 1
            chosen.add(idx)
    plan.indices = sorted(chosen)
    return plan


def recover_checker_pick(decision: dict, err: float, tol: float = 1e-6):
    """A level's chosen move recovered from its Sage-benchmark error.

    The score caches store each level's equity error vs the Sage reference
    (best_eq - chosen_eq); the chosen move is the benchmark move whose equity
    equals best_eq - err. Returns the board tuple, or None if ambiguous (a tie).
    """
    moves = decision.get("moves") or []
    if not moves:
        return None
    target = moves[0]["equity"] - err
    matches = [m for m in moves if abs(m["equity"] - target) < tol]
    return tuple(matches[0]["board"]) if len(matches) == 1 else None


def xg_level_picks(paths: BenchmarkPaths, level: str, seed: int,
                   decisions: list[dict]) -> dict[str, dict]:
    """{key: {'checker': board} | {'cube': (should_double, should_take)}} for one
    XG level's batch-analyze file (the level's #1 move / cube action per decision)."""
    path = paths.xg_level_file(level, seed)
    if not path.exists():
        return {}
    tx = xg_file.XgArchive.load(path).get("temp.xg")
    index = GameFileIndex(tx)
    out: dict[str, dict] = {}
    for d in decisions:
        off = find_decision_record(index, paths, d)
        if off is None:
            continue
        if d["kind"] == "checker":
            rec = xg_file.parse_move_record(tx, off)
            if rec["moves"]:
                best = max(rec["moves"], key=lambda m: m["eval"][6])
                out[d["key"]] = {"checker": tuple(best["board"])}
        else:
            rec = xg_file.parse_cube_record(tx, off)
            out[d["key"]] = {"cube": (rec["flag_double"] > 0,
                                      rec["equity_dt"] <= rec["equity_dp"])}
    return out


# ---------------------------------------------------------------------------
# Result extraction (shared by harvest modes)
# ---------------------------------------------------------------------------


def checker_cache_record(decision: dict, move_rec: dict, xgr: bytes | None,
                         level: str, source: dict) -> dict:
    """Cache entry for a checker decision from a (possibly rolled-out) record."""
    moves = []
    for i, m in enumerate(move_rec["moves"]):
        entry = {
            "board": list(m["board"]),
            "equity": float(m["eval"][6]),
            "probs": xg_file.xg_eval_to_probs(m["eval"]),
            "eval_level": xg_file.player_level_label(m["level"]),
        }
        ri = move_rec["rollout_indices"][i]
        if ri >= 0 and xgr:
            ctx = xg_file.parse_rollout_context(xgr, ri)
            if ctx["rolled"] > 0:  # a 0-game context carries no rollout data
                # XG's checker-move rollout Result1 is from the POST-MOVE
                # perspective (opponent on roll), so it is the negation of the
                # mover-perspective equity Sage stores. Negate the equity and
                # invert the probabilities to match Sage's convention (and XG's
                # own stored N-ply eval, which IS mover-perspective).
                # Verified empirically: mean|-XG - Sage| = 0.008 vs 0.47 raw.
                entry.update({
                    "eval_level": "rollout",
                    "equity": -float(ctx["result_nd"][6]),
                    "probs": invert_probs(xg_file.xg_eval_to_probs(ctx["result_nd"])),
                    "trials": ctx["rolled"],
                    "ci": ctx["ci"],
                    "duration": ctx["duration"],
                })
        moves.append(entry)
    return {
        "key": decision["key"],
        "kind": "checker",
        "level": level,
        "board": list(decision["board"]),
        "dice": list(decision["dice"]),
        "analyze_m": xg_file.player_level_label(move_rec["analyze_m"]),
        "moves": moves,
        "source": source,
    }


def cube_cache_record(decision: dict, cube_rec: dict, xgr: bytes | None,
                      level: str, source: dict) -> dict:
    """Cache entry for a cube decision from a (possibly rolled-out) record."""
    out = {
        "key": decision["key"],
        "kind": "cube",
        "level": level,
        "board": list(decision["board"]),
        "eval_level": xg_file.player_level_label(cube_rec["level"]),
        "equity_nd": cube_rec["equity_nd"],
        "equity_dt": cube_rec["equity_dt"],
        "equity_dp": cube_rec["equity_dp"],
        "probs": xg_file.xg_eval_to_probs(cube_rec["eval_nd"]),
        "probs_dt": xg_file.xg_eval_to_probs(cube_rec["eval_dt"]),
        "should_double": cube_rec["flag_double"] > 0,
        "should_take": cube_rec["equity_dt"] <= cube_rec["equity_dp"],
        "is_beaver": cube_rec["is_beaver"] != 0,
        "source": source,
    }
    ri = cube_rec["rollout_index"]
    if ri >= 0 and xgr:
        ctx = xg_file.parse_rollout_context(xgr, ri)
        if ctx["rolled"] > 0 and ctx["rolled2"] > 0:
            nd, dt = float(ctx["result_nd"][6]), float(ctx["result_dt"][6])
            dp = cube_rec["equity_dp"]
            out.update({
                "eval_level": "rollout",
                "equity_nd": nd,
                "equity_dt": dt,
                "probs": xg_file.xg_eval_to_probs(ctx["result_nd"]),
                "probs_dt": xg_file.xg_eval_to_probs(ctx["result_dt"]),
                # recommendations recomputed from the rollout equities (the
                # pre-rollout flag_double reflects the old eval)
                "should_double": min(dt, dp) > nd,
                "should_take": dt <= dp,
                "trials_nd": ctx["rolled"],
                "trials_dt": ctx["rolled2"],
                "ci_nd": ctx["ci"],
                "ci_dt": ctx["ci2"],
                "duration": ctx["duration"],
            })
    return out


def group_by_seed(decisions: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for d in decisions:
        out.setdefault(d["seed"], []).append(d)
    return out
