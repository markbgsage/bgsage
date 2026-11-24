# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Run luck-adjusted head-to-head money games with the Open Sage engine."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from hashlib import sha256
from itertools import islice
import json
from math import fsum, isfinite, sqrt
from multiprocessing import get_context
import os
from pathlib import Path
from random import Random
from statistics import NormalDist, covariance, mean, stdev
import sys
from time import perf_counter, process_time, thread_time

import bgbot_cpp

from .analyzer import BgBotAnalyzer
from .board import STARTING_BOARD, flip_board
from .gnubg import GNUBG_PLIES_BY_LEVEL, GnuBgPolicy, gnubg_version, resolve_gnubg_cli
from .simulation import GameConfig, SimulationLimitExceeded, simulate_game
from .weights import PRODUCTION_MODEL
from .xg_h2h import XG_LEVELS, play_xg_games


_LEVEL_ALIASES = {
    "1p": "1ply", "2p": "2ply", "3p": "3ply", "4p": "4ply",
    "1t": "truncated1", "2t": "truncated2", "3t": "truncated3",
    "r": "rollout",
}
_BOT_NAMES = ("sage", "gnubg", "xg")
_ROLLS = {(lo, hi) for hi in range(1, 7) for lo in range(1, hi + 1)}
_OPENING_ROLLS = sorted((lo, hi) for lo, hi in _ROLLS if lo != hi)
_NORMAL_95 = NormalDist().inv_cdf(0.975)


def _level_name(level: str) -> str:
    level = level.lower()
    return _LEVEL_ALIASES.get(level, level)


class _LuckScorer:
    """Score chance events using the ND roll table of a fixed 2-ply analyzer."""

    def __init__(self, reference, config: GameConfig):
        self.reference = reference
        self.config = config
        self.opening = None

    def _table(self, board, cube_value, cube_owner):
        result = self.reference.cube_action(
            board, cube_value=cube_value, cube_owner=cube_owner,
            jacoby=self.config.jacoby, beaver=False,
            max_cube_value=self.config.max_cube_value, incl_2ply_details=True,
        )
        rows = (result.details or {}).get("nd", [])
        scores = {
            tuple(sorted((row["die1"], row["die2"]))): row["cubeful_equity"]
            for row in rows
        }
        if (len(rows) != len(_ROLLS) or scores.keys() != _ROLLS
                or not all(isfinite(value) for value in scores.values())):
            raise ValueError("Luck scoring requires a complete, finite 21-roll ND table")
        return scores

    def _opening_table(self):
        a_board = list(STARTING_BOARD)
        b_board = flip_board(a_board)
        a_scores = self._table(a_board, 1, "centered")
        b_scores = a_scores if a_board == b_board else self._table(b_board, 1, "centered")
        # Each of the 15 non-doubles can be rolled by either opener, with
        # equal probability. Convert B's branch to A's perspective first.
        values = {(1, roll): a_scores[roll] for roll in _OPENING_ROLLS}
        values.update({(2, roll): -b_scores[roll] for roll in _OPENING_ROLLS})
        baseline = fsum(values.values()) / len(values)
        return {event: value - baseline for event, value in values.items()}

    def score(self, game: dict) -> float:
        if self.opening is None:
            self.opening = self._opening_table()
        opening_turn, *rest = game["turns"]
        opening_roll = tuple(sorted(opening_turn["dice"]))
        deltas = [self.opening[(opening_turn["player"], opening_roll)]]
        for turn in rest:
            if turn["dice"] is None:  # A cube pass occurs before the roll.
                break
            cube = turn["checker_cube_value"]
            scores = self._table(turn["board"], cube, turn["checker_cube_owner"])
            baseline = fsum(value * (1 if lo == hi else 2)
                            for (lo, hi), value in scores.items()) / 36
            roll = tuple(sorted(turn["dice"]))
            sign = 1 if turn["player"] == 1 else -1
            deltas.append(sign * cube * (scores[roll] - baseline))
        return fsum(deltas)


class _GameRunner:
    def __init__(self, player_a, player_b, config, beta, bot_a="sage", bot_b="sage", gnubg_cli=None,
                 *, xg_policy=None):
        def create_player(bot, level):
            if bot == "xg":
                return xg_policy
            if bot == "gnubg":
                return GnuBgPolicy(level, gnubg_cli)
            return BgBotAnalyzer(eval_level=level, parallel_threads=1)

        self.players = (create_player(bot_a, player_a), create_player(bot_b, player_b))
        self.scorer = _LuckScorer(BgBotAnalyzer(eval_level="2ply", parallel_threads=1), config)
        self.config = config
        self.beta = beta
        self.cpu_clock = thread_time if xg_policy is not None else process_time

    def play(self, seed: int) -> dict:
        started = self.cpu_clock()
        wall_start = perf_counter()
        try:
            game = simulate_game(seed=seed, config=self.config, players=self.players)
        except SimulationLimitExceeded as exc:
            raise SimulationLimitExceeded(f"Seed {seed}: {exc}") from exc
        wall_played = perf_counter()
        played = self.cpu_clock()
        luck = self.scorer.score(game)
        scored = self.cpu_clock()
        wall_scored = perf_counter()
        raw = game["points"] if game["winner"] == 1 else -game["points"]
        return {
            "seed": seed, "start_player": game["start_player"],
            "winner": game["winner"], "win_type": game["win_type"],
            "termination": game["termination"], "cube_at_end": game["cube_at_end"],
            "rolls": sum(turn["dice"] is not None for turn in game["turns"]),
            "raw_points": raw, "luck_points": luck, "adjusted_points": raw - self.beta * luck,
            "play_cpu_seconds": played - started, "luck_cpu_seconds": scored - played,
            "play_elapsed_seconds": wall_played - wall_start,
            "luck_elapsed_seconds": wall_scored - wall_played,
        }


_worker: _GameRunner | None = None


def _initialize_worker(*args):
    global _worker
    _worker = _GameRunner(*args)


def _play_in_worker(seed):
    assert _worker is not None
    return _worker.play(seed)


def _available_cpus() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _worker_count(workers: int | None, games: int, *, uses_xg: bool = False) -> int:
    if workers is None:
        workers = _available_cpus()
    elif type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer or None for automatic selection")
    if sys.platform == "win32" and not uses_xg:
        workers = min(workers, 61)  # ProcessPoolExecutor limit on Windows.
    return min(workers, games)


def _parallel_games(pool, seeds, workers, *, play=_play_in_worker):
    remaining = enumerate(seeds)
    pending = {pool.submit(play, seed): index
               for index, seed in islice(remaining, 2 * workers)}
    results = {}
    while pending:
        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            results[pending.pop(future)] = future.result()
        for index, seed in islice(remaining, len(finished)):
            pending[pool.submit(play, seed)] = index
    return [results[index] for index in range(len(results))]


def _game_seeds(seed: int, games: int):
    # simulate_game currently accepts 31-bit seeds. Avoid collisions within a
    # run and preserve the seed prefix when rerunning at a larger game count.
    rng = Random(seed)
    used = set()
    while len(used) < games:
        game_seed = rng.getrandbits(31)
        if game_seed not in used:
            used.add(game_seed)
            yield game_seed


def _estimate(values):
    average = mean(values)
    sd = stdev(values)
    se = sd / sqrt(len(values))
    return {"mean_ppg": average, "sd_ppg": sd, "se_ppg": se,
            "ci95_ppg": [average - _NORMAL_95 * se, average + _NORMAL_95 * se]}


def _summarize(records, *, external_bot=False):
    raw = [game["raw_points"] for game in records]
    luck = [game["luck_points"] for game in records]
    adjusted = [game["adjusted_points"] for game in records]
    raw_stats, luck_stats, adjusted_stats = map(_estimate, (raw, luck, adjusted))
    raw_se, adj_se = raw_stats["se_ppg"], adjusted_stats["se_ppg"]
    variance_ratio = (raw_se / adj_se) ** 2 if adj_se > 0 else None
    play_time = fsum(game["play_cpu_seconds"] for game in records)
    score_time = fsum(game["luck_cpu_seconds"] for game in records)
    total_cpu = play_time + score_time
    summary = {
        "raw": raw_stats, "luck": luck_stats, "adjusted": adjusted_stats,
        "raw_luck_sample_covariance": covariance(raw, luck),
        "se_reduction_ppg": raw_se - adj_se,
        "se_reduction_percent": 100 * (1 - adj_se / raw_se) if raw_se > 0 else None,
        "variance_ratio": variance_ratio,
        "play_cpu_seconds": play_time, "luck_cpu_seconds": score_time,
        "estimated_cpu_speedup_at_equal_se": (
            variance_ratio * play_time / total_cpu
            if not external_bot and variance_ratio is not None and total_cpu > 0 else None
        ),
    }
    if "play_elapsed_seconds" in records[0]:
        play_elapsed = fsum(game["play_elapsed_seconds"] for game in records)
        luck_elapsed = fsum(game["luck_elapsed_seconds"] for game in records)
        total_elapsed = play_elapsed + luck_elapsed
        summary.update(
            play_worker_seconds=play_elapsed, luck_worker_seconds=luck_elapsed,
            estimated_worker_time_speedup_at_equal_se=(
                variance_ratio * play_elapsed / total_elapsed
                if variance_ratio is not None and total_elapsed > 0 else None
            ),
        )
    return summary


def run_head_to_head(
    *, player_a: str = "1ply", player_b: str = "2ply", games: int = 100,
    seed: int = 42, workers: int | None = None, beta: float = 1.0, jacoby: bool = True,
    max_cube_value: int = 1024, max_turns: int = 1000, bot_a: str = "sage",
    bot_b: str = "sage", gnubg_cli: str | None = None, xg_profile: str | None = None,
    xg_exe: str | None = None, xg_template: str | None = None,
) -> dict:
    """Run Sage, GNUbg, or XG policies and return results in player A's points."""
    player_a, player_b = _level_name(player_a), _level_name(player_b)
    players = []
    for bot, level in ((bot_a, player_a), (bot_b, player_b)):
        if bot not in _BOT_NAMES:
            raise ValueError(f"Unknown bot {bot!r}. Choose Sage, GNUbg or XG.")
        if level not in _LEVEL_ALIASES.values():
            raise ValueError("Choose an evaluation level from 1P–4P, 1T–3T, or R")
        info = {"bot": bot, "eval_level": level}
        if bot == "gnubg":
            if level not in GNUBG_PLIES_BY_LEVEL:
                raise ValueError("GNUbg requires an evaluation level from 1P to 4P")
            info["native_plies"] = GNUBG_PLIES_BY_LEVEL[level]
        if bot == "xg" and level not in XG_LEVELS:
            raise ValueError("XG requires a level from 1P to 4P or 1T to 3T")
        players.append(info)
    uses_xg = "xg" in (bot_a, bot_b)
    if uses_xg and {bot_a, bot_b} != {"sage", "xg"}:
        raise ValueError("XG head-to-head pairs one XG seat with Sage")
    if uses_xg and not jacoby:
        raise ValueError("XG head-to-head uses the audited Jacoby-on rules")
    if type(games) is not int or not 2 <= games <= (1 << 31):
        raise ValueError("games must be between 2 and 2**31")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isfinite(beta):
        raise ValueError("beta must be finite")
    workers = _worker_count(workers, games, uses_xg=uses_xg)
    config = GameConfig(jacoby=jacoby, max_cube_value=max_cube_value, max_turns=max_turns)
    external_bot = bot_a != "sage" or bot_b != "sage"
    gnubg = None
    if "gnubg" in (bot_a, bot_b):
        gnubg_cli = resolve_gnubg_cli(gnubg_cli)
        gnubg = {"executable": gnubg_cli, "version": gnubg_version(gnubg_cli),
                 "executable_sha256": sha256(Path(gnubg_cli).read_bytes()).hexdigest()}

    started = perf_counter()
    seeds = _game_seeds(seed, games)
    args = (player_a, player_b, config, beta, bot_a, bot_b, gnubg_cli)
    xg = None
    if uses_xg:
        records, xg = play_xg_games(
            seeds, workers, lambda policy: _GameRunner(*args, xg_policy=policy), _parallel_games,
            level=player_a if bot_a == "xg" else player_b,
            profile=xg_profile, executable=xg_exe, template=xg_template, max_cube=max_cube_value,
        )
    elif workers == 1:
        runner = _GameRunner(*args)
        records = [runner.play(game_seed) for game_seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                                 initializer=_initialize_worker, initargs=args) as pool:
            records = _parallel_games(pool, seeds, workers)

    result = {
        "schema_version": 1, "model": PRODUCTION_MODEL,
        "engine_sha256": sha256(Path(bgbot_cpp.__file__).read_bytes()).hexdigest(),
        "player_a": players[0], "player_b": players[1],
        "reference": "sage_2ply_nd", "beta": beta, "seed": seed, "workers": workers,
        "rules": {"jacoby": jacoby, "beaver": False, "max_cube_value": max_cube_value,
                  "max_turns": max_turns},
        "game_count": games, "summary": _summarize(records, external_bot=external_bot), "games": records,
        "wall_seconds": perf_counter() - started,
    }
    if gnubg is not None:
        result["gnubg"] = gnubg
        result["cpu_timing_scope"] = "H2H worker processes only"
    if xg is not None:
        result["xg"] = xg
        result["cpu_timing_scope"] = "H2H game threads only"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    levels = tuple(_LEVEL_ALIASES.values())
    parser.add_argument("--a", dest="player_a", type=_level_name, choices=levels, default="1ply",
                        help="evaluation level for A, whose perspective is reported")
    parser.add_argument("--b", dest="player_b", type=_level_name, choices=levels, default="2ply",
                        help="evaluation level for B")
    parser.add_argument("--bot-a", choices=_BOT_NAMES, default="sage", help="engine for A (default: sage)")
    parser.add_argument("--bot-b", choices=_BOT_NAMES, default="sage", help="engine for B (default: sage)")
    parser.add_argument("--gnubg-cli", help="GNU Backgammon command-line executable")
    parser.add_argument("--xg-profile", help="exact existing XG custom profile name")
    parser.add_argument("--xg-exe", help="path to the installed eXtreme Gammon 2 executable")
    parser.add_argument("--xg-template", help="path to a complete .xg game for Batch Analyze record templates")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int,
                        help="concurrent game streams (default: one per available CPU)")
    parser.add_argument("--beta", type=float, default=1.0, help="fixed multiplier on total game luck")
    parser.add_argument("--jacoby", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cube-value", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=1000)
    args = parser.parse_args()
    try:
        result = run_head_to_head(**vars(args))
    except (ValueError, OSError, RuntimeError, SimulationLimitExceeded) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
