# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Typed data structures for the Open Sage bot engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchInfo:
    """Match state from the player on roll's perspective.

    When away1=0 and away2=0, the game is a money (unlimited) game.
    """

    away1: int = 0           # Points player needs to win (0 = money game)
    away2: int = 0           # Points opponent needs to win (0 = money game)
    is_crawford: bool = False

    @property
    def is_money(self) -> bool:
        return self.away1 == 0 and self.away2 == 0

    @property
    def is_post_crawford(self) -> bool:
        return not self.is_crawford and (self.away1 == 1 or self.away2 == 1)


@dataclass
class Probabilities:
    """Five probability outputs from the neural network.

    All probabilities are from the perspective of the player who just moved
    (post-move, pre-opponent-roll).
    """

    win: float              # P(any win) = P(single) + P(gammon) + P(backgammon)
    gammon_win: float       # P(gammon or backgammon win)
    backgammon_win: float   # P(backgammon win only)
    gammon_loss: float      # P(gammon or backgammon loss)
    backgammon_loss: float  # P(backgammon loss only)

    def __post_init__(self) -> None:
        # NN outputs and rollout estimates can produce slightly inconsistent
        # nested probabilities (e.g. P(gammon_win) marginally above P(win))
        # due to estimation noise. Clamp so downstream callers always see
        # backgammon ≤ gammon ≤ (win or 1-win).
        if self.gammon_win > self.win:
            self.gammon_win = self.win
        if self.backgammon_win > self.gammon_win:
            self.backgammon_win = self.gammon_win
        loss = 1.0 - self.win
        if self.gammon_loss > loss:
            self.gammon_loss = loss
        if self.backgammon_loss > self.gammon_loss:
            self.backgammon_loss = self.gammon_loss

    def to_list(self) -> list[float]:
        return [self.win, self.gammon_win, self.backgammon_win,
                self.gammon_loss, self.backgammon_loss]

    @classmethod
    def from_list(cls, probs: list[float]) -> Probabilities:
        return cls(*probs[:5])

    @staticmethod
    def clamp_list(probs: list[float]) -> list[float]:
        """Return a new 5-element list with the nested-probability invariants enforced.

        Same logic as :meth:`__post_init__`, but operates on a raw list for
        callers that haven't wrapped their probs in a :class:`Probabilities`.
        """
        win, gw, bw, gl, bl = probs[:5]
        if gw > win:
            gw = win
        if bw > gw:
            bw = gw
        loss = 1.0 - win
        if gl > loss:
            gl = loss
        if bl > gl:
            bl = gl
        return [win, gw, bw, gl, bl]

    @property
    def equity(self) -> float:
        """Cubeless equity derived from probabilities."""
        return (2.0 * self.win - 1.0
                + self.gammon_win - self.gammon_loss
                + self.backgammon_win - self.backgammon_loss)


@dataclass
class MoveAnalysis:
    """Analysis of a single candidate move."""

    board: list[int]                     # 26-element post-move board
    equity: float                        # Cubeful equity (or cubeless if cubeful=False)
    cubeless_equity: float               # Cubeless equity
    probs: Probabilities                 # Post-move probabilities
    equity_diff: float                   # Difference from best move (0.0 for best)
    eval_level: str                      # "1-ply", "2-ply", "3-ply", ..., "Rollout"
    player_game_plan: str | None = None  # Game plan after this move (opt-in)
    opponent_game_plan: str | None = None
    std_error: float | None = None       # Rollout standard error
    prob_std_errors: list[float] | None = None  # Per-probability standard errors


@dataclass
class CheckerPlayResult:
    """Result of checker play analysis."""

    moves: list[MoveAnalysis]   # Sorted best-first by equity
    board: list[int]            # Original pre-move board
    die1: int
    die2: int
    eval_level: str


@dataclass
class CubeActionResult:
    """Result of cube action analysis."""

    probs: Probabilities        # Pre-roll cubeless probabilities
    cubeless_equity: float
    equity_nd: float            # No Double / Take equity
    equity_dt: float            # Double / Take equity (or Double / Beaver if is_beaver)
    equity_dp: float            # Double / Pass equity (+1.0 money game, MET-based for match)
    should_double: bool
    should_take: bool
    optimal_equity: float
    optimal_action: str         # "No Double", "Double/Take", "Double/Pass", "Double/Beaver"
    eval_level: str
    is_beaver: bool = False     # True if opponent would beaver (equity_dt = DB equity)
    cubeless_se: float | None = None  # Rollout cubeless standard error
    equity_nd_se: float | None = None  # Rollout ND cubeful standard error
    equity_dt_se: float | None = None  # Rollout DT cubeful standard error
    details: dict | None = None  # 2-ply details {"nd": [...], "dt": [...]} (only when incl_2ply_details=True)


@dataclass
class RollEquity:
    """Equity of one dice roll's best checker play, an input to luck.

    ``equity`` is the cubeful equity after the best play for this roll, from the
    perspective of the player on roll. ``weight`` is 1 for doubles and 2 for
    non-doubles (each non-double roll happens two ways among the 36 dice
    combinations), so a weighted average over all rolls is the expected equity.
    """

    die1: int
    die2: int
    equity: float
    weight: int


@dataclass
class LuckResult:
    """How lucky an actual roll was, in equity units, from the roller's view.

    ``luck = actual_equity - average_equity``: the equity of the best play with
    the roll that happened, minus the weight-averaged equity over every possible
    roll from the same position. Positive means the roll was lucky (it beats an
    average roll), negative means unlucky; over many rolls luck averages to zero.
    """

    luck: float
    actual_equity: float        # Equity of the best play with the roll that happened
    average_equity: float       # Weight-averaged equity over all possible rolls
    ply: int                    # Effective ply of the per-roll equities
    level_label: str            # Human label for ``ply`` (e.g. "2-ply")
    per_roll: list[RollEquity]  # Rolls considered (doubles excluded for an opening roll)


@dataclass
class PostMoveAnalysis:
    """Result of evaluating a post-move position (right before the opponent's turn).

    Probabilities are from the perspective of the player who just moved.
    """

    probs: Probabilities        # Post-move cubeless probabilities
    cubeless_equity: float      # Cubeless equity
    cubeful_equity: float       # Cubeful equity (Janowski), or same as cubeless if cubeful=False
    eval_level: str             # "1-ply", "2-ply", "3-ply", ..., "Rollout"
    cubeless_se: float | None = None  # Rollout standard error of cubeless equity
    cubeful_se: float | None = None   # Rollout standard error of cubeful equity


@dataclass
class GamePlanResult:
    """Game plan classification for a position."""

    player: str     # "purerace", "racing", "attacking", "priming", "anchoring"
    opponent: str   # Same values, from the opponent's perspective


@dataclass
class GameStats:
    """Statistics from simulated games."""

    n_games: int
    wins: int
    gammons: int
    backgammons: int
    ppg: float
