"""Typed data structures for the bgsage backgammon engine."""

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

    def to_list(self) -> list[float]:
        return [self.win, self.gammon_win, self.backgammon_win,
                self.gammon_loss, self.backgammon_loss]

    @classmethod
    def from_list(cls, probs: list[float]) -> Probabilities:
        return cls(*probs[:5])

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
    eval_level: str                      # "0-ply", "1-ply", "2-ply", ..., "Rollout"
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
    equity_dt: float            # Double / Take equity
    equity_dp: float            # Double / Pass equity (+1.0 money game, MET-based for match)
    should_double: bool
    should_take: bool
    optimal_equity: float
    optimal_action: str         # "No Double", "Double/Take", "Double/Pass"
    eval_level: str
    cubeless_se: float | None = None  # Rollout standard error


@dataclass
class PostMoveAnalysis:
    """Result of evaluating a post-move position (right before the opponent's turn).

    Probabilities are from the perspective of the player who just moved.
    """

    probs: Probabilities        # Post-move cubeless probabilities
    cubeless_equity: float      # Cubeless equity
    cubeful_equity: float       # Cubeful equity (Janowski), or same as cubeless if cubeful=False
    eval_level: str             # "0-ply", "1-ply", "2-ply", ..., "Rollout"


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
