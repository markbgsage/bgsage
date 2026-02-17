"""bgsage — a neural-network backgammon engine.

Provides position analysis (checker play and cube decisions) at multiple
evaluation levels: raw NN (0-ply), N-ply search, and Monte Carlo rollout.

Quick start::

    from bgsage import BgBotAnalyzer, STARTING_BOARD

    analyzer = BgBotAnalyzer()
    result = analyzer.checker_play(STARTING_BOARD, 3, 1)
    for m in result.moves[:3]:
        print(f"{m.equity:+.3f}  {m.probs.win:.1%}")
"""

from .analyzer import BgBotAnalyzer, create_analyzer
from .board import (
    STARTING_BOARD,
    check_game_over,
    classify_game_plan,
    flip_board,
    invert_probs,
    is_crashed,
    is_race,
    possible_moves,
    possible_single_die_moves,
)
from .types import (
    CheckerPlayResult,
    CubeActionResult,
    GameStats,
    MoveAnalysis,
    Probabilities,
)
from .weights import WeightConfig

__all__ = [
    "BgBotAnalyzer",
    "create_analyzer",
    "WeightConfig",
    "Probabilities",
    "MoveAnalysis",
    "CheckerPlayResult",
    "CubeActionResult",
    "GameStats",
    "STARTING_BOARD",
    "flip_board",
    "possible_moves",
    "possible_single_die_moves",
    "check_game_over",
    "classify_game_plan",
    "is_race",
    "is_crashed",
    "invert_probs",
]
