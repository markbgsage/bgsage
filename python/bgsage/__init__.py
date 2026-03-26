"""bgsage — the Open Sage bot engine, a neural-network backgammon engine.

Provides position analysis (checker play and cube decisions) at multiple
evaluation levels: raw NN (1-ply), N-ply search, and Monte Carlo rollout.

Quick start::

    from bgsage import BgBotAnalyzer, STARTING_BOARD

    analyzer = BgBotAnalyzer()
    result = analyzer.checker_play(STARTING_BOARD, 3, 1)
    for m in result.moves[:3]:
        print(f"{m.equity:+.3f}  {m.probs.win:.1%}")
"""

from .analyzer import BgBotAnalyzer, RolloutCancelled, create_analyzer
from .batch import PositionEval, batch_checker_play, batch_cube_action, batch_evaluate, batch_post_move_evaluate
from .board import (
    STARTING_BOARD,
    can_double_match,
    check_game_over,
    classify_game_plan,
    classify_game_plans,
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
    GamePlanResult,
    GameStats,
    MatchInfo,
    MoveAnalysis,
    PostMoveAnalysis,
    Probabilities,
)
from .gnubg import GnuBgAnalyzer
from .weights import MODELS, PRODUCTION_MODEL, WeightConfig, WeightConfigPair, default_weights

# Re-export TrialEvalConfig from the C++ bindings for convenient access
try:
    import bgbot_cpp as _cpp
    TrialEvalConfig = _cpp.TrialEvalConfig
except (ImportError, AttributeError):
    TrialEvalConfig = None

__all__ = [
    "BgBotAnalyzer",
    "RolloutCancelled",
    "GnuBgAnalyzer",
    "batch_checker_play",
    "batch_cube_action",
    "batch_evaluate",
    "batch_post_move_evaluate",
    "create_analyzer",
    "WeightConfig",
    "WeightConfigPair",
    "default_weights",
    "PRODUCTION_MODEL",
    "MODELS",
    "Probabilities",
    "MoveAnalysis",
    "PostMoveAnalysis",
    "CheckerPlayResult",
    "CubeActionResult",
    "GamePlanResult",
    "GameStats",
    "MatchInfo",
    "PositionEval",
    "STARTING_BOARD",
    "flip_board",
    "possible_moves",
    "possible_single_die_moves",
    "check_game_over",
    "classify_game_plan",
    "classify_game_plans",
    "is_race",
    "is_crashed",
    "invert_probs",
    "can_double_match",
    "TrialEvalConfig",
]
