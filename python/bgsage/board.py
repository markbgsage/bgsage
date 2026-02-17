"""Board utilities — thin wrappers around bgbot_cpp functions.

These are importable without creating an analyzer, useful for game logic,
move generation, and position classification.
"""

from __future__ import annotations

import bgbot_cpp

#: Standard backgammon starting position.
STARTING_BOARD: list[int] = [
    0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,
]


def flip_board(board: list[int]) -> list[int]:
    """Flip a board to the other player's perspective."""
    return bgbot_cpp.flip_board(board)


def possible_moves(board: list[int], die1: int, die2: int) -> list[list[int]]:
    """Return all legal resulting board positions for the given dice.

    Each element is a 26-element board representing one distinct final position.
    Returns an empty list when no legal moves exist (e.g. dancing on the bar).
    """
    return [list(b) for b in bgbot_cpp.possible_moves(board, die1, die2)]


def possible_single_die_moves(
    board: list[int], die: int
) -> list[dict[str, object]]:
    """Return legal single-die moves for click-to-move UI.

    Each dict has keys: ``from`` (int), ``to`` (int), ``board`` (list[int]).
    """
    return bgbot_cpp.possible_single_die_moves(board, die)


def check_game_over(board: list[int]) -> int:
    """Check if the game is over.

    Returns:
        0 if not over; +1/-1 single win/loss; +2/-2 gammon; +3/-3 backgammon.
        Positive means the player whose perspective the board is in has won.
    """
    return bgbot_cpp.check_game_over(board)


def classify_game_plan(board: list[int]) -> str:
    """Classify the game plan for the player on roll.

    Returns one of: ``"purerace"``, ``"racing"``, ``"attacking"``,
    ``"priming"``, ``"anchoring"``.
    """
    gp = bgbot_cpp.classify_game_plan(board)
    return gp.name.lower() if hasattr(gp, "name") else str(gp).lower()


def is_race(board: list[int]) -> bool:
    """Return True if contact is broken (pure race position)."""
    return bgbot_cpp.is_race(board)


def is_crashed(board: list[int]) -> bool:
    """Return True if the position is classified as crashed."""
    return bgbot_cpp.is_crashed(board)


def invert_probs(probs: list[float]) -> list[float]:
    """Invert probabilities from one player's perspective to the other.

    ``[P(win), P(gw), P(bw), P(gl), P(bl)]`` becomes
    ``[1-P(win), P(gl), P(bl), P(gw), P(bw)]``.
    """
    return [
        1.0 - probs[0],
        probs[3],
        probs[4],
        probs[1],
        probs[2],
    ]
