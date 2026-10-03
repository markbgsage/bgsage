# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Seeded, recorded games and matches using the public analysis interface.

Python owns rules and records; legal moves, terminal detection and decisions use
the native engine. No app, network, account or persistence dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from .board import STARTING_BOARD, check_game_over, flip_board, possible_moves


class SimulationCancelled(Exception):
    """A caller cancelled between decisions; no fabricated final score."""


class SimulationLimitExceeded(Exception):
    """A game reached its turn limit without a result."""


def _integer(name, value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')


@dataclass(frozen=True)
class GameConfig:
    """Custom start is a pre-roll board in ``player_on_roll``'s perspective.

    ``start_board=None`` uses a standard opening contest (reroll ties; no cube
    offer before the opening checker play). An explicit board uses ordinary
    dice including doubles. Cube ownership is relative to that initial mover.
    Beavers/raccoons, resignations and automatic opening doubles are excluded.
    """
    start_board: tuple[int, ...] | None = None
    player_on_roll: int = 1
    cube_value: int = 1
    cube_owner: str = 'centered'
    jacoby: bool = True
    max_cube_value: int = 1024
    max_turns: int = 1000
    match_length: int = 0
    score1: int = 0
    score2: int = 0
    is_crawford: bool = False

    def __post_init__(self):
        _integer('player_on_roll', self.player_on_roll, 1, 2)
        _integer('max_turns', self.max_turns, 1, 10000)
        _integer('match_length', self.match_length, 0, 99)
        _integer('cube_value', self.cube_value, 1, 1024)
        _integer('max_cube_value', self.max_cube_value, 2, 1024)
        for name in ('cube_value', 'max_cube_value'):
            value = getattr(self, name)
            if value & (value - 1):
                raise ValueError(f'{name} must be a power of two')
        if self.cube_value > self.max_cube_value:
            raise ValueError('cube_value exceeds max_cube_value')
        if self.cube_owner not in ('centered', 'player', 'opponent'):
            raise ValueError('invalid cube_owner')
        if type(self.jacoby) is not bool or type(self.is_crawford) is not bool:
            raise ValueError('jacoby and is_crawford must be booleans')
        for name in ('score1', 'score2'):
            _integer(name, getattr(self, name), 0, max(0, self.match_length - 1))
        if self.is_crawford and (not self.match_length or
                self.match_length - 1 not in (self.score1, self.score2) or
                self.cube_value != 1 or self.cube_owner != 'centered'):
            raise ValueError('invalid Crawford context')
        if self.start_board is None:
            if self.cube_value != 1 or self.cube_owner != 'centered':
                raise ValueError('a standard opening starts with a centered cube at 1')
        else:
            board = tuple(self.start_board)
            object.__setattr__(self, 'start_board', board)
            if len(board) != 26 or any(type(p) is not int for p in board):
                raise ValueError('start_board must contain 26 integers')
            if board[0] < 0 or board[25] < 0 or (
                    board[25] + sum(max(0, p) for p in board[1:25]) > 15 or
                    board[0] + sum(max(0, -p) for p in board[1:25]) > 15):
                raise ValueError('invalid checker counts')
            if (not (board[25] + sum(max(0, p) for p in board[1:25])) or
                    not (board[0] + sum(max(0, -p) for p in board[1:25])) or
                    check_game_over(list(board))):
                raise ValueError('start_board must be nonterminal')


def _players(players):
    if players is None:
        from .analyzer import BgBotAnalyzer
        players = (BgBotAnalyzer(parallel_threads=1),) * 2
    if len(players) != 2:
        raise ValueError('players must contain two analyzer-compatible policies')
    return players


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise SimulationCancelled('simulation cancelled')


def simulate_game(*, seed: int = 42, config: GameConfig | None = None,
                  players=None, cancel: Callable[[], bool] | None = None) -> dict:
    """Return a JSON-serializable game with every cube/checker decision.

    Policies implement ``checker_play`` and ``cube_action`` like BgBotAnalyzer.
    The responder policy evaluates the doubler's board: ``should_take`` is the
    opponent's response, as in the public cube API. Returned moves are checked
    against native legal moves. Same seed/policies/engine build reproduce dice
    and decisions. Cancellation is checked before each engine call.
    """
    _integer('seed', seed, 0, 2147483647)
    cfg = config or GameConfig()
    players = _players(players)
    rng = random.Random(seed)
    board = list(cfg.start_board or STARTING_BOARD)
    active = cfg.player_on_roll
    owner, cube = cfg.cube_owner, cfg.cube_value
    opening = cfg.start_board is None
    opening_dice = None
    if opening:
        while True:
            _check_cancel(cancel)
            d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
            if d1 != d2:
                break
        active = 1 if d1 > d2 else 2
        if active == 2:
            board = flip_board(board)
        opening_dice = [d1, d2] if active == 1 else [d2, d1]
    turns = []
    start_board = list(board)
    def finish(winner, multiplier, reason):
        if not cfg.match_length and cfg.jacoby and owner == 'centered':
            multiplier = 1
        return {'schema_version': 1, 'seed': seed, 'winner': winner,
                'win_type': {1: 'single', 2: 'gammon', 3: 'backgammon'}[multiplier],
                'termination': reason, 'points': cube * multiplier,
                'cube_at_end': cube, 'start_board': start_board,
                'start_player': turns[0]['player'],
                'start_scores': [cfg.score1, cfg.score2],
                'match_length': cfg.match_length, 'is_crawford': cfg.is_crawford,
                'jacoby': cfg.jacoby and not cfg.match_length, 'beaver': False,
                'max_cube_value': cfg.max_cube_value, 'turns': turns}
    for number in range(cfg.max_turns):
        _check_cancel(cancel)
        scores = (cfg.score1, cfg.score2) if active == 1 else (cfg.score2, cfg.score1)
        context = dict(cube_value=cube, cube_owner=owner,
                       away1=cfg.match_length - scores[0] if cfg.match_length else 0,
                       away2=cfg.match_length - scores[1] if cfg.match_length else 0,
                       is_crawford=cfg.is_crawford,
                       jacoby=cfg.jacoby and not cfg.match_length, beaver=False,
                       max_cube_value=cfg.max_cube_value)
        turn = {'turn': number, 'player': active, 'board': list(board),
                **context, 'cube_action': None, 'dice': None, 'post_board': None}
        turns.append(turn)
        if not opening and not cfg.is_crawford and owner != 'opponent' and cube < cfg.max_cube_value:
            offer = players[active - 1].cube_action(list(board), **context)
            turn['cube_action'] = 'no_double'
            if offer.should_double:
                _check_cancel(cancel)
                response = players[2 - active].cube_action(list(board), **context)
                if not response.should_take:
                    turn['cube_action'] = 'double/pass'
                    return finish(active, 1, 'pass')
                turn['cube_action'] = 'double/take'
                cube *= 2
                owner = 'opponent'  # taker owns it; doubler is still on roll
                context.update(cube_value=cube, cube_owner=owner)
        dice = opening_dice if opening else [rng.randint(1, 6), rng.randint(1, 6)]
        opening = False
        _check_cancel(cancel)
        legal = possible_moves(board, *dice)
        result = players[active - 1].checker_play(list(board), *dice, **context)
        if not result.moves or list(result.moves[0].board) not in legal:
            raise ValueError('policy returned an illegal checker move')
        post = list(result.moves[0].board)
        turn.update(dice=list(dice), post_board=post,
                    checker_cube_value=cube, checker_cube_owner=owner)
        outcome = check_game_over(post)
        if outcome:
            return finish(active if outcome > 0 else 3 - active, abs(outcome), 'bearoff')
        board = flip_board(post)
        active = 3 - active
        owner = {'centered': 'centered', 'player': 'opponent', 'opponent': 'player'}[owner]
    raise SimulationLimitExceeded(f'game exceeded {cfg.max_turns} turns')


def simulate_match(*, match_length: int = 5, seed: int = 42, players=None,
                   max_turns: int = 1000, max_cube_value: int = 1024,
                   max_total_turns: int = 10000,
                   cancel: Callable[[], bool] | None = None) -> dict:
    """Play a complete match from 0-0, with one Crawford game then post-Crawford.

    Game points are raw cube times win multiplier. Scores are capped at the
    target; ``points_awarded`` records the actual score increment separately.
    Games have derived seeds; each can be replayed independently.
    """
    _integer('match_length', match_length, 1, 99)
    _integer('seed', seed, 0, 2147483647)
    _integer('max_total_turns', max_total_turns, 1, 100000)
    # Validate limits before allocating/loading analyzers.
    GameConfig(match_length=match_length, max_turns=max_turns, max_cube_value=max_cube_value)
    players = _players(players)
    scores, games, crawford_done = [0, 0], [], False
    rng = random.Random(seed)
    turns_used = 0
    while max(scores) < match_length:
        _check_cancel(cancel)
        if turns_used >= max_total_turns:
            raise SimulationLimitExceeded('match exceeded total turn limit')
        crawford = not crawford_done and match_length - 1 in scores
        game = simulate_game(seed=rng.randrange(2147483648), players=players,
                             config=GameConfig(match_length=match_length,
                                 score1=scores[0], score2=scores[1], is_crawford=crawford,
                                 jacoby=False, max_turns=min(max_turns, max_total_turns - turns_used),
                                 max_cube_value=max_cube_value),
                             cancel=cancel)
        side = game['winner'] - 1
        game['points_awarded'] = min(game['points'], match_length - scores[side])
        scores[side] += game['points_awarded']
        game['end_scores'] = list(scores)
        games.append(game)
        turns_used += len(game.get('turns', []))
        crawford_done |= crawford
    return {'schema_version': 1, 'seed': seed, 'match_length': match_length,
            'winner': 1 if scores[0] == match_length else 2,
            'final_scores': scores, 'games': games}
