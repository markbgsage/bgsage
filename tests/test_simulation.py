"""Rules tests plus real native-engine seeded transcript replay."""
from types import SimpleNamespace as NS
import pytest
from bgsage import BgBotAnalyzer, GameConfig, STARTING_BOARD, simulate_game, simulate_match
from bgsage import simulation as sim


class Policy:
    def __init__(self, double=False, take=True):
        self.double, self.take, self.calls = double, take, []

    def cube_action(self, board, **context):
        self.calls.append(('cube', context))
        return NS(should_double=self.double, should_take=self.take)

    def checker_play(self, board, *dice, **context):
        self.calls.append(('checker', context))
        return NS(moves=[NS(board=sim.possible_moves(board, *dice)[0])])


def test_real_seeded_game_replay():
    a = BgBotAnalyzer(parallel_threads=1)
    result = simulate_game(seed=173, players=(a, a))
    assert result == simulate_game(seed=173, players=(a, a))
    assert result['turns'][0]['cube_action'] is None
    assert result['turns'][0]['dice'][0] != result['turns'][0]['dice'][1]
    for turn in result['turns']:
        if turn['post_board'] is not None:
            assert turn['post_board'] in sim.possible_moves(turn['board'], *turn['dice'])
    assert result['points'] >= 1 and result['termination'] in ('pass', 'bearoff')


def test_standard_opening_can_start_either_player():
    players = (Policy(), Policy())
    results = [simulate_game(seed=seed, players=players) for seed in range(8)]
    assert {r['start_player'] for r in results} == {1, 2}
    for r in results:
        assert r['turns'][0]['cube_action'] is None


def test_pass_uses_undoubled_cube_and_responder_policy():
    proposer, responder = Policy(double=True), Policy(take=False)
    result = simulate_game(players=(proposer, responder),
        config=GameConfig(start_board=tuple(STARTING_BOARD), cube_value=4, cube_owner='player'))
    assert (result['winner'], result['points'], result['termination']) == (1, 4, 'pass')
    assert result['turns'][0]['dice'] is None
    assert not any(k == 'checker' for k, _ in proposer.calls)
    assert responder.calls[0][1]['cube_owner'] == 'player'


def test_take_flips_ownership_and_never_redoubles_nonowner():
    p1, p2 = Policy(double=True), Policy(double=True)
    result = simulate_game(players=(p1, p2),
        config=GameConfig(start_board=tuple(STARTING_BOARD), max_cube_value=4))
    turns = result['turns']
    assert turns[0]['checker_cube_value'] == 2
    assert turns[0]['checker_cube_owner'] == 'opponent'
    assert turns[1]['cube_owner'] == 'player'
    assert turns[1]['checker_cube_value'] == 4
    assert all(t['cube_action'] is None for t in turns[2:])


@pytest.mark.parametrize('jacoby,multiplier', [(True, 1), (False, 3)])
def test_jacoby_only_when_undoubled(monkeypatch, jacoby, multiplier):
    # Keep native legality; isolate terminal scoring only.
    original = sim.check_game_over
    monkeypatch.setattr(sim, 'check_game_over', lambda board: 0 if board == STARTING_BOARD else 3)
    result = simulate_game(players=(Policy(), Policy()),
        config=GameConfig(start_board=tuple(STARTING_BOARD), jacoby=jacoby))
    assert result['points'] == multiplier
    monkeypatch.setattr(sim, 'check_game_over', original)


def test_crawford_once_then_post_crawford_and_score_capping(monkeypatch):
    calls = []
    def game(**kwargs):
        c = kwargs['config']; calls.append(c)
        # 0-0 -> 4-0 -> 4-1 (Crawford) -> 4-5 (post-Crawford)
        winner, points = [(1, 4), (2, 1), (2, 8)][len(calls)-1]
        return {'winner': winner, 'points': points}
    monkeypatch.setattr(sim, 'simulate_game', game)
    result = simulate_match(match_length=5, players=(Policy(), Policy()))
    assert [c.is_crawford for c in calls] == [False, True, False]
    assert [(c.score1, c.score2) for c in calls] == [(0, 0), (4, 0), (4, 1)]
    assert result['final_scores'] == [4, 5]
    assert result['games'][-1]['points'] == 8
    assert result['games'][-1]['points_awarded'] == 4
    assert all(not c.jacoby for c in calls)


def test_one_point_match_is_crawford():
    result = simulate_match(match_length=1, players=(Policy(double=True), Policy(double=True)))
    assert len(result['games']) == 1 and result['games'][0]['is_crawford']
    assert all(t['cube_action'] is None for t in result['games'][0]['turns'])


def test_match_total_turn_budget(monkeypatch):
    def game(**kwargs):
        return {'winner': 1, 'points': 1, 'turns': [None] * 3}
    monkeypatch.setattr(sim, 'simulate_game', game)
    with pytest.raises(sim.SimulationLimitExceeded, match='total turn limit'):
        simulate_match(match_length=5, players=(Policy(), Policy()), max_total_turns=3)


def test_custom_start_mover_and_crawford_context():
    p1, p2 = Policy(double=True), Policy(double=True)
    result = simulate_game(players=(p1, p2), config=GameConfig(
        start_board=tuple(STARTING_BOARD), player_on_roll=2,
        match_length=5, score1=4, score2=2, is_crawford=True))
    assert result['start_player'] == 2
    assert result['turns'][0]['away1'] == 3 and result['turns'][0]['away2'] == 1
    assert result['turns'][1]['away1'] == 1 and result['turns'][1]['away2'] == 3
    assert all(t['cube_action'] is None and not t['jacoby'] and not t['beaver'] for t in result['turns'])


def test_cancellation_limit_illegal_policy():
    with pytest.raises(sim.SimulationCancelled):
        simulate_game(cancel=lambda: True)
    with pytest.raises(sim.SimulationLimitExceeded):
        simulate_game(players=(Policy(), Policy()), config=GameConfig(max_turns=1))
    bad = NS(checker_play=lambda *a, **k: NS(moves=[NS(board=[0]*26)]))
    with pytest.raises(ValueError, match='illegal checker'):
        simulate_game(players=(bad, bad))


@pytest.mark.parametrize('kwargs', [dict(seed=True), dict(seed=-1)])
def test_seed_validation(kwargs):
    with pytest.raises(ValueError):
        simulate_game(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(cube_value=3), dict(max_cube_value=3),
    dict(start_board=[0]*26), dict(start_board=[16]+[0]*25), dict(score1=1),
    dict(is_crawford=True), dict(player_on_roll=0), dict(max_turns=True)])
def test_context_validation(kwargs):
    with pytest.raises(ValueError):
        GameConfig(**kwargs)
