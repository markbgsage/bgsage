"""Evaluate a position with Stage 5 and GNUbg side-by-side.

Supports both cube action and checker play analysis, for money games or match play.
Evaluates at 1-ply through 4-ply plus rollout (Stage 5 only), and prints a comparison table.

Usage:
    # Cube action (money game, centered cube)
    python bgsage/scripts/eval_position.py cube \
        --checkers "0,2,3,2,2,1,3,1,0,0,0,0,1,0,-1,-1,0,0,0,-3,-2,-3,-2,-2,-1,0"

    # Cube action (match play: 5-point match, player 3 pts, opp 0 pts)
    python bgsage/scripts/eval_position.py cube \
        --checkers "0,2,3,2,2,1,3,1,0,0,0,0,1,0,-1,-1,0,0,0,-3,-2,-3,-2,-2,-1,0" \
        --match 5 --score 3 0

    # Cube action with player owning cube=2
    python bgsage/scripts/eval_position.py cube \
        --checkers "..." --cube-value 2 --cube-owner player

    # Checker play
    python bgsage/scripts/eval_position.py checker \
        --checkers "0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0" \
        --dice 3 1

    # Checker play (match play)
    python bgsage/scripts/eval_position.py checker \
        --checkers "..." --dice 3 1 --match 5 --score 3 0
"""

import sys
import os
import argparse
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(project_dir)
sys.path.insert(0, os.path.join(project_dir, 'python'))
sys.path.insert(0, os.path.join(root_dir, 'build_msvc'))
sys.path.insert(0, os.path.join(root_dir, 'build'))

cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
if os.path.isdir(cuda_x64):
    os.add_dll_directory(cuda_x64)
for d in [os.path.join(root_dir, 'build_msvc'), os.path.join(root_dir, 'build')]:
    if os.path.isdir(d):
        os.add_dll_directory(d)



# ---------------------------------------------------------------------------
# Cube action analysis
# ---------------------------------------------------------------------------

def run_cube_analysis(checkers, cube_value, cube_owner_str, match_length,
                      player_score, opp_score, is_crawford):
    import bgbot_cpp
    from bgsage.weights import WeightConfig

    w = WeightConfig.default()
    w.validate()

    owner_map = {
        "centered": bgbot_cpp.CubeOwner.CENTERED,
        "player": bgbot_cpp.CubeOwner.PLAYER,
        "opponent": bgbot_cpp.CubeOwner.OPPONENT,
    }
    owner = owner_map[cube_owner_str]

    # Match play params
    if match_length and match_length > 0:
        away1 = match_length - player_score
        away2 = match_length - opp_score
    else:
        away1 = away2 = 0

    game_plan = bgbot_cpp.classify_game_plan(checkers)
    is_race = bgbot_cpp.is_race(checkers)

    # Print header
    print("=" * 80)
    print("Cube Action Analysis")
    print("=" * 80)
    print(f"Checkers: {checkers}")
    print(f"Game plan: {game_plan}, is_race: {is_race}")
    print(f"Cube: {cube_value} ({cube_owner_str})")
    if match_length and match_length > 0:
        print(f"Match: {match_length}-point, player {player_score}pts, opp {opp_score}pts "
              f"(player {away1}-away, opp {away2}-away)")
        print(f"Crawford: {is_crawford}")
    else:
        print("Game: Unlimited (money)")
    print()

    # --- Stage 5 ---
    sage = {}

    # 1-ply
    t0 = time.time()
    r = bgbot_cpp.evaluate_cube_decision(
        checkers, cube_value, owner, *w.weight_args,
        away1=away1, away2=away2, is_crawford=is_crawford)
    sage['1ply'] = {'result': r, 'time': time.time() - t0}

    # 2-4 ply
    for n_ply in [2, 3, 4]:
        t0 = time.time()
        r = bgbot_cpp.cube_decision_nply(
            checkers, cube_value, owner, n_plies=n_ply,
            purerace_weights=w.purerace, racing_weights=w.racing,
            attacking_weights=w.attacking, priming_weights=w.priming,
            anchoring_weights=w.anchoring,
            n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
            n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
            n_hidden_anchoring=w.n_hidden_anchoring,
            away1=away1, away2=away2, is_crawford=is_crawford)
        sage[f'{n_ply}ply'] = {'result': r, 'time': time.time() - t0}

    # Rollout
    t0 = time.time()
    r = bgbot_cpp.cube_decision_rollout(
        checkers, cube_value, owner,
        purerace_weights=w.purerace, racing_weights=w.racing,
        attacking_weights=w.attacking, priming_weights=w.priming,
        anchoring_weights=w.anchoring,
        n_hidden_purerace=w.n_hidden_purerace, n_hidden_racing=w.n_hidden_racing,
        n_hidden_attacking=w.n_hidden_attacking, n_hidden_priming=w.n_hidden_priming,
        n_hidden_anchoring=w.n_hidden_anchoring,
        n_trials=1296, truncation_depth=0, decision_ply=1,
        n_threads=0, seed=42,
        away1=away1, away2=away2, is_crawford=is_crawford)
    sage['rollout'] = {'result': r, 'time': time.time() - t0}

    # --- GNUbg ---
    from bgsage.gnubg import GnuBgAnalyzer

    gnubg = {}
    for n_ply in [1, 2, 3, 4]:
        analyzer = GnuBgAnalyzer(eval_level=f'{n_ply}ply', timeout=300)
        t0 = time.time()
        try:
            cr = analyzer.cube_action(
                checkers, cube_value=cube_value, cube_owner=cube_owner_str,
                away1=away1, away2=away2, is_crawford=is_crawford)
            gnubg[f'{n_ply}ply'] = {
                'result': {
                    'p_win': cr.probs.win, 'p_gw': cr.probs.gammon_win,
                    'p_bw': cr.probs.backgammon_win,
                    'p_gl': cr.probs.gammon_loss, 'p_bl': cr.probs.backgammon_loss,
                    'equity_cubeless': cr.cubeless_equity,
                    'equity_nd': cr.equity_nd, 'equity_dt': cr.equity_dt,
                    'equity_dp': cr.equity_dp,
                    'optimal_action': cr.optimal_action,
                },
                'time': time.time() - t0,
            }
        except Exception as e:
            print(f"WARNING: GNUbg {n_ply}-ply failed: {e}")

    # --- Print tables ---
    _print_cube_tables(sage, gnubg)


def _action_str(should_double, should_take):
    if should_double:
        return "D/T" if should_take else "D/P"
    return "No Double"


def _print_cube_tables(sage, gnubg):
    levels = ['1ply', '2ply', '3ply', '4ply', 'rollout']

    print()
    print("Cubeless Probabilities:")
    print(f"{'Level':<10} {'Engine':<8} {'P(win)':>8} {'P(gw)':>8} {'P(bw)':>8} {'P(gl)':>8} {'P(bl)':>8} {'Eq':>9}")
    print("-" * 69)
    for level in levels:
        label = 'Rollout' if level == 'rollout' else level.replace('ply', '-ply')
        if level in sage:
            p = sage[level]['result']['probs']
            eq = sage[level]['result']['cubeless_equity']
            print(f"{label:<10} {'Sage':<8} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[3]:8.4f} {p[4]:8.4f} {eq:+9.4f}")
        if level in gnubg:
            r = gnubg[level]['result']
            eq = r['equity_cubeless']
            print(f"{label:<10} {'GNUbg':<8} {r['p_win']:8.4f} {r['p_gw']:8.4f} {r['p_bw']:8.4f} {r['p_gl']:8.4f} {r['p_bl']:8.4f} {eq:+9.4f}")

    print()
    print("Cubeful Equities:")
    print(f"{'Level':<10} {'Engine':<8} {'ND':>9} {'DT':>9} {'DP':>9} {'Action':<16} {'Time':>7}")
    print("-" * 68)

    levels_cf = ['1ply', '2ply', '3ply', '4ply', 'rollout']
    for level in levels_cf:
        label = 'Rollout' if level == 'rollout' else level.replace('ply', '-ply')
        if level in sage:
            r = sage[level]['result']
            t = sage[level]['time']
            action = _action_str(r['should_double'], r['should_take'])
            print(f"{label:<10} {'Sage':<8} {r['equity_nd']:+9.4f} {r['equity_dt']:+9.4f} {r['equity_dp']:+9.4f} {action:<16} {t:6.2f}s")
        if level in gnubg:
            r = gnubg[level]['result']
            t = gnubg[level]['time']
            nd = f"{r['equity_nd']:+9.4f}" if r['equity_nd'] is not None else "     N/A"
            dt_v = f"{r['equity_dt']:+9.4f}" if r['equity_dt'] is not None else "     N/A"
            dp_v = f"{r['equity_dp']:+9.4f}" if r['equity_dp'] is not None else "     N/A"
            action = r['optimal_action'] or "?"
            # Trim long GNUbg action strings
            if len(action) > 15:
                action = action[:15]
            print(f"{label:<10} {'GNUbg':<8} {nd} {dt_v} {dp_v} {action:<16} {t:6.2f}s")


# ---------------------------------------------------------------------------
# Checker play analysis
# ---------------------------------------------------------------------------

def run_checker_analysis(checkers, die1, die2, cube_value, cube_owner_str,
                         match_length, player_score, opp_score, is_crawford):
    import bgbot_cpp
    from bgsage import BgBotAnalyzer
    from bgsage.gnubg import GnuBgAnalyzer
    from bgsage.weights import WeightConfig

    w = WeightConfig.default()

    # Match play params
    if match_length and match_length > 0:
        away1 = match_length - player_score
        away2 = match_length - opp_score
    else:
        away1 = away2 = 0

    game_plan = bgbot_cpp.classify_game_plan(checkers)
    is_race = bgbot_cpp.is_race(checkers)

    print("=" * 80)
    print("Checker Play Analysis")
    print("=" * 80)
    print(f"Checkers: {checkers}")
    print(f"Dice: {die1}-{die2}")
    print(f"Game plan: {game_plan}, is_race: {is_race}")
    print(f"Cube: {cube_value} ({cube_owner_str})")
    if match_length and match_length > 0:
        print(f"Match: {match_length}-point, player {player_score}pts, opp {opp_score}pts "
              f"(player {away1}-away, opp {away2}-away)")
        print(f"Crawford: {is_crawford}")
    else:
        print("Game: Unlimited (money)")
    print()

    # --- Stage 5 at each level ---
    sage = {}
    for level_str in ['1ply', '2ply', '3ply', '4ply']:
        analyzer = BgBotAnalyzer(eval_level=level_str, cubeful=(away1 > 0))
        t0 = time.time()
        result = analyzer.checker_play(
            checkers, die1, die2, cube_value=cube_value, cube_owner=cube_owner_str,
            away1=away1, away2=away2, is_crawford=is_crawford)
        sage[level_str] = {'result': result, 'time': time.time() - t0}

    # --- GNUbg at each level ---
    gnubg = {}
    for n_ply in [1, 2, 3, 4]:
        analyzer = GnuBgAnalyzer(eval_level=f'{n_ply}ply', timeout=300)
        t0 = time.time()
        result = analyzer.checker_play(
            checkers, die1, die2, cube_value=cube_value, cube_owner=cube_owner_str,
            away1=away1, away2=away2, is_crawford=is_crawford)
        gnubg[f'{n_ply}ply'] = {'result': result, 'time': time.time() - t0}

    # --- Print tables ---
    _print_checker_tables(sage, gnubg)


def _print_checker_tables(sage, gnubg):
    # Show top 5 moves at each level
    levels = ['1ply', '2ply', '3ply', '4ply']
    for level in levels:
        label = level.replace('ply', '-ply')
        print(f"\n{label}:")
        print(f"  {'Rank':<5} {'Engine':<8} {'Equity':>9} {'Diff':>9} {'P(win)':>8} {'P(gw)':>8} {'P(gl)':>8} {'Time':>7}")
        print(f"  {'-' * 64}")

        if level in sage:
            r = sage[level]
            moves = r['result'].moves[:5]
            for i, m in enumerate(moves):
                t_str = f"{r['time']:6.2f}s" if i == 0 else ""
                print(f"  {i+1:<5} {'Sage':<8} {m.equity:+9.4f} {m.equity_diff:+9.4f} "
                      f"{m.probs.win:8.4f} {m.probs.gammon_win:8.4f} {m.probs.gammon_loss:8.4f} {t_str}")

        if level in gnubg:
            r = gnubg[level]
            moves = r['result'].moves[:5]
            for i, m in enumerate(moves):
                t_str = f"{r['time']:6.2f}s" if i == 0 else ""
                print(f"  {i+1:<5} {'GNUbg':<8} {m.equity:+9.4f} {m.equity_diff:+9.4f} "
                      f"{m.probs.win:8.4f} {m.probs.gammon_win:8.4f} {m.probs.gammon_loss:8.4f} {t_str}")

    # Summary: best move at each level
    print(f"\n{'=' * 80}")
    print("Best Move Summary:")
    print(f"{'Level':<10} {'Engine':<8} {'Equity':>9} {'P(win)':>8}")
    print("-" * 40)
    for lvl in levels:
        label = lvl.replace('ply', '-ply')
        if lvl in sage and sage[lvl]['result'].moves:
            m = sage[lvl]['result'].moves[0]
            print(f"{label:<10} {'Sage':<8} {m.equity:+9.4f} {m.probs.win:8.4f}")
        if lvl in gnubg and gnubg[lvl]['result'].moves:
            m = gnubg[lvl]['result'].moves[0]
            print(f"{label:<10} {'GNUbg':<8} {m.equity:+9.4f} {m.probs.win:8.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_checkers(s):
    """Parse checkers from a string like '0,2,3,...' or '0 2 3 ...'."""
    s = s.strip().strip('[]')
    if ',' in s:
        return [int(x.strip()) for x in s.split(',')]
    return [int(x) for x in s.split()]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a position with Stage 5 and GNUbg side-by-side.')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Common args
    for name in ['cube', 'checker']:
        sub = subparsers.add_parser(name)
        sub.add_argument('--checkers', type=str, required=True,
                         help='26-element board array (comma or space separated)')
        sub.add_argument('--cube-value', type=int, default=1)
        sub.add_argument('--cube-owner', choices=['centered', 'player', 'opponent'],
                         default='centered')
        sub.add_argument('--match', type=int, default=0,
                         help='Match length (0 = unlimited/money game)')
        sub.add_argument('--score', type=int, nargs=2, default=[0, 0],
                         metavar=('PLAYER', 'OPP'),
                         help='Player and opponent scores')
        sub.add_argument('--crawford', action='store_true',
                         help='This is the Crawford game')

    # Checker-specific
    checker_sub = subparsers.choices['checker']
    checker_sub.add_argument('--dice', type=int, nargs=2, required=True,
                             metavar=('DIE1', 'DIE2'))

    args = parser.parse_args()
    checkers = parse_checkers(args.checkers)

    if len(checkers) != 26:
        print(f"ERROR: checkers must have 26 elements, got {len(checkers)}")
        sys.exit(1)

    if args.mode == 'cube':
        run_cube_analysis(
            checkers, args.cube_value, args.cube_owner,
            args.match, args.score[0], args.score[1], args.crawford)
    elif args.mode == 'checker':
        run_checker_analysis(
            checkers, args.dice[0], args.dice[1],
            args.cube_value, args.cube_owner,
            args.match, args.score[0], args.score[1], args.crawford)


if __name__ == '__main__':
    main()
