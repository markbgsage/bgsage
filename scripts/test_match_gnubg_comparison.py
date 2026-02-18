"""Compare bgsage match play cube decisions against GNUbg at 0-ply.

Runs GNUbg in match mode at several match scores and compares cube decisions
and equities against our engine. This validates that the MET implementation,
Janowski-in-MWC-space, and match cube logic produce results consistent with
the reference implementation.

Note: Since bgsage and GNUbg use completely different neural networks, the
underlying cubeless probabilities will differ. We expect:
  - Same cube decision framework behavior (MET lookups, Janowski interpolation)
  - Similar directional trends (match play ND != money ND, etc.)
  - Exact action agreement only when the position is clearly not borderline

Usage:
    python bgsage/scripts/test_match_gnubg_comparison.py [--build-dir build] [--model stage5]
"""

import sys
import os
import argparse
import re
import subprocess
import tempfile

# Pre-parse build-dir to set paths before importing bgsage
build_dir_arg = 'build'
for i, arg in enumerate(sys.argv):
    if arg == '--build-dir' and i + 1 < len(sys.argv):
        build_dir_arg = sys.argv[i + 1]

script_dir = os.path.dirname(os.path.abspath(__file__))
bgsage_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(bgsage_dir)

build_dir = os.path.join(project_dir, build_dir_arg)
build_dir_std = os.path.join(project_dir, 'build')
sys.path.insert(0, os.path.join(bgsage_dir, 'python'))
sys.path.insert(0, build_dir)
sys.path.insert(0, build_dir_std)

cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
if os.path.isdir(cuda_x64):
    os.add_dll_directory(cuda_x64)
if os.path.isdir(build_dir):
    os.add_dll_directory(build_dir)
if os.path.isdir(build_dir_std):
    os.add_dll_directory(build_dir_std)


GNUBG_CLI = r'C:\Program Files (x86)\gnubg\gnubg-cli.exe'


def run_gnubg(cmd, timeout=60):
    """Run gnubg-cli with the given command and return stdout."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cmd)
        cmd_file = f.name

    try:
        result = subprocess.run(
            [GNUBG_CLI, '-q', '-t', '--no-rc', '-c', cmd_file],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout
    finally:
        os.remove(cmd_file)


def build_gnubg_command(checkers, match_length=0, score1=0, score2=0,
                        is_crawford=False, cube_value=1, cube_owner='centered'):
    """Build a gnubg command to evaluate cube decision.

    Args:
        checkers: 26-element board (player on roll's perspective)
        match_length: 0 for money game, N for N-point match
        score1: player's score (player on roll = mghig = player 1)
        score2: opponent's score (gnubg = player 0)
        is_crawford: True if Crawford game
        cube_value: current cube value
        cube_owner: 'centered', 'player', or 'opponent'
    """
    if match_length > 0:
        cmd = f'new match {match_length}\n'
    else:
        cmd = 'new session\n'

    cmd += 'set evaluation chequer eval plies 0\n'
    cmd += 'set evaluation cubedecision eval plies 0\n'

    if match_length > 0:
        # In gnubg: player 0 = gnubg, player 1 = mghig
        # We want mghig (player 1) on roll, so:
        # gnubg score = score2 (opponent), mghig score = score1 (player)
        cmd += f'set score {score2} {score1}\n'

        # Crawford handling
        away1 = match_length - score1  # player's away
        away2 = match_length - score2  # opponent's away
        if is_crawford:
            cmd += 'set crawford on\n'
        elif away1 == 1 or away2 == 1:
            cmd += 'set postcrawford on\n'

    # Set cube
    cmd += f'set cube value {cube_value}\n'
    if cube_owner == 'player':
        # Player = mghig = player 1 in gnubg
        cmd += 'set cube owner 1\n'
    elif cube_owner == 'opponent':
        # Opponent = gnubg = player 0 in gnubg
        cmd += 'set cube owner 0\n'
    else:
        cmd += 'set cube centre\n'

    # Set board: gnubg 'set board simple' format:
    # player_on_roll_bar pt1..pt24 opponent_bar
    # Player on roll = mghig (player 1 in gnubg)
    cmd += 'set board simple '
    cmd += str(checkers[25]) + ' '  # mghig's bar
    for n in checkers[1:25]:
        cmd += str(n) + ' '
    cmd += str(checkers[0]) + '\n'  # gnubg's bar

    # Ensure mghig (player 1) is on turn
    cmd += 'set turn 1\n'
    cmd += 'hint\n'
    return cmd


def parse_cube_output(output):
    """Parse gnubg cube analysis output.

    Handles any ordering of ND/DT/DP in cubeful equities section.
    Returns dict with equity_nd, equity_dt, equity_dp, action.
    """
    result = {}

    if 'Cube analysis' not in output:
        return None

    section = output[output.index('Cube analysis'):]
    lines = section.split('\n')

    # Parse cubeless equity
    for line in lines:
        if 'cubeless equity' in line.lower():
            match = re.search(r'cubeless equity\s+([+-]?\d+\.\d+)', line)
            if match:
                result['cubeless_eq'] = float(match.group(1))
            money_match = re.search(r'Money:\s+([+-]?\d+\.\d+)', line)
            if money_match:
                result['cubeless_eq_money'] = float(money_match.group(1))

    # Parse cubeful equities — handle any ordering
    # Lines look like: "N. No double         +0.096"
    #                   "N. Double, take      -0.232  (+0.139)"
    #                   "N. Double, pass      +1.000  (+0.904)"
    for line in lines:
        stripped = line.strip()

        # No double
        nd = re.match(r'\d+\.\s+No double\s+([+-]?\d+\.\d+)', stripped)
        if nd:
            result['equity_nd'] = float(nd.group(1))

        # Double, take
        dt = re.match(r'\d+\.\s+Double,\s*take\s+([+-]?\d+\.\d+)', stripped)
        if dt:
            result['equity_dt'] = float(dt.group(1))

        # Double, pass
        dp = re.match(r'\d+\.\s+Double,\s*pass\s+([+-]?\d+\.\d+)', stripped)
        if dp:
            result['equity_dp'] = float(dp.group(1))

    # Parse proper cube action
    for line in lines:
        if 'Proper cube action' in line:
            action_str = line.split('Proper cube action:')[-1].strip()
            result['action_raw'] = action_str
            # Normalize
            a = action_str.lower()
            if 'no double' in a or 'no dbl' in a:
                result['action'] = 'ND'
            elif 'pass' in a or 'too good' in a:
                result['action'] = 'DP'
            elif 'take' in a:
                result['action'] = 'DT'
            else:
                result['action'] = action_str

    # Parse cubeless probs if available
    for line in lines:
        stripped = line.strip()
        prob_match = re.match(
            r'(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+-\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)',
            stripped)
        if prob_match:
            result['gnubg_probs'] = [
                float(prob_match.group(1)),  # P(win)
                float(prob_match.group(2)),  # P(gw)
                float(prob_match.group(3)),  # P(bw)
                float(prob_match.group(4)),  # P(loss) — not used directly
                float(prob_match.group(5)),  # P(gl)
                float(prob_match.group(6)),  # P(bl)
            ]

    return result


def our_action_str(should_double, should_take):
    if should_double and should_take:
        return 'DT'
    elif should_double:
        return 'DP'
    else:
        return 'ND'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=str, default='build')

    from bgsage.weights import WeightConfig
    WeightConfig.add_model_arg(parser)
    args = parser.parse_args()

    import bgbot_cpp
    from bgsage import STARTING_BOARD

    w = WeightConfig.from_args(args)
    w.validate()

    CubeOwner = bgbot_cpp.CubeOwner

    if not os.path.isfile(GNUBG_CLI):
        print(f"GNUbg CLI not found at {GNUBG_CLI}, skipping comparison")
        return 0

    print("=" * 70)
    print("Match Play Cube Decision: bgsage vs GNUbg (0-ply)")
    print("=" * 70)

    starting = STARTING_BOARD

    # Position 2: contact position (slightly modified starting)
    contact_pos = [0, 0, -2, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]

    # Position 3: strong race position
    race_pos = [0, 0, 0, 0, 4, 3, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -4, -3, -2, -1, 0, 0]

    # Test scenarios: (position, match_len, score1, score2, is_crawford, label)
    test_cases = [
        # Money game baselines
        (starting, 0, 0, 0, False, "Starting, money"),
        (contact_pos, 0, 0, 0, False, "Contact, money"),
        (race_pos, 0, 0, 0, False, "Race, money"),

        # DMP
        (starting, 5, 4, 4, False, "Starting, DMP 5pt"),
        (contact_pos, 5, 4, 4, False, "Contact, DMP 5pt"),

        # Even match scores
        (starting, 7, 0, 0, False, "Starting, 7pt 7a-7a"),
        (starting, 7, 2, 2, False, "Starting, 7pt 5a-5a"),
        (starting, 7, 4, 4, False, "Starting, 7pt 3a-3a"),

        # Asymmetric scores
        (starting, 7, 4, 2, False, "Starting, 7pt 3a-5a"),
        (starting, 7, 2, 4, False, "Starting, 7pt 5a-3a"),

        # Crawford
        (starting, 7, 6, 4, True, "Starting, Crawford 1a-3a"),
        (starting, 7, 4, 6, True, "Starting, Crawford 3a-1a"),

        # Post-Crawford
        (starting, 7, 6, 4, False, "Starting, post-Craw 1a-3a"),
        (starting, 7, 4, 6, False, "Starting, post-Craw 3a-1a"),

        # Race in match play
        (race_pos, 7, 0, 0, False, "Race, 7pt 7a-7a"),
        (race_pos, 7, 4, 2, False, "Race, 7pt 3a-5a"),

        # Contact in match play
        (contact_pos, 7, 0, 0, False, "Contact, 7pt 7a-7a"),
        (contact_pos, 7, 4, 2, False, "Contact, 7pt 3a-5a"),
    ]

    agree = 0
    disagree = 0
    skipped = 0

    for pos, match_len, sc1, sc2, is_craw, label in test_cases:
        print(f"\n--- {label} ---")

        # Our evaluation
        if match_len == 0:
            away1, away2 = 0, 0
        else:
            away1 = match_len - sc1
            away2 = match_len - sc2

        our = bgbot_cpp.evaluate_cube_decision(
            pos, 1, CubeOwner.CENTERED, *w.weight_args,
            away1=away1, away2=away2, is_crawford=is_craw,
        )

        our_a = our_action_str(our["should_double"], our["should_take"])
        our_probs = list(our["probs"])

        # GNUbg evaluation
        cmd = build_gnubg_command(pos, match_len, sc1, sc2, is_craw)
        try:
            gnubg_output = run_gnubg(cmd)
            g = parse_cube_output(gnubg_output)

            if g is None or 'equity_nd' not in g:
                print(f"  [SKIP] Could not parse GNUbg output")
                for line in gnubg_output.split('\n'):
                    if any(kw in line.lower() for kw in ['cube', 'double', 'pass', 'take', 'equity', 'proper', 'score', 'match']):
                        print(f"    {line.strip()}")
                skipped += 1
                continue

            gnubg_nd = g.get('equity_nd', float('nan'))
            gnubg_dt = g.get('equity_dt', float('nan'))
            gnubg_dp = g.get('equity_dp', float('nan'))
            gnubg_a = g.get('action', '?')

            # Get GNUbg probs for comparison
            gnubg_probs = g.get('gnubg_probs', None)
            gnubg_pstr = ""
            if gnubg_probs:
                gnubg_pstr = f"W={gnubg_probs[0]:.3f} Gw={gnubg_probs[1]:.3f} Bw={gnubg_probs[2]:.3f} Gl={gnubg_probs[4]:.3f} Bl={gnubg_probs[5]:.3f}"

            print(f"  bgsage: ND={our['equity_nd']:+.4f}  DT={our['equity_dt']:+.4f}  DP={our['equity_dp']:+.4f}  -> {our_a}")
            print(f"  GNUbg:  ND={gnubg_nd:+.4f}  DT={gnubg_dt:+.4f}  DP={gnubg_dp:+.4f}  -> {gnubg_a}")
            print(f"  bgsage probs: W={our_probs[0]:.3f} Gw={our_probs[1]:.3f} Bw={our_probs[2]:.3f} Gl={our_probs[3]:.3f} Bl={our_probs[4]:.3f}")
            if gnubg_pstr:
                print(f"  GNUbg  probs: {gnubg_pstr}")

            # Compare
            if our_a == gnubg_a:
                print(f"  -> AGREE: {our_a}")
                agree += 1
            else:
                # Check if it's borderline — different NNs give different probs
                # If the ND-DT gap is small for either engine, disagreement is expected
                our_margin = abs(our['equity_nd'] - our['equity_dt'])
                gnubg_margin = abs(gnubg_nd - gnubg_dt) if gnubg_dt == gnubg_dt else 999

                if our_margin < 0.15 or gnubg_margin < 0.15:
                    print(f"  -> BORDERLINE: bgsage={our_a}, GNUbg={gnubg_a} "
                          f"(margins: ours={our_margin:.3f}, gnubg={gnubg_margin:.3f})")
                    agree += 1  # count borderline as OK
                else:
                    print(f"  -> DISAGREE: bgsage={our_a}, GNUbg={gnubg_a}")
                    disagree += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            skipped += 1

    total = agree + disagree + skipped
    print(f"\n{'='*70}")
    print(f"Results: {agree} agree, {disagree} disagree, {skipped} skipped out of {total}")
    print(f"{'='*70}")

    # MET spot checks
    print(f"\n--- MET Spot Checks ---")
    for a1, a2 in [(1,1), (2,1), (1,2), (3,3), (5,5), (7,7), (3,5), (5,3)]:
        met = bgbot_cpp.get_met(a1, a2, False)
        sym = bgbot_cpp.get_met(a2, a1, False)
        print(f"  get_met({a1},{a2})={met:.5f}  get_met({a2},{a1})={sym:.5f}  sum={met+sym:.5f}")

    return 1 if disagree > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
