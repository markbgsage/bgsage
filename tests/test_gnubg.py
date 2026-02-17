"""
Tests for the gnubg interface.

Run with:
    python python/tests/test_gnubg.py

Tests:
1. Endgame: both players have 2 on their 1-point -> player on roll wins (equity +1)
2. Gammon certain: player has 2 on 1-point, opponent has not borne off -> P(win)=1, P(gw)=1
3. Compare gnubg 2-ply vs Stage 4 2-ply on top-100 worst 0-ply positions
"""

import os
import sys
import time
import math
import unittest

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64):
        os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'python'))

from bgsage.gnubg import post_move_analytics, post_move_analytics_many


def flip_board(checkers):
    """Flip a board to the other player's perspective."""
    flipped = [0] * 26
    flipped[0] = checkers[25]
    flipped[25] = checkers[0]
    for i in range(1, 25):
        flipped[i] = -checkers[25 - i]
    return flipped


class TestGnubgEndgame(unittest.TestCase):
    """Test simple endgame positions where the answer is known exactly.

    post_move_analytics takes a post-move board from the MOVER's perspective.
    """

    def test_both_on_1point_player_wins(self):
        """Both players have 2 checkers on their own 1-point.
        Player just moved and it's their post-move position. Player is the mover
        and will bear off first on their next turn -> P(win) should be very high.

        But this is a post-move board: the opponent will roll next. Since both
        have 2 on their 1-point, the opponent will also bear off in one roll.
        Who wins depends on who rolls first — and that's the opponent.

        So: opponent bears off next, then player bears off. The opponent wins!
        P(win) for the mover should be 0 here. Let's use a game-over case instead.
        """
        # Mover just bore off their last checker. Post-move: mover has 0 checkers.
        # Opponent still has 2 on point 24 (their 1-point).
        checkers = [0] * 26
        checkers[24] = -2  # opponent: 2 on point 24

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3,
                               msg="P(win) should be 1.0 (game over, mover won)")
        self.assertAlmostEqual(result['probs'][1], 0.0, places=3,
                               msg="P(gw) should be 0.0 (not a gammon)")
        self.assertAlmostEqual(result['equity'], 1.0, places=3)

    def test_gammon_certain(self):
        """Mover just bore off their last checker. Opponent has all 15 still on board.
        P(win)=1, P(gw)=1, P(bw)=0 (opponent not in mover's home or bar).
        """
        checkers = [0] * 26
        checkers[24] = -15  # opponent: 15 on point 24 (their 1-point)

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][1], 1.0, places=3,
                               msg="P(gw) should be 1.0 (gammon)")
        self.assertAlmostEqual(result['probs'][2], 0.0, places=3,
                               msg="P(bw) should be 0.0 (not backgammon)")
        self.assertAlmostEqual(result['equity'], 2.0, places=3)

    def test_backgammon_certain(self):
        """Mover bore off all. Opponent has 15 on bar -> backgammon."""
        checkers = [0] * 26
        checkers[0] = 15    # opponent: 15 on bar

        result = post_move_analytics(checkers, n_plies=0)

        self.assertAlmostEqual(result['probs'][0], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][1], 1.0, places=3)
        self.assertAlmostEqual(result['probs'][2], 1.0, places=3,
                               msg="P(bw) should be 1.0 (backgammon)")
        self.assertAlmostEqual(result['equity'], 3.0, places=3)

    def test_non_terminal_race(self):
        """Non-terminal race position: mover has 2 on 1-pt, opponent has 2 on their
        1-pt (our pt 24). Post-move, opponent rolls next. Opponent will bear off in
        1 roll, then mover bears off in 1 roll. Both bear off -> opponent wins.
        P(win) for mover should be 0.0.
        """
        checkers = [0] * 26
        checkers[1] = 2     # mover: 2 on point 1
        checkers[24] = -2   # opponent: 2 on their 1-point

        result = post_move_analytics(checkers, n_plies=0)

        # Opponent rolls next and bears off. Then mover rolls and bears off.
        # Opponent wins -> mover P(win) = 0
        self.assertAlmostEqual(result['probs'][0], 0.0, places=3,
                               msg="P(win) should be 0.0 (opponent bears off first)")
        self.assertAlmostEqual(result['equity'], -1.0, places=3)

    def test_near_race(self):
        """Mover has 2 on 1-pt. Opponent has 2 on their 2-pt (our pt 23).
        Post-move, opponent rolls next. Opponent needs specific rolls to bear off
        both in one turn. Should be close to P(win)=0 but not exactly 0.
        """
        checkers = [0] * 26
        checkers[1] = 2     # mover: 2 on point 1
        checkers[23] = -2   # opponent: 2 on their 2-point (our pt 23)

        result = post_move_analytics(checkers, n_plies=0)

        # Opponent needs doubles or a roll including a 2 to bear off both.
        # Most rolls bear off both. P(win) for mover should be small.
        self.assertLess(result['probs'][0], 0.5,
                        msg="P(win) should be < 0.5 (opponent almost certain to bear off)")
        # Verify equity is internally consistent
        eq = (2.0 * result['probs'][0] - 1.0 + result['probs'][1] - result['probs'][3]
              + result['probs'][2] - result['probs'][4])
        self.assertAlmostEqual(result['equity'], eq, places=3)


class TestGnubgVsStage4(unittest.TestCase):
    """Compare gnubg 2-ply with Stage 4 2-ply on top-100 worst 0-ply positions."""

    @classmethod
    def setUpClass(cls):
        """Load models and find top-100 positions."""
        import bgbot_cpp
        from bgsage.data import load_benchmark_file

        cls.bgbot_cpp = bgbot_cpp

        DATA_DIR = os.path.join(project_dir, 'data')
        MODELS_DIR = os.path.join(project_dir, 'models')

        cls.NH = {'purerace': 120, 'racing': 250, 'attacking': 250,
                  'priming': 250, 'anchoring': 250}

        # Load Stage 4 weights (fall back to Stage 3 .best files)
        cls.weights = {}
        for t in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
            for prefix in ['sl_s4_', 'sl_']:
                path = os.path.join(MODELS_DIR, f'{prefix}{t}.weights.best')
                if os.path.exists(path):
                    cls.weights[t] = path
                    break
            else:
                raise FileNotFoundError(f"No weights found for {t}")

        # Load contact + crashed benchmarks
        contact_file = os.path.join(DATA_DIR, 'contact.bm')
        crashed_file = os.path.join(DATA_DIR, 'crashed.bm')

        print("Loading benchmarks...")
        scenarios_contact = load_benchmark_file(contact_file)
        scenarios_crashed = load_benchmark_file(crashed_file)

        # Score at 0-ply to find worst positions
        print("Scoring at 0-ply to find top-100 worst...")
        w = cls.weights
        nh = cls.NH
        errors_contact = bgbot_cpp.score_benchmarks_per_scenario_5nn(
            scenarios_contact,
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'])
        errors_crashed = bgbot_cpp.score_benchmarks_per_scenario_5nn(
            scenarios_crashed,
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'])

        # Combine and sort to find top 100
        all_errors = []
        for i, err in enumerate(errors_contact):
            all_errors.append((err, 'contact', i))
        for i, err in enumerate(errors_crashed):
            all_errors.append((err, 'crashed', i))
        all_errors.sort(key=lambda x: -x[0])

        top_100 = all_errors[:100]

        # Extract the actual checker positions for these scenarios
        cls.test_positions = []
        cls.position_sources = []

        contact_indices = {e[2] for e in top_100 if e[1] == 'contact'}
        crashed_indices = {e[2] for e in top_100 if e[1] == 'crashed'}

        from bgsage.data import board_from_gnubg_position_string

        def read_bm_positions(filepath, indices):
            """Read specific positions from a .bm file."""
            positions = {}
            with open(filepath, 'r') as f:
                move_idx = 0
                for line in f:
                    if not line.startswith('m '):
                        continue
                    if move_idx in indices:
                        bits = line.split()
                        start_board = board_from_gnubg_position_string(bits[1])
                        die1 = int(bits[2])
                        die2 = int(bits[3])
                        # The first ranked board is the best move result (flipped post-move)
                        best_board_flipped = board_from_gnubg_position_string(bits[4])
                        positions[move_idx] = {
                            'start_board': start_board,
                            'die1': die1,
                            'die2': die2,
                            'best_board_flipped': best_board_flipped,
                        }
                    move_idx += 1
            return positions

        contact_positions = read_bm_positions(contact_file, contact_indices)
        crashed_positions = read_bm_positions(crashed_file, crashed_indices)

        for err, source, idx in top_100:
            if source == 'contact':
                pos = contact_positions[idx]
            else:
                pos = crashed_positions[idx]
            cls.test_positions.append(pos)
            cls.position_sources.append((source, idx, err))

        print(f"  Found {len(cls.test_positions)} test positions")

        # Create Stage 4 2-ply strategy
        cls.multipy_2 = bgbot_cpp.create_multipy_5nn(
            w['purerace'], w['racing'], w['attacking'], w['priming'], w['anchoring'],
            nh['purerace'], nh['racing'], nh['attacking'], nh['priming'], nh['anchoring'],
            n_plies=2)

    def test_top100_comparison(self):
        """Compare gnubg 2-ply vs Stage 4 2-ply on top-100 positions.

        The .bm best_board_flipped is the flipped post-move board (opponent's
        perspective). We un-flip it to get the post-move board from the mover's
        perspective, then pass that to both gnubg and our NN.

        Both gnubg (via post_move_analytics) and our NN (via evaluate_board)
        return probabilities from the mover's perspective.
        """
        bgbot_cpp = self.bgbot_cpp

        gnubg_results = []
        s4_results = []
        game_plans = []

        # Un-flip all boards to get post-move positions from mover's perspective
        post_move_boards = [flip_board(pos['best_board_flipped'])
                            for pos in self.test_positions]

        # gnubg: evaluate post-move boards at 2-ply
        print("\nEvaluating 100 positions with gnubg 2-ply...")
        t0 = time.perf_counter()
        gnubg_results = post_move_analytics_many(post_move_boards, n_plies=2,
                                                  max_workers=8, timeout=120)
        t_gnubg = time.perf_counter() - t0
        print(f"  gnubg 2-ply: {t_gnubg:.1f}s ({t_gnubg/len(post_move_boards)*1000:.0f}ms per position)")

        # Stage 4: evaluate post-move boards with start_board for context
        print("Evaluating 100 positions with Stage 4 2-ply...")
        t0 = time.perf_counter()

        for i, pos in enumerate(self.test_positions):
            self.multipy_2.clear_cache()
            r = self.multipy_2.evaluate_board(post_move_boards[i], pos['start_board'])
            s4_results.append({'probs': list(r['probs']), 'equity': r['equity']})

            gp = bgbot_cpp.classify_game_plan(pos['start_board'])
            game_plans.append(gp)

        t_s4 = time.perf_counter() - t0
        print(f"  Stage 4 2-ply: {t_s4:.1f}s ({t_s4/len(post_move_boards)*1000:.0f}ms per position)")

        # Compare probabilities
        prob_names = ['P(win)', 'P(gw)', 'P(bw)', 'P(gl)', 'P(bl)']

        diffs = {i: [] for i in range(5)}
        eq_diffs = []
        gp_diffs = {}
        gp_eq_diffs = {}
        big_diff_count = {i: 0 for i in range(5)}
        big_diff_positions = []

        for idx in range(len(self.test_positions)):
            gnubg_probs = gnubg_results[idx]['probs']
            s4_probs = s4_results[idx]['probs']
            gnubg_eq = gnubg_results[idx]['equity']
            s4_eq = s4_results[idx]['equity']

            gp_name = game_plans[idx]

            if gp_name not in gp_diffs:
                gp_diffs[gp_name] = {i: [] for i in range(5)}
                gp_eq_diffs[gp_name] = []

            eq_diff = s4_eq - gnubg_eq
            eq_diffs.append(eq_diff)
            gp_eq_diffs[gp_name].append(eq_diff)

            has_big = False
            for i in range(5):
                d = s4_probs[i] - gnubg_probs[i]
                diffs[i].append(d)
                gp_diffs[gp_name][i].append(d)
                if abs(d) > 0.02:
                    big_diff_count[i] += 1
                    has_big = True

            if has_big:
                big_diff_positions.append((idx, gnubg_probs, s4_probs, gp_name,
                                           gnubg_eq, s4_eq))

        # Print report
        print("\n" + "=" * 80)
        print("COMPARISON: gnubg 2-ply vs Stage 4 2-ply (top 100 worst 0-ply positions)")
        print("=" * 80)

        print(f"\n{'Metric':<12} {'RMS':>8} {'Mean':>8} {'MaxAbs':>8} {'|d|>0.02':>10}")
        print("-" * 50)
        for i in range(5):
            rms = math.sqrt(sum(d**2 for d in diffs[i]) / len(diffs[i]))
            mean = sum(diffs[i]) / len(diffs[i])
            maxabs = max(abs(d) for d in diffs[i])
            print(f"{prob_names[i]:<12} {rms:>8.4f} {mean:>+8.4f} {maxabs:>8.4f} {big_diff_count[i]:>8}/{len(diffs[i])}")

        rms_eq = math.sqrt(sum(d**2 for d in eq_diffs) / len(eq_diffs))
        mean_eq = sum(eq_diffs) / len(eq_diffs)
        maxabs_eq = max(abs(d) for d in eq_diffs)
        big_eq_count = sum(1 for d in eq_diffs if abs(d) > 0.02)
        print(f"{'Equity':<12} {rms_eq:>8.4f} {mean_eq:>+8.4f} {maxabs_eq:>8.4f} {big_eq_count:>8}/{len(eq_diffs)}")

        # Per-gameplan breakdown
        print(f"\nPer-Game-Plan RMS Differences:")
        gp_header = f"{'GamePlan':<12} {'N':>4} {'P(win)':>8} {'P(gw)':>8} {'P(bw)':>8} {'P(gl)':>8} {'P(bl)':>8} {'Equity':>8}"
        print(gp_header)
        print("-" * len(gp_header))

        for gp_name in sorted(gp_diffs.keys()):
            n = len(gp_diffs[gp_name][0])
            if n == 0:
                continue
            rms_vals = []
            for i in range(5):
                rms = math.sqrt(sum(d**2 for d in gp_diffs[gp_name][i]) / n)
                rms_vals.append(rms)
            rms_eq_gp = math.sqrt(sum(d**2 for d in gp_eq_diffs[gp_name]) / n)
            print(f"{gp_name:<12} {n:>4} {rms_vals[0]:>8.4f} {rms_vals[1]:>8.4f} "
                  f"{rms_vals[2]:>8.4f} {rms_vals[3]:>8.4f} {rms_vals[4]:>8.4f} {rms_eq_gp:>8.4f}")

        # Show positions with biggest differences
        if big_diff_positions:
            print(f"\nPositions with |prob diff| > 0.02 ({len(big_diff_positions)} positions):")
            print(f"{'#':>4} {'GP':<12} {'Prob':>8} {'gnubg':>8} {'S4':>8} {'Diff':>8}")
            print("-" * 55)
            big_diff_positions.sort(
                key=lambda x: max(abs(x[1][i] - x[2][i]) for i in range(5)),
                reverse=True)
            for idx, gp, sp, gp_name, geq, seq in big_diff_positions[:20]:
                worst_i = max(range(5), key=lambda i: abs(gp[i] - sp[i]))
                d = sp[worst_i] - gp[worst_i]
                print(f"{idx:>4} {gp_name:<12} {prob_names[worst_i]:>8} "
                      f"{gp[worst_i]:>8.4f} {sp[worst_i]:>8.4f} {d:>+8.4f}")

        # Investigate really large differences (> 0.10)
        very_big = [(idx, gp, sp, gpn, geq, seq) for idx, gp, sp, gpn, geq, seq
                    in big_diff_positions if max(abs(gp[i] - sp[i]) for i in range(5)) > 0.10]
        if very_big:
            print(f"\n*** {len(very_big)} positions with |diff| > 0.10 ***")
            for idx, gp, sp, gp_name, geq, seq in very_big[:5]:
                pos = self.test_positions[idx]
                source, bm_idx, err_0ply = self.position_sources[idx]
                print(f"\n  Position #{idx} (from {source}[{bm_idx}], 0-ply err={err_0ply*1000:.1f}):")
                print(f"    Start board:      {pos['start_board']}")
                print(f"    Post-move board:  {flip_board(pos['best_board_flipped'])}")
                print(f"    Dice: {pos['die1']},{pos['die2']}")
                print(f"    Game plan: {gp_name}")
                print(f"    gnubg probs: {[f'{p:.4f}' for p in gp]}")
                print(f"    S4    probs: {[f'{p:.4f}' for p in sp]}")
                print(f"    gnubg eq: {geq:+.4f}, S4 eq: {seq:+.4f}, diff: {seq-geq:+.4f}")

        # Summary stats
        print(f"\nSummary:")
        print(f"  Total positions compared: {len(self.test_positions)}")
        print(f"  gnubg time: {t_gnubg:.1f}s, Stage 4 time: {t_s4:.1f}s")
        print(f"  Equity RMS diff: {rms_eq:.4f}")
        print(f"  P(win) RMS diff: {math.sqrt(sum(d**2 for d in diffs[0]) / len(diffs[0])):.4f}")
        total_big = sum(big_diff_count.values())
        print(f"  Total prob diffs > 0.02: {total_big} / {len(self.test_positions) * 5}")

        if very_big:
            print(f"\n  NOTE: {len(very_big)} positions have probability differences > 0.10")
            print(f"        These are the hardest positions (top-100 worst 0-ply errors)")
            print(f"        and large differences are expected on these challenging positions.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
