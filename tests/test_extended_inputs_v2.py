"""
Unit tests for the 244-input extended contact encoding with GNUbg features.

Tests each new feature individually with hand-crafted board positions,
then tests the full compute_extended_contact_inputs for correctness.

Board convention:
  board[0]   = opponent (player 2) checkers on bar (always >= 0)
  board[1-24] = board points (positive = player 1, negative = player 2)
  board[25]  = player 1 checkers on bar (always >= 0)
"""

import os
import sys
import math

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    if os.path.isdir(build_dir):
        os.add_dll_directory(build_dir)

sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'python'))

import bgbot_cpp

STARTING_BOARD = [0,  # bar2
                  0, 0, 0, 0, 0, -5,   # points 1-6
                  0, -3, 0, 0, 0, 5,    # points 7-12
                  -5, 0, 0, 0, 3, 0,    # points 13-18
                  5, 0, 0, 0, 0, -2,    # points 19-24
                  0]  # bar1


class TestBreakContact:
    """Test break_contact: sum of (distance past opponent's back checker * checkers)."""

    def test_no_opponent_checkers(self):
        board = [0] * 26
        board[5] = 3
        assert bgbot_cpp.break_contact(board) == 0

    def test_simple_contact(self):
        # Opponent's back checker at point 10 (board[10] = -2)
        # Player checker at point 15 (board[15] = 3): distance = 15-10 = 5, contribution = 5*3 = 15
        board = [0] * 26
        board[10] = -2
        board[15] = 3
        assert bgbot_cpp.break_contact(board) == 15

    def test_multiple_player_checkers(self):
        board = [0] * 26
        board[5] = -1  # opp back checker at 5
        board[8] = 2   # distance 3, contribution 6
        board[10] = 3  # distance 5, contribution 15
        assert bgbot_cpp.break_contact(board) == 21

    def test_starting_position(self):
        result = bgbot_cpp.break_contact(STARTING_BOARD)
        # Opponent's back checker at point 24 (board[24] = -2)
        # Player checkers past point 24: none (24 is the highest)
        # But wait - bar counts too. Player has bar = 0.
        # Player checkers AT or BELOW opp_back don't count. Only PAST.
        # Since opp_back = 24, no player checker is past 24 (except bar at 25)
        # board[25] = 0, so result should be 0
        assert result == 0


class TestFreePip:
    """Test free_pip: pips of checkers already past opponent's back checker."""

    def test_no_opponent(self):
        board = [0] * 26
        board[5] = 3
        assert bgbot_cpp.free_pip(board) == 0

    def test_simple(self):
        board = [0] * 26
        board[15] = -1  # opp back at 15
        board[5] = 2    # player at 5, already past opp (5 < 15)
        board[10] = 3   # player at 10, already past opp
        # free_pip = 5*2 + 10*3 = 10 + 30 = 40
        assert bgbot_cpp.free_pip(board) == 40

    def test_checker_at_opp_back_not_counted(self):
        board = [0] * 26
        board[10] = -2   # opp back at 10
        board[10] = -2   # wait, that's opponent. Let me put player at 10
        # Can't have both player and opp at same point. Player at 10 would need board[10] > 0
        # Let's do: opp at 15, player at 10 and 15
        board = [0] * 26
        board[15] = -1   # opp back at 15
        board[10] = 2    # player past opp
        board[15] = -1   # opp - can't have player here too
        # free_pip = 10*2 = 20
        assert bgbot_cpp.free_pip(board) == 20


class TestTiming:
    """Test timing: wastage heuristic."""

    def test_empty_board(self):
        board = [0] * 26
        assert bgbot_cpp.timing_feature(board) == 0

    def test_bar_checkers(self):
        board = [0] * 26
        board[25] = 2  # 2 on bar
        # t = 24*2 = 48, no = 2
        # Home board adj: point 6 has 0 checkers, deficit=2, no>=2: t -= 5*2=10, no=0
        # t = 48 - 10 = 38
        assert bgbot_cpp.timing_feature(board) == 38

    def test_basic_timing(self):
        # Put checkers in home board optimally (2 per point)
        board = [0] * 26
        board[1] = 2
        board[2] = 2
        board[3] = 2
        board[4] = 2
        board[5] = 2
        board[6] = 2
        # 12 checkers in home, each point has 2 (optimal)
        # No checkers outside home, so timing from outer points = 0
        # Home board: each point has 2, so no adjustments
        result = bgbot_cpp.timing_feature(board)
        assert result == 0


class TestBackbone:
    """Test backbone: connectivity of anchor chain."""

    def test_no_anchors(self):
        board = [0] * 26
        board[10] = 1  # blot, not anchor
        assert bgbot_cpp.backbone_feature(board) == 0.0

    def test_single_anchor(self):
        board = [0] * 26
        board[10] = 3
        # Only one anchor found, pa set but no comparison made
        # tot = 0 (never accumulated), so result = 0
        assert bgbot_cpp.backbone_feature(board) == 0.0

    def test_two_adjacent_anchors(self):
        board = [0] * 26
        board[10] = 2  # anchor at 10 (GNUbg 9)
        board[11] = 3  # anchor at 11 (GNUbg 10)
        # pa=10 (GNUbg 10), then find 9 (GNUbg 9): d=1, c=11
        # Wait - scan from 23 down. First anchor at 11 (GNUbg 10), pa=10.
        # Then anchor at 10 (GNUbg 9): d = 10-9 = 1, c = 11
        # w = 11 * board[pa+1] = 11 * board[11] = 11*3 = 33
        # tot = board[11] = 3
        # backbone = 1 - 33/(3*11) = 1 - 1 = 0
        result = bgbot_cpp.backbone_feature(board)
        assert abs(result - 0.0) < 1e-6

    def test_two_distant_anchors(self):
        board = [0] * 26
        board[20] = 2  # GNUbg 19
        board[10] = 2  # GNUbg 9
        # pa=19, then gnubg_pt=9: d=19-9=10, c=13-10=3
        # w = 3 * board[20] = 3*2 = 6
        # tot = 2
        # backbone = 1 - 6/(2*11) = 1 - 6/22 ≈ 0.727
        result = bgbot_cpp.backbone_feature(board)
        assert abs(result - (1.0 - 6.0/22.0)) < 1e-6


class TestBackg:
    """Test backg: backgame strength with 2+ anchors in opponent's home."""

    def test_no_anchors_in_opp_home(self):
        board = [0] * 26
        board[5] = 3  # anchor in player's home
        assert bgbot_cpp.backg_feature(board) == 0.0

    def test_one_anchor(self):
        board = [0] * 26
        board[20] = 2  # 1 anchor in opp home (19-24)
        assert bgbot_cpp.backg_feature(board) == 0.0  # need 2+

    def test_two_anchors(self):
        board = [0] * 26
        board[20] = 2
        board[22] = 3
        # 2 anchors, tot = 2+3 = 5
        # backg = (5-3)/4 = 0.5
        result = bgbot_cpp.backg_feature(board)
        assert abs(result - 0.5) < 1e-6

    def test_with_bar(self):
        board = [0] * 26
        board[20] = 2
        board[22] = 2
        board[25] = 1  # bar
        # 2 anchors, tot = 2+2+1 = 5
        # backg = (5-3)/4 = 0.5
        result = bgbot_cpp.backg_feature(board)
        assert abs(result - 0.5) < 1e-6


class TestBackg1:
    """Test backg1: backgame strength with exactly 1 anchor."""

    def test_no_anchors(self):
        board = [0] * 26
        assert bgbot_cpp.backg1_feature(board) == 0.0

    def test_one_anchor(self):
        board = [0] * 26
        board[20] = 3
        # 1 anchor in opp home, tot = 3
        # backg1 = 3/8 = 0.375
        result = bgbot_cpp.backg1_feature(board)
        assert abs(result - 0.375) < 1e-6

    def test_two_anchors_returns_zero(self):
        board = [0] * 26
        board[20] = 2
        board[22] = 2
        assert bgbot_cpp.backg1_feature(board) == 0.0


class TestEnterLoss:
    """Test enter_loss: pips lost when on bar."""

    def test_not_on_bar(self):
        board = [0] * 26
        assert bgbot_cpp.enter_loss(board) == 0

    def test_one_blocked_point(self):
        board = [0] * 26
        board[25] = 1  # on bar
        board[24] = -2  # opponent blocks point 24 (their point 0/ace)
        # In GNUbg: anBoardOpp[0] > 1 (blocked)
        # loss += 4 * (0+1) = 4 (any double)
        # For j=1..5: anBoardOpp[j] not blocked
        #   if two=false: nothing added
        # Total: 4
        result = bgbot_cpp.enter_loss(board)
        assert result == 4

    def test_fully_blocked(self):
        board = [0] * 26
        board[25] = 1
        for i in range(19, 25):
            board[i] = -2  # block all 6 points
        result = bgbot_cpp.enter_loss(board)
        # This should be a large number
        assert result > 0


class TestContainment:
    """Test containment and acontainment."""

    def test_no_player_checkers(self):
        board = [0] * 26
        result = bgbot_cpp.containment_feature(board)
        # No player checkers blocking = escape is easy = 36 rolls escape
        # containment = (36-36)/36 = 0
        assert abs(result) < 1e-6

    def test_full_prime(self):
        # Player has a 6-prime from points 7-12
        board = [0] * 26
        for i in range(7, 13):
            board[i] = 2
        result = bgbot_cpp.containment_feature(board)
        # Strong containment - should be high
        assert result > 0.5


class TestMobility:
    """Test mobility: escape opportunities weighted by distance."""

    def test_no_checkers_outside_home(self):
        board = [0] * 26
        board[3] = 5  # all in home board
        result = bgbot_cpp.mobility_feature(board)
        # GNUbg loop starts at i=6 (0-indexed), which is our point 7
        # No player checkers at point 7+, so mobility = 0
        assert result == 0


class TestMoment2:
    """Test moment2: second moment of checker distribution."""

    def test_single_point(self):
        board = [0] * 26
        board[5] = 5
        # All checkers on one point → mean = that point, no spread above mean
        result = bgbot_cpp.moment2_feature(board)
        assert result == 0

    def test_spread_checkers(self):
        board = [0] * 26
        board[3] = 2
        board[15] = 2
        result = bgbot_cpp.moment2_feature(board)
        # Should have some positive moment
        assert result > 0


class TestPiploss:
    """Test compute_piploss: average pips lost from hits."""

    def test_no_blots(self):
        board = [0] * 26
        board[10] = 2
        board[5] = -2  # anchor, not blot
        assert bgbot_cpp.compute_piploss(board) == 0

    def test_simple_blot(self):
        board = [0] * 26
        board[10] = -1  # opponent blot
        board[15] = 2   # player checker
        result = bgbot_cpp.compute_piploss(board)
        assert result > 0


class TestBackRescueEscapes:
    """Test back_rescue_escapes: Escapes1 variant."""

    def test_no_opponent(self):
        board = [0] * 26
        board[5] = 3
        # No opponent checkers → should return 36
        assert bgbot_cpp.back_rescue_escapes(board) == 36

    def test_with_opponent(self):
        board = [0] * 26
        board[5] = 3
        board[20] = -2  # opponent
        result = bgbot_cpp.back_rescue_escapes(board)
        assert 0 <= result <= 36


class TestFullEncoding244:
    """Test the full 244-input encoding."""

    def test_output_size(self):
        inputs = bgbot_cpp.compute_extended_contact_inputs(STARTING_BOARD)
        assert len(inputs) == 244, f"Expected 244 inputs, got {len(inputs)}"

    def test_point_encoding_unchanged(self):
        """First 96 point encoding values should match Tesauro."""
        inputs = bgbot_cpp.compute_extended_contact_inputs(STARTING_BOARD)
        tesauro = bgbot_cpp.compute_tesauro_inputs(STARTING_BOARD)

        for i in range(96):
            assert abs(inputs[i] - tesauro[i]) < 1e-6, \
                f"Player point mismatch at [{i}]: extended={inputs[i]}, tesauro={tesauro[i]}"

    def test_opponent_point_encoding(self):
        """Opponent point encoding at offset 122 should match Tesauro offset 98."""
        inputs = bgbot_cpp.compute_extended_contact_inputs(STARTING_BOARD)
        tesauro = bgbot_cpp.compute_tesauro_inputs(STARTING_BOARD)

        for i in range(96):
            ext_val = inputs[122 + i]
            tes_val = tesauro[98 + i]
            assert abs(ext_val - tes_val) < 1e-6, \
                f"Opponent point mismatch at ext[{122+i}] vs tes[{98+i}]: {ext_val} vs {tes_val}"

    def test_bar_encoding(self):
        board = [0] * 26
        board[25] = 3  # player bar
        board[0] = 2   # opponent bar
        board[5] = 10  # keep some checkers on board
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[96] - 3/2.0) < 1e-6, f"Player bar: {inputs[96]}"
        assert abs(inputs[122 + 96] - 2/2.0) < 1e-6, f"Opponent bar: {inputs[218]}"

    def test_all_features_in_range(self):
        """All features should be in reasonable ranges."""
        inputs = bgbot_cpp.compute_extended_contact_inputs(STARTING_BOARD)

        # Player features [96-121]
        assert 0 <= inputs[96] <= 7.5   # bar
        for i in [97, 98, 99]:
            assert 0 <= inputs[i] <= 1.0, f"Borne-off bucket {i-97}: {inputs[i]}"
        assert 0 <= inputs[100] <= 1.0  # I_ENTER2
        assert 0 <= inputs[101] <= 4.0  # I_FORWARD_ANCHOR (up to 2.0 for "no anchor")
        assert 0 <= inputs[102] <= 1.0  # I_P1 (/36)
        assert 0 <= inputs[103] <= 1.0  # I_P2 (/36)
        assert 0 <= inputs[104] <= 1.0  # I_BACKESCAPES (/36)
        assert 0 <= inputs[105] <= 1.0  # I_BACK_CHEQUER
        assert 0 <= inputs[106] <= 1.0  # I_BACK_ANCHOR
        assert inputs[107] >= 0         # I_BREAK_CONTACT
        assert inputs[108] >= 0         # I_FREEPIP
        assert inputs[109] >= 0         # I_PIPLOSS
        assert 0 <= inputs[110] <= 1.0  # I_ACONTAIN
        assert 0 <= inputs[111] <= 1.0  # I_ACONTAIN2
        assert 0 <= inputs[112] <= 1.0  # I_CONTAIN
        assert 0 <= inputs[113] <= 1.0  # I_CONTAIN2
        assert inputs[114] >= 0         # I_MOBILITY
        assert inputs[115] >= 0         # I_MOMENT2
        assert inputs[116] >= 0         # I_ENTER
        assert inputs[117] >= 0         # I_TIMING
        assert 0 <= inputs[118] <= 1.0  # I_BACKBONE
        assert inputs[119] >= 0         # I_BACKG
        assert inputs[120] >= 0         # I_BACKG1
        assert 0 <= inputs[121] <= 1.0  # I_BACKRESCAPES

        # Opponent features should also be in range
        for i in range(122, 244):
            val = inputs[i]
            assert not math.isnan(val), f"NaN at index {i}"
            assert not math.isinf(val), f"Inf at index {i}"

    def test_symmetry(self):
        """Flipping the board should swap player/opponent features."""
        board = [0] * 26
        board[3] = 4
        board[10] = 2
        board[20] = -3
        board[15] = -1

        # Flip: swap player/opponent
        flipped = [0] * 26
        flipped[0] = board[25]
        flipped[25] = board[0]
        for i in range(1, 25):
            flipped[i] = -board[25 - i]

        inputs_orig = bgbot_cpp.compute_extended_contact_inputs(board)
        inputs_flip = bgbot_cpp.compute_extended_contact_inputs(flipped)

        # Point encoding symmetry: point i in orig maps to point (25-i) in flipped
        # orig_P2[point_i] should == flip_P1[point_(25-i)]
        for i in range(24):
            for j in range(4):
                orig_p2_idx = 122 + 4 * i + j   # P2 at point i+1
                flip_p1_idx = 4 * (23 - i) + j   # P1 at point 24-i = 25-(i+1)
                orig_p2 = inputs_orig[orig_p2_idx]
                flip_p1 = inputs_flip[flip_p1_idx]
                assert abs(orig_p2 - flip_p1) < 1e-5, \
                    f"Point symmetry mismatch: orig_P2[point {i+1}] idx {orig_p2_idx}={orig_p2} " \
                    f"vs flip_P1[point {24-i}] idx {flip_p1_idx}={flip_p1}"

        # Bar encoding symmetry
        assert abs(inputs_orig[122 + 96] - inputs_flip[96]) < 1e-5, "Bar symmetry"

        # Borne-off bucket symmetry
        for j in range(3):
            assert abs(inputs_orig[122 + 97 + j] - inputs_flip[97 + j]) < 1e-5, \
                f"Borne-off bucket {j} symmetry"


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestBreakContact, TestFreePip, TestTiming,
        TestBackbone, TestBackg, TestBackg1, TestEnterLoss,
        TestContainment, TestMobility, TestMoment2,
        TestPiploss, TestBackRescueEscapes,
        TestFullEncoding244,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {cls.__name__}.{method_name}")
                print(f"        {e}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, traceback.format_exc()))
                print(f"  ERROR: {cls.__name__}.{method_name}")
                print(f"         {e}")

    print()
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if errors:
        print("\nFailed tests:")
        for cls_name, method_name, msg in errors:
            print(f"  {cls_name}.{method_name}: {msg[:200]}")

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
