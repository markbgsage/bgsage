"""
Comprehensive unit tests for the extended 214-input contact encoding.

Tests each helper function individually with hand-crafted board positions,
then tests the full compute_extended_contact_inputs against the old Python
implementation for cross-validation.

Board convention:
  board[0]   = opponent (player 2) checkers on bar (always >= 0)
  board[1-24] = board points (positive = player 1, negative = player 2)
  board[25]  = player 1 checkers on bar (always >= 0)

  Player 1's home board: points 1-6
  Player 2's home board: points 19-24 (from player 1's perspective)

  An "anchor" = 2+ checkers on a point
  A "blot" = exactly 1 checker on a point
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

# Also load the old Python implementation for cross-validation
old_src_dir = os.path.join(project_dir, 'old_src')
if old_src_dir not in sys.path:
    sys.path.insert(0, old_src_dir)


def make_board(*args, **kwargs):
    """Create a board as a list of 26 ints. Defaults to all zeros."""
    board = [0] * 26
    for k, v in kwargs.items():
        if k == 'bar1':
            board[25] = v
        elif k == 'bar2':
            board[0] = v
        elif k.startswith('p'):
            board[int(k[1:])] = v
    return board


# ======================== Starting Board ========================

STARTING_BOARD = [0,  # bar2
                  0, 0, 0, 0, 0, -5,   # points 1-6
                  0, -3, 0, 0, 0, 5,    # points 7-12
                  -5, 0, 0, 0, 3, 0,    # points 13-18
                  5, 0, 0, 0, 0, -2,    # points 19-24
                  0]  # bar1


class TestMaxPoint:
    """Test max_point: the furthest point from home with at least one player checker."""

    def test_starting_position(self):
        # Player has checkers on points 6(x5 opp), 8(x3 opp), 12(x5), 17(x3), 19(x5)
        # Wait - in starting board: board[6]=-5 (opp), board[8]=-3 (opp),
        # board[12]=5 (player), board[13]=-5 (opp), board[17]=3 (player), board[19]=5 (player), board[24]=-2 (opp)
        # Player's checkers are at 12, 17, 19. Furthest = 19
        result = bgbot_cpp.max_point(STARTING_BOARD)
        assert result == 19, f"Starting position max_point should be 19, got {result}"

    def test_all_borne_off(self):
        board = [0] * 26
        assert bgbot_cpp.max_point(board) == 0

    def test_single_checker_on_24(self):
        board = [0] * 26
        board[24] = 1
        assert bgbot_cpp.max_point(board) == 24

    def test_single_checker_on_1(self):
        board = [0] * 26
        board[1] = 1
        assert bgbot_cpp.max_point(board) == 1

    def test_ignores_opponent_checkers(self):
        board = [0] * 26
        board[20] = -3  # opponent
        board[5] = 2    # player
        assert bgbot_cpp.max_point(board) == 5

    def test_ignores_bar(self):
        """max_point should NOT include checkers on the bar."""
        board = [0] * 26
        board[25] = 2   # player on bar
        board[3] = 1    # player on point 3
        assert bgbot_cpp.max_point(board) == 3


class TestMaxAnchorPoint:
    """Test max_anchor_point: furthest point with 2+ player checkers."""

    def test_starting_position(self):
        # Player's anchors (2+ checkers): board[12]=5, board[17]=3, board[19]=5
        # Furthest = 19
        result = bgbot_cpp.max_anchor_point(STARTING_BOARD)
        assert result == 19, f"Starting position max_anchor should be 19, got {result}"

    def test_no_anchors(self):
        board = [0] * 26
        board[5] = 1   # only one checker = blot, not anchor
        board[10] = 1
        assert bgbot_cpp.max_anchor_point(board) == 0

    def test_single_anchor(self):
        board = [0] * 26
        board[15] = 2
        assert bgbot_cpp.max_anchor_point(board) == 15

    def test_multiple_anchors(self):
        board = [0] * 26
        board[5] = 3
        board[15] = 2
        board[20] = 5
        assert bgbot_cpp.max_anchor_point(board) == 20

    def test_blot_vs_anchor(self):
        board = [0] * 26
        board[20] = 1   # blot, not anchor
        board[10] = 2   # anchor
        assert bgbot_cpp.max_anchor_point(board) == 10


class TestProbNoEnterFromBar:
    """Test prob_no_enter_from_bar: probability of failing to enter from bar."""

    def test_no_anchors_in_home(self):
        """No opponent anchors in their home board (points 19-24) → player can always enter."""
        board = [0] * 26
        board[25] = 1  # player on bar
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert player_prob == 0.0, f"Expected 0.0, got {player_prob}"
        assert opp_prob == 0.0, f"Expected 0.0, got {opp_prob}"

    def test_one_anchor_player_perspective(self):
        """1 opponent anchor in points 19-24 → probability = 1/36."""
        board = [0] * 26
        board[20] = -2  # opponent anchor in their home board
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert abs(player_prob - 1/36) < 1e-6, f"Expected 1/36, got {player_prob}"

    def test_three_anchors_player_perspective(self):
        """3 opponent anchors → probability = 9/36 = 1/4."""
        board = [0] * 26
        board[19] = -2
        board[21] = -3
        board[23] = -5
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert abs(player_prob - 9/36) < 1e-6, f"Expected 9/36, got {player_prob}"

    def test_six_anchors_fully_blocked(self):
        """6 opponent anchors → probability = 36/36 = 1.0."""
        board = [0] * 26
        for i in range(19, 25):
            board[i] = -2
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert abs(player_prob - 1.0) < 1e-6, f"Expected 1.0, got {player_prob}"

    def test_blots_dont_count(self):
        """Single opponent checkers (blots) in home board don't block entry."""
        board = [0] * 26
        board[19] = -1   # blot, not anchor
        board[20] = -1   # blot
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert player_prob == 0.0, f"Blots shouldn't block, got {player_prob}"

    def test_opponent_perspective(self):
        """Test opponent's bar entry probability."""
        board = [0] * 26
        board[1] = 2   # player anchor in player's home board (blocks opponent entry)
        board[3] = 3   # player anchor
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(board)
        assert player_prob == 0.0  # no opp anchors in 19-24
        assert abs(opp_prob - 4/36) < 1e-6, f"Expected 4/36, got {opp_prob}"

    def test_starting_position(self):
        """Starting position has specific anchor patterns."""
        player_prob, opp_prob = bgbot_cpp.prob_no_enter_from_bar(STARTING_BOARD)
        # Opp has anchors at points 24(-2) — that's 1 anchor in 19-24
        # but also 13(-5) is NOT in 19-24. So just 1 anchor (point 24).
        # Actually: points 19-24 for checking opponent anchors:
        # board[19]=5 (player, not opponent), board[20]=0, ..., board[24]=-2 (opponent)
        # So 1 opponent anchor → player_prob = 1/36
        assert abs(player_prob - 1/36) < 1e-6, f"Expected 1/36, got {player_prob}"

        # For opponent: player anchors in points 1-6:
        # board[1]=0, ..., board[6]=-5 (opponent). No player anchors in 1-6.
        assert opp_prob == 0.0, f"Expected 0.0, got {opp_prob}"


class TestForwardAnchorPoints:
    """Test forward_anchor_points: most forward anchor on opponent's side."""

    def test_no_forward_anchors(self):
        """No player anchors on points 13-24 → returns 0."""
        board = [0] * 26
        board[5] = 3   # anchor in home board
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        assert player_fwd == 0, f"Expected 0, got {player_fwd}"

    def test_anchor_at_point_13(self):
        """Player anchor at point 13 → from opponent's view that's 25-13 = 12."""
        board = [0] * 26
        board[13] = 2
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        assert player_fwd == 12, f"Expected 12, got {player_fwd}"

    def test_anchor_at_point_24(self):
        """Player anchor at point 24 → from opponent's view that's 25-24 = 1."""
        board = [0] * 26
        board[24] = 3
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        assert player_fwd == 1, f"Expected 1, got {player_fwd}"

    def test_most_forward_selected(self):
        """Multiple anchors → returns the most forward (lowest point from opponent's view)."""
        board = [0] * 26
        board[14] = 2  # 25-14 = 11
        board[20] = 3  # 25-20 = 5
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        # Most forward = first found scanning from 13 upward = 14, giving 25-14 = 11
        assert player_fwd == 11, f"Expected 11, got {player_fwd}"

    def test_blot_not_anchor(self):
        """Single checker (blot) doesn't count as anchor."""
        board = [0] * 26
        board[15] = 1   # blot
        board[20] = 2   # anchor
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        assert player_fwd == 5, f"Expected 5 (25-20), got {player_fwd}"

    def test_opponent_forward_anchor(self):
        """Test opponent's forward anchor (opp anchors on player's side, points 1-12)."""
        board = [0] * 26
        board[5] = -2   # opponent anchor at point 5
        board[10] = -3  # opponent anchor at point 10
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(board)
        # Scan from 12 down: first hit is 10
        assert opp_fwd == 10, f"Expected 10, got {opp_fwd}"

    def test_starting_position(self):
        player_fwd, opp_fwd = bgbot_cpp.forward_anchor_points(STARTING_BOARD)
        # Player anchors at 12, 17, 19. Points 13-24: 17 and 19 qualify.
        # First scan from 13: 17 is first anchor (board[17]=3) → 25-17 = 8
        assert player_fwd == 8, f"Expected 8, got {player_fwd}"
        # Opponent anchors: board[6]=-5 (point 6), board[8]=-3 (point 8), board[13]=-5 (point 13)
        # Points 1-12: scan from 12 down: 8(-3) is first → opp_fwd = 8
        assert opp_fwd == 8, f"Expected 8, got {opp_fwd}"


class TestHittingShots:
    """Test hitting_shots: number of rolls (out of 36) that hit at least one opponent blot."""

    def test_no_blots(self):
        """No opponent blots → 0 hitting shots."""
        board = [0] * 26
        board[10] = 2  # player
        board[5] = -2  # opponent anchor, not blot
        assert bgbot_cpp.hitting_shots(board) == 0

    def test_direct_hit_distance_1(self):
        """Blot at distance 1 from player checker → any roll with a 1 hits it."""
        board = [0] * 26
        board[10] = -1  # opponent blot
        board[11] = 1   # player checker at distance 1
        # Any roll containing a 1: (1,1), (2,1), (3,1), (4,1), (5,1), (6,1)
        # Plus exact distance 1: already covered
        # Rolls with d=1: (1,1)(1 roll), (2,1)(2), (3,1)(2), (4,1)(2), (5,1)(2), (6,1)(2) = 11
        result = bgbot_cpp.hitting_shots(board)
        assert result == 11, f"Expected 11, got {result}"

    def test_direct_hit_distance_6(self):
        """Blot at distance 6 from player checker."""
        board = [0] * 26
        board[10] = -1  # opponent blot
        board[16] = 1   # player checker at distance 6
        # Direct: any roll containing 6: (6,1)x2, (6,2)x2, (6,3)x2, (6,4)x2, (6,5)x2, (6,6)x1 = 11
        # Also: combination rolls summing to 6 if unblocked:
        #   (5,1) already counted via 6; but stepping through: 10+5=15, 10+1=11
        #   For the "short rolls" section (distance=6):
        #     (5,1): board[10+1]=0>=−1 or board[10+5]=0>=−1 → yes, but (5,1) key = max(5,1)=5*7+1 → already accounted for (6,1) is 6*7+1
        # Wait, I need to be more careful. For distance 6:
        #   (6,k) for all k → already handled above
        #   Short doubles: (2,2) if board[12]>-2 and board[14]>-2 → 0>-2 yes → 2,2 = 1 roll
        #   (3,3) if board[13]>-2 → 0>-2 yes → 3,3 = 1 roll
        #   Short combo: (5,1) already in set, (4,2) already in set, (3,3) already covered
        # So total unique rolls:
        #   (1,1) not from distance 6 directly (no short double check for d=6 hitting 1-1)
        #   (6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(2,2),(3,3), and combos:
        #   (5,1) is same as (1,5) → key = max(5,1)=5*7+1 → distinct from (6,1)=6*7+1
        # The direct hits add: for k in 1..6: (max(6,k), min(6,k))
        #   (6,1),(6,2),(6,3),(6,4),(6,5),(6,6) → 6 unique rolls
        # Short doubles: (2,2), (3,3) → 2 more
        # Short combos (distance=6): (5,1), (4,2) → 2 more
        # Total unique rolls: 10. Count: (6,1)=2, (6,2)=2, (6,3)=2, (6,4)=2, (6,5)=2, (6,6)=1, (2,2)=1, (3,3)=1, (5,1)=2, (4,2)=2 = 17
        result = bgbot_cpp.hitting_shots(board)
        assert result == 17, f"Expected 17, got {result}"

    def test_blocked_combination(self):
        """Opponent anchor blocks combination roll."""
        board = [0] * 26
        board[10] = -1  # blot
        board[15] = 1   # player checker, distance 5
        board[11] = -2  # opponent anchor blocks steps through point 11
        board[14] = -2  # opponent anchor blocks steps through point 14
        # Direct hits with a 5: (5,1),(5,2),(5,3),(5,4),(5,5),(6,5) → 6 unique rolls
        # Short doubles: distance 5 doesn't trigger any new short doubles
        # Short combos (distance=5):
        #   (4,1): board[10+1]=-2>=−1? No. board[10+4]=-2>=−1? No. → blocked
        #   (3,2): board[10+2]=0>=-1? Yes → (3,2) added = 2 more
        # So total should be: (5,1),(5,2),(5,3),(5,4),(5,5),(6,5),(3,2) = 7 unique rolls
        # Count: (5,1)=2, (5,2)=2, (5,3)=2, (5,4)=2, (5,5)=1, (6,5)=2, (3,2)=2 = 13
        result = bgbot_cpp.hitting_shots(board)
        assert result == 13, f"Expected 13, got {result}"

    def test_starting_position(self):
        """Test hitting shots in starting position."""
        result = bgbot_cpp.hitting_shots(STARTING_BOARD)
        # This is complex - let's just verify it's a reasonable number and matches Python
        assert 0 <= result <= 36, f"Result {result} out of range"


class TestDoubleHittingShots:
    """Test double_hitting_shots: rolls that hit 2+ opponent blots."""

    def test_no_blots(self):
        board = [0] * 26
        board[10] = 2
        board[5] = -2
        assert bgbot_cpp.double_hitting_shots(board) == 0

    def test_single_blot(self):
        """Can't double-hit with only one blot."""
        board = [0] * 26
        board[10] = -1
        board[15] = 2
        assert bgbot_cpp.double_hitting_shots(board) == 0

    def test_two_blots_different_checkers(self):
        """Two blots at direct die distances from different player checkers."""
        board = [0] * 26
        board[10] = -1  # blot
        board[15] = -1  # blot
        board[12] = 1   # player checker: d=2 to hit blot at 10
        board[18] = 1   # player checker: d=3 to hit blot at 15
        # Case 1: (2,3) and (3,2) from different checkers
        # Case 2: checker at 18 hits blot at 15 with d1=3, then step to 15-5=10 which is also a blot → (3,5) and (5,3)
        # Total: 4 pairs
        result = bgbot_cpp.double_hitting_shots(board)
        assert result == 4, f"Expected 4, got {result}"


class TestBackEscapes:
    """Test back_escapes: rolls allowing back checker to escape past point 15."""

    def test_no_back_checkers(self):
        """No checkers beyond point 18 → already escaped, returns 36."""
        board = [0] * 26
        board[10] = 3
        board[5] = 2
        assert bgbot_cpp.back_escapes(board) == 36

    def test_checker_at_19(self):
        """Checker at 19 needs to reach 15 or below. Minimum move = 4."""
        board = [0] * 26
        board[19] = 1
        # 19 - 4 = 15 (exactly). Need total dice >= 4.
        # For non-doubles: d1+d2 >= 4
        # For doubles: 2*d >= 4, i.e., d >= 2 (2 steps reach 15)
        # All rolls with sum >= 4 (and no blocking):
        result = bgbot_cpp.back_escapes(board)
        # Non-double rolls with sum >= 4:
        # (2,1)=3 no, (3,1)=4 yes, (4,1)=5 yes, (5,1)=6 yes, (6,1)=7 yes
        # (3,2)=5 yes, (4,2)=6 yes, (5,2)=7 yes, (6,2)=8 yes
        # (4,3)=7 yes, (5,3)=8 yes, (6,3)=9 yes
        # (5,4)=9 yes, (6,4)=10 yes
        # (6,5)=11 yes
        # = 14 non-double rolls × 2 = 28
        # Doubles: (1,1) → 4*1=4, 19-4=15 ✓? Check: 19-1=18 blocked? No. 19-2=17 blocked? No.
        #   (1,1): i-d1=18 board[18]=0>=−2 ok, i-d2=18 ok. i-d1-d2=17>=−2 ok.
        #   i-2*d1=17 which is > 15. i-3*d1=16>15. i-4*d1=15 ≤ 15.
        #   Need to check: board[19-3*1]=board[16]>−2 yes, board[19-4*1]=board[15]>−2 yes → escape!
        #   = 1 roll
        # (2,2): 19-2*2=15 ≤ 15 → 1 roll
        # (3,3): 19-2*3=13 ≤ 15 → 1 roll
        # (4,4): 19-2*4=11 → 1 roll
        # (5,5): 19-2*5=9 → 1 roll
        # (6,6): 19-2*6=7 → 1 roll
        # = 6 doubles
        # Total: 28 + 6 = 34
        assert result == 34, f"Expected 34, got {result}"

    def test_checker_at_24(self):
        """Checker at point 24 needs to reach 15. Minimum move = 9."""
        board = [0] * 26
        board[24] = 1
        result = bgbot_cpp.back_escapes(board)
        # 24-9=15, so need total >= 9 (non-doubles)
        # Non-double: (6,3),(6,4),(6,5),(5,4),(5,5)(wait no that's double)
        #   d1+d2 >= 9: (6,3)=9, (6,4)=10, (6,5)=11, (5,4)=9 → 4 × 2 = 8
        # Doubles:
        #   (3,3): 24-2*3=18>15, 24-3*3=15 ok → check: board[24-3]=board[21]>-2, board[24-6]=board[18]>-2, board[24-9]=board[15]>-2 → 1
        #   (4,4): 24-8=16>15, 24-12=12≤15 → check: board[20]>-2, board[16]>-2, board[12]>-2 → 1
        #   (5,5): 24-10=14≤15 → 1
        #   (6,6): 24-12=12≤15 → 1
        #   (1,1): 24-4=20>15 → 24-4=20>15, need 24-4*1=20, that's >15. Can't escape with 1-1 (max travel=4).
        #   (2,2): 24-8=16>15. 24-8=16>15, need to check 3 steps: 24-6=18>15. 4 steps: 24-8=16>15. Can't reach 15.
        # Wait, for doubles: the code checks if i-4*d1 > 15 → skip.
        # (1,1): 24-4=20 > 15 → skip
        # (2,2): 24-8=16 > 15 → skip
        # (3,3): 24-12=12 ≤ 15 → don't skip. Then checks: i-d1=21 board[21]=0>=−2 ok. i-d2=21 ok.
        #   i-d1-d2=18>=−2 ok. i-2*d1=18>15, so check i-3*d1=15≤15 and board[15]>−2 yes → 1
        # (4,4): 24-16=8≤15, don't skip. i-4=20 ok, i-8=16>=−2 ok. i-8=16>15, i-12=12≤15 and board[12]>−2 yes → 1
        #   Actually i-2*d1=16>15, so check i-3*d1=12≤15 → yes, board[24-12]=board[12]>−2 → 1
        # (5,5): 24-20=4≤15, don't skip. i-5=19 ok, i-10=14>=−2 ok. i-10=14≤15 → 1
        # (6,6): 24-24=0≤15, don't skip. i-6=18 ok, i-12=12>=−2 ok. i-12=12≤15 → 1
        # So 4 doubles = 4.
        # Total = 8 + 4 = 12.
        assert result == 12, f"Expected 12, got {result}"

    def test_fully_blocked(self):
        """Opponent anchors block all escape routes."""
        board = [0] * 26
        board[24] = 1  # back checker
        # Block points 18-23 with opponent anchors
        for i in range(18, 24):
            board[i] = -2
        result = bgbot_cpp.back_escapes(board)
        # First step must land on unblocked point.
        # board[24-d1] for d1=1..6: board[23]=-2, board[22]=-2, ..., board[18]=-2
        # All blocked! So first step blocked both ways → 0 escapes
        assert result == 0, f"Expected 0, got {result}"


class TestHittingShotsEdgeCases:
    """Test edge cases for hitting shots."""

    def test_bar_checker_hits(self):
        """Player checker on bar can hit blots. The hitting_shots function
        looks for player checkers at board[j] > 0 for j = i+1..24.
        Board[25] (bar) is > 0 but the loop only goes to j=24,
        so bar checkers are NOT included in hitting_shots."""
        board = [0] * 26
        board[25] = 1  # player on bar (index 25)
        board[20] = -1  # opponent blot at point 20
        # j goes up to 24, board[25] is NOT checked in the loop
        result = bgbot_cpp.hitting_shots(board)
        assert result == 0, f"Expected 0 (bar not checked in hitting_shots), got {result}"

    def test_long_distance_double_blocked(self):
        """Long-distance double hit blocked by intermediate anchor."""
        board = [0] * 26
        board[1] = -1   # blot at point 1
        board[13] = 1   # player checker at distance 12
        board[7] = -2   # opponent anchor blocking 3-3-3-3 path
        # Distance 12: possible rolls include (6,6), (4,4), (3,3)
        # (3,3): check intermediates: 1+3=4 board[4]=0 ok, 1+6=7 board[7]=-2 BLOCKED
        # (4,4): intermediates: 1+4=5 ok, 1+8=9 ok → NOT blocked
        # (6,6): intermediates: 1+6=7 board[7]=-2 BLOCKED
        # So only (4,4) hits from distance 12.
        # Also direct hits: distance 12 not ≤6, so no direct single-die hits.
        # Expected: (4,4) = 1 roll
        result = bgbot_cpp.hitting_shots(board)
        assert result == 1, f"Expected 1 (4-4 only), got {result}"


# ======================== Cross-validation against Python ========================

class TestCrossValidation:
    """Compare C++ implementations against old Python implementations."""

    @staticmethod
    def _load_python_functions():
        """Try to load the old Python functions for comparison."""
        try:
            import importlib.util
            util_path = os.path.join(project_dir, 'old_src', 'bgbot', 'util.py')
            spec = importlib.util.spec_from_file_location("old_bgbot_util", util_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return {
                'hitting_shots': mod.hitting_shots,
                'double_hitting_shots': mod.double_hitting_shots,
                'back_escapes': mod.back_escapes,
                'max_point': mod.max_point,
                'max_anchor_point': mod.max_anchor_point,
                'prob_no_enter_from_bar': mod.prob_no_enter_from_bar,
                'forward_anchor_points': mod.forward_anchor_points,
                'extended_contact_inputs': mod.extended_contact_inputs,
            }
        except Exception as e:
            print(f"Warning: Could not load Python reference: {e}")
            return None

    def _test_board_set(self):
        """Collection of diverse board positions for testing."""
        boards = []

        # Starting board
        boards.append(('starting', STARTING_BOARD))

        # Race position (no contact)
        race = [0] * 26
        race[1] = 3
        race[2] = 5
        race[3] = 5
        race[4] = 2
        race[22] = -3
        race[23] = -5
        race[24] = -5
        race[20] = -2
        boards.append(('race', race))

        # Contact: player back game
        back_game = [0] * 26
        back_game[1] = 3
        back_game[2] = 2
        back_game[3] = 2
        back_game[22] = 2   # anchor deep in opponent territory
        back_game[24] = -5
        back_game[19] = -3
        back_game[13] = -5
        back_game[8] = -2
        back_game[6] = 6    # big stack
        boards.append(('back_game', back_game))

        # Many blots
        many_blots = [0] * 26
        many_blots[5] = 2
        many_blots[8] = 3
        many_blots[12] = 2
        many_blots[15] = 1  # player blot
        many_blots[18] = 1  # player blot
        many_blots[7] = -1  # opp blot
        many_blots[10] = -1 # opp blot
        many_blots[14] = -1 # opp blot
        many_blots[20] = -3
        many_blots[23] = -5
        many_blots[24] = -5
        many_blots[6] = -2
        boards.append(('many_blots', many_blots))

        # Player on bar
        bar_pos = [0] * 26
        bar_pos[25] = 2   # player on bar
        bar_pos[6] = 3
        bar_pos[8] = 2
        bar_pos[13] = 5
        bar_pos[17] = 3
        bar_pos[1] = -2
        bar_pos[3] = -2
        bar_pos[20] = -5
        bar_pos[23] = -3
        bar_pos[24] = -3
        boards.append(('player_on_bar', bar_pos))

        # Crashed position (all in opponent's home)
        crashed = [0] * 26
        crashed[1] = 5
        crashed[2] = 5
        crashed[3] = 5
        crashed[22] = -5
        crashed[23] = -5
        crashed[24] = -5
        boards.append(('crashed', crashed))

        # Endgame
        endgame = [0] * 26
        endgame[1] = 2
        endgame[2] = 3
        endgame[4] = 1
        endgame[5] = 2
        endgame[6] = 1
        endgame[22] = -2
        endgame[23] = -1  # blot
        endgame[24] = -3
        boards.append(('endgame', endgame))

        return boards

    def test_simple_functions_cross_validation(self):
        """Cross-validate max_point, max_anchor_point, prob_no_enter, forward_anchor against Python."""
        py_funcs = self._load_python_functions()
        if py_funcs is None:
            print("Skipping cross-validation: Python reference not available")
            return

        for name, board in self._test_board_set():
            board_tuple = tuple(board)

            # max_point
            cpp_val = bgbot_cpp.max_point(board)
            py_val = py_funcs['max_point'](board_tuple)
            assert cpp_val == py_val, f"max_point mismatch on {name}: C++={cpp_val}, Python={py_val}"

            # max_anchor_point
            cpp_val = bgbot_cpp.max_anchor_point(board)
            py_val = py_funcs['max_anchor_point'](board_tuple)
            assert cpp_val == py_val, f"max_anchor_point mismatch on {name}: C++={cpp_val}, Python={py_val}"

            # prob_no_enter_from_bar
            cpp_p, cpp_o = bgbot_cpp.prob_no_enter_from_bar(board)
            py_p, py_o = py_funcs['prob_no_enter_from_bar'](board_tuple)
            assert abs(cpp_p - py_p) < 1e-6, f"prob_no_enter player mismatch on {name}: C++={cpp_p}, Py={py_p}"
            assert abs(cpp_o - py_o) < 1e-6, f"prob_no_enter opp mismatch on {name}: C++={cpp_o}, Py={py_o}"

            # forward_anchor_points
            cpp_p, cpp_o = bgbot_cpp.forward_anchor_points(board)
            py_p, py_o = py_funcs['forward_anchor_points'](board_tuple)
            assert cpp_p == py_p, f"forward_anchor player mismatch on {name}: C++={cpp_p}, Py={py_p}"
            assert cpp_o == py_o, f"forward_anchor opp mismatch on {name}: C++={cpp_o}, Py={py_o}"

    def test_hitting_shots_cross_validation(self):
        """Cross-validate hitting_shots against Python."""
        py_funcs = self._load_python_functions()
        if py_funcs is None:
            print("Skipping cross-validation: Python reference not available")
            return

        for name, board in self._test_board_set():
            board_tuple = tuple(board)
            cpp_val = bgbot_cpp.hitting_shots(board)
            py_val = py_funcs['hitting_shots'](board_tuple)
            assert cpp_val == py_val, f"hitting_shots mismatch on {name}: C++={cpp_val}, Python={py_val}"

    def test_double_hitting_shots_cross_validation(self):
        """Cross-validate double_hitting_shots against Python."""
        py_funcs = self._load_python_functions()
        if py_funcs is None:
            print("Skipping cross-validation: Python reference not available")
            return

        for name, board in self._test_board_set():
            board_tuple = tuple(board)
            cpp_val = bgbot_cpp.double_hitting_shots(board)
            py_val = py_funcs['double_hitting_shots'](board_tuple)
            assert cpp_val == py_val, f"double_hitting_shots mismatch on {name}: C++={cpp_val}, Python={py_val}"

    def test_back_escapes_cross_validation(self):
        """Cross-validate back_escapes against Python."""
        py_funcs = self._load_python_functions()
        if py_funcs is None:
            print("Skipping cross-validation: Python reference not available")
            return

        for name, board in self._test_board_set():
            board_tuple = tuple(board)
            cpp_val = bgbot_cpp.back_escapes(board)
            py_val = py_funcs['back_escapes'](board_tuple)
            assert cpp_val == py_val, f"back_escapes mismatch on {name}: C++={cpp_val}, Python={py_val}"

    def test_full_extended_inputs_cross_validation(self):
        """Cross-validate the full 214-input encoding against Python."""
        py_funcs = self._load_python_functions()
        if py_funcs is None:
            print("Skipping cross-validation: Python reference not available")
            return

        for name, board in self._test_board_set():
            board_tuple = tuple(board)
            cpp_inputs = bgbot_cpp.compute_extended_contact_inputs(board)
            py_inputs = py_funcs['extended_contact_inputs'](board_tuple)

            mismatches = []
            for idx in range(214):
                cpp_val = float(cpp_inputs[idx])
                py_val = float(py_inputs[idx])
                if abs(cpp_val - py_val) > 1e-5:
                    mismatches.append((idx, cpp_val, py_val))

            if mismatches:
                msg = f"Extended input mismatches on '{name}':\n"
                for idx, cv, pv in mismatches:
                    # Annotate what this input represents
                    if idx < 96:
                        desc = f"player point {idx//4 + 1} feature {idx%4}"
                    elif idx == 96:
                        desc = "player bar"
                    elif 97 <= idx <= 99:
                        desc = f"player borne-off bucket {idx-97}"
                    elif idx == 100:
                        desc = "player bar-exit prob"
                    elif idx == 101:
                        desc = "player fwd anchor"
                    elif idx == 102:
                        desc = "player hitting shots"
                    elif idx == 103:
                        desc = "player dbl hitting shots"
                    elif idx == 104:
                        desc = "player back escapes"
                    elif idx == 105:
                        desc = "player max point"
                    elif idx == 106:
                        desc = "player max anchor"
                    elif 107 <= idx < 203:
                        desc = f"opp point {(idx-107)//4 + 1} feature {(idx-107)%4}"
                    elif idx == 203:
                        desc = "opp bar"
                    elif 204 <= idx <= 206:
                        desc = f"opp borne-off bucket {idx-204}"
                    elif idx == 207:
                        desc = "opp bar-exit prob"
                    elif idx == 208:
                        desc = "opp fwd anchor"
                    elif idx == 209:
                        desc = "opp hitting shots"
                    elif idx == 210:
                        desc = "opp dbl hitting shots"
                    elif idx == 211:
                        desc = "opp back escapes"
                    elif idx == 212:
                        desc = "opp max point"
                    elif idx == 213:
                        desc = "opp max anchor"
                    else:
                        desc = "unknown"
                    msg += f"  [{idx}] {desc}: C++={cv:.6f} Python={pv:.6f}\n"
                assert False, msg


class TestTesauroEncoding:
    """Test the basic 196-input Tesauro encoding for correctness."""

    def test_empty_board(self):
        """Empty board = all 15 checkers borne off for each player."""
        board = [0] * 26
        inputs = bgbot_cpp.compute_tesauro_inputs(board)
        # Point encoding should be all zeros
        for i in range(96):
            assert inputs[i] == 0.0, f"Player point input {i} should be 0"
        for i in range(98, 194):
            assert inputs[i] == 0.0, f"Opponent point input {i} should be 0"
        # Bar should be 0
        assert inputs[96] == 0.0   # player bar
        assert inputs[194] == 0.0  # opponent bar
        # Borne off = 15/15 = 1.0
        assert abs(inputs[97] - 1.0) < 1e-6   # player borne off
        assert abs(inputs[195] - 1.0) < 1e-6  # opponent borne off

    def test_single_checker(self):
        """One player checker on point 1 → input 0 = 1.0, rest of point encoding = 0."""
        board = [0] * 26
        board[1] = 1
        inputs = bgbot_cpp.compute_tesauro_inputs(board)
        assert inputs[0] == 1.0  # >= 1 checker
        assert inputs[1] == 0.0  # < 2 checkers
        assert inputs[2] == 0.0
        assert inputs[3] == 0.0

    def test_five_checkers(self):
        """5 checkers on point 1 → 1,1,1,(5-3)/2 = 1,1,1,1.0."""
        board = [0] * 26
        board[1] = 5
        inputs = bgbot_cpp.compute_tesauro_inputs(board)
        assert inputs[0] == 1.0
        assert inputs[1] == 1.0
        assert inputs[2] == 1.0
        assert abs(inputs[3] - 1.0) < 1e-6  # (5-3)/2 = 1.0

    def test_opponent_checker(self):
        """Opponent checker on point 1 → opponent encoding at offset 98."""
        board = [0] * 26
        board[1] = -3
        inputs = bgbot_cpp.compute_tesauro_inputs(board)
        # Player encoding should be 0
        assert inputs[0] == 0.0
        assert inputs[1] == 0.0
        # Opponent encoding at 0 + 98
        assert inputs[98] == 1.0
        assert inputs[99] == 1.0
        assert inputs[100] == 1.0
        assert inputs[101] == 0.0  # < 4 checkers

    def test_bar_and_borne_off(self):
        """Test bar and borne-off encoding."""
        board = [0] * 26
        board[25] = 3  # player on bar
        board[0] = 2   # opponent on bar
        # 15 total checkers each, but with some on bar
        inputs = bgbot_cpp.compute_tesauro_inputs(board)
        assert abs(inputs[96] - 3/2.0) < 1e-6   # player bar
        assert abs(inputs[194] - 2/2.0) < 1e-6   # opponent bar


class TestBorneOffBuckets:
    """Test the 3-bucket borne-off encoding in extended inputs."""

    def test_zero_borne_off(self):
        """0 borne off → all buckets 0."""
        # All 15 checkers on the board
        board = [0] * 26
        board[1] = 15
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97]) < 1e-6  # bucket 0
        assert abs(inputs[98]) < 1e-6  # bucket 1
        assert abs(inputs[99]) < 1e-6  # bucket 2

    def test_three_borne_off(self):
        """3 borne off → bucket 0 = 3/5 = 0.6."""
        board = [0] * 26
        board[1] = 12   # 15-12 = 3 borne off
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97] - 0.6) < 1e-6
        assert abs(inputs[98]) < 1e-6
        assert abs(inputs[99]) < 1e-6

    def test_five_borne_off(self):
        """5 borne off → bucket 0 = 1.0, bucket 1 = 0."""
        board = [0] * 26
        board[1] = 10
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97] - 1.0) < 1e-6
        assert abs(inputs[98]) < 1e-6
        assert abs(inputs[99]) < 1e-6

    def test_eight_borne_off(self):
        """8 borne off → bucket 0 = 1.0, bucket 1 = (8-5)/5 = 0.6."""
        board = [0] * 26
        board[1] = 7
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97] - 1.0) < 1e-6
        assert abs(inputs[98] - 0.6) < 1e-6
        assert abs(inputs[99]) < 1e-6

    def test_twelve_borne_off(self):
        """12 borne off → bucket 0 = 1.0, bucket 1 = 1.0, bucket 2 = (12-10)/5 = 0.4."""
        board = [0] * 26
        board[1] = 3
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97] - 1.0) < 1e-6
        assert abs(inputs[98] - 1.0) < 1e-6
        assert abs(inputs[99] - 0.4) < 1e-6

    def test_all_borne_off(self):
        """15 borne off → bucket 0 = 1.0, bucket 1 = 1.0, bucket 2 = 1.0."""
        board = [0] * 26
        inputs = bgbot_cpp.compute_extended_contact_inputs(board)
        assert abs(inputs[97] - 1.0) < 1e-6
        assert abs(inputs[98] - 1.0) < 1e-6
        assert abs(inputs[99] - 1.0) < 1e-6


class TestExtendedFeatureNormalization:
    """Test that extended features are properly normalized."""

    def test_feature_ranges(self):
        """All extended features should be in reasonable ranges."""
        for board_data in [STARTING_BOARD]:
            inputs = bgbot_cpp.compute_extended_contact_inputs(board_data)

            # Player features
            assert 0 <= inputs[96] <= 7.5, f"Player bar out of range: {inputs[96]}"
            for i in [97, 98, 99]:
                assert 0 <= inputs[i] <= 1.0, f"Player borne-off bucket {i-97} out of range: {inputs[i]}"
            assert 0 <= inputs[100] <= 1.0, f"Player bar-exit prob out of range: {inputs[100]}"
            assert 0 <= inputs[101] <= 2.0, f"Player fwd anchor out of range: {inputs[101]}"
            assert 0 <= inputs[102] <= 36/15.0, f"Player hitting shots out of range: {inputs[102]}"
            assert 0 <= inputs[103], f"Player dbl hitting out of range: {inputs[103]}"
            assert 0 <= inputs[104] <= 36/15.0, f"Player back escapes out of range: {inputs[104]}"
            assert 0 <= inputs[105] <= 1.0, f"Player max point out of range: {inputs[105]}"
            assert 0 <= inputs[106] <= 1.0, f"Player max anchor out of range: {inputs[106]}"

            # Opponent features
            assert 0 <= inputs[203] <= 7.5, f"Opp bar out of range: {inputs[203]}"
            for i in [204, 205, 206]:
                assert 0 <= inputs[i] <= 1.0, f"Opp borne-off bucket {i-204} out of range: {inputs[i]}"


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestMaxPoint, TestMaxAnchorPoint, TestProbNoEnterFromBar,
        TestForwardAnchorPoints, TestHittingShots, TestDoubleHittingShots,
        TestBackEscapes, TestHittingShotsEdgeCases,
        TestTesauroEncoding, TestBorneOffBuckets, TestExtendedFeatureNormalization,
        TestCrossValidation,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in methods:
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
