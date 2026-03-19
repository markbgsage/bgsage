"""Tests for the one-sided bearoff database."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

import bgbot_cpp
from bgsage import BgBotAnalyzer
from bgsage.weights import bearoff_db_path

N_POSITIONS = 54264

def load_db():
    d = bgbot_cpp.BearoffDB()
    path = bearoff_db_path()
    assert path is not None, "Bearoff DB file not found"
    assert d.load(path), "Failed to load bearoff DB"
    return d


def test_position_indexing_roundtrip():
    """Every index should roundtrip through index_to_position and position_index."""
    failed = 0
    for idx in range(N_POSITIONS):
        checkers = bgbot_cpp.BearoffDB.index_to_position(idx)
        idx2 = bgbot_cpp.BearoffDB.position_index(checkers)
        if idx != idx2:
            failed += 1
            if failed <= 3:
                print(f"  FAIL: index {idx} -> checkers {checkers} -> index {idx2}")
    assert failed == 0, f"{failed} roundtrip failures"
    print(f"  PASS: All {N_POSITIONS} indices roundtrip correctly")


def test_is_bearoff():
    db = load_db()

    # Valid bearoff
    board = [0, 5, 3, 2, 0, 0, 0, 0,0,0,0,0,0, 0,0,0,0,0,0, -5,-3,-2,0,0,0, 0]
    assert db.is_bearoff(board), "Should be bearoff"

    # Bar checker
    board_bar = [0, 5, 3, 2, 0, 0, 0, 0,0,0,0,0,0, 0,0,0,0,0,0, -5,-3,-2,0,0,0, 1]
    assert not db.is_bearoff(board_bar), "Bar checker should not be bearoff"

    # Outfield checker
    board_out = [0, 4, 3, 2, 0, 0, 0, 1, 0,0,0,0,0, 0,0,0,0,0,0, -5,-3,-2,0,0,0, 0]
    assert not db.is_bearoff(board_out), "Outfield checker should not be bearoff"

    # Starting position
    board_start = [0,-2,0,0,0,0,5, 0,3,0,0,0,-5, 5,0,0,0,-3,0, -5,0,0,0,0,2, 0]
    assert not db.is_bearoff(board_start), "Starting position should not be bearoff"

    # All borne off
    assert db.is_bearoff([0]*26), "Empty board should be bearoff"

    print("  PASS: is_bearoff checks")


def test_one_sided_lookups():
    db = load_db()

    # 1 checker on point 1: mean_rolls = 1.0
    board1 = [0,1,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,-1, 0]
    mean = db.get_mean_rolls(board1, 0)
    assert abs(mean - 1.0) < 0.01, f"Expected mean=1.0, got {mean}"

    # EPC for 1 on pt 1
    epc = db.lookup_epc(board1, 0)
    assert abs(epc - 49/6) < 0.01, f"Expected EPC=8.1667, got {epc}"

    # Terminal: mean_rolls = 0
    mean_t = db.get_mean_rolls([0]*26, 0)
    assert abs(mean_t) < 0.001, f"Expected terminal mean=0, got {mean_t}"

    print("  PASS: one-sided lookups")


def test_two_sided_probabilities():
    db = load_db()

    # Symmetric: player on roll advantage
    board = [0,5,5,5,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,-5,-5,-5,0, 0]
    probs = db.lookup_probs(board)
    assert probs[0] > 0.5, f"Player on roll should win > 50%, got {probs[0]:.4f}"

    # Massive advantage: 1 on pt 1 vs 15 on opp pt 6
    board2 = [0,1,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, -15,0,0,0,0,0, 0]
    probs2 = db.lookup_probs(board2)
    assert probs2[0] > 0.999, f"Expected P(win) ~1.0, got {probs2[0]:.4f}"
    assert probs2[1] > 0.999, f"Expected P(gw) ~1.0, got {probs2[1]:.4f}"

    # No backgammons
    board3 = [0,3,3,3,3,3,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,-3,-3,-3,-3,-3, 0]
    probs3 = db.lookup_probs(board3)
    assert probs3[2] == 0.0, f"P(bw) should be 0, got {probs3[2]}"
    assert probs3[4] == 0.0, f"P(bl) should be 0, got {probs3[4]}"

    # No gammon loss when player has checkers already off
    board4 = [0,5,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,-15, 0]
    probs4 = db.lookup_probs(board4)
    assert probs4[3] == 0.0, f"P(gl) should be 0 when player has checkers off, got {probs4[3]}"

    # Win prob in range
    board5 = [0,3,2,3,2,3,2, 0,0,0,0,0,0, 0,0,0,0,0,0, -2,-3,-2,-3,-2,-3, 0]
    probs5 = db.lookup_probs(board5)
    assert 0 <= probs5[0] <= 1
    assert 0 <= probs5[1] <= probs5[0]  # gammon_win <= win
    eq = 2*probs5[0] - 1 + probs5[1] - probs5[3]
    assert -2 <= eq <= 2

    print("  PASS: two-sided probabilities")


def test_analyzer_integration():
    analyzer = BgBotAnalyzer(eval_level="1ply")

    # EPC works
    board = [0,3,3,3,3,3,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,-3,-3,-3,-3,-3, 0]
    epc = analyzer.epc(board, 0)
    assert epc is not None and epc > 0, f"Expected positive EPC, got {epc}"

    # Non-bearoff returns None
    board_start = [0,-2,0,0,0,0,5, 0,3,0,0,0,-5, 5,0,0,0,-3,0, -5,0,0,0,0,2, 0]
    assert analyzer.epc(board_start, 0) is None

    # Disabled
    analyzer_no_db = BgBotAnalyzer(eval_level="1ply", bearoff_db=False)
    assert analyzer_no_db._bearoff_db is None

    print("  PASS: analyzer integration")


def test_4ply_cubeless_matches_db():
    """4-ply cube action cubeless P(win) must exactly match the bearoff DB value."""
    db = load_db()
    board = [0,1,0,0,3,2,2,0,0,0,0,0,0,0,0,0,0,0,0,-4,-1,0,0,-2,-1,0]
    exact_probs = db.lookup_probs(board, post_move=False)
    exact_pwin = exact_probs[0]

    analyzer = BgBotAnalyzer(eval_level='4ply', bearoff_db=True)
    cube = analyzer.cube_action(board, cube_value=1, cube_owner='centered')

    assert cube.probs.win == exact_pwin, (
        f"4-ply P(win) {cube.probs.win:.6f} != DB exact {exact_pwin:.6f}")
    print(f"  PASS: 4-ply P(win) = {cube.probs.win:.6f} matches DB exactly")


def test_roller_pp_cubeless_matches_db():
    """XG Roller++ equivalent cubeless P(win) must exactly match the bearoff DB value."""
    db = load_db()
    board = [0,1,0,0,3,2,2,0,0,0,0,0,0,0,0,0,0,0,0,-4,-1,0,0,-2,-1,0]
    exact_probs = db.lookup_probs(board, post_move=False)
    exact_pwin = exact_probs[0]

    analyzer = BgBotAnalyzer(eval_level='truncated3', bearoff_db=True)
    cube = analyzer.cube_action(board, cube_value=1, cube_owner='centered')

    assert cube.probs.win == exact_pwin, (
        f"Roller++ P(win) {cube.probs.win:.6f} != DB exact {exact_pwin:.6f}")
    print(f"  PASS: Roller++ P(win) = {cube.probs.win:.6f} matches DB exactly")


if __name__ == "__main__":
    tests = [
        ("Position indexing roundtrip", test_position_indexing_roundtrip),
        ("is_bearoff", test_is_bearoff),
        ("One-sided lookups", test_one_sided_lookups),
        ("Two-sided probabilities", test_two_sided_probabilities),
        ("Analyzer integration", test_analyzer_integration),
        ("4-ply cubeless matches DB", test_4ply_cubeless_matches_db),
        ("Roller++ cubeless matches DB", test_roller_pp_cubeless_matches_db),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
