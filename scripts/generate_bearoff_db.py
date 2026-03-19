#!/usr/bin/env python3
"""Generate the one-sided bearoff database for 15 checkers on 6 points.

Usage:
    python scripts/generate_bearoff_db.py [--output data/bearoff_1sided.db]

The database covers all 54,264 positions (C(21,6)) and stores:
- Bearoff distribution: P(all off in exactly k rolls) for k=0..31
- Mean rolls to bear off (for EPC calculation)
- Gammon distribution: P(0 off after k rolls) for positions with all 15 on board

Total size: ~4.7 MB.
"""

import sys
import os
import time
import argparse

# Add build directory to path for bgbot_cpp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")

import bgbot_cpp


def main():
    parser = argparse.ArgumentParser(description="Generate bearoff database")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), '..', 'data', 'bearoff_1sided.db'),
        help="Output path for the database file")
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)

    print("Generating bearoff database (15 checkers, 6 points, 54264 positions)...")
    db = bgbot_cpp.BearoffDB()

    t0 = time.time()
    db.generate()
    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if db.save(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved to {output_path} ({size_mb:.2f} MB)")
    else:
        print("ERROR: Failed to save database")
        sys.exit(1)

    # Validate: reload and check known positions
    print("\nValidating...")
    db2 = bgbot_cpp.BearoffDB()
    if not db2.load(output_path):
        print("ERROR: Failed to reload database")
        sys.exit(1)

    # Test 1: 1 checker on point 1 → mean_rolls = 1.0 (any roll bears it off)
    board_1on1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mean = db2.get_mean_rolls(board_1on1, 0)
    epc = db2.lookup_epc(board_1on1, 0)
    print(f"  1 checker on point 1: mean_rolls={mean:.4f} (expected 1.0), EPC={epc:.4f} (expected {49/6:.4f})")
    assert abs(mean - 1.0) < 0.01, f"Expected mean_rolls=1.0, got {mean}"

    # Test 2: Position index roundtrip
    for idx in [0, 1, 100, 1000, 10000, 54263]:
        checkers = bgbot_cpp.BearoffDB.index_to_position(idx)
        idx2 = bgbot_cpp.BearoffDB.position_index(checkers)
        assert idx == idx2, f"Roundtrip failed for index {idx}: checkers={checkers}, got back {idx2}"
    print("  Position index roundtrip: OK (6 test indices)")

    # Test 3: is_bearoff for a valid bearoff position
    # Player has 5 checkers on point 1, opponent has 5 on point 24 (opp's point 1)
    board_bearoff = [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0]
    assert db2.is_bearoff(board_bearoff), "Expected is_bearoff=True for home board race"
    print("  is_bearoff for home board race: OK")

    # Test 4: NOT a bearoff (checker on point 7)
    board_not_bearoff = [0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0]
    assert not db2.is_bearoff(board_not_bearoff), "Expected is_bearoff=False for checker on point 7"
    print("  is_bearoff for outfield checker: OK")

    # Test 5: Two-sided probability lookup
    probs = db2.lookup_probs(board_bearoff)
    print(f"  Two-sided probs (5v5 on pt 1 vs pt 24): P(win)={probs[0]:.4f}, "
          f"P(gw)={probs[1]:.4f}, P(bw)={probs[2]:.4f}, P(gl)={probs[3]:.4f}, P(bl)={probs[4]:.4f}")
    # Player on roll should win most of the time with equal positions
    assert probs[0] > 0.5, f"Expected P(win) > 0.5 for player on roll, got {probs[0]}"

    # Test 6: All checkers borne off (terminal)
    board_terminal = [0] * 26
    # This is both sides borne off — player has 0 on board, opp has 0 on board
    probs_terminal = db2.lookup_probs(board_terminal)
    mean_terminal = db2.get_mean_rolls(board_terminal, 0)
    print(f"  Terminal (all off): mean_rolls={mean_terminal:.4f}, P(win)={probs_terminal[0]:.4f}")

    print("\nAll validations passed!")


if __name__ == "__main__":
    main()
