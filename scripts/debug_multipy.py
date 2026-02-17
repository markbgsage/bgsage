"""
Debug multi-ply: trace through specific scenarios to find the 1-ply bug.
Focus on purerace to eliminate game plan complexity.
"""

import os
import sys

# Setup import paths
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
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')

NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250


def get_best_weights():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    paths = {}
    for t in types:
        path = os.path.join(MODELS_DIR, f'sl_{t}.weights.best')
        if os.path.exists(path):
            paths[t] = path
        else:
            return None
    return paths


def main():
    weights = get_best_weights()
    if weights is None:
        print("Cannot find all weight files.")
        sys.exit(1)

    # Create strategies
    multipy_0 = bgbot_cpp.create_multipy_5nn(
        weights['purerace'], weights['racing'],
        weights['attacking'], weights['priming'], weights['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=0)

    multipy_1 = bgbot_cpp.create_multipy_5nn(
        weights['purerace'], weights['racing'],
        weights['attacking'], weights['priming'], weights['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=1,
        filter_max_moves=100,
        filter_threshold=10.0)  # No filtering

    # Load a few purerace scenarios
    bm_file = os.path.join(DATA_DIR, 'purerace.bm')
    scenarios = load_benchmark_file(bm_file, step=11100)  # ~10 scenarios
    n = scenarios.size()
    print(f"Loaded {n} purerace scenarios\n")

    # Get the scenario data by reading the raw file
    # Actually, let's just use the C++ functions directly
    # For each scenario, generate candidates, evaluate at 0-ply and 1-ply

    # First, let's create a simple test position manually.
    # A pure race: P1 has 15 checkers on point 1, P2 has 15 on point 24
    # (extremely simple — P1 needs to bear off 15 from point 1)
    print("=" * 60)
    print("TEST 1: Simple symmetric race position")
    print("=" * 60)
    board = [0] * 26
    board[4] = 5  # P1: 5 checkers on point 4
    board[5] = 5  # P1: 5 on point 5
    board[6] = 5  # P1: 5 on point 6
    board[20] = -5  # P2: 5 on point 20
    board[21] = -5
    board[22] = -5

    print(f"Board: {board}")
    print(f"is_race: {bgbot_cpp.is_race(board)}")

    # Evaluate at 0-ply
    res0 = multipy_0.evaluate_board(board, board)
    print(f"\n0-ply evaluation:")
    print(f"  probs: {[f'{p:.4f}' for p in res0['probs']]}")
    print(f"  equity: {res0['equity']:.4f}")

    # Evaluate at 1-ply
    multipy_1.clear_cache()
    res1 = multipy_1.evaluate_board(board, board)
    print(f"\n1-ply evaluation:")
    print(f"  probs: {[f'{p:.4f}' for p in res1['probs']]}")
    print(f"  equity: {res1['equity']:.4f}")

    # Flip and evaluate (opponent's perspective)
    flipped = bgbot_cpp.flip_board(board)
    print(f"\nFlipped board: {flipped}")
    res0_flip = multipy_0.evaluate_board(flipped, flipped)
    print(f"0-ply flipped:")
    print(f"  probs: {[f'{p:.4f}' for p in res0_flip['probs']]}")
    print(f"  equity: {res0_flip['equity']:.4f}")

    # Check: inverted flipped should approximately equal original
    inv = bgbot_cpp.invert_probs_py(res0_flip['probs'])
    inv_eq = bgbot_cpp.NeuralNetwork.compute_equity(inv)
    print(f"Inverted flipped probs: {[f'{p:.4f}' for p in inv]}")
    print(f"Inverted equity: {inv_eq:.4f}")
    print(f"Original equity: {res0['equity']:.4f}")
    print(f"Sum (should be ~0 for symmetric): {res0['equity'] + res0_flip['equity']:.4f}")

    # Now test a concrete move selection scenario
    print("\n" + "=" * 60)
    print("TEST 2: Move selection comparison")
    print("=" * 60)

    # Create a slightly asymmetric position
    board2 = [0] * 26
    board2[1] = 2  # P1 bearing off
    board2[2] = 3
    board2[3] = 3
    board2[4] = 4
    board2[5] = 3
    board2[20] = -3
    board2[21] = -3
    board2[22] = -3
    board2[23] = -3
    board2[24] = -3
    print(f"Board2: {board2}")
    print(f"is_race: {bgbot_cpp.is_race(board2)}")

    # Roll 6-5
    d1, d2 = 6, 5
    candidates = bgbot_cpp.possible_moves(board2, d1, d2)
    print(f"\nDice: {d1}-{d2}, {len(candidates)} legal moves")

    # Evaluate each candidate at 0-ply and 1-ply
    print(f"\n{'Idx':>3} {'0-ply eq':>10} {'1-ply eq':>10} {'diff':>10}")
    print("-" * 40)
    multipy_1.clear_cache()

    equities_0 = []
    equities_1 = []
    for i, cand in enumerate(candidates):
        res0 = multipy_0.evaluate_board(cand, board2)
        res1 = multipy_1.evaluate_board(cand, board2)
        equities_0.append(res0['equity'])
        equities_1.append(res1['equity'])
        diff = res1['equity'] - res0['equity']
        print(f"{i:3d} {res0['equity']:10.4f} {res1['equity']:10.4f} {diff:+10.4f}")

    best_0 = max(range(len(candidates)), key=lambda i: equities_0[i])
    best_1 = max(range(len(candidates)), key=lambda i: equities_1[i])
    print(f"\n0-ply best: candidate {best_0} (eq={equities_0[best_0]:.4f})")
    print(f"1-ply best: candidate {best_1} (eq={equities_1[best_1]:.4f})")
    print(f"Same choice: {'YES' if best_0 == best_1 else 'NO'}")

    # Now trace through the 1-ply evaluation of the 0-ply best candidate
    print("\n" + "=" * 60)
    print(f"TEST 3: Trace 1-ply for candidate {best_0}")
    print("=" * 60)
    cand = candidates[best_0]
    print(f"Candidate board: {cand}")

    # Step 1: Flip to opponent's perspective
    opp_board = bgbot_cpp.flip_board(cand)
    print(f"Opp board (flipped): {opp_board}")

    # Step 2: For each dice roll, what does opponent see?
    ALL_ROLLS = [
        (1,1,1), (2,2,1), (3,3,1), (4,4,1), (5,5,1), (6,6,1),
        (1,2,2), (1,3,2), (1,4,2), (1,5,2), (1,6,2),
        (2,3,2), (2,4,2), (2,5,2), (2,6,2),
        (3,4,2), (3,5,2), (3,6,2),
        (4,5,2), (4,6,2),
        (5,6,2)
    ]

    total_weight = 0
    sum_eq = 0.0
    sum_probs = [0.0] * 5

    for d1, d2, weight in ALL_ROLLS:
        opp_cands = bgbot_cpp.possible_moves(opp_board, d1, d2)

        if len(opp_cands) == 1:
            opp_best = opp_cands[0]
        else:
            # Find opponent's best move at 0-ply
            best_opp_eq = -999.0
            best_idx = 0
            for j, oc in enumerate(opp_cands):
                res = multipy_0.evaluate_board(oc, opp_board)
                if res['equity'] > best_opp_eq:
                    best_opp_eq = res['equity']
                    best_idx = j
            opp_best = opp_cands[best_idx]

        # Flip back to player 1's perspective
        back_to_p1 = bgbot_cpp.flip_board(opp_best)

        # Evaluate at 0-ply (base case for 1-ply search)
        res = multipy_0.evaluate_board(back_to_p1, back_to_p1)
        p1_eq = res['equity']
        p1_probs = res['probs']

        total_weight += weight
        sum_eq += weight * p1_eq
        for k in range(5):
            sum_probs[k] += weight * p1_probs[k]

    avg_eq = sum_eq / 36.0
    avg_probs = [p / 36.0 for p in sum_probs]
    expected_eq = bgbot_cpp.NeuralNetwork.compute_equity(avg_probs)

    print(f"\nManual 1-ply avg probs: {[f'{p:.4f}' for p in avg_probs]}")
    print(f"Manual 1-ply avg equity (from probs): {expected_eq:.4f}")
    print(f"Manual 1-ply avg equity (from eq sum): {avg_eq:.4f}")

    # Compare with C++ 1-ply evaluation
    multipy_1.clear_cache()
    cpp_res = multipy_1.evaluate_board(cand, board2)
    print(f"\nC++ 1-ply probs: {[f'{p:.4f}' for p in cpp_res['probs']]}")
    print(f"C++ 1-ply equity: {cpp_res['equity']:.4f}")
    print(f"Match: {'YES' if abs(expected_eq - cpp_res['equity']) < 0.001 else 'NO'}")

    # Also check: what does 0-ply evaluate the candidate as?
    res0_cand = multipy_0.evaluate_board(cand, board2)
    print(f"\n0-ply equity of same candidate: {res0_cand['equity']:.4f}")
    print(f"1-ply equity of same candidate: {cpp_res['equity']:.4f}")
    print(f"1-ply < 0-ply: {cpp_res['equity'] < res0_cand['equity']}")
    print(f"(Expected: 1-ply slightly lower than 0-ply because it accounts for opponent response)")


if __name__ == '__main__':
    main()
