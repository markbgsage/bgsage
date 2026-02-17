"""
Debug: verify perspective equivalence.
Does evaluate(flip(board)) give invert(evaluate(board))?
The answer should be NO because the NN evaluates post-move positions.
"""

import os
import sys

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

MODELS_DIR = os.path.join(project_dir, 'models')
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250


def main():
    types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
    wpaths = {t: os.path.join(MODELS_DIR, f'sl_{t}.weights.best') for t in types}

    multipy_0 = bgbot_cpp.create_multipy_5nn(
        wpaths['purerace'], wpaths['racing'],
        wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=0)

    positions = [
        [0, 2, 3, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -3, -3, -3, 0],
        [0, 0, 0, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -5, -5, 0, 0, 0],
        [0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -5, -5, 0],
    ]

    print("=" * 70)
    print("Test 1: evaluate(board) vs invert(evaluate(flip(board)))")
    print("If NN is a true value function, these should match.")
    print("If NN is post-move biased, they will differ systematically.")
    print("=" * 70)

    for i, board in enumerate(positions):
        flipped = bgbot_cpp.flip_board(board)

        res1 = multipy_0.evaluate_board(board, board)
        res2 = multipy_0.evaluate_board(flipped, flipped)
        inv = bgbot_cpp.invert_probs_py(res2['probs'])
        eq_inv = bgbot_cpp.NeuralNetwork.compute_equity(inv)

        print(f"\nPosition {i}:")
        print(f"  evaluate(board):              probs={[f'{p:.5f}' for p in res1['probs']]}  eq={res1['equity']:.5f}")
        print(f"  evaluate(flip(board)):         probs={[f'{p:.5f}' for p in res2['probs']]}  eq={res2['equity']:.5f}")
        print(f"  invert(evaluate(flip(board))): probs={[f'{p:.5f}' for p in inv]}  eq={eq_inv:.5f}")
        print(f"  DIFF: evaluate vs inverted = {res1['equity'] - eq_inv:.5f}")
        print(f"  SUM:  evaluate + flipped   = {res1['equity'] + res2['equity']:.5f} (should be ~0 if value function)")

    # The critical test: after opponent moves
    print("\n\n" + "=" * 70)
    print("Test 2: After opponent move, compare two methods")
    print("Method A: evaluate(opp_best) from opp perspective, invert to P1")
    print("Method B: evaluate(flip(opp_best)) from P1 perspective directly")
    print("=" * 70)

    board = positions[0]
    cands = bgbot_cpp.possible_moves(board, 3, 1)
    if not cands:
        print("No candidates!")
        return
    cand = cands[0]
    opp_board = bgbot_cpp.flip_board(cand)

    opp_cands = bgbot_cpp.possible_moves(opp_board, 4, 2)
    print(f"\nP1 candidate: {cand[:7]}...")
    print(f"Opp board: {opp_board[:7]}...")
    print(f"Opponent has {len(opp_cands)} moves with 4-2")

    total_a = 0.0
    total_b = 0.0
    for j in range(min(5, len(opp_cands))):
        opp_move = opp_cands[j]

        # Method A: evaluate opp_move from opp's perspective, invert
        res_a = multipy_0.evaluate_board(opp_move, opp_board)
        inv_a = bgbot_cpp.invert_probs_py(res_a['probs'])
        eq_a = bgbot_cpp.NeuralNetwork.compute_equity(inv_a)

        # Method B: flip opp_move to P1's perspective, evaluate directly
        back_to_p1 = bgbot_cpp.flip_board(opp_move)
        res_b = multipy_0.evaluate_board(back_to_p1, back_to_p1)

        diff = abs(eq_a - res_b['equity'])
        total_a += eq_a
        total_b += res_b['equity']

        print(f"\n  Opp move {j}:")
        print(f"    Method A (eval opp, invert): P1 eq = {eq_a:.5f}")
        print(f"    Method B (flip back, eval):  P1 eq = {res_b['equity']:.5f}")
        print(f"    Diff: {diff:.5f}  {'MATCH' if diff < 0.001 else 'MISMATCH'}")

    if len(opp_cands) > 0:
        n = min(5, len(opp_cands))
        print(f"\n  Average A: {total_a/n:.5f}")
        print(f"  Average B: {total_b/n:.5f}")
        print(f"  Average diff: {abs(total_a/n - total_b/n):.5f}")

    # Test 3: Does the mismatch explain the benchmark regression?
    print("\n\n" + "=" * 70)
    print("Test 3: Systematic direction of mismatch")
    print("=" * 70)
    print("If Method B (flip-eval) systematically gives HIGHER P1 equity than")
    print("Method A (eval-invert), it means the NN overestimates the just-moved player.")
    print("This could explain why 1-ply using Method B distorts rankings.")

    higher_b = 0
    total_diff = 0.0
    n_tests = 0

    for board in positions:
        for d1 in range(1, 7):
            for d2 in range(d1, 7):
                cands = bgbot_cpp.possible_moves(board, d1, d2)
                if not cands:
                    continue
                cand = cands[0]
                opp_board = bgbot_cpp.flip_board(cand)
                opp_cands = bgbot_cpp.possible_moves(opp_board, d1, d2)
                if not opp_cands:
                    continue

                for oc in opp_cands[:3]:
                    # Method A
                    res_a = multipy_0.evaluate_board(oc, opp_board)
                    inv_a = bgbot_cpp.invert_probs_py(res_a['probs'])
                    eq_a = bgbot_cpp.NeuralNetwork.compute_equity(inv_a)

                    # Method B
                    back = bgbot_cpp.flip_board(oc)
                    res_b = multipy_0.evaluate_board(back, back)

                    n_tests += 1
                    d = res_b['equity'] - eq_a
                    total_diff += d
                    if d > 0:
                        higher_b += 1

    if n_tests > 0:
        print(f"  Tests: {n_tests}")
        print(f"  Method B higher: {higher_b}/{n_tests} ({100*higher_b/n_tests:.1f}%)")
        print(f"  Average diff (B - A): {total_diff/n_tests:.5f}")
        print(f"  {'Method B systematically HIGHER' if total_diff/n_tests > 0 else 'Method B systematically LOWER'}")


if __name__ == '__main__':
    main()
