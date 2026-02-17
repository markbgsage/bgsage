"""
Debug multi-ply: find positions with multiple candidates where 1-ply
changes the ranking vs 0-ply.
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
from bgsage.data import board_from_gnubg_position_string

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


def parse_bm_scenarios(filepath, max_scenarios=200):
    """Parse raw .bm file into Python-readable scenarios."""
    scenarios = []
    with open(filepath) as f:
        for line in f:
            if not line.startswith('m '):
                continue
            if len(scenarios) >= max_scenarios:
                break
            bits = line.split()
            start_board = board_from_gnubg_position_string(bits[1])
            die1, die2 = int(bits[2]), int(bits[3])
            ranked_boards = []
            ranked_errors = []
            idx = 4
            while idx < len(bits):
                board = board_from_gnubg_position_string(bits[idx])
                ranked_boards.append(board)
                if idx + 1 < len(bits):
                    ranked_errors.append(float(bits[idx + 1]))
                else:
                    ranked_errors.append(0.0)
                idx += 2
            scenarios.append({
                'start_board': start_board,
                'die1': die1, 'die2': die2,
                'ranked_boards': ranked_boards,
                'ranked_errors': ranked_errors,
            })
    return scenarios


def main():
    weights = get_best_weights()

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
        filter_threshold=10.0)

    bm_file = os.path.join(DATA_DIR, 'purerace.bm')
    scenarios = parse_bm_scenarios(bm_file, max_scenarios=500)
    print(f"Parsed {len(scenarios)} purerace scenarios")

    count = 0
    disagree = 0
    total_0ply_err = 0.0
    total_1ply_err = 0.0
    worse_count = 0
    better_count = 0
    same_count = 0

    for s in scenarios:
        board = s['start_board']
        d1, d2 = s['die1'], s['die2']
        candidates = bgbot_cpp.possible_moves(board, d1, d2)

        if len(candidates) <= 1:
            count += 1
            continue

        count += 1

        # Evaluate all candidates at 0-ply
        equities_0 = []
        for c in candidates:
            res = multipy_0.evaluate_board(c, board)
            equities_0.append(res['equity'])

        # Evaluate all candidates at 1-ply
        multipy_1.clear_cache()
        equities_1 = []
        for c in candidates:
            res = multipy_1.evaluate_board(c, board)
            equities_1.append(res['equity'])

        best_0 = max(range(len(candidates)), key=lambda i: equities_0[i])
        best_1 = max(range(len(candidates)), key=lambda i: equities_1[i])

        # Flip chosen boards and match against ranked list
        chosen_0 = bgbot_cpp.flip_board(candidates[best_0])
        chosen_1 = bgbot_cpp.flip_board(candidates[best_1])

        err_0 = None
        err_1 = None
        for i, rb in enumerate(s['ranked_boards']):
            if chosen_0 == rb:
                err_0 = 0.0 if i == 0 else s['ranked_errors'][i]
                break
        for i, rb in enumerate(s['ranked_boards']):
            if chosen_1 == rb:
                err_1 = 0.0 if i == 0 else s['ranked_errors'][i]
                break

        # If move not found in ranked list, use worst error
        if err_0 is None:
            err_0 = s['ranked_errors'][-1] if len(s['ranked_errors']) > 1 else 0.0
        if err_1 is None:
            err_1 = s['ranked_errors'][-1] if len(s['ranked_errors']) > 1 else 0.0

        total_0ply_err += err_0
        total_1ply_err += err_1

        if best_0 != best_1:
            disagree += 1
            if err_1 > err_0:
                worse_count += 1
            elif err_1 < err_0:
                better_count += 1
            else:
                same_count += 1

            if disagree <= 5:
                print(f"Scenario {count}: dice={d1},{d2}, {len(candidates)} candidates")
                print(f"  0-ply equities: {[f'{e:.4f}' for e in equities_0[:8]]}")
                print(f"  1-ply equities: {[f'{e:.4f}' for e in equities_1[:8]]}")
                print(f"  0-ply best: {best_0} (eq={equities_0[best_0]:.4f})")
                print(f"  1-ply best: {best_1} (eq={equities_1[best_1]:.4f})")
                print(f"  0-ply error: {err_0:.6f}")
                print(f"  1-ply error: {err_1:.6f}")
                print(f"  1-ply {'BETTER' if err_1 < err_0 else 'WORSE' if err_1 > err_0 else 'SAME'}")
                print()

    avg_0 = total_0ply_err / count * 1000 if count > 0 else 0
    avg_1 = total_1ply_err / count * 1000 if count > 0 else 0
    print(f"\nSummary ({count} scenarios):")
    print(f"  Disagreements: {disagree} ({100*disagree/count:.1f}%)")
    print(f"  When disagree: better={better_count}, worse={worse_count}, same={same_count}")
    print(f"  0-ply avg ER: {avg_0:.2f}")
    print(f"  1-ply avg ER: {avg_1:.2f}")
    print(f"  Difference: {avg_1 - avg_0:+.2f}")


if __name__ == '__main__':
    main()
