"""
Split benchmark and training data by game plan classification.

Takes contact.bm + crashed.bm and splits into racing.bm, attacking.bm, priming.bm, anchoring.bm.
Takes contact-train-data + crashed-train-data and splits similarly.
Also creates purerace.bm (copy of race.bm) and purerace-train-data (copy of race-train-data).

5-NN categories:
  purerace  — pure race positions (is_race=true), 80h, 196 Tesauro inputs
  racing    — game plan racing with contact still present, 120h, 214 extended inputs
  attacking — attacking game plan, 120h, 214 extended inputs
  priming   — priming game plan, 120h, 214 extended inputs
  anchoring — anchoring game plan, 120h, 214 extended inputs
"""

import os
import sys
import time

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
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import board_from_gnubg_position_string

DATA_DIR = os.path.join(project_dir, 'data')


def split_benchmark_files():
    """Split contact.bm + crashed.bm into racing.bm, attacking.bm, priming.bm, anchoring.bm.
    Also create purerace.bm as a copy of race.bm."""
    print('=== Splitting Benchmark Files ===')

    # Create purerace.bm as copy of race.bm (pure race positions)
    race_bm_path = os.path.join(DATA_DIR, 'race.bm')
    purerace_bm_path = os.path.join(DATA_DIR, 'purerace.bm')
    if os.path.exists(race_bm_path):
        import shutil
        shutil.copy2(race_bm_path, purerace_bm_path)
        with open(race_bm_path, 'r') as f:
            move_count = sum(1 for line in f if line.startswith('m '))
        print(f'  Created purerace.bm: copied {move_count} scenarios from race.bm')
    else:
        print(f'  WARNING: race.bm not found, cannot create purerace.bm')

    # Read all lines from contact + crashed
    all_lines = []
    for filename in ['contact.bm', 'crashed.bm']:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f'  WARNING: {filepath} not found, skipping')
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()
        move_lines = [line for line in lines if line.startswith('m ')]
        print(f'  Loaded {len(move_lines)} scenarios from {filename}')
        all_lines.extend(move_lines)

    print(f'  Total scenarios from contact+crashed: {len(all_lines)}')

    # Classify each scenario by game plan
    # classify_game_plan now returns "purerace" for pure races, "racing" for game-plan racing
    buckets = {'purerace': [], 'racing': [], 'attacking': [], 'priming': [], 'anchoring': []}
    t0 = time.time()

    for line in all_lines:
        bits = line.split()
        board = board_from_gnubg_position_string(bits[1])
        gp = bgbot_cpp.classify_game_plan(board)
        buckets[gp].append(line)

    elapsed = time.time() - t0
    print(f'  Classification took {elapsed:.1f}s')

    total = len(all_lines)
    for gp_name, lines in buckets.items():
        pct = 100.0 * len(lines) / total if total > 0 else 0
        print(f'  {gp_name:10s}: {len(lines):6d} ({pct:5.1f}%)')

    # Write output files for racing, attacking, priming, anchoring
    # (purerace positions from contact/crashed are discarded — they're already in purerace.bm)
    if buckets['purerace']:
        print(f'  Note: {len(buckets["purerace"])} purerace positions from contact/crashed discarded '
              f'(already in purerace.bm)')
    for gp_name in ['racing', 'attacking', 'priming', 'anchoring']:
        outpath = os.path.join(DATA_DIR, f'{gp_name}.bm')
        with open(outpath, 'w') as f:
            for line in buckets[gp_name]:
                f.write(line)
        print(f'  Wrote {len(buckets[gp_name])} scenarios to {outpath}')
    print()


def split_training_data():
    """Split contact-train-data + crashed-train-data by game plan.
    Also create purerace-train-data as copy of race-train-data."""
    print('=== Splitting Training Data ===')

    # Create purerace-train-data as copy of race-train-data (pure race positions)
    race_td_path = os.path.join(DATA_DIR, 'race-train-data')
    purerace_td_path = os.path.join(DATA_DIR, 'purerace-train-data')
    if os.path.exists(race_td_path):
        import shutil
        shutil.copy2(race_td_path, purerace_td_path)
        with open(race_td_path, 'r') as f:
            line_count = sum(1 for line in f
                           if line.strip() and not line.startswith('#') and len(line.split()) == 6)
        print(f'  Created purerace-train-data: copied {line_count} positions from race-train-data')
    else:
        print(f'  WARNING: race-train-data not found, cannot create purerace-train-data')

    # Read all lines from contact + crashed
    all_lines = []
    for filename in ['contact-train-data', 'crashed-train-data']:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f'  WARNING: {filepath} not found, skipping')
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data_lines = [line for line in lines
                      if line.strip() and not line.startswith('#') and len(line.split()) == 6]
        print(f'  Loaded {len(data_lines)} positions from {filename}')
        all_lines.extend(data_lines)

    print(f'  Total positions from contact+crashed: {len(all_lines)}')

    # Classify each position by game plan
    buckets = {'purerace': [], 'racing': [], 'attacking': [], 'priming': [], 'anchoring': []}
    errors = 0
    t0 = time.time()

    for line in all_lines:
        parts = line.split()
        pos_str = parts[0]
        if len(pos_str) != 20:
            errors += 1
            continue
        try:
            board = board_from_gnubg_position_string(pos_str)
        except (ValueError, IndexError):
            errors += 1
            continue
        gp = bgbot_cpp.classify_game_plan(board)
        buckets[gp].append(line)

    elapsed = time.time() - t0
    print(f'  Classification took {elapsed:.1f}s')
    if errors > 0:
        print(f'  Parse errors: {errors}')

    total = sum(len(v) for v in buckets.values())
    for gp_name, lines in buckets.items():
        pct = 100.0 * len(lines) / total if total > 0 else 0
        print(f'  {gp_name:10s}: {len(lines):7d} ({pct:5.1f}%)')

    # Write output files for racing, attacking, priming, anchoring
    # (purerace positions from contact/crashed are discarded — they're already in purerace-train-data)
    if buckets['purerace']:
        print(f'  Note: {len(buckets["purerace"])} purerace positions from contact/crashed discarded '
              f'(already in purerace-train-data)')
    for gp_name in ['racing', 'attacking', 'priming', 'anchoring']:
        outpath = os.path.join(DATA_DIR, f'{gp_name}-train-data')
        with open(outpath, 'w') as f:
            for line in buckets[gp_name]:
                f.write(line)
        print(f'  Wrote {len(buckets[gp_name])} positions to {outpath}')
    print()


if __name__ == '__main__':
    split_benchmark_files()
    split_training_data()
    print('Done.')
