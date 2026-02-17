"""
Run baseline benchmarks: score the existing 3-NN strategy against the new game plan benchmark files.
This gives us baseline scores for comparison with the new 4-NN strategy.
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
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')

# Best 3-NN weights
CONTACT_WEIGHTS = os.path.join(MODELS_DIR, 'sl_contact.weights.best')
CRASHED_WEIGHTS = os.path.join(MODELS_DIR, 'td_multi_350k_crashed.weights')
RACE_WEIGHTS = os.path.join(MODELS_DIR, 'td_multi_350k_race.weights')

N_HIDDEN_CONTACT = 120
N_HIDDEN_CRASHED = 120
N_HIDDEN_RACE = 80


def main():
    # Verify weights exist
    for name, path in [('contact', CONTACT_WEIGHTS), ('crashed', CRASHED_WEIGHTS), ('race', RACE_WEIGHTS)]:
        if not os.path.exists(path):
            print(f'ERROR: {name} weights not found: {path}')
            sys.exit(1)

    print('=== Baseline Benchmarks (3-NN Strategy) ===')
    print(f'  Contact: {CONTACT_WEIGHTS} ({N_HIDDEN_CONTACT}h)')
    print(f'  Crashed: {CRASHED_WEIGHTS} ({N_HIDDEN_CRASHED}h)')
    print(f'  Race:    {RACE_WEIGHTS} ({N_HIDDEN_RACE}h)')
    print()

    # Score against old-style benchmarks
    print('--- Old-style benchmarks (step=1) ---')
    for bm_name in ['contact', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            print(f'  {bm_name}: (not found)')
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        t_load = time.time() - t0
        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_3nn(
            scenarios, CONTACT_WEIGHTS, CRASHED_WEIGHTS, RACE_WEIGHTS,
            N_HIDDEN_CONTACT, N_HIDDEN_CRASHED, N_HIDDEN_RACE)
        t_score = time.time() - t0
        print(f'  {bm_name:10s}: {result.score():8.2f}  '
              f'({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')
    print()

    # Score against new game plan benchmarks
    print('--- Game Plan benchmarks (step=1) ---')
    for bm_name in ['attacking', 'priming', 'anchoring', 'race']:
        bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
        if not os.path.exists(bm_path):
            print(f'  {bm_name}: (not found)')
            continue
        t0 = time.time()
        scenarios = load_benchmark_file(bm_path)
        t_load = time.time() - t0
        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_3nn(
            scenarios, CONTACT_WEIGHTS, CRASHED_WEIGHTS, RACE_WEIGHTS,
            N_HIDDEN_CONTACT, N_HIDDEN_CRASHED, N_HIDDEN_RACE)
        t_score = time.time() - t0
        print(f'  {bm_name:10s}: {result.score():8.2f}  '
              f'({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')
    print()


if __name__ == '__main__':
    main()
