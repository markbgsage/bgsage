"""
Run GNUbg benchmarks using PubEval strategy via C++ engine.

Compares results against published values from the blog:
  - PubEval (Tesauro weights): Contact 44.2, Crashed 51.3, Race 5.54
  - PubEval (ListNet weights): should be modestly better
"""

import os
import sys
import time

# Setup import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')

if sys.platform == 'win32' and os.path.isdir(build_dir):
    os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'data')

BENCHMARK_TYPES = ['contact', 'crashed', 'race']

# Blog-reported PubEval scores (Tesauro weights)
EXPECTED_TESAURO = {
    'contact': 44.2,
    'crashed': 51.3,
    'race': 5.54,
}


def run_benchmarks(weights_name='TESAURO', step=1):
    """Run all three benchmarks and print results."""

    weights = getattr(bgbot_cpp.PubEvalWeights, weights_name)

    print(f'=== PubEval ({weights_name}) Benchmark Scores ===')
    print(f'(step={step}, using every {step}th scenario)')
    print()

    results = {}
    total_time = 0

    for bm_type in BENCHMARK_TYPES:
        filepath = os.path.join(DATA_DIR, f'{bm_type}.bm')
        if not os.path.exists(filepath):
            print(f'  WARNING: {filepath} not found, skipping')
            continue

        # Load and parse
        t0 = time.time()
        scenarios = load_benchmark_file(filepath, step=step)
        t_load = time.time() - t0

        # Score in C++
        t0 = time.time()
        result = bgbot_cpp.score_benchmarks_pubeval(scenarios, weights)
        t_score = time.time() - t0

        score = result.score()
        results[bm_type] = score
        total_time += t_load + t_score

        expected = EXPECTED_TESAURO.get(bm_type, '?')
        print(f'  {bm_type:8s}: {score:8.2f}  '
              f'(expected ~{expected}, '
              f'{result.count} scenarios, '
              f'load {t_load:.1f}s + score {t_score:.1f}s)')

    print()
    print(f'  Total time: {total_time:.1f}s')
    print()

    if weights_name == 'TESAURO':
        print('  Blog-reported PubEval (Tesauro) scores:')
        print(f'    Contact: {EXPECTED_TESAURO["contact"]}')
        print(f'    Crashed: {EXPECTED_TESAURO["crashed"]}')
        print(f'    Race:    {EXPECTED_TESAURO["race"]}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run GNUbg benchmarks')
    parser.add_argument('--weights', choices=['TESAURO', 'LISTNET'], default='TESAURO')
    parser.add_argument('--step', type=int, default=1,
                       help='Use every Nth scenario (default: 1 = all)')
    args = parser.parse_args()

    run_benchmarks(weights_name=args.weights, step=args.step)
