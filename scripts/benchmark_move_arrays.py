# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Compare exact array batching with the prior 21-call construction.

Uses a private move-generator probe or an explicitly selected full extension.
It does not replace bgbot_cpp or claim end-to-end video throughput.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START = (0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5,
         5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0)
ROLLS = tuple((a, b) for a in range(1, 7) for b in range(a, 7))


def flip(board):
    return (board[25],) + tuple(-value for value in board[24:0:-1]) + (board[0],)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=256)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--reference-engine', type=Path,
                        help='Optional snapshotted existing bgbot_cpp extension inside this repository')
    parser.add_argument('--extension', type=Path,
                        help='Optional private full bgbot_cpp extension; default is the move-only probe')
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Engine benchmark output must stay inside the engine repository')
    out.mkdir(parents=True, exist_ok=False)
    if args.extension:
        extension_path = args.extension.resolve()
        if not extension_path.is_relative_to(ROOT):
            raise ValueError('Private extension must stay inside this repository')
        if args.reference_engine:
            raise ValueError('Load only one full extension per process; compare full builds in separate runs')
        extension_spec = importlib.util.spec_from_file_location('bgbot_cpp', extension_path)
        probe = importlib.util.module_from_spec(extension_spec)
        extension_spec.loader.exec_module(probe)
    else:
        sys.path.insert(0, str(ROOT / 'build/video_move_batch'))
        import _bgsage_move_array_probe as probe
    baseline = probe
    dll_handles = []
    if args.reference_engine:
        reference_path = args.reference_engine.resolve()
        if not reference_path.is_relative_to(ROOT):
            raise ValueError('Reference extension must be snapshotted inside this repository')
        if sys.platform == 'win32' and os.environ.get('CUDA_PATH'):
            for suffix in ('bin/x64', 'bin'):
                directory = Path(os.environ['CUDA_PATH']) / suffix
                if directory.is_dir():
                    dll_handles.append(os.add_dll_directory(str(directory)))
        reference_spec = importlib.util.spec_from_file_location('bgbot_cpp', reference_path)
        baseline = importlib.util.module_from_spec(reference_spec)
        reference_spec.loader.exec_module(baseline)
    spec = importlib.util.spec_from_file_location('move_arrays_benchmark', ROOT / 'python/bgsage/move_arrays.py')
    arrays = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('bgbot_cpp')
    try:
        sys.modules['bgbot_cpp'] = probe
        spec.loader.exec_module(arrays)
    finally:
        if previous is None:
            sys.modules.pop('bgbot_cpp', None)
        else:
            sys.modules['bgbot_cpp'] = previous
    paths = [Path(probe.__file__), Path(__file__)] + [ROOT / name for name in (
        'cpp/pybind/move_arrays.h', 'cpp/move_array_probe/bindings.cpp',
        'cpp/move_array_probe/CMakeLists.txt', 'cpp/src/moves.cpp',
        'cpp/src/board.cpp', 'cpp/include/bgbot/types.h', 'cpp/include/bgbot/moves.h',
        'python/bgsage/move_arrays.py')]
    if args.reference_engine:
        paths.append(reference_path)
    if args.extension:
        paths += [ROOT / name for name in ('CMakeLists.txt', 'cpp/CMakeLists_cpu.txt', 'cpp/pybind/bindings.cpp')]
    digest = lambda: {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before = digest()
    rng, board, positions = random.Random(78915), START, []
    for _ in range(args.positions):
        positions.append(board)
        first, second = rng.choice(ROLLS)
        board = flip(rng.choice(probe.possible_moves(board, first, second)))
        if not any(value > 0 for value in board[1:]) or not any(value < 0 for value in board[1:25]):
            board = START

    def old(board):
        found = {}
        for k, (first, second) in enumerate(ROLLS):
            # Preserve the public possible_moves wrapper's historical copy.
            for row in [list(row) for row in baseline.possible_moves(board, first, second)]:
                key = tuple(row)
                found[key] = found.get(key, 0) | (1 << k)
        return np.asarray(list(found), dtype=np.int8), np.fromiter(found.values(), np.int32, len(found))

    endpoint_count = 0
    for board in positions:
        expected, actual = old(board), arrays.possible_moves_all(board)
        if not all(np.array_equal(x, y) for x, y in zip(expected, actual)):
            raise AssertionError('Native batching changed endpoint order or roll aliases')
        endpoint_count += len(actual[0])
    measurements = []
    for index in range(args.rounds):
        order = ('old', 'batched') if index % 2 == 0 else ('batched', 'old')
        for mode in order:
            function = old if mode == 'old' else arrays.possible_moves_all
            start = time.perf_counter()
            rows = sum(len(function(board)[0]) for board in positions)
            measurements.append(dict(round=index, mode=mode, seconds=time.perf_counter()-start,
                                     endpoint_rows=rows))
    medians = {mode: statistics.median(row['seconds'] for row in measurements if row['mode'] == mode)
               for mode in ('old', 'batched')}
    after = digest()
    result = dict(seed=78915, positions=len(positions), endpoint_rows=endpoint_count,
                  exact_order_and_aliases=True, measurements=measurements, median_seconds=medians,
                  speedup=medians['old']/medians['batched'], source_hashes_before=before,
                  source_hashes_after=after, unchanged=before == after, python=sys.version,
                  platform=platform.platform(), numpy=np.__version__, extension=probe.__file__,
                  reference_extension=baseline.__file__,
                  qualification=('Private full CPU extension' if args.extension else 'Private move-only extension')
                  + ' microbenchmark; no installed binary change, models, video, AWS or end-to-end throughput claim.')
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('positions','endpoint_rows','exact_order_and_aliases','median_seconds','speedup','unchanged')}, indent=2))
    if before != after:
        raise RuntimeError('Source changed during benchmark')


if __name__ == '__main__':
    main()
