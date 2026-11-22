# SPDX-License-Identifier: MPL-2.0
"""Compare native builds on adjacent calls, alternating which build runs first."""
import argparse
from dataclasses import asdict
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from zipfile import ZipFile

from sample import DATA, ROOT, file_identity, inside_repo, load_engine, write_json


def worker(pipe, extension, threads, mask):
    from multi_ply_rollout_truth import set_cpu_mask
    set_cpu_mask(mask)
    engine, _ = load_engine(extension)
    weights = engine.WeightConfig.from_model('stage11')
    analyzers = {}
    while (case := pipe.recv()) is not None:
        level = case['level']
        if level not in analyzers:
            analyzers[level] = engine.BgBotAnalyzer(weights=weights, eval_level=level, cubeful=True,
                parallel_threads=threads, bearoff_db=str(ROOT / 'data/bearoff_1sided.db'))
        analyzer = analyzers[level]
        analyzer.set_seed(case['seed'])
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        result = getattr(analyzer, case['api'])(**case['inputs'])
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        pipe.send((wall, cpu, asdict(result)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--mask', type=lambda value: int(value, 0), default=0xf00)
    args = parser.parse_args()
    out = inside_repo(args.output)
    out.mkdir(parents=True, exist_ok=False)
    ctx = mp.get_context('spawn')
    workers = {}
    study = dict(threads=args.threads, cpu_mask=hex(args.mask), shuffle_seed=20261121,
                 binaries={}, cases=[])
    for name, path in [('control', args.control), ('candidate', args.candidate)]:
        path = path.resolve()
        study['binaries'][name] = file_identity(path)
        receive, send = ctx.Pipe()
        proc = ctx.Process(target=worker, args=(send, path, args.threads, args.mask))
        proc.start()
        send.close()
        workers[name] = receive, proc
    cases = []
    try:
        with ZipFile(DATA / 'experiments/truncated_16t/full_outputs.zip') as z:
            for level in ['2T', '3T']:
                seen = set()
                for line in z.read(f'{level}-candidate/results.jsonl').splitlines():
                    case = json.loads(line)
                    key = case['api'], case['mode'], case['source_game_plan']
                    if key not in seen:
                        seen.add(key)
                        cases.append(case)
        random.Random(study['shuffle_seed']).shuffle(cases)
        with gzip.open(out / 'calls.jsonl.gz', 'wt', encoding='utf-8') as raw:
            for i, case in enumerate(cases):
                row = {k: case[k] for k in ('level', 'api', 'position_id', 'seed')}
                order = list(workers) if i % 2 == 0 else list(reversed(workers))
                row['order'] = order
                for name in order:
                    pipe, _ = workers[name]
                    pipe.send(case)
                    wall, cpu, analytics = pipe.recv()
                    row[name] = dict(wall_seconds=wall, process_cpu_seconds=cpu, exact=analytics == case['analytics'])
                    raw.write(json.dumps(dict(arm=name, **{k: case[k] for k in ('level', 'api', 'position_id', 'seed', 'inputs')},
                        **row[name], analytics=analytics), separators=(',', ':'), allow_nan=False) + '\n')
                study['cases'].append(row)
                write_json(out / 'measurements.json', study)
                print(i + 1, row['level'], row['api'], row['position_id'], round(row['control']['wall_seconds'], 3),
                    round(row['candidate']['wall_seconds'], 3), flush=True)
    finally:
        for pipe, proc in workers.values():
            if proc.is_alive():
                pipe.send(None)
            pipe.close()
            proc.join()


if __name__ == '__main__':
    main()
