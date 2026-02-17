"""
Run a batch of convergence experiments sequentially in a single process.
This avoids process-kill issues from running separate background processes.
"""

import os
import sys
import time

# Setup paths
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

N_HIDDEN = 80
TARGET = 23.0
BM_STEP = 10


def load_scenarios():
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    return load_benchmark_file(bm_path, step=BM_STEP)


def run_serial(n_games, alpha, name, scenarios, resume=''):
    """Run serial TD, return (result, cumulative_history_with_offsets)."""
    r = bgbot_cpp.td_train(
        n_games=n_games, alpha=alpha, n_hidden=N_HIDDEN,
        eps=0.1, seed=42, benchmark_interval=10000,
        model_name=name, models_dir=MODELS_DIR,
        resume_from=resume, scenarios=scenarios)
    return r


def run_parallel(n_games, alpha, name, scenarios, n_workers=8, batch=8, resume=''):
    """Run parallel TD, return result."""
    r = bgbot_cpp.td_train_parallel(
        n_games=n_games, alpha=alpha, n_hidden=N_HIDDEN,
        n_workers=n_workers, batch_size=batch, sum_deltas=False,
        eps=0.1, seed=42, benchmark_interval=10000,
        model_name=name, models_dir=MODELS_DIR,
        resume_from=resume, scenarios=scenarios)
    return r


def weights_path(name):
    return os.path.join(MODELS_DIR, f"{name}.weights")


def find_hit(history, target, offset=0.0):
    """Find cumulative time when target first hit."""
    for e in history:
        if e.contact_score <= target and e.contact_score > 0:
            return e.elapsed_seconds + offset
    return None


def print_phase(label, result, offset=0.0):
    print(f"\n  --- {label} ---")
    for e in result.history:
        t = e.elapsed_seconds + offset
        print(f"    Game {e.game_number:>7d}  contact={e.contact_score:6.2f}  time={t:.0f}s")
    print(f"  Phase time: {result.total_seconds:.0f}s")


def run_experiment(exp_name, phases, scenarios):
    """Run a multi-phase experiment.
    phases: list of (label, 'serial'|'parallel', n_games, alpha, kwargs_dict)
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*70}")

    cumulative = 0.0
    prev_weights = ''
    hit_time = None

    for i, (label, mode, n_games, alpha, kwargs) in enumerate(phases):
        name = f"exp_{exp_name}_p{i}"
        resume = prev_weights

        if mode == 'serial':
            r = run_serial(n_games, alpha, name, scenarios, resume=resume)
        elif mode == 'parallel':
            r = run_parallel(n_games, alpha, name, scenarios,
                            n_workers=kwargs.get('workers', 8),
                            batch=kwargs.get('batch', 8),
                            resume=resume)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print_phase(label, r, offset=cumulative)

        if hit_time is None:
            hit_time = find_hit(r.history, TARGET, offset=cumulative)
            if hit_time is not None:
                print(f"\n  ** TARGET {TARGET} FIRST HIT at {hit_time:.0f}s **")

        cumulative += r.total_seconds
        prev_weights = weights_path(name)

    final_score = r.history[-1].contact_score if r.history else 999
    print(f"\n  TOTAL TIME: {cumulative:.0f}s ({cumulative/60:.1f} min)")
    print(f"  FINAL SCORE: {final_score:.2f}")
    if hit_time:
        print(f"  TIME TO TARGET: {hit_time:.0f}s ({hit_time/60:.1f} min)")
    else:
        print(f"  TARGET NOT HIT")

    return {
        'name': exp_name,
        'total_time': cumulative,
        'final_score': final_score,
        'hit_time': hit_time,
    }


def main():
    print("Loading benchmark scenarios...")
    scenarios = load_scenarios()
    print(f"  Loaded {len(scenarios)} scenarios")
    print(f"  Hidden: {N_HIDDEN}, Target: {TARGET}")

    results = []

    # E7: 3-phase alpha (serial only)
    results.append(run_experiment("E7_3phase", [
        ("100k @ alpha=0.1",  'serial', 100000, 0.1,  {}),
        ("100k @ alpha=0.05", 'serial', 100000, 0.05, {}),
        ("400k @ alpha=0.02", 'serial', 400000, 0.02, {}),
    ], scenarios))

    # E11: 50k warmup -> 150k par(8w) -> 100k serial@0.05 -> 300k serial@0.02
    results.append(run_experiment("E11_hybrid_3alpha", [
        ("50k serial @ 0.1",       'serial',   50000,  0.1,  {}),
        ("150k par(8w,8b) @ 0.1",  'parallel', 150000, 0.1,  {'workers': 8, 'batch': 8}),
        ("100k serial @ 0.05",     'serial',   100000, 0.05, {}),
        ("300k serial @ 0.02",     'serial',   300000, 0.02, {}),
    ], scenarios))

    # E12: 50k@0.1 + 200k@0.05 + 400k@0.02 (more 0.05, less 0.1)
    results.append(run_experiment("E12_more_05", [
        ("50k @ alpha=0.1",  'serial', 50000,  0.1,  {}),
        ("200k @ alpha=0.05",'serial', 200000, 0.05, {}),
        ("400k @ alpha=0.02",'serial', 400000, 0.02, {}),
    ], scenarios))

    # E9: 10k warmup -> 200k par(8w)@0.1 -> 400k serial@0.02
    results.append(run_experiment("E9_10k_warmup", [
        ("10k serial @ 0.1",       'serial',   10000,  0.1,  {}),
        ("200k par(8w,8b) @ 0.1",  'parallel', 200000, 0.1,  {'workers': 8, 'batch': 8}),
        ("400k serial @ 0.02",     'serial',   400000, 0.02, {}),
    ], scenarios))

    # E10: 100k warmup -> 200k par(8w)@0.1 -> 400k serial@0.02
    results.append(run_experiment("E10_100k_warmup", [
        ("100k serial @ 0.1",      'serial',   100000, 0.1,  {}),
        ("200k par(8w,8b) @ 0.1",  'parallel', 200000, 0.1,  {'workers': 8, 'batch': 8}),
        ("400k serial @ 0.02",     'serial',   400000, 0.02, {}),
    ], scenarios))

    # E13: 50k warmup -> 150k par(8w)@0.1 -> serial@0.05 until target
    # (like E11 but 200k@0.05 instead of 100k)
    results.append(run_experiment("E13_long_05", [
        ("50k serial @ 0.1",       'serial',   50000,  0.1,  {}),
        ("150k par(8w,8b) @ 0.1",  'parallel', 150000, 0.1,  {'workers': 8, 'batch': 8}),
        ("200k serial @ 0.05",     'serial',   200000, 0.05, {}),
        ("200k serial @ 0.02",     'serial',   200000, 0.02, {}),
    ], scenarios))

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY (target: contact <= {TARGET})")
    print(f"{'='*70}")
    print(f"{'Experiment':<35} {'Hit Time':>10} {'Final':>8} {'Total':>8}")
    print(f"{'-'*35} {'-'*10} {'-'*8} {'-'*8}")

    # Include E1 baseline from prior run
    print(f"{'E1: serial 200k@0.1+400k@0.02':<35} {'765s':>10} {'22.42':>8} {'1234s':>8}  (prior run)")

    for r in sorted(results, key=lambda x: x['hit_time'] or 99999):
        hit_str = f"{r['hit_time']:.0f}s" if r['hit_time'] else "NOT HIT"
        print(f"{r['name']:<35} {hit_str:>10} {r['final_score']:>8.2f} {r['total_time']:>8.0f}s")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
