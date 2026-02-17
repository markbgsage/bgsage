"""
Experiment: Find fastest wall-clock convergence to a target contact score.

Each experiment function returns (wall_clock_seconds, final_contact_score, history)
where history is a list of (games, contact_score, elapsed_seconds) tuples.

All experiments use single-NN Tesauro inputs (196 inputs).
"""

import os
import sys
import time
import json
from datetime import datetime

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


def load_scenarios(step=10):
    """Load contact benchmark scenarios."""
    bm_path = os.path.join(DATA_DIR, 'contact.bm')
    return load_benchmark_file(bm_path, step=step)


def run_serial_phase(n_games, alpha, n_hidden, model_name, scenarios,
                     resume_from='', seed=42, benchmark_interval=10000,
                     target=None):
    """Run serial TD training phase. Returns (result, hit_target_at_seconds)."""
    result = bgbot_cpp.td_train(
        n_games=n_games, alpha=alpha, n_hidden=n_hidden,
        eps=0.1, seed=seed, benchmark_interval=benchmark_interval,
        model_name=model_name, models_dir=MODELS_DIR,
        resume_from=resume_from, scenarios=scenarios)

    # Check if target was hit
    hit_time = None
    if target is not None:
        for entry in result.history:
            if entry.contact_score <= target and entry.contact_score > 0:
                hit_time = entry.elapsed_seconds
                break

    return result, hit_time


def run_parallel_phase(n_games, alpha, n_hidden, model_name, scenarios,
                       n_workers=0, batch_size=8, resume_from='', seed=42,
                       benchmark_interval=10000, target=None):
    """Run parallel TD training phase. Returns (result, hit_target_at_seconds)."""
    result = bgbot_cpp.td_train_parallel(
        n_games=n_games, alpha=alpha, n_hidden=n_hidden,
        n_workers=n_workers, batch_size=batch_size,
        sum_deltas=False,
        eps=0.1, seed=seed, benchmark_interval=benchmark_interval,
        model_name=model_name, models_dir=MODELS_DIR,
        resume_from=resume_from, scenarios=scenarios)

    hit_time = None
    if target is not None:
        for entry in result.history:
            if entry.contact_score <= target and entry.contact_score > 0:
                hit_time = entry.elapsed_seconds
                break

    return result, hit_time


def find_target_time(history_entries, target, time_offset=0.0):
    """Find when target was first hit in history entries.
    Returns elapsed seconds from experiment start, or None."""
    for entry in history_entries:
        if entry.contact_score <= target and entry.contact_score > 0:
            return entry.elapsed_seconds + time_offset
    return None


def print_history(history_entries, label="", time_offset=0.0):
    """Print training history."""
    if label:
        print(f"\n--- {label} ---")
    for entry in history_entries:
        t = entry.elapsed_seconds + time_offset
        print(f"  Game {entry.game_number:>7d}  contact={entry.contact_score:6.2f}  time={t:.1f}s")


def experiment_pure_serial(n_hidden, target, scenarios, name="serial"):
    """Pure serial: 200k@0.1 + 400k@0.02"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name} (pure serial, {n_hidden}h)")
    print(f"{'='*60}")

    t_start = time.time()
    all_history = []

    # Phase 1
    model = f"exp_{name}"
    r1, _ = run_serial_phase(200000, 0.1, n_hidden, model, scenarios,
                              benchmark_interval=10000, target=target)
    all_history.extend(r1.history)
    phase1_time = r1.total_seconds
    print_history(r1.history, f"Phase 1: 200k@0.1")

    # Check if already hit target
    hit = find_target_time(r1.history, target)
    if hit is not None:
        wall = time.time() - t_start
        print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
        return wall, r1.history[-1].contact_score, all_history

    # Phase 2
    resume = os.path.join(MODELS_DIR, f"{model}.weights")
    model2 = f"exp_{name}_p2"
    r2, _ = run_serial_phase(400000, 0.02, n_hidden, model2, scenarios,
                              resume_from=resume, benchmark_interval=10000,
                              target=target)
    all_history.extend(r2.history)
    print_history(r2.history, f"Phase 2: 400k@0.02", time_offset=phase1_time)

    hit = find_target_time(r2.history, target, time_offset=phase1_time)
    wall = time.time() - t_start
    if hit is not None:
        print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
    else:
        final = r2.history[-1].contact_score if r2.history else 999
        print(f"\n** TARGET {target} NOT HIT. Final: {final:.2f} (wall {wall:.1f}s) **")

    return wall, all_history[-1].contact_score if all_history else 999, all_history


def experiment_hybrid(n_hidden, target, scenarios,
                      warmup_games, warmup_alpha,
                      parallel_games, parallel_alpha, parallel_workers, parallel_batch,
                      serial2_games, serial2_alpha,
                      name="hybrid"):
    """Hybrid: serial warmup -> parallel -> serial refinement."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name} ({n_hidden}h)")
    print(f"  Warmup: {warmup_games//1000}k@{warmup_alpha}")
    print(f"  Parallel: {parallel_games//1000}k@{parallel_alpha}, {parallel_workers}w, batch={parallel_batch}")
    print(f"  Serial2: {serial2_games//1000}k@{serial2_alpha}")
    print(f"{'='*60}")

    t_start = time.time()
    cumulative_time = 0.0
    all_history = []

    # Phase 1: Serial warmup
    model = f"exp_{name}_warmup"
    r1, _ = run_serial_phase(warmup_games, warmup_alpha, n_hidden, model, scenarios,
                              benchmark_interval=10000, target=target)
    cumulative_time += r1.total_seconds
    all_history.extend(r1.history)
    print_history(r1.history, f"Warmup: {warmup_games//1000}k@{warmup_alpha}")

    hit = find_target_time(r1.history, target)
    if hit is not None:
        wall = time.time() - t_start
        print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
        return wall, r1.history[-1].contact_score, all_history

    # Phase 2: Parallel
    if parallel_games > 0:
        resume = os.path.join(MODELS_DIR, f"{model}.weights")
        model2 = f"exp_{name}_parallel"
        r2, _ = run_parallel_phase(parallel_games, parallel_alpha, n_hidden, model2, scenarios,
                                    n_workers=parallel_workers, batch_size=parallel_batch,
                                    resume_from=resume, benchmark_interval=10000, target=target)
        phase2_offset = cumulative_time
        cumulative_time += r2.total_seconds
        all_history.extend(r2.history)
        print_history(r2.history, f"Parallel: {parallel_games//1000}k@{parallel_alpha}", time_offset=phase2_offset)

        hit = find_target_time(r2.history, target, time_offset=phase2_offset)
        if hit is not None:
            wall = time.time() - t_start
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
            return wall, r2.history[-1].contact_score, all_history

    # Phase 3: Serial refinement
    if serial2_games > 0:
        resume = os.path.join(MODELS_DIR, f"{model2 if parallel_games > 0 else model}.weights")
        model3 = f"exp_{name}_refine"
        r3, _ = run_serial_phase(serial2_games, serial2_alpha, n_hidden, model3, scenarios,
                                  resume_from=resume, benchmark_interval=10000, target=target)
        phase3_offset = cumulative_time
        cumulative_time += r3.total_seconds
        all_history.extend(r3.history)
        print_history(r3.history, f"Refine: {serial2_games//1000}k@{serial2_alpha}", time_offset=phase3_offset)

        hit = find_target_time(r3.history, target, time_offset=phase3_offset)
        if hit is not None:
            wall = time.time() - t_start
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
            return wall, r3.history[-1].contact_score, all_history

    wall = time.time() - t_start
    final = all_history[-1].contact_score if all_history else 999
    print(f"\n** TARGET {target} NOT HIT. Final: {final:.2f} (wall {wall:.1f}s) **")
    return wall, final, all_history


def run_single_experiment(exp_id, n_hidden, target, scenarios):
    """Run a single experiment by ID. Returns (name, wall_time, final_score)."""

    if exp_id == 1:
        # Baseline: pure serial
        wall, score, _ = experiment_pure_serial(
            n_hidden, target, scenarios, name="E1_serial")
        return "E1: Pure serial 200k@0.1+400k@0.02", wall, score

    elif exp_id == 2:
        # More alpha=0.1: 400k@0.1 + 200k@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=400000, warmup_alpha=0.1,
            parallel_games=0, parallel_alpha=0, parallel_workers=0, parallel_batch=0,
            serial2_games=200000, serial2_alpha=0.02,
            name="E2_serial_long_p1")
        return "E2: Serial 400k@0.1+200k@0.02", wall, score

    elif exp_id == 3:
        # Hybrid: 50k serial -> 350k parallel(8w,8b) -> 200k serial@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=50000, warmup_alpha=0.1,
            parallel_games=350000, parallel_alpha=0.1, parallel_workers=8, parallel_batch=8,
            serial2_games=200000, serial2_alpha=0.02,
            name="E3_hybrid_8w")
        return "E3: 50k serial + 350k par(8w,8b) + 200k serial@0.02", wall, score

    elif exp_id == 4:
        # Hybrid wider: 50k serial -> 350k parallel(16w,16b) -> 200k serial@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=50000, warmup_alpha=0.1,
            parallel_games=350000, parallel_alpha=0.1, parallel_workers=16, parallel_batch=16,
            serial2_games=200000, serial2_alpha=0.02,
            name="E4_hybrid_16w")
        return "E4: 50k serial + 350k par(16w,16b) + 200k serial@0.02", wall, score

    elif exp_id == 5:
        # Short warmup, aggressive parallel, then serial finish
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=30000, warmup_alpha=0.1,
            parallel_games=200000, parallel_alpha=0.1, parallel_workers=8, parallel_batch=8,
            serial2_games=400000, serial2_alpha=0.02,
            name="E5_short_warmup")
        return "E5: 30k serial + 200k par(8w,8b) + 400k serial@0.02", wall, score

    elif exp_id == 6:
        # Long warmup, skip parallel, direct serial
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=100000, warmup_alpha=0.1,
            parallel_games=0, parallel_alpha=0, parallel_workers=0, parallel_batch=0,
            serial2_games=500000, serial2_alpha=0.02,
            name="E6_serial_100k_500k")
        return "E6: Serial 100k@0.1+500k@0.02", wall, score

    elif exp_id == 7:
        # 3-phase alpha: 100k@0.1, 100k@0.05, 400k@0.02
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: E7 (3-phase alpha, {n_hidden}h)")
        print(f"{'='*60}")
        t_start = time.time()
        cumulative = 0.0

        r1, _ = run_serial_phase(100000, 0.1, n_hidden, "exp_E7_p1", scenarios, benchmark_interval=10000, target=target)
        cumulative += r1.total_seconds
        print_history(r1.history, "Phase 1: 100k@0.1")
        hit = find_target_time(r1.history, target)
        if hit:
            return "E7: 3-phase 100k@0.1+100k@0.05+400k@0.02", time.time()-t_start, r1.history[-1].contact_score

        resume1 = os.path.join(MODELS_DIR, "exp_E7_p1.weights")
        r2, _ = run_serial_phase(100000, 0.05, n_hidden, "exp_E7_p2", scenarios, resume_from=resume1, benchmark_interval=10000, target=target)
        cumulative += r2.total_seconds
        print_history(r2.history, "Phase 2: 100k@0.05", time_offset=cumulative-r2.total_seconds)
        hit = find_target_time(r2.history, target, time_offset=cumulative-r2.total_seconds)
        if hit:
            return "E7: 3-phase 100k@0.1+100k@0.05+400k@0.02", time.time()-t_start, r2.history[-1].contact_score

        resume2 = os.path.join(MODELS_DIR, "exp_E7_p2.weights")
        r3, _ = run_serial_phase(400000, 0.02, n_hidden, "exp_E7_p3", scenarios, resume_from=resume2, benchmark_interval=10000, target=target)
        cumulative += r3.total_seconds
        print_history(r3.history, "Phase 3: 400k@0.02", time_offset=cumulative-r3.total_seconds)
        hit = find_target_time(r3.history, target, time_offset=cumulative-r3.total_seconds)
        wall = time.time() - t_start
        final = r3.history[-1].contact_score if r3.history else 999
        if hit:
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
        else:
            print(f"\n** TARGET NOT HIT. Final: {final:.2f} (wall {wall:.1f}s) **")
        return "E7: 3-phase 100k@0.1+100k@0.05+400k@0.02", wall, final

    elif exp_id == 8:
        # Hybrid: 100k serial@0.1 -> 300k parallel(8w,8b)@0.05 -> 200k serial@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=100000, warmup_alpha=0.1,
            parallel_games=300000, parallel_alpha=0.05, parallel_workers=8, parallel_batch=8,
            serial2_games=200000, serial2_alpha=0.02,
            name="E8_hybrid_alpha_sched")
        return "E8: 100k serial@0.1 + 300k par(8w)@0.05 + 200k serial@0.02", wall, score

    elif exp_id == 9:
        # 10k warmup -> parallel(8w)@0.1 -> serial@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=10000, warmup_alpha=0.1,
            parallel_games=200000, parallel_alpha=0.1, parallel_workers=8, parallel_batch=8,
            serial2_games=400000, serial2_alpha=0.02,
            name="E9_10k_warmup")
        return "E9: 10k serial + 200k par(8w,8b)@0.1 + 400k serial@0.02", wall, score

    elif exp_id == 10:
        # 100k warmup -> parallel(8w)@0.1 -> serial@0.02
        wall, score, _ = experiment_hybrid(
            n_hidden, target, scenarios,
            warmup_games=100000, warmup_alpha=0.1,
            parallel_games=200000, parallel_alpha=0.1, parallel_workers=8, parallel_batch=8,
            serial2_games=400000, serial2_alpha=0.02,
            name="E10_100k_warmup")
        return "E10: 100k serial + 200k par(8w,8b)@0.1 + 400k serial@0.02", wall, score

    elif exp_id == 11:
        # Aggressive: 50k warmup -> 150k par(8w)@0.1 -> 100k serial@0.05 -> 300k serial@0.02
        # Tests intermediate alpha + parallel + warmup variation
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: E11 (50k warmup -> par -> 0.05 -> 0.02, {n_hidden}h)")
        print(f"{'='*60}")
        t_start = time.time()
        cumulative = 0.0

        r1, _ = run_serial_phase(50000, 0.1, n_hidden, "exp_E11_warmup", scenarios, benchmark_interval=10000, target=target)
        cumulative += r1.total_seconds
        print_history(r1.history, "Warmup: 50k@0.1")
        hit = find_target_time(r1.history, target)
        if hit:
            return "E11: 50k serial + 150k par(8w)@0.1 + 100k serial@0.05 + 300k serial@0.02", time.time()-t_start, r1.history[-1].contact_score

        resume1 = os.path.join(MODELS_DIR, "exp_E11_warmup.weights")
        r2, _ = run_parallel_phase(150000, 0.1, n_hidden, "exp_E11_par", scenarios, n_workers=8, batch_size=8, resume_from=resume1, benchmark_interval=10000, target=target)
        phase2_offset = cumulative
        cumulative += r2.total_seconds
        print_history(r2.history, "Parallel: 150k@0.1 (8w,8b)", time_offset=phase2_offset)
        hit = find_target_time(r2.history, target, time_offset=phase2_offset)
        if hit:
            return "E11: 50k serial + 150k par(8w)@0.1 + 100k serial@0.05 + 300k serial@0.02", time.time()-t_start, r2.history[-1].contact_score

        resume2 = os.path.join(MODELS_DIR, "exp_E11_par.weights")
        r3, _ = run_serial_phase(100000, 0.05, n_hidden, "exp_E11_mid", scenarios, resume_from=resume2, benchmark_interval=10000, target=target)
        phase3_offset = cumulative
        cumulative += r3.total_seconds
        print_history(r3.history, "Mid: 100k@0.05", time_offset=phase3_offset)
        hit = find_target_time(r3.history, target, time_offset=phase3_offset)
        if hit:
            return "E11: 50k serial + 150k par(8w)@0.1 + 100k serial@0.05 + 300k serial@0.02", time.time()-t_start, r3.history[-1].contact_score

        resume3 = os.path.join(MODELS_DIR, "exp_E11_mid.weights")
        r4, _ = run_serial_phase(300000, 0.02, n_hidden, "exp_E11_refine", scenarios, resume_from=resume3, benchmark_interval=10000, target=target)
        phase4_offset = cumulative
        cumulative += r4.total_seconds
        print_history(r4.history, "Refine: 300k@0.02", time_offset=phase4_offset)
        hit = find_target_time(r4.history, target, time_offset=phase4_offset)
        wall = time.time() - t_start
        final = r4.history[-1].contact_score if r4.history else 999
        if hit:
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
        else:
            print(f"\n** TARGET NOT HIT. Final: {final:.2f} (wall {wall:.1f}s) **")
        return "E11: 50k serial + 150k par(8w)@0.1 + 100k serial@0.05 + 300k serial@0.02", wall, final

    elif exp_id == 12:
        # Pure serial 3-phase with LESS alpha=0.1 and more 0.05
        # 50k@0.1 + 200k@0.05 + 400k@0.02
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: E12 (3-phase: 50k@0.1+200k@0.05+400k@0.02, {n_hidden}h)")
        print(f"{'='*60}")
        t_start = time.time()
        cumulative = 0.0

        r1, _ = run_serial_phase(50000, 0.1, n_hidden, "exp_E12_p1", scenarios, benchmark_interval=10000, target=target)
        cumulative += r1.total_seconds
        print_history(r1.history, "Phase 1: 50k@0.1")

        resume1 = os.path.join(MODELS_DIR, "exp_E12_p1.weights")
        r2, _ = run_serial_phase(200000, 0.05, n_hidden, "exp_E12_p2", scenarios, resume_from=resume1, benchmark_interval=10000, target=target)
        phase2_offset = cumulative
        cumulative += r2.total_seconds
        print_history(r2.history, "Phase 2: 200k@0.05", time_offset=phase2_offset)
        hit = find_target_time(r2.history, target, time_offset=phase2_offset)
        if hit:
            wall = time.time() - t_start
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
            return "E12: 50k@0.1+200k@0.05+400k@0.02", wall, r2.history[-1].contact_score

        resume2 = os.path.join(MODELS_DIR, "exp_E12_p2.weights")
        r3, _ = run_serial_phase(400000, 0.02, n_hidden, "exp_E12_p3", scenarios, resume_from=resume2, benchmark_interval=10000, target=target)
        phase3_offset = cumulative
        cumulative += r3.total_seconds
        print_history(r3.history, "Phase 3: 400k@0.02", time_offset=phase3_offset)
        hit = find_target_time(r3.history, target, time_offset=phase3_offset)
        wall = time.time() - t_start
        final = r3.history[-1].contact_score if r3.history else 999
        if hit:
            print(f"\n** TARGET {target} HIT at {hit:.1f}s (wall {wall:.1f}s) **")
        else:
            print(f"\n** TARGET NOT HIT. Final: {final:.2f} (wall {wall:.1f}s) **")
        return "E12: 50k@0.1+200k@0.05+400k@0.02", wall, final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, required=True, help='Experiment ID to run')
    parser.add_argument('--hidden', type=int, default=80, help='Hidden nodes')
    parser.add_argument('--target', type=float, default=23.0, help='Target contact score')
    args = parser.parse_args()

    print(f"Loading benchmark scenarios (step=10)...")
    scenarios = load_scenarios(step=10)
    print(f"  Loaded {len(scenarios)} scenarios")

    name, wall, score = run_single_experiment(args.exp, args.hidden, args.target, scenarios)

    print(f"\n{'='*60}")
    print(f"RESULT: {name}")
    print(f"  Wall clock: {wall:.1f}s ({wall/60:.1f} min)")
    print(f"  Final contact: {score:.2f}")
    print(f"  Target: {args.target}")
    print(f"  Hit target: {'YES' if score <= args.target else 'NO'}")
    print(f"{'='*60}")
