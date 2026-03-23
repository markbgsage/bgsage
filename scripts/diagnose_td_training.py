"""
Diagnose TD training divergence with 100h/200h (Stage 5 Small).

Experiments:
1. Single-NN 200h TD (20k games) — basic sanity check
2. 5-NN Stage 5 (200h/400h) TD (20k games) — known-good architecture
3. 5-NN Stage 4 (120h/250h) TD (20k games) — smaller, known to work
4. 5-NN Stage 5 Small (100h/200h) TD (20k games) — reproduce issue
5. 5-NN Stage 5 Small (100h/200h) TD with lower alpha=0.01 (20k games) — test LR hypothesis
"""

import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))

build_dirs = [
    os.path.join(project_dir, 'build_msvc'),
    os.path.join(project_dir, 'build'),
]

if sys.platform == 'win32':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)
    for d in build_dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)

for d in reversed(build_dirs):
    if os.path.isdir(d):
        sys.path.insert(0, d)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))

import bgbot_cpp
from bgsage.data import load_benchmark_file

DATA_DIR = os.path.join(project_dir, 'bgsage', 'data')
MODELS_DIR = os.path.join(project_dir, 'bgsage', 'models')

N_GAMES = 20000
BENCHMARK_INTERVAL = 1000

# Load benchmarks
print("Loading benchmarks (step=10)...")
benchmarks = {}
for bm_name in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
    bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
    if os.path.exists(bm_path):
        benchmarks[bm_name] = load_benchmark_file(bm_path, step=10)
        print(f"  {bm_name}: {benchmarks[bm_name].size()} scenarios")

contact_bm = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'), step=10)
print(f"  contact: {contact_bm.size()} scenarios")
print()


def run_5nn_experiment(label, n_hidden_pr, n_hidden_contact, alpha, model_prefix):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'  PureRace: {n_hidden_pr}h, Contact NNs: {n_hidden_contact}h, alpha={alpha}')
    print(f'{"="*70}')

    t0 = time.time()
    result = bgbot_cpp.td_train_gameplan(
        n_games=N_GAMES,
        alpha=alpha,
        n_hidden_purerace=n_hidden_pr,
        n_hidden_racing=n_hidden_contact,
        n_hidden_attacking=n_hidden_contact,
        n_hidden_priming=n_hidden_contact,
        n_hidden_anchoring=n_hidden_contact,
        eps=0.1,
        seed=42,
        benchmark_interval=BENCHMARK_INTERVAL,
        model_name=model_prefix,
        models_dir=MODELS_DIR,
        purerace_benchmark=benchmarks.get('purerace'),
        attacking_benchmark=benchmarks.get('attacking'),
        priming_benchmark=benchmarks.get('priming'),
        anchoring_benchmark=benchmarks.get('anchoring'),
        race_benchmark=benchmarks.get('racing'),
    )
    elapsed = time.time() - t0
    print(f'\n  {label}: {result.games_played} games in {elapsed:.1f}s')

    # Print weight file sizes
    for plan in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
        path = os.path.join(MODELS_DIR, f'{model_prefix}_{plan}.weights')
        if os.path.exists(path):
            print(f'    {plan}: {os.path.getsize(path)} bytes')

    # Print final history entries
    csv_path = os.path.join(MODELS_DIR, f'{model_prefix}.history.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        print(f'  History ({len(lines)-1} entries):')
        for line in lines[-6:]:
            print(f'    {line.strip()}')

    return result


def run_single_nn_experiment(label, n_hidden, alpha, model_prefix):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'  Single NN: {n_hidden}h (196 Tesauro inputs), alpha={alpha}')
    print(f'{"="*70}')

    t0 = time.time()
    result = bgbot_cpp.td_train(
        n_games=N_GAMES,
        alpha=alpha,
        n_hidden=n_hidden,
        eps=0.1,
        seed=42,
        benchmark_interval=BENCHMARK_INTERVAL,
        model_name=model_prefix,
        models_dir=MODELS_DIR,
        benchmark_scenarios=contact_bm,
    )
    elapsed = time.time() - t0
    print(f'\n  {label}: {result.games_played} games in {elapsed:.1f}s')

    path = os.path.join(MODELS_DIR, f'{model_prefix}.weights')
    if os.path.exists(path):
        print(f'    Weight file: {os.path.getsize(path)} bytes')

    return result


# Experiment 1: Single-NN 200h
run_single_nn_experiment(
    "Exp 1: Single-NN 200h baseline",
    n_hidden=200, alpha=0.1, model_prefix='diag_single200h')

# Experiment 2: 5-NN Stage 5 (200h/400h) — known good
run_5nn_experiment(
    "Exp 2: 5-NN Stage 5 (200h/400h) — known good",
    n_hidden_pr=200, n_hidden_contact=400, alpha=0.1, model_prefix='diag_s5')

# Experiment 3: 5-NN Stage 4 (120h/250h) — known good, smaller
run_5nn_experiment(
    "Exp 3: 5-NN Stage 4 (120h/250h) — known good, smaller",
    n_hidden_pr=120, n_hidden_contact=250, alpha=0.1, model_prefix='diag_s4')

# Experiment 4: 5-NN Stage 5 Small (100h/200h) — reproduce issue
run_5nn_experiment(
    "Exp 4: 5-NN Stage 5 Small (100h/200h) — reproduce issue",
    n_hidden_pr=100, n_hidden_contact=200, alpha=0.1, model_prefix='diag_s5s')

# Experiment 5: 5-NN Stage 5 Small with lower alpha
run_5nn_experiment(
    "Exp 5: 5-NN Stage 5 Small (100h/200h) alpha=0.01",
    n_hidden_pr=100, n_hidden_contact=200, alpha=0.01, model_prefix='diag_s5s_lo')

# Experiment 6: 5-NN 150h/300h — midpoint
run_5nn_experiment(
    "Exp 6: 5-NN midpoint (150h/300h)",
    n_hidden_pr=150, n_hidden_contact=300, alpha=0.1, model_prefix='diag_mid')

print(f'\n{"="*70}')
print(f'  ALL EXPERIMENTS COMPLETE')
print(f'{"="*70}')
print()
print("Summary of final contact_score (attacking ER) at 20k games:")
for prefix, label in [
    ('diag_single200h', 'Single-NN 200h'),
    ('diag_s5', 'Stage 5 (200h/400h)'),
    ('diag_s4', 'Stage 4 (120h/250h)'),
    ('diag_s5s', 'Stage 5 Small (100h/200h)'),
    ('diag_s5s_lo', 'S5S alpha=0.01'),
    ('diag_mid', 'Midpoint (150h/300h)'),
]:
    csv_path = os.path.join(MODELS_DIR, f'{prefix}.history.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            last = lines[-1].strip().split(',')
            print(f'  {label:30s}: {float(last[1]):.1f}')
    else:
        # Single NN uses different CSV format, just check weight file
        path = os.path.join(MODELS_DIR, f'{prefix}.weights')
        if os.path.exists(path):
            print(f'  {label:30s}: (single-NN, check log)')
