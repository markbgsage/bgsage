"""
Single diagnostic experiment runner. Run with:
  python diag_exp.py <experiment_id>

Experiments:
  1 = Single-NN 200h (baseline)
  2 = 5-NN Stage 5 (200h/400h)
  3 = 5-NN Stage 4 (120h/250h)
  4 = 5-NN Stage 5 Small (100h/200h)
  5 = 5-NN S5S alpha=0.01
  6 = 5-NN midpoint (150h/300h)
"""

import os
import sys
import time

exp_id = int(sys.argv[1])

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
print(f"Experiment {exp_id}: Loading benchmarks...", flush=True)
benchmarks = {}
for bm_name in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
    bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
    if os.path.exists(bm_path):
        benchmarks[bm_name] = load_benchmark_file(bm_path, step=10)

contact_bm = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'), step=10)

EXPERIMENTS = {
    1: {
        'label': 'Single-NN 200h',
        'type': 'single',
        'n_hidden': 200,
        'alpha': 0.1,
        'prefix': 'diag_single200h',
    },
    2: {
        'label': '5-NN Stage 5 (200h/400h)',
        'type': '5nn',
        'n_hidden_pr': 200, 'n_hidden_contact': 400,
        'alpha': 0.1,
        'prefix': 'diag_s5',
    },
    3: {
        'label': '5-NN Stage 4 (120h/250h)',
        'type': '5nn',
        'n_hidden_pr': 120, 'n_hidden_contact': 250,
        'alpha': 0.1,
        'prefix': 'diag_s4',
    },
    4: {
        'label': '5-NN Stage 5 Small (100h/200h)',
        'type': '5nn',
        'n_hidden_pr': 100, 'n_hidden_contact': 200,
        'alpha': 0.1,
        'prefix': 'diag_s5s',
    },
    5: {
        'label': '5-NN S5S alpha=0.01',
        'type': '5nn',
        'n_hidden_pr': 100, 'n_hidden_contact': 200,
        'alpha': 0.01,
        'prefix': 'diag_s5s_lo',
    },
    6: {
        'label': '5-NN midpoint (150h/300h)',
        'type': '5nn',
        'n_hidden_pr': 150, 'n_hidden_contact': 300,
        'alpha': 0.1,
        'prefix': 'diag_mid',
    },
}

exp = EXPERIMENTS[exp_id]
print(f"\nStarting: {exp['label']}", flush=True)

t0 = time.time()

if exp['type'] == 'single':
    result = bgbot_cpp.td_train(
        n_games=N_GAMES,
        alpha=exp['alpha'],
        n_hidden=exp['n_hidden'],
        eps=0.1,
        seed=42,
        benchmark_interval=BENCHMARK_INTERVAL,
        model_name=exp['prefix'],
        models_dir=MODELS_DIR,
        benchmark_scenarios=contact_bm,
    )
else:
    result = bgbot_cpp.td_train_gameplan(
        n_games=N_GAMES,
        alpha=exp['alpha'],
        n_hidden_purerace=exp['n_hidden_pr'],
        n_hidden_racing=exp['n_hidden_contact'],
        n_hidden_attacking=exp['n_hidden_contact'],
        n_hidden_priming=exp['n_hidden_contact'],
        n_hidden_anchoring=exp['n_hidden_contact'],
        eps=0.1,
        seed=42,
        benchmark_interval=BENCHMARK_INTERVAL,
        model_name=exp['prefix'],
        models_dir=MODELS_DIR,
        purerace_benchmark=benchmarks.get('purerace'),
        attacking_benchmark=benchmarks.get('attacking'),
        priming_benchmark=benchmarks.get('priming'),
        anchoring_benchmark=benchmarks.get('anchoring'),
        race_benchmark=benchmarks.get('racing'),
    )

elapsed = time.time() - t0
print(f"\nDone: {exp['label']} - {result.games_played} games in {elapsed:.1f}s", flush=True)

# Print final history
csv_path = os.path.join(MODELS_DIR, f"{exp['prefix']}.history.csv")
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    print(f"History ({len(lines)-1} entries):")
    for line in lines[-6:]:
        print(f"  {line.strip()}")
