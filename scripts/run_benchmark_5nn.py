"""Quick benchmark script for 5-NN game plan weights."""

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

# Weight files - use 1200k weights
model_name = sys.argv[1] if len(sys.argv) > 1 else 'sl'

purerace_w  = os.path.join(MODELS_DIR, f'{model_name}_purerace.weights.best')
racing_w    = os.path.join(MODELS_DIR, f'{model_name}_racing.weights.best')
attacking_w = os.path.join(MODELS_DIR, f'{model_name}_attacking.weights.best')
priming_w   = os.path.join(MODELS_DIR, f'{model_name}_priming.weights.best')
anchoring_w = os.path.join(MODELS_DIR, f'{model_name}_anchoring.weights.best')

hp, hr, ha, hpp, hn = 120, 250, 250, 250, 250

print(f'=== Benchmark: {model_name} ===')
print(f'  PureRace:  {purerace_w} ({hp}h)')
print(f'  Racing:    {racing_w} ({hr}h)')
print(f'  Attacking: {attacking_w} ({ha}h)')
print(f'  Priming:   {priming_w} ({hpp}h)')
print(f'  Anchoring: {anchoring_w} ({hn}h)')
print()

# Game plan benchmarks
print('--- Game Plan benchmarks (full) ---')
for bm_type in ['purerace', 'racing', 'attacking', 'priming', 'anchoring']:
    bm_path = os.path.join(DATA_DIR, f'{bm_type}.bm')
    if not os.path.exists(bm_path):
        print(f'  {bm_type:10s}: (not found)')
        continue
    t0 = time.time()
    scenarios = load_benchmark_file(bm_path)
    t_load = time.time() - t0
    t0 = time.time()
    result = bgbot_cpp.score_benchmarks_5nn(
        scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        hp, hr, ha, hpp, hn)
    t_score = time.time() - t0
    print(f'  {bm_type:10s}: {result.score():8.2f}  ({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')

# Old-style benchmarks
print()
print('--- Old-style benchmarks (for comparison) ---')
for bm_name in ['contact', 'crashed', 'race']:
    bm_path = os.path.join(DATA_DIR, f'{bm_name}.bm')
    if not os.path.exists(bm_path):
        print(f'  {bm_name:10s}: (not found)')
        continue
    t0 = time.time()
    scenarios = load_benchmark_file(bm_path)
    t_load = time.time() - t0
    t0 = time.time()
    result = bgbot_cpp.score_benchmarks_5nn(
        scenarios, purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
        hp, hr, ha, hpp, hn)
    t_score = time.time() - t0
    print(f'  {bm_name:10s}: {result.score():8.2f}  ({result.count} scenarios, load {t_load:.1f}s + score {t_score:.1f}s)')

# vs PubEval
print()
print('=== vs PubEval (10k games) ===')
t0 = time.time()
stats = bgbot_cpp.play_games_5nn_vs_pubeval(
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    hp, hr, ha, hpp, hn,
    n_games=10000, seed=42)
t_games = time.time() - t0
print(f'  PPG: {stats.avg_ppg():+.3f}  ({stats.n_games} games in {t_games:.1f}s)')
print(f'  P1: {stats.p1_wins}W {stats.p1_gammons}G {stats.p1_backgammons}B')
print(f'  P2: {stats.p2_wins}W {stats.p2_gammons}G {stats.p2_backgammons}B')

# Self-play outcome distribution
print()
print('=== Self-play outcome distribution (10k games) ===')
t0 = time.time()
ss = bgbot_cpp.play_games_5nn_vs_self(
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    hp, hr, ha, hpp, hn,
    n_games=10000, seed=42)
t_sp = time.time() - t0
total = ss.n_games
singles = ss.p1_wins + ss.p2_wins
gammons = ss.p1_gammons + ss.p2_gammons
backgammons = ss.p1_backgammons + ss.p2_backgammons
print(f'  Single: {singles:4d} ({100*singles/total:.1f}%)  '
      f'Gammon: {gammons:4d} ({100*gammons/total:.1f}%)  '
      f'Backgammon: {backgammons:3d} ({100*backgammons/total:.1f}%)  '
      f'({total} games in {t_sp:.1f}s)')
