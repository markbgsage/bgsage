"""Quick test of ultra-tight (3, 0.03) filter on full benchmark."""
import os, sys, time
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
build_dir = os.path.join(project_dir, 'build')
if sys.platform == 'win32':
    cuda_x64 = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64'
    if os.path.isdir(cuda_x64): os.add_dll_directory(cuda_x64)
    if os.path.isdir(build_dir): os.add_dll_directory(build_dir)
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(project_dir, 'bgsage', 'python'))
import bgbot_cpp
from bgsage.data import load_benchmark_file
DATA_DIR = os.path.join(project_dir, 'data')
MODELS_DIR = os.path.join(project_dir, 'models')
NH_PR, NH_RC, NH_AT, NH_PM, NH_AN = 120, 250, 250, 250, 250
N_THREADS = 24

types = ['purerace', 'racing', 'attacking', 'priming', 'anchoring']
wpaths = {t: os.path.join(MODELS_DIR, f'sl_{t}.weights.best') for t in types}
contact_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'contact.bm'))
purerace_scenarios = load_benchmark_file(os.path.join(DATA_DIR, 'purerace.bm'))

configs = [
    (2, 0.02, "ULTRA2(2, 0.02)"),
    (3, 0.03, "ULTRA3(3, 0.03)"),
    (4, 0.04, "TIGHT (4, 0.04)"),
    (5, 0.08, "TINY  (5, 0.08)"),
]

print(f"{'Config':<24} {'Contact':>10} {'Time':>8} {'PureRace':>10} {'Time':>8}")
print("-" * 65)

for max_moves, threshold, label in configs:
    multipy = bgbot_cpp.create_multipy_5nn(
        wpaths['purerace'], wpaths['racing'],
        wpaths['attacking'], wpaths['priming'], wpaths['anchoring'],
        NH_PR, NH_RC, NH_AT, NH_PM, NH_AN,
        n_plies=1, filter_max_moves=max_moves, filter_threshold=threshold)

    t0 = time.perf_counter()
    rc = bgbot_cpp.score_benchmarks_multipy(contact_scenarios, multipy, N_THREADS)
    tc = time.perf_counter() - t0
    multipy.clear_cache()

    t0 = time.perf_counter()
    rr = bgbot_cpp.score_benchmarks_multipy(purerace_scenarios, multipy, N_THREADS)
    tr = time.perf_counter() - t0
    multipy.clear_cache()

    print(f"{label:<24} {rc.score():>10.2f} {tc:>7.1f}s {rr.score():>10.2f} {tr:>7.1f}s")
