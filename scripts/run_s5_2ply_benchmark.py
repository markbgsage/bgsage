"""Run Stage 5 2-ply contact + race benchmarks, output to stdout."""
import bgbot_cpp, os, sys, time
sys.path.insert(0, 'python')
from bgsage.data import load_benchmark_file

models = 'models'
w5 = {
    'pr': os.path.join(models, 'sl_s5_purerace.weights.best'),
    'rc': os.path.join(models, 'sl_s5_racing.weights.best'),
    'at': os.path.join(models, 'sl_s5_attacking.weights.best'),
    'pm': os.path.join(models, 'sl_s5_priming.weights.best'),
    'an': os.path.join(models, 'sl_s5_anchoring.weights.best'),
}
NH5 = (200, 400, 400, 400, 400)

contact = load_benchmark_file('data/contact.bm')
race = load_benchmark_file('data/race.bm')

multipy2 = bgbot_cpp.create_multipy_5nn(
    w5['pr'], w5['rc'], w5['at'], w5['pm'], w5['an'], *NH5,
    n_plies=2, filter_max_moves=5, filter_threshold=0.08)

# 2-ply contact
t0 = time.perf_counter()
r = bgbot_cpp.score_benchmarks_multipy(contact, multipy2, 0)
elapsed = time.perf_counter() - t0
ch = multipy2.cache_hits()
cm = multipy2.cache_misses()
print(f'Stage 5 2-ply contact: {r.score():.2f}  ({elapsed:.1f}s)')
print(f'  Cache: {multipy2.cache_size()} entries, {ch} hits / {ch+cm} lookups ({ch/max(ch+cm,1)*100:.1f}%)')
sys.stdout.flush()

# 2-ply race
multipy2.clear_cache()
t0 = time.perf_counter()
r = bgbot_cpp.score_benchmarks_multipy(race, multipy2, 0)
elapsed = time.perf_counter() - t0
ch = multipy2.cache_hits()
cm = multipy2.cache_misses()
print(f'Stage 5 2-ply race: {r.score():.2f}  ({elapsed:.1f}s)')
print(f'  Cache: {multipy2.cache_size()} entries, {ch} hits / {ch+cm} lookups ({ch/max(ch+cm,1)*100:.1f}%)')
