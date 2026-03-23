# Model Benchmarks

Historical benchmark results for all trained models. When a new model stage is
trained, append its results to the tables below.

## Summary Table (1-ply)

| Model | Contact ER | Race ER | vs PubEval | Self-Play (S/G/B%) |
|-------|-----------|---------|------------|-------------------|
| Single NN 80h (Tesauro 196) | 19.12 | — | — | — |
| Single NN 120h (Tesauro 196) | 17.95 | — | — | — |
| Stage 3 (5-NN, 120h/250h, 244 inputs) | 10.47 | 1.03 | +0.641 | 73.4/25.7/0.9 |
| Stage 4 (5-NN, 120h/250h, 244 inputs, per-NN gpw) | 10.31 | 1.03 | +0.646 | 72.9/26.0/1.1 |
| **Stage 5 (5-NN, 200h/400h, 244 inputs, per-NN gpw)** | **9.87** | **0.95** | **+0.633** | **75.0/24.2/0.8** |
| Stage 6 (5-NN, 100h/300h, 244 inputs, per-NN gpw) | 10.09 | 1.00 | +0.624 | 73.1/26.0/0.9 |

**Targets:** Contact < 10.5, Race < 0.643, vs PubEval > +0.63

## Per-Plan Benchmark Detail

| Model | PureRace | Racing | Attacking | Priming | Anchoring | Crashed |
|-------|----------|--------|-----------|---------|-----------|---------|
| Stage 3 | 0.98 | 6.13 | 8.49 | 9.17 | 10.78 | — |
| Stage 4 | 1.03 | 6.14 | 8.51 | 9.34 | 11.43 | 6.63 |
| **Stage 5** | **0.95** | **5.74** | **8.74** | **8.59** | **11.06** | **6.44** |
| Stage 6 | 1.00 | 5.90 | 8.73 | 9.58 | 11.34 | 6.84 |

## Model Details

### Single NN (Capacity Experiment)

Single neural network with 196 Tesauro inputs. All training data combined
(contact+crashed+race). Trained with TD 50k@0.1 then SL 200ep@1.0.
Benchmarked on contact.bm only (no race.bm or vs PubEval).

| Hidden | Contact ER | Weight Files |
|--------|-----------|-------------|
| 40 | 22.17 | `models/capacity_40h.best` |
| 80 | 19.12 | `models/capacity_80h.best` |
| 120 | 17.95 | `models/capacity_120h.best` |
| 160 | 17.33 | `models/capacity_160h.best` |
| 200 | 17.65 | `models/capacity_200h.best` |
| 250 | 16.93 | `models/capacity_250h.best` |
| 400 | 16.05 | `models/capacity_400h.best` |

Loading: `bgbot_cpp.score_benchmarks_nn(scenarios, weight_path, n_hidden)`

### Stage 3 (5-NN, 120h purerace / 250h contact, 244 inputs)

First 5-NN game plan architecture with extended 244-input encoding.
TD: 200k@0.1 + 1M@0.02. SL: 200ep@20 -> 500ep@6.3 -> 500ep@2.0 (gpw=2.0 uniform).

| Network | Weight File |
|---------|-------------|
| PureRace (120h, 196 inputs) | `models/sl_purerace.weights.best` |
| Racing (250h, 244 inputs) | `models/sl_racing.weights.best` |
| Attacking (250h, 244 inputs) | `models/sl_attacking.weights.best` |
| Priming (250h, 244 inputs) | `models/sl_priming.weights.best` |
| Anchoring (250h, 244 inputs) | `models/sl_anchoring.weights.best` |

Loading:
```python
bgbot_cpp.score_benchmarks_5nn(scenarios,
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    120, 250, 250, 250, 250)
```

### Stage 4 (5-NN, 120h purerace / 250h contact, 244 inputs, per-NN gpw)

Same architecture as Stage 3 but with optimal per-NN game plan weights
(gpw=5.0 for racing/attacking/priming, gpw=1.5 for anchoring).
PureRace reused from Stage 3.

| Network | Weight File |
|---------|-------------|
| PureRace (120h, 196 inputs) | `models/sl_s4_purerace.weights.best` |
| Racing (250h, 244 inputs) | `models/sl_s4_racing.weights.best` |
| Attacking (250h, 244 inputs) | `models/sl_s4_attacking.weights.best` |
| Priming (250h, 244 inputs) | `models/sl_s4_priming.weights.best` |
| Anchoring (250h, 244 inputs) | `models/sl_s4_anchoring.weights.best` |

Loading:
```python
bgbot_cpp.score_benchmarks_5nn(scenarios,
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    120, 250, 250, 250, 250)
```

### Stage 5 (5-NN, 200h purerace / 400h contact, 244 inputs, per-NN gpw) — CURRENT BEST

Larger hidden layers. TD: 200k@0.1 + ~890k@0.02.
SL: per-NN schedules with gpw=5.0 (racing/attacking/priming), gpw=1.5 (anchoring).

| Network | Weight File |
|---------|-------------|
| PureRace (200h, 196 inputs) | `models/sl_s5_purerace.weights.best` |
| Racing (400h, 244 inputs) | `models/sl_s5_racing.weights.best` |
| Attacking (400h, 244 inputs) | `models/sl_s5_attacking.weights.best` |
| Priming (400h, 244 inputs) | `models/sl_s5_priming.weights.best` |
| Anchoring (400h, 244 inputs) | `models/sl_s5_anchoring.weights.best` |

Loading:
```python
bgbot_cpp.score_benchmarks_5nn(scenarios,
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    200, 400, 400, 400, 400)
```

Training script: `python/run_stage5_training.py`

### Stage 6 (5-NN, 100h purerace / 300h contact, 244 inputs, per-NN gpw)

Mid-size model between Stage 5 Small (100h/200h) and Stage 5 (200h/400h).
TD: 200k@0.1 + 1M@0.02. SL: same schedule as Stage 5, except Priming uses gpw=2.0
(gpw=5.0 caused divergence at 300h, similar to Racing at 200h in S5S).
Racing also uses gpw=2.0.

| Network | Weight File |
|---------|-------------|
| PureRace (100h, 196 inputs) | `models/sl_s6_purerace.weights.best` |
| Racing (300h, 244 inputs) | `models/sl_s6_racing.weights.best` |
| Attacking (300h, 244 inputs) | `models/sl_s6_attacking.weights.best` |
| Priming (300h, 244 inputs) | `models/sl_s6_priming.weights.best` |
| Anchoring (300h, 244 inputs) | `models/sl_s6_anchoring.weights.best` |

Loading:
```python
bgbot_cpp.score_benchmarks_5nn(scenarios,
    purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
    100, 300, 300, 300, 300)
```

Training script: `scripts/run_stage6_training.py`

## Benchmark PR (Performance Rating)

Benchmark PR measures average equity error per decision against 2-ply 1296-trial
VR rollout reference. Lower is better. Analogous to XG's Performance Rating.

Data: 200 games, 8,667 decisions (6,326 scored after excluding single-candidate),
25,074 unique positions rolled out on AWS ECS.
Decisions use TINY filter (top 5 within 0.08). Trivial positions excluded (spread < 0.001).
Formula: PR = mean(error) * 500. Scoring generates all legal moves, strategy picks best,
looks up rollout equity in filtered set (worst equity penalty if not found).

| Model | 1-ply | 2-ply |
|-------|-------|-------|
| Stage 3 | 2.76 | 2.12 |
| Stage 4 | 2.57 | 1.97 |
| **Stage 5** | **2.45** | **1.81** |

Per-plan detail (Stage 5 1-ply):

| PureRace | Racing | Attacking | Priming | Anchoring |
|----------|--------|-----------|---------|-----------|
| 0.55     | 2.01   | 2.43      | 2.71    | 3.37      |

N decisions: 1,145 purerace, 1,247 racing, 1,516 attacking, 848 priming, 1,570 anchoring.

Scripts: `python/generate_benchmark_pr.py`, `python/score_benchmark_pr.py`

## Multi-Ply Results (Stage 3 weights, 24 threads, AVX2)

| Metric | 1-ply | 2-ply (TINY) | 3-ply (TINY) |
|--------|-------|--------------|--------------|
| Contact ER | 10.47 | 8.71 | 7.19 |
| Race ER | 1.03 | 0.84 | 0.48 |

## Multi-Ply Contact ER Comparison (Full 107,484 scenarios)

| Model | 1-ply | 2-ply | 3-ply | Time (3-ply) |
|-------|-------|-------|-------|-------------|
| **Stage 5** (200h/400h) | **9.87** | **8.42** | **7.65** | 1,628s |
| Stage 6 (100h/300h) | 10.09 | 8.80 | 7.97 | 341s |

Stage 6 is significantly faster at 3-ply (~4.8x) due to smaller matrices (300h vs 400h),
while maintaining competitive accuracy (7.97 vs 7.65, a 4% gap).

## XG Roller-Style Truncated Rollout Benchmark (Stage 5)

Top-100 worst 1-ply scenarios from 207,484 total (contact + crashed).
Higher ER = harder positions. Lower ER = better strategy.
Script: `scripts/run_top100_safe.py`

Settings key: `Nt` = N trials, `trunc=D` = truncation depth (half-moves),
`dp=P` = decision ply for move selection, `late=P@M` = switch to ply P after
half-move M. See `CLAUDE.md` Truncated Rollouts section for full parameter docs.

Benchmark run: 2026-03-18, 16 threads (RTX 4070S / Windows), Stage 5 production model.
Rollout levels run in separate subprocesses to avoid OOM from cache accumulation.

| Strategy | Settings | ER | Time |
|----------|----------|------|------|
| 1-ply | - | 541.10 | 0.1s |
| 2-ply | TINY filter | 355.24 | 0.3s |
| 3-ply | TINY filter | 338.78 | 3.7s |
| 4-ply | TINY filter | 333.59 | 23.0s |
| XG Roller | 42t, trunc=5, dp=1 | 346.04 | 18.3s |
| XG Roller+ | 360t, trunc=7, dp=2, late=1@2 | 327.82 | 71.9s |
| XG Roller++ | 360t, trunc=5, dp=3, late=2@2 | 325.14 | 156.4s |

**Key observations:**
- **XG Roller++ (dp=3)** is the strongest level at 325.14, beating 4-ply (333.59)
  by a significant margin.
- **XG Roller+ (dp=2)** at 327.82 is close to Roller++ at 40% of the cost.
- **XG Roller (dp=1)** at 346.04 is weaker than 3-ply (338.78) on these worst-case
  positions — the higher move count of crashed positions makes 1-ply move selection
  insufficient even with Monte Carlo averaging.
- **ER values are fully deterministic** across runs (same positions, same model,
  same algorithms). Only timing varies with system load and thread count.
- Roller/Roller+ ER values differ from prior runs (346.04 vs 336.12 for Roller,
  327.82 vs 316.51 for Roller+) due to VR optimizations: thinned VR at odd
  ultra-late moves and 1-ply move1 selection. These trade slight accuracy for
  ~5x faster cube decision rollouts.

**Crash fixes (2026-03-18):** Previous Roller++ crashes were caused by two
pre-existing issues: (1) PosCache at 2M entries (64MB/thread) caused OOM when
16+ threads accumulated thread-local caches — reduced to 512K entries (16MB).
(2) 4MB thread stacks overflowed on deep 3-ply recursion with crashed positions
generating 90+ legal moves — increased to 8MB.

## XG Roller-Style Truncated Rollout Benchmark (Stage 6)

Top-100 worst 1-ply scenarios from 207,484 total (contact + crashed).
Script: `scripts/run_top100_safe.py --model stage6`

Benchmark run: 2026-03-23, 16 threads (RTX 4070S / Windows), Stage 6 model.

| Strategy | Settings | ER | Time |
|----------|----------|------|------|
| 1-ply | - | 558.56 | 0.0s |
| 2-ply | TINY filter | 335.29 | 0.2s |
| 3-ply | TINY filter | 348.46 | 2.8s |
| 4-ply | TINY filter | 353.17 | 13.2s |
| XG Roller | 42t, trunc=5, dp=1 | 325.63 | 34.1s |
| XG Roller+ | 360t, trunc=7, dp=2, late=1@2 | 331.86 | 79.9s |
| XG Roller++ | 360t, trunc=5, dp=3, late=2@2 | 320.03 | 523.7s |

**Note:** XG Roller++ was run separately with 4 threads (vs 16 for other levels)
to avoid thread-local PosCache accumulation crashes at 3-ply decision depth.

## Contact Benchmark by Evaluation Level (Stage 5)

Full contact.bm benchmark (107,484 scenarios) for N-ply evaluators. Truncated
rollout levels use subsampled data (step=100, 1,075 scenarios) due to per-process
memory limits — N-ply calibration on the same subsample shown for comparison.

Benchmark run: 2026-03-19, 16 threads (N-ply) / 8 threads (1T) / 4 threads (2T),
RTX 4070S / Windows, Stage 5 production model.

### Full Dataset (107,484 scenarios)

| Level | Contact ER | Time |
|-------|-----------|------|
| 1-ply | 9.87 | 0.8s |
| 2-ply | 8.42 | 16s |
| 3-ply | 7.65 | 1,628s |

### Subsampled (1,075 scenarios, step=100)

| Level | Contact ER | Time | Notes |
|-------|-----------|------|-------|
| 1-ply | 10.64 | 0.0s | calibration |
| 2-ply | 8.72 | 0.2s | calibration |
| 3-ply | 7.33 | 4.2s | calibration |
| 4-ply | 7.80 | 85s | calibration |
| **1T (XG Roller)** | **9.24** | **160s** | 42t, trunc=5, dp=1 |
| **2T (XG Roller+)** | **6.77** | **1,028s** | 360t, trunc=7, dp=2, late=1@2 |

**Key observations:**
- **2T (XG Roller+)** at ER=6.77 is the strongest evaluator tested, beating 3-ply
  (7.33 on same subsample) by 8% and even beating the 4-ply subsample (7.80).
- **1T (XG Roller)** at ER=9.24 is between 1-ply (10.64) and 2-ply (8.72) on the
  subsample, consistent with its dp=1 move selection.
- **4-ply subsample anomaly:** 4-ply (7.80) is weaker than 3-ply (7.33) on this
  small subsample, likely due to sampling variance (1,075 scenarios). The full
  107k 4-ply benchmark was not run due to time constraints (~24h estimated).
- **Subsample tracking:** N-ply ERs on the step=100 subsample track within ~10% of
  full-dataset values (1-ply: 10.64 vs 9.87, 2-ply: 8.72 vs 8.42, 3-ply: 7.33
  vs 7.65), validating the subsample as representative for rollout comparison.
