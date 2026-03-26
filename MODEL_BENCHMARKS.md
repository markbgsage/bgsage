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
| Stage 7 (17-NN pair, 100h/300h, 244 inputs, per-pair gpw) | 9.76 | 1.00 | — | — |
| Stage 8 (17-NN pair, 100h/400h, S5 fallback, per-pair gpw) | **9.49** | **1.00** | **+0.633** | 72.8/26.1/1.0 |

**Targets:** Contact < 10.5, Race < 0.643, vs PubEval > +0.63

## Per-Plan Benchmark Detail

| Model | PureRace | Racing | Attacking | Priming | Anchoring | Crashed |
|-------|----------|--------|-----------|---------|-----------|---------|
| Stage 3 | 0.98 | 6.13 | 8.49 | 9.17 | 10.78 | — |
| Stage 4 | 1.03 | 6.14 | 8.51 | 9.34 | 11.43 | 6.63 |
| **Stage 5** | **0.95** | **5.74** | **8.74** | **8.59** | **11.06** | **6.44** |
| Stage 6 | 1.00 | 5.90 | 8.73 | 9.58 | 11.34 | 6.84 |
| Stage 7 (pair) | 1.00 | — | — | — | — | — |
| Stage 8 (pair+S5) | 1.00 | 5.26 | 7.95 | 8.33 | 10.83 | 5.92 |

**Note:** Stages 7 and 8 use 17-NN pair strategy (player x opponent game plan), so
per-plan ERs are not directly comparable to single-plan models. See pair-filtered
benchmarks below. Stage 8 uses S5 fallback: any pair NN worse than S5 is replaced
with the corresponding S5 plan weights.

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

| Model | 1-ply | 2-ply | 3-ply | 4-ply | Time (3/4-ply) |
|-------|-------|-------|-------|-------|----------------|
| **Stage 5** (200h/400h) | **9.87** | **8.42** | **7.65** | — | 1,628s |
| Stage 6 (100h/300h) | 10.09 | 8.80 | 7.97 | — | 341s |
| Stage 7 (17-NN pair, 100h/300h) | 9.76 | 8.60 | 7.93 | 7.82 | 419s / 1,016s |

Stage 6 is significantly faster at 3-ply (~4.8x) due to smaller matrices (300h vs 400h),
while maintaining competitive accuracy (7.97 vs 7.65, a 4% gap).

Stage 7 beats Stage 5 at 1-ply (9.76 vs 9.87) thanks to pair specialization, but S5's
larger 400h hidden layers pull ahead at 2-ply+ where multi-ply search compensates for
the lack of opponent-awareness in the single-plan strategy.

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

## Full Rollout Cube Decision Accuracy (Stage 5)

Cubeful rollout of a single reference position comparing different evaluation
strengths for checker play and cube decisions during trial games. All configs use
1296 trials (36² for full stratification), 16 threads, `ultra_late_threshold=9999`
(no ply reductions — configured strategies used for every move throughout the game).

**Reference position:** `[2,2,0,0,0,0,4,1,2,0,1,0,-4,3,0,0,0,-3,0,-4,-2,0,1,0,1,0]`,
centered cube at 1, money game (Jacoby on).

Benchmark run: 2026-03-23, RTX 4070S / Windows, Stage 5 production model.

| Config | Checker | Cube | ND eq | ND SE | DT eq | DT SE | CL eq | CL SE | P(win) | P(gw) | P(gl) | Time |
|--------|---------|------|------:|------:|------:|------:|------:|------:|-------:|------:|------:|-----:|
| Fast, no VR | 1-ply | 1-ply | +0.564 | 0.094 | +0.491 | 0.118 | +0.411 | 0.039 | 0.5850 | 0.3613 | 0.1161 | 0.4s |
| Fast, VR | 1-ply | 1-ply | +0.544 | 0.019 | +0.528 | 0.022 | +0.435 | 0.006 | 0.5943 | 0.3680 | 0.1179 | 6.2s |
| Medium | 2-ply | 3-ply | +0.541 | 0.022 | +0.518 | 0.029 | +0.447 | 0.006 | 0.5987 | 0.3687 | 0.1178 | 76s |
| **3p** | **3-ply** | **3-ply** | **+0.599** | **0.018** | **+0.610** | **0.021** | **+0.461** | **0.006** | **0.6037** | **0.3724** | **0.1162** | **98s** |
| Default | 3-ply | 1T | +0.600 | 0.018 | +0.605 | 0.022 | +0.460 | 0.006 | 0.6041 | 0.3718 | 0.1177 | 858s |
| **XG ref** | **3-ply** | **1T** | **+0.598** | | **+0.606** | | **+0.449** | **0.014** | **0.6007** | **0.3801** | **0.1156** | |

**Key observations:**
- **VR is essential:** No-VR has CL SE=0.039 vs VR CL SE=0.006 (6.5x reduction).
  Cubeful SEs also dramatically improved. VR adds ~15x runtime (0.4s → 6.2s at 1-ply)
  but the SE improvement is worth it for any serious analysis.
- **3-ply checker+cube matches XG:** ND +0.599 vs XG +0.598, DT +0.610 vs XG +0.606.
  This confirms our cubeful rollout produces the same quality as XG's full rollout.
- **3-ply cube ≈ 1T cube, 9x faster:** 3p (98s) and Default/1T (858s) give nearly
  identical cubeful equities, but 3-ply cube is 9x faster. The inner truncated
  rollout for cube decisions adds little accuracy over 3-ply cubeful recursion.
- **`ultra_late_threshold` matters:** Previous runs with the default threshold of 2
  (dropping to 1-ply at move 2+) showed VR appearing to bias results. The real cause
  was that N-ply strategies were only used for the first 2 moves, making "3-ply"
  rollouts essentially 1-ply. Setting `ultra_late_threshold=9999` fixed this entirely.

**Reproducing:**

```bash
python bgsage/scripts/rollout_cube_bench.py
```

Edit the `configs` list in the script to select which configurations to run. Each
config specifies `checker` and `cube` `TrialEvalConfig` objects and
`ultra_late_threshold`.

## Stage 7 (17-NN Pair Strategy, 100h purerace / 300h contact)

17 NNs selected by (player, opponent) game plan pair. 14 distinct weight files —
4 rare pairs (prim_prim, prim_anch, anch_prim, anch_anch) share one NN.
PureRace reused from Stage 6 (same 100h architecture).

TD: 300k@0.1 + 1.5M@0.02. SL: 4-phase schedule with per-pair optimal GPW values
determined by pair-filtered benchmark scan.

| Network | GPW | Weight File |
|---------|-----|-------------|
| PureRace (100h, 196 inputs) | — | `models/sl_s7_purerace.weights.best` |
| Racing/Racing (300h, 244 inputs) | 7.0 | `models/sl_s7_race_race.weights.best` |
| Racing/Attacking (300h) | 10.0 | `models/sl_s7_race_att.weights.best` |
| Racing/Priming (300h) | 5.0 | `models/sl_s7_race_prim.weights.best` |
| Racing/Anchoring (300h) | 7.0 | `models/sl_s7_race_anch.weights.best` |
| Attacking/Racing (300h) | 5.0 | `models/sl_s7_att_race.weights.best` |
| Attacking/Attacking (300h) | 7.0 | `models/sl_s7_att_att.weights.best` |
| Attacking/Priming (300h) | 5.0 | `models/sl_s7_att_prim.weights.best` |
| Attacking/Anchoring (300h) | 10.0 | `models/sl_s7_att_anch.weights.best` |
| Priming/Racing (300h) | 7.0 | `models/sl_s7_prim_race.weights.best` |
| Priming/Attacking (300h) | 7.0 | `models/sl_s7_prim_att.weights.best` |
| Priming/Anchoring (shared, 300h) | 5.0 | `models/sl_s7_prim_anch.weights.best` |
| Anchoring/Racing (300h) | 7.0 | `models/sl_s7_anch_race.weights.best` |
| Anchoring/Attacking (300h) | 7.0 | `models/sl_s7_anch_att.weights.best` |

Loading:
```python
bgbot_cpp.score_benchmarks_pair(scenarios, weight_paths, hidden_sizes)
```

Training scripts: `scripts/run_s7_training.py`, `scripts/run_s7_sl_training.py`,
`scripts/run_s7_sl_phase34.py`

### S7 Pair-Filtered Benchmark Detail

Each NN benchmarked on positions matching its specific (player, opponent) game plan
pair, compared against S6's single-plan NN on the same subset.

| NN | Freq | S7 Pair ER | S6 Plan ER | Δ |
|----|------|-----------|-----------|---|
| race_race | 10.0% | **7.26** | 7.97 | -0.71 |
| race_att | 5.7% | **4.72** | 5.00 | -0.28 |
| race_prim | 9.5% | **4.38** | 4.51 | -0.13 |
| race_anch | 11.3% | **5.26** | 5.82 | -0.56 |
| att_race | 5.7% | **5.39** | 6.99 | -1.60 |
| att_att | 3.9% | **9.90** | 9.96 | -0.06 |
| att_prim | 6.3% | **8.31** | 8.33 | -0.02 |
| att_anch | 6.8% | **9.98** | 10.57 | -0.59 |
| prim_race | 9.5% | **8.50** | 9.40 | -0.90 |
| prim_att | 6.3% | **9.71** | 10.04 | -0.33 |
| prim_anch (shared) | 6.8% | **9.15** | 9.60 | -0.45 |
| anch_race | 11.3% | **11.79** | 11.84 | -0.05 |
| anch_att | 6.8% | **10.36** | 10.72 | -0.36 |
| **Weighted avg** | | **7.85** | **8.53** | **-0.68** |

All 13 NNs beat S6 baseline. Weighted average improvement: -0.68 ER (8.0%).

## XG Roller-Style Truncated Rollout Benchmark (Stage 7)

Top-100 worst 1-ply scenarios from 207,484 total (contact + crashed), selected
using S7 pair strategy at 1-ply.

Benchmark run: 2026-03-24, RTX 4070S / Windows, Stage 7 pair model.
N-ply levels run at 16 threads; rollout levels at 4 threads (16 threads causes
segfault with pair strategy due to thread-local cache accumulation — see CLAUDE.md).

| Strategy | Settings | ER | Time |
|----------|----------|------|------|
| 1-ply | — | 408.00 | <1s |
| 2-ply | TINY filter | 250.74 | <1s |
| 3-ply | TINY filter | 243.32 | 2.8s |
| 4-ply | TINY filter | 242.65 | 5.5s |
| XG Roller | 42t, trunc=5, dp=1 | 233.04 | 23.5s |
| XG Roller+ | 360t, trunc=7, dp=2, late=1@2 | 223.51 | 240s |
| XG Roller++ | 360t, trunc=5, dp=3, late=2@2 | 225.17 | 492s |

**Note:** S7 top-100 ERs are not directly comparable to S5/S6 top-100 ERs because
the top-100 worst positions differ between models (selected by each model's own
1-ply errors). S7's lower absolute ERs partly reflect that its worst 1-ply errors
are less severe than S5/S6's worst errors.

## Stage 8 (17-NN Pair Strategy, 100h purerace / 400h contact, S5 fallback)

17 NNs selected by (player, opponent) game plan pair. Same architecture as Stage 7
but with 400h contact NNs (matching Stage 5 hidden size). 14 distinct weight files —
4 rare pairs share one NN (same sharing as S7). PureRace reused from Stage 7.

TD: 300k@0.1 + 1.5M@0.02. GPW scan: [2, 5, 7, 10, 12] with pair-filtered benchmarks.
SL: 4-phase schedule (100ep@20 + 200ep@10 + 200ep@3.1 + 500ep@1.0) with per-pair
optimal GPW values. S5 fallback: any pair NN with worse pair-filtered ER than S5 is
replaced with the corresponding S5 plan weights, guaranteeing no regressions.

Training script: `scripts/run_s8_training.py`

### S8 Standard Benchmarks (1-ply)

Benchmark run: 2026-03-25, RTX 4070S / Windows.

| Benchmark | S8 | S5 (400h) | S7 (300h) |
|-----------|------|-----------|-----------|
| PureRace | 1.00 | 0.95 | 1.00 |
| Racing | 5.26 | 5.74 | 5.90 |
| Attacking | 7.95 | 8.74 | 8.73 |
| Priming | 8.33 | 8.59 | 9.58 |
| Anchoring | 10.83 | 11.06 | 11.34 |
| **Contact** | **9.49** | **9.87** | **9.76** |
| Crashed | 5.92 | 6.44 | — |
| Race | 1.00 | 0.95 | 1.00 |
| vs PubEval | +0.633 | +0.633 | — |

### S8 Pair-Filtered Benchmarks vs S5 (1-ply)

Each pair NN is scored only on benchmark positions matching its (player, opponent)
game plan pair. S5 uses the player's plan NN on the same subset. "S5" in the GPW
column means the S5 fallback was applied (S8 pair NN was worse, replaced with S5
plan weights).

| NN | Count | GPW | S8 ER | S5 ER | Delta |
|----|-------|-----|-------|-------|-------|
| race_race | 16901 | 7.0 | 7.10 | 7.57 | -0.48 |
| race_att | 17425 | 12.0 | 4.30 | 4.44 | -0.14 |
| race_prim | 12514 | S5 | 4.49 | 4.49 | 0.00 |
| race_anch | 25648 | 12.0 | 5.09 | 5.20 | -0.11 |
| att_race | 15944 | 10.0 | 5.62 | 7.24 | -1.62 |
| att_att | 10369 | S5 | 9.56 | 9.56 | 0.00 |
| att_prim | 9326 | 12.0 | 7.93 | 8.05 | -0.12 |
| att_anch | 10088 | 12.0 | 9.98 | 10.31 | -0.34 |
| prim_race | 20775 | 12.0 | 7.95 | 8.24 | -0.29 |
| prim_att | 13243 | S5 | 9.10 | 9.10 | 0.00 |
| prim_anch (shared) | 9548 | 5.0 | 8.78 | 8.82 | -0.04 |
| anch_race | 29037 | 5.0 | 11.25 | 11.52 | -0.26 |
| anch_att | 16666 | 5.0 | 10.26 | 10.31 | -0.05 |
| **Weighted avg** | **207484** | | **7.77** | **8.05** | **-0.28** |

10 of 13 pair NNs beat S5; 3 replaced with S5 weights (race_prim, att_att, prim_att).
Biggest win: att_race (-1.62), where pair specialization helps the Attacking NN
recognize Racing opponents.

### S8 Multi-Ply Contact Benchmarks

| Level | S8 Contact ER | S5 Contact ER |
|-------|--------------|--------------|
| 1-ply | 9.49 | 9.87 |
| 2-ply | 8.44 | — |
| 3-ply | 7.76 | — |
| 4-ply | 7.66 | — |

### S8 Top-100 Worst Positions Benchmark

Top-100 worst 1-ply scenarios from 207,484 total (contact + crashed), selected
using S8 pair strategy at 1-ply.

Benchmark run: 2026-03-25, RTX 4070S / Windows, Stage 8 pair model (after S5 fallback).
N-ply and XG Roller/Roller+ at 16 threads; XG Roller++ at 4 threads (segfaults at 8+
threads with pair strategy at 3-ply decision depth).

| Strategy | Settings | ER | Time |
|----------|----------|------|------|
| 1-ply | — | 406.15 | <1s |
| 2-ply | TINY filter | 248.57 | <1s |
| 3-ply | TINY filter | 282.54 | 2.1s |
| 4-ply | TINY filter | 275.99 | 2.7s |
| XG Roller | 42t, trunc=5, dp=1 | 250.95 | 40.9s |
| XG Roller+ | 360t, trunc=7, dp=2, late=1@2 | 234.37 | 172.1s |
| XG Roller++ | 360t, trunc=5, dp=3, late=2@2 | 232.51 | 741.4s |

**Note:** S8 top-100 ERs are not directly comparable to S5/S6/S7 top-100 ERs because
the top-100 worst positions differ between models (selected by each model's own
1-ply errors).
