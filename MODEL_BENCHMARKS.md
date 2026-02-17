# Model Benchmarks

Historical benchmark results for all trained models. When a new model stage is
trained, append its results to the tables below.

## Summary Table (0-ply)

| Model | Contact ER | Race ER | vs PubEval | Self-Play (S/G/B%) |
|-------|-----------|---------|------------|-------------------|
| Single NN 80h (Tesauro 196) | 19.12 | — | — | — |
| Single NN 120h (Tesauro 196) | 17.95 | — | — | — |
| Stage 3 (5-NN, 120h/250h, 244 inputs) | 10.47 | 1.03 | +0.641 | 73.4/25.7/0.9 |
| Stage 4 (5-NN, 120h/250h, 244 inputs, per-NN gpw) | 10.31 | 1.03 | +0.646 | 72.9/26.0/1.1 |
| **Stage 5 (5-NN, 200h/400h, 244 inputs, per-NN gpw)** | **9.87** | **0.95** | **+0.633** | **75.0/24.2/0.8** |

**Targets:** Contact < 10.5, Race < 0.643, vs PubEval > +0.63

## Per-Plan Benchmark Detail

| Model | PureRace | Racing | Attacking | Priming | Anchoring | Crashed |
|-------|----------|--------|-----------|---------|-----------|---------|
| Stage 3 | 0.98 | 6.13 | 8.49 | 9.17 | 10.78 | — |
| Stage 4 | 1.03 | 6.14 | 8.51 | 9.34 | 11.43 | 6.63 |
| **Stage 5** | **0.95** | **5.74** | **8.74** | **8.59** | **11.06** | **6.44** |

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

## Benchmark PR (Performance Rating)

Benchmark PR measures average equity error per decision against 1-ply 1296-trial
VR rollout reference. Lower is better. Analogous to XG's Performance Rating.

Data: 200 games, 8,667 decisions (6,326 scored after excluding single-candidate),
25,074 unique positions rolled out on AWS ECS.
Decisions use TINY filter (top 5 within 0.08). Trivial positions excluded (spread < 0.001).
Formula: PR = mean(error) * 500. Scoring generates all legal moves, strategy picks best,
looks up rollout equity in filtered set (worst equity penalty if not found).

| Model | 0-ply | 1-ply |
|-------|-------|-------|
| Stage 3 | 2.76 | 2.12 |
| Stage 4 | 2.57 | 1.97 |
| **Stage 5** | **2.45** | **1.81** |

Per-plan detail (Stage 5 0-ply):

| PureRace | Racing | Attacking | Priming | Anchoring |
|----------|--------|-----------|---------|-----------|
| 0.55     | 2.01   | 2.43      | 2.71    | 3.37      |

N decisions: 1,145 purerace, 1,247 racing, 1,516 attacking, 848 priming, 1,570 anchoring.

Scripts: `python/generate_benchmark_pr.py`, `python/score_benchmark_pr.py`

## Multi-Ply Results (Stage 3 weights, 24 threads, AVX2)

| Metric | 0-ply | 1-ply (TINY) | 2-ply (TINY) |
|--------|-------|--------------|--------------|
| Contact ER | 10.47 | 8.71 | 7.19 |
| Race ER | 1.03 | 0.84 | 0.48 |
