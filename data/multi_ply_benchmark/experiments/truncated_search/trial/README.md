# Trial-body and opening-cache experiments

Five exact-work optimizations were tested at engine revision `60f93556137a7a5c4be671502f4b3f521998ff94`. The measured speedup is the control's total wall time divided by the variant's total wall time. Each row uses the same positions and seeds on both sides. A value above 1 favors the variant.

| Change | 2T | 3T |
| --- | ---: | ---: |
| Reuse the variance-reduction winner and its probabilities when building the move-1 table | 0.923× | 1.008× |
| Skip move generation for an opening play already published to the cache | 0.991× | 1.002× |
| Rank only the driving cube in the base pair networks' variance-reduction selector | 0.942× | 0.974× |
| Generate and score rolls consecutively, retaining candidate storage for the played roll | 0.963× | 1.024× |
| Combine the four changes above | 0.960× | 1.003× |
| Cache the weighted cubeful variance-reduction mean by move-1 opening roll and branch | 1.000× | 0.968× |

We discarded these five trial-loop variants. Across the seven benchmark positions the largest observed gain at either level was 2.4%, from streaming on 3T. That implementation took 3.8% more wall time on 2T. Every variant returned the same complete analytics as its four-thread control, including checker ordering, cube recommendations, probabilities, equities and standard errors. The four-thread control also agreed with the archived 16-thread control to within `1.20e-7` across every numeric field, with identical discrete results.

## Measurement

`screen.py` calls the public Stage 11 analyzer with the one-sided bearoff database, standard truncated presets and cubeful evaluation. The seven positions cover all five game plans, unlimited and match scoring, and the three analytics APIs. Their IDs and selection are in `CASES`. Each position uses the seed from the full truncated benchmark. Each arm has a dedicated process and analyzer. One arm runs at a time on logical CPUs 0–3, with four evaluator threads. The runner shuffles the order within each position using seed 7917. Timings include the public analytics call and exclude analyzer creation and JSON serialization.

The first screen includes a control, four individual changes and their combination, with two repeats at each level, for 168 calls. The move-1 mean experiment has a fresh matching control and three repeats at each level, for 84 calls. The control and variants used MSVC Release, AVX2, fast floating point and link-time code generation. Each binary linked the same baseline library and Python binding with its own compiled rollout implementation. These figures measure end-to-end performance on the four assigned logical CPUs.

| Files | Contents |
| --- | --- |
| `screen.json` and `mean-screen.json` | Binary SHA-256 hashes, repetitions, aggregate and median paired speedups, exact comparisons |
| `screen.jsonl.gz` and `mean-screen.jsonl.gz` | Every measured call, including request, seed, wall seconds and complete analytics |
| `variants.zip` | Standalone unified diffs for the five measured implementations and the four-change combination |

## Reproduction

Use a build-only export of the baseline revision for each variant. From the engine directory the following PowerShell example prepares the mean-cache source. Apply its patch to a fresh export. For the control, build the unmodified export.

```powershell
$study = "data/multi_ply_benchmark/experiments/truncated_search/trial"
Expand-Archive "$study/variants.zip" build/trial-patches
git archive --format=zip --output=build/trial-baseline.zip 60f93556137a7a5c4be671502f4b3f521998ff94
Expand-Archive build/trial-baseline.zip build/trial-mean-src
git apply --directory=build/trial-mean-src build/trial-patches/mean.patch
```

Build both source exports using the engine's CPU-only Release CMake configuration with Python bindings enabled. Use the same compiler, Python, pybind11 and flags on both sides. Run from the engine directory with the Stage 11 weights and bearoff database installed. Substitute the paths of the two built extensions.

```powershell
python "$study/screen.py" --output build/trial-mean-run --threads 4 --mask 0xf --repeats 3 --build "base=$baseExtension" --build "mean=$meanExtension"
python "$study/summarize.py" build/trial-mean-run mean-screen --destination build/trial-mean-summary
```

For the four-idea screen, pass `base`, `prefill`, `opening`, `driver`, `stream` and `prefill-opening-driver-stream` in that order with `--repeats 2`. The optional `--reference` argument to `summarize.py` accepts the directory containing the original `2T-candidate/results.jsonl` and `3T-candidate/results.jsonl` files.
