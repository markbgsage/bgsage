# Warm 2T computation with checker progress

This check compares the corrected original Linux build at `139427d`, the intermediate build at `82e6397`, and the selected engine on the same sixteen logical CPUs. Each set contains one fixed checker or cube position per game plan and per match or unlimited mode. The ten checker calls request both trial and finalization callbacks. The ten cube calls use the standard API defaults. A separate warm analyzer for each build repeats those checker positions without callbacks. The callbacks count notifications in process. The timed interval excludes network traffic, external progress reporting, queueing, cache lookups in an application, model loading and result serialization.

Each analyzer uses the built-in 2T preset, its default seed of 42 for the entire session, and the same held-out cube call for warm-up. The build order rotates and reverses for each API and callback setting. Every call runs to completion before the next starts. The full Linux benchmark resets the seed per position. [Full-corpus timings](../results.md) · [History and original cache correction](../historical-linux/README.md)

| 10 warm 2T calls | Corrected original | Intermediate | Selected | Original / selected | Intermediate / selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checker with progress | 11.682 s | 4.182 s | 2.759 s | 4.23× | 1.52× |
| Cube | 4.154 s | 1.521 s | 0.987 s | 4.21× | 1.54× |
| Checker without progress | 11.638 s | 4.183 s | 2.718 s | 4.28× | 1.54× |

All ten checker decisions and ten cube recommendations agreed across the builds. Each build returned identical checker analytics with and without a progress listener. No scored probability or equity moved materially farther from or closer to rollout truth. The largest direct difference between the selected build and either comparison was 0.000000719 for a probability and 0.000000775 for an equity. [All 90 public responses](calls.jsonl.gz) · [Results by API](summary.json) · [Position roster, seeds, binaries and assets](settings.json)

The checker callback counts matter for applications that publish each update. The original and intermediate builds each delivered 1,306 trial notifications and 46 finalization notifications across ten checker calls. The selected build delivered 3,481 and 121. Those extra notifications reflect the shorter trial assignments. The 1.52× measurement includes the cost of calling Python and incrementing counters. A listener that serializes or transmits each notification adds its own cost.

With the three Linux builds in the locations recorded by the study, run from the engine root. The output directory must be new.

```sh
python3 -B data/multi_ply_benchmark/experiments/truncated_pass2/app-style/measure.py \
    --output logs/truncated_pass2/warm-2t-repeat
```
