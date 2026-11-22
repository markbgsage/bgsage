# CPU contact encoder

The selected encoder reduced aggregate public-call latency by 18.1% at 2P, 15.7% at 3P and 18.5% at 4P in its independent four-worker Linux screen. Every one of the 660 complete analytics responses equaled the release at `82e6397`, and all measured changes in error against the frozen rollout reference were zero. The 2P and 3P screens include all 100 checker, 100 cube and 100 post-move calls. The 4P screen sampled two positions from each game-plan/source cell for each API. [Per-API results](lanes_screen/score.json) · [Full analytics and timings](lanes_screen/outputs.zip)

| Level | Public calls | Control, seconds | Selected encoder, seconds |
| --- | ---: | ---: | ---: |
| 2P | 300 | 0.843 | 0.691 |
| 3P | 300 | 7.792 | 6.570 |
| 4P | 60 | 28.190 | 22.977 |

## Where the time went

On 44,282 real benchmark descendants the original encoder spent a median 343 ns on the two players' single-hit, double-hit and pip-loss features, including point-mask construction. The whole contact vector took 612 ns. The selected encoder took 82 ns and 308 ns respectively. These are four standalone runs on one logical CPU per arm. Frequency and other system activity varied between runs. The public comparisons alternated the order of releases by position, pinned four workers to logical CPUs 4–7, and excluded startup, output serialization and disk I/O. [Probe](probe.cpp) · [Control measurements](ctrl_native.json) · [Selected measurements](bar_reach_lanes_native.json)

Three changes reduce work on the same trained inputs.

- **Bar entries.** The pip-loss feature takes the highest valid target for each roll. Its target masks can be constructed directly from the open entry points, indirect paths and permitted direct hits. The one-checker case preserves the reference's smaller-die-first priority and its distinct entry test on doubles. With multiple checkers on the bar, only direct entries can contribute. This change alone reduced the paired 4P sample from 44.36 to 40.84 seconds. [Isolated public comparisons](screen/score.json)
- **Reachable blots.** With no checker on the bar, blots at or beyond the rearmost checker can contribute to none of the three features. Removing them before testing for zero or multiple blots avoids the remaining calculation when possible. [Native measurements](reach_native.json)
- **Parallel rolls.** On AVX2 each lane represents an entire target bit mask for one roll. Testing the unlike rolls in two groups reduced the same paired 4P sample to 37.37 seconds when combined with the first two changes. The selected implementation also evaluates the six doubles in one group. The highest target of a 24-bit mask is recoverable exactly from its binary32 exponent. The scalar implementation evaluates the same rolls on CPUs without the AVX2 path. [Intermediate full results](screen/outputs.zip) · [Selected implementation](../../../../../cpp/src/encoding.cpp)

## Verification and replay

All 63,910,432 bytes of 244-input contact features agreed on 65,482 corpus boards across every native variant. A second deterministic generator independently checked all three integer shot features on 262,144 boards against the original scalar board functions. It covers zero through three checkers on either bar. The final encoder and the original were also compiled with AVX2 and FMA disabled on x86-64. Their complete feature hashes agree with each other and with the AVX2 binaries. [Scalar results](scalar.json) · [Native checks](probe.cpp)

The study uses the repository-local GCC 15.2 Linux toolchain and qualified object and input corpora from the preceding CPU studies. Run the commands from the engine root.

```bash
study=data/multi_ply_benchmark/experiments/search_pass3/evaluator
base=build/det_eval2/linux_release/guarded
python3 -B "$study/build_linux.py" ctrl "$base" --probe
python3 -B "$study/run.py" ctrl bar reach simd bar_simd bar_reach_simd bar_reach_lanes
python3 -B "$study/scalar.py"
python3 -B "$study/build_linux.py" lanes build/search_pass3/evaluator/source_bar_reach_lanes
python3 -B data/multi_ply_benchmark/experiments/search_pass2/screen_checker.py \
  --arm control "$base" "$base/python" --arm lanes build/search_pass3/evaluator/lanes build/search_pass3/evaluator/lanes/python \
  --cores 4 5 6 7 --apis checker_play cube_action post_move_analytics --levels 2ply 3ply 4ply \
  --per-group 10 --deep-per-group 2 --repeats 1 --output build/search_pass3/lanes-replay
python3 -B "$study/score.py" build/search_pass3/lanes-replay
```
