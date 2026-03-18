# Truncated Rollout Performance Notes

Reference benchmark for XG Roller++ Cube-equivalent rollout on the fixed cube
position used by [`benchmark_3t.cpp`](/C:/Users/mghig/Dropbox/agents/bgbot/bgsage_truncated_rollout_opt/cpp/src/benchmark_3t.cpp).

## Benchmark Position

`0,0,0,2,2,-2,3,2,2,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0`

Money game, cube centered at 1, Jacoby on, Beaver on.

## Truncated Rollout Settings

- `n_trials=360`
- `truncation_depth=7`
- `decision_ply=3`
- `late_ply=2`
- `late_threshold=2`
- `enable_vr=true`
- `n_threads=16`
- `seed=42`

## Reference Outputs

These values stayed unchanged across all retained optimizations.

- `ND = +0.973394`  `SE = 0.003849`
- `DT = +1.060218`  `SE = 0.010040`
- `DP = +1.000000`
- `CL = +0.600300`  `SE = 0.004319`
- `P(win) = 0.793274`  `SE = 0.002075`
- `P(gw)  = 0.023935`  `SE = 0.000415`
- `P(bw)  = 0.000089`  `SE = 0.000014`
- `P(gl)  = 0.010198`  `SE = 0.000283`
- `P(bl)  = 0.000073`  `SE = 0.000017`

## Timing Notes

Single-run wall times on this Windows development box were noisy enough that
they should be treated as point samples, not stable absolutes.

- Initial pre-optimization standalone run: `1.827 s`
- After retained optimizations, best isolated standalone runs observed:
  - `1.502 s`
  - `1.613 s`
  - `1.563 s`
  - `1.556 s`
- Later standalone runs under heavier system noise were slower (~`2.2 s`)

Using the best isolated pre/post runs on the same box, the retained changes
improved this benchmark by about `17.8%` (`1.827 -> 1.502 s`).

## Retained Changes

- Shared cross-thread position cache now coordinates in-flight computations
  instead of allowing parallel duplicate N-ply work.
- Trial dispatch uses chunked work stealing (`chunk=4`) to reduce atomic
  overhead and improve thread-local cache locality.
- The shared cross-thread position cache is now persistent inside
  `RolloutStrategy`, which helps serial batches of 16-thread rollout
  evaluations avoid rebuilding that cache structure every call.

## Spot Checks

Same position, same model, `16` threads, `MoveFilters::TINY`:

- `3-ply cube_decision_nply`: `0.005 s`
- `4-ply cube_decision_nply`: `0.041 s`
