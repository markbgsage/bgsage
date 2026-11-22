# Keeping legal boards alive during nested rollouts

A public cube-action call with a nested checker evaluator on `cube-money-attacking-01` returned **Double/Take** with one worker and **No Double** with two at checkpoint `1ecca390`. Both configurations used four outer trials, three half-moves, 1P checker and horizon defaults, and seed 1720888050. The checker override started a two-game, one-half-move child rollout. The dice and trial count stayed fixed.

The parent generated legal boards and passed them to its checker policy. Before it consumed the winning index, the child rollout generated its own moves on the same thread into that live buffer. The serial and parallel opening-prefill paths use different scheduling, so the overwritten candidate lists could lead to different results. The same problem affected later trial moves even when both schedules agreed.

| Build and policy | One worker, ND / DT | Two workers, ND / DT | Actions |
| --- | ---: | ---: | --- |
| Checkpoint, nested checker | 0.340444 / 0.398978 | 0.400287 / 0.386464 | Double/Take and No Double |
| Optimization build before buffer fix, nested checker | 0.340444 / 0.398978 | 0.400288 / 0.386464 | Double/Take and No Double |
| Fixed, nested checker | 0.372515 / 0.337019 | 0.372515 / 0.337019 | Both No Double |
| Fixed, nested late checker | 0.372515 / 0.337019 | 0.372515 / 0.337019 | Both No Double |
| Fixed, nested cube evaluator | 0.385224 / 0.347976 | 0.385224 / 0.347976 | Both No Double |

In the fixed build every returned field matches across worker counts for each of the three configurations, including the five probabilities and all standard errors. The first two rows used independent binaries. The [archive of all 18 public results](outputs.zip) includes complete inputs and binary hashes. The [regression tests](../../../../../../tests/test_rollout_prefill.py) cover checked-in values for both checker configurations and require serial/parallel equality for the cube override.

The generator and trial each own their active buffers. An inner rollout allocates a separate buffer when the worker has already lent its idle buffer to an outer call. Ordinary sequential trials reuse the worker's existing storage. The change affects buffer ownership and preserves the caller's original candidate ordering, evaluator, dice, and cube state. These four-game observations test execution determinism. The main timing study measures accuracy against the existing 1,296-game references. [Rollout algorithm](../../../../../../ROLLOUT.md)

Run the [reproducer](reproduce.py) with the extension and Python source for the desired build, starting at the engine root.

```sh
python data/multi_ply_benchmark/experiments/truncated_pass2/dispatch/nested/reproduce.py \
  --extension build/ENGINE/bgbot_cpp.EXTENSION --python-dir build/ENGINE/python \
  --output logs/nested-rollout.json
```
