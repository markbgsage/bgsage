# Native neural base verification

The probe compares the actual C++ methods exported by two Linux builds of `bgbot_cpp`. Both build directories must contain their matching `cpp/` headers. Use the same Python ABI, toolchain, and compilation environment as the engines.

```sh
python3 data/multi_ply_benchmark/experiments/full_rollout_cpu/dense/check.py \
  --control build/truncated_pass2/shipped \
  --candidate build/truncated_pass2/full-recip-release \
  --compile-commands build/logs/linux_cpu/final-gcc/compile_commands.json \
  --output logs/full_rollout_opt/dense-release.json
```

Each case creates the same seeded network in both engines. The probe records the full saved hidden sums, the copy of the input features, the five probabilities returned by the base evaluation, and the five probabilities after changing every thirteenth input. It also checks that the candidate's preparation-only method produces identical state and identical incremental probabilities to its output-producing method. The fixtures cover 196- and 244-input networks, short and non-vector-aligned dimensions, stack and heap hidden buffers, and sparse, dense, and all-zero inputs. Reports contain the two extension hashes and hashes of the raw results.

With `full-hot-dense` as the candidate, the same probe isolates the base preparation change. All 98,352 floats were byte-identical to the control. [Standalone base results](standalone.json)

The sigmoid probe recompiles the actual eight- and sixteen-lane functions from each source export with the engine's release flags. On an AVX-512F Linux CPU it compares 1,048,576 finite inputs, half drawn from [−6, 6] and half from [−120, 120].

```sh
python3 data/multi_ply_benchmark/experiments/full_rollout_cpu/dense/check_sigmoid.py \
  --tree wide build/truncated_pass2/full-wide-release \
  --tree split_reciprocal build/truncated_pass2/full-recip-release \
  --compile-commands build/logs/linux_cpu/final-gcc/compile_commands.json \
  --output logs/full_rollout_opt/sigmoid.json
```

Across 288 final-build cases, both kernels returned identical saved state and base predictions. The split reciprocal reproduced all 98,352 returned floats. The standard wide reciprocal differed in 66 incremental predictions with a maximum absolute difference of 0.00000023842. The separate sigmoid probe found zero differences for the split reciprocal and a maximum difference of 0.00000017882 for the standard wide reciprocal. [Split network](release_split.json) · [Wide network](release_wide.json) · [Sigmoid results](release_sigmoid.json)
