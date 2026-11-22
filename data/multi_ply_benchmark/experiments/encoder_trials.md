# Contact encoder and neural-network experiments

The November 21, 2026 profile singled out contact features as a larger cost than
the typical neural-network inference on sibling moves. A standalone MSVC probe
used the 1,422 candidate boards from the 80 contact checker positions in the
fixed corpus. Within each position, the first candidate supplied the neural
network's base sums and the remaining candidates used its sparse delta path.
The first pass measured 2,907 ns per extended encoding, 1,509 ns per full base
evaluation, and 562 ns per inference averaged across siblings. One call to
`double_hitting_shots` averaged 355 ns. Contact encoding calls it on both sides.
These are one-thread component measurements with preloaded inputs and one
244-input, 400-hidden contact net. They exclude routing and move generation.

| Experiment | Implementation and result | Decision |
| --- | --- | --- |
| A. Reuse blocked points in escape calculations | Build a 24-bit mask for each side. Each mobility, containment, and rescue query reads a 12-point window with one shift, mask, and table lookup. The same masks serve the flipped board. | Retained with B and the explicit squared-count normalization below. |
| B. Precompute double-hit geometry | For each of the 24 × 24 possible distances to two blots, store the die and number of steps once in a constexpr table. The inner loop still checks every original blocker and checker-source condition. Median isolated double-shot time across three launches fell from 497 to 177 ns. | Retained. |
| C. Eliminate duplicate hidden-layer accumulation | In `forward_save_base`, accumulate only in the saved hidden buffer, then copy once before sigmoid. The compiler's floating-point contraction changed a few NN outputs by at most 2.38e-7. Median sibling inference was 865 ns versus 804 ns in the baseline. | Discarded. |
| D. SIMD detection of sparse input differences | Compare eight sibling inputs with AVX2, iterate changed mask bits in original column order, and apply the same delta updates. Full feature and NN output hashes matched. Median sibling time was 678 ns versus 804 ns, but timings overlapped under concurrent load. | Experimental only. No end-to-end gain was established for this change. |
| E. Cache contact features per exact board | A 512-entry thread-local table held a full comparison board and its two 26-feature blocks. Compared with A+B, full checker time rose from 21.74 to 22.19 s, cube time fell from 4.66 to 4.47 s, and post-move rose from 1.23 to 1.28 s. All outputs matched. | Discarded. Its additional allocation and lookups did not improve the three calls in aggregate. |

**Whole-engine comparison.** The final A+B executable and a fresh baseline
were built from the same Git revision with the same MSVC toolchain and release
flags. Both used the original Python package, Stage 11 weights and bearoff
database, cubeful 3-ply, two requested threads, and CPU affinity 8–11. Each
arm ran all 100 positions per API. The case ID's numeric suffix modulo five
split the positions into five equal shards balanced across source and game
plan. Each shard ran the APIs in checker, cube, post-move order in a fresh
process. Arm order alternated between shards. Timings cover public calls only.
Other local work ran on the machine, so speed estimates vary across passes.

| API | Baseline seconds | Retained encoder seconds | Full outputs equal |
| --- | ---: | ---: | ---: |
| Checker play | 45.080 | 36.745 | 100/100 |
| Cube action | 7.595 | 6.520 | 100/100 |
| Post-move | 2.390 | 1.638 | 100/100 |
| All three | 55.065 | 44.902 | 300/300 |

**Numeric checks.** Whole-program optimization initially changed how the
compiler rounded squared containment probabilities. Across 31,989 real child
boards, the largest feature difference was one float ULP, confined to feature
indices 113, 233, and 235. Expressing the squared term as the square of the
raw roll count multiplied by `1.0f / 1296.0f` restored the baseline rounding.
All 7,805,316 floats in the final native feature dump were byte-identical
(SHA-256 `ba7991df5c18121dc0ccd7ff7e0f90562b5e708a1215ebf3a4e6eca9274c7a92`).
The boards comprise the 1,422 legal endpoints and their legal replies on
one deterministic roll per endpoint. Both existing extended-input test suites
also passed (95 tests).

The [experiment data](encoder_trials.json) lists runs in execution order,
three launches per standalone microprobe variant, binary hashes, per-shard
wall times, and output digests. The two stages of A+B and the cache trial used
different binaries in the same private build directory. The variant identity
section distinguishes them. The fixed rollout-reference study records absolute
accuracy separately. Exact equality to the original 300 outputs also preserves
each of their errors against any common rollout reference.

Subsequent tests measured the sparse-input scan with the improved contact
encoder and four-column accumulation on the complete public calls. The
combined neural changes passed the whole-call and rollout timing checks.
[CPU neural study](cpu_nn/README.md)
