# Linux x86-64 3-ply validation

The optimized Linux build chose the same checker move and cube action as the original on every fixed position. Checker rankings also matched in full. Across the three public APIs, the largest change was 0.0000001193 in a cubeless probability and 0.0000009537 in an equity. Both are below the respective 0.0025 and 0.005 materiality thresholds.

## Wall time

Each row covers all 100 positions for that API. The timer surrounds a normal cubeful `BgBotAnalyzer` call with Stage 11 weights and the exact bearoff database. A new process loaded each build. At two threads the original ran first. At sixteen threads the optimized build ran first. The timings include one complete pass per build and thread count.

| Threads | API | Original seconds | Optimized seconds | Speedup | Faster positions |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | Checker play | 20.008 | 7.760 | 2.58× | 100/100 |
| 2 | Cube action | 12.541 | 0.992 | 12.64× | 100/100 |
| 2 | Post-move | 1.051 | 0.559 | 1.88× | 97/100 |
| 16 | Checker play | 6.232 | 2.684 | 2.32× | 99/100 |
| 16 | Cube action | 11.683 | 0.500 | 23.37× | 100/100 |
| 16 | Post-move | 0.431 | 0.263 | 1.64× | 93/100 |

This was an Ubuntu 26.04 WSL2 guest on an eight-core, sixteen-logical-processor Ryzen 7 9700X. GCC 15.2.0 compiled both CPU engines with the project's Release flags `-O3 -ffast-math -funroll-loops -mavx2 -mfma`. Python 3.14.4 loaded both native extensions through the normal public APIs. The model weights and database were on the Windows-mounted `/mnt/c` filesystem. The original cube-action path reloaded weights for every call, making its wall time sensitive to that filesystem. Native AWS instance latency and storage performance remain unmeasured. [Per-case and aggregate results](results.json)

## Numerical results

The Linux builds matched exactly on all result fields for 124 of 300 calls. The remaining 176 differed only in floating-point values. Both thread counts produced the same outputs within each build.

| API | Identical complete calls | Largest probability change | Largest equity change | Material changes |
| --- | ---: | ---: | ---: | ---: |
| Checker play | 26/100 | 1.20 × 10⁻⁷ | 7.75 × 10⁻⁷ | 0 |
| Cube action | 46/100 | 1.20 × 10⁻⁷ | 9.54 × 10⁻⁷ | 0 |
| Post-move | 52/100 | 5.97 × 10⁻⁸ | 3.88 × 10⁻⁷ | 0 |

There were 15,372 floating-point fields per pass. Of these, 2,780 differed. The absolute difference between two predictions bounds how much their absolute error can differ against any shared rollout reference. As a result, none of these changes can move the tested values materially farther from that reference.

The native probe compared 28,812,080 encoded features across 65,482 boards. Four extended contact features differed for some boards, at zero-based indices 111, 113, 233 and 235. These are the containment statistics for the two sides. The maximum change was 5.97 × 10⁻⁸. When each build used its own encoded inputs, Stage 11 sibling probabilities could change by up to 2.39 × 10⁻⁷. Feeding the same original input vectors into both kernels gave bit-identical results for all 7,110 outputs from 1,422 real siblings. A further 2,100 outputs also matched exactly across 35 combinations of network dimensions.

## Source and artifacts

The control's C++ and Python sources came directly from engine commit `139427d826dd67798226e7d23c003c90315306ba`. All 73 files in that export were checked against their Git blobs. The optimized arm used the integrated engine and Python sources. The same model weights and bearoff database were used in every arm. Each archive record includes the position, public arguments, complete returned analytics and elapsed wall time. The archive also includes per-field comparisons, per-build source fingerprints, compiler commands and the native probe recipe. [All 1,200 public results and build records](full_outputs.zip) · [Compiler, binary and fixture identities](provenance.json)
