# Low-ply calls on two CPUs and two filesystems

The original native cube entry point constructed its evaluator from file paths on every call, even when the caller reused a Python analyzer. This makes its wall time sensitive to the cost of opening and reading model files. The main Linux timing panel ran with weights under WSL's Windows-mounted checkout and 16 available logical CPUs.

This experiment compares that source of variation at 1P and 2P using only two logical CPUs. Each arm creates one cubeful Stage 11 analyzer, then runs all 100 checker, 100 cube and 100 post-move calls. Identical model files are available on both the Windows-mounted checkout and a Linux-native temporary filesystem. Every model file is read and hashed before timing, so the comparison includes the cost of reopening warm files. Both engines use the same database and position inputs. The process begins each arm with a fresh analyzer, gives the native evaluator a two-thread maximum, and excludes analyzer construction and JSON serialization from its measured interval.

The comparison uses the unmodified `139427d` and the verified candidate binary from this study. Each engine and storage configuration runs in its own process. The 2P arm order reverses the 1P order. All 1,200 pairs are checked for exact equality of the public inputs and full analytics between storage locations. [Measured totals and build identities](summary.json) · [All 2,400 calls](full_outputs.zip)

| Weight storage | Preset | Original → selected, 300 calls | Overall speedup | Checker | Cube | Post-move |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Disk-backed Linux (`/var/tmp`) | 1P | 2.074857 → 0.012075 s | 171.83× | 31.92× | 1,568.33× | 1.80× |
| Disk-backed Linux (`/var/tmp`) | 2P | 3.802976 → 0.449765 s | 8.46× | 4.72× | 38.47× | 7.77× |
| Windows mount (`v9fs`) | 1P | 8.144691 → 0.011556 s | 704.83× | 33.93× | 7,761.04× | 1.76× |
| Windows mount (`v9fs`) | 2P | 9.851578 → 0.433081 s | 22.75× | 4.88× | 171.65× | 8.41× |

On disk-backed Linux, average engine time for one 1P cube call fell from 17.398 ms to 0.011 ms. For a 1P post-move call it fell from 9.0 µs to 5.0 µs. The mix of requests and the filesystem both affect the overall speedup.

A separate run placed the same weights on Linux `tmpfs`. Its overall speedups were 175.73× at 1P and 8.61× at 2P. All 2,400 inputs and full analytics agreed across the two runs. Comparing engine revisions, every checker ranking and cube recommendation agreed. Zero of the 4,724 rollout-referenced fields at each level moved materially closer or farther. [Memory-filesystem timings](tmpfs/summary.json) · [Full outputs from the memory-filesystem run](tmpfs/full_outputs.zip)

Reproduce from the engine root under WSL with a Windows-mounted checkout, a disk-backed Linux `/var/tmp` and the source trees and extensions referenced by the archived qualification manifests. The output directory must be new. The recorded run reported `ext2/ext3` for `/var/tmp`. To repeat the memory-filesystem measurement, use a `tmpfs` path for `--native-parent` and a separate output directory.

```sh
study=data/multi_ply_benchmark/experiments/truncated_pass2
python3 -B "$study/inline/measure.py" measure --native-parent /var/tmp --output logs/inline-filesystem-check
```
