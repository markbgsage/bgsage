# Thread width for 2P pure races

On the 16-logical-CPU Linux host, evaluating the 21 root rolls serially cut the pure-race cube-action subtotal by 59% and post-move by 68% across four complete benchmark repetitions. On Windows the 16-thread release beat serial on both APIs. The selected patch therefore serializes Linux pure-race 2P roots only. Public checker analysis can still distribute endpoints across workers. [Selected source patch](serial-linux.patch)

## Pure-race screen

Each entry is the median wall time in milliseconds for all 20 positions of one API, across five repetitions with `--race-only`. Each analyzer requested 16 workers. The candidate widths cap parallelism at the two 2P roll-dispatch entry points. The checkpoint column uses the preceding release binary. Every width reproduced all 300 complete checkpoint responses per platform, including all legal checker moves. [Linux full outputs](sweep_linux/outputs.zip) · [Linux timing breakdown](sweep_linux/summary.json) · [Windows full outputs](../race-width-windows/outputs.zip) · [Windows timing breakdown](../race-width-windows/summary.json)

| Platform and API | Checkpoint | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linux checker | 11.114 | 11.194 | 10.438 | 10.684 | 10.396 | 9.804 |
| Linux cube | 6.428 | 2.227 | 3.584 | 3.577 | 4.010 | 6.320 |
| Linux post-move | 5.043 | 1.345 | 2.137 | 2.113 | 2.725 | 5.137 |
| Windows checker | 3.160 | 3.188 | 3.384 | 3.061 | 3.422 | 3.289 |
| Windows cube | 1.275 | 2.146 | 1.636 | 1.172 | 1.176 | 1.225 |
| Windows post-move | 0.985 | 1.346 | 0.993 | 0.730 | 0.776 | 1.040 |

## All benchmark positions

The Linux A/B covered all 100 positions for each of the three APIs, repeated four times. Times are the sums in seconds across those repetitions. Both candidate widths reproduced all 1,200 complete checkpoint responses, so their probability and equity errors against the fixed rollout truth were unchanged. The other 80 positions per API took 0.17086 versus 0.17098 seconds on cube and 0.09901 versus 0.09907 seconds on post-move, checkpoint versus serial. [Full outputs](full_linux/outputs.zip) · [Per-repeat timing and exact counts](full_linux/summary.json)

| Linux 2P | Checkpoint | Width 1 | Width 4 |
| --- | ---: | ---: | ---: |
| Checker, all | 0.53636 | 0.53703 | 0.52918 |
| Cube, all | 0.19748 | 0.18191 | 0.18400 |
| Post-move, all | 0.11966 | 0.10574 | 0.10636 |
| Cube, pure-race subset | 0.02662 | 0.01093 | 0.01314 |
| Post-move, pure-race subset | 0.02065 | 0.00667 | 0.00780 |

## Reproduction

The runner starts a fresh analyzer in each arm and repetition, warms it with one contact checker position, then shuffles positions with a recorded seed. It rotates arm order in blocks of ten positions. It times the public call and serializes the full analytics afterward. The archives include the original inputs. [Runner](screen.py) · [Linux binary hashes and arguments](sweep_linux/metadata.json) · [Windows binary hashes and arguments](../race-width-windows/metadata.json)

The following commands use the qualified Linux release objects and toolchain from the CPU studies. Run them from the engine root. The Linux width-four arm used a fixed cap. The experimental build reads `BGSAGE_RACE_TEST_THREADS` for the other widths. The final release contains the fixed Linux rule. [Experimental patch](experiment.patch) · [Fixed width-four patch](four.patch)

```bash
study=data/multi_ply_benchmark/experiments/search_pass3/race2
work=build/search_pass3/evaluator
base=$work/lanes_final
for spec in 'raceenv experiment.patch' 'race4 four.patch' 'race1linux serial-linux.patch'; do
    read -r arm patchfile <<< "$spec"
    src="$work/source-replay-$arm"
    mkdir "$src" || exit 1
    cp -R "$base/cpp" "$base/python" "$base/CMakeLists.txt" "$src/"
    patch -d "$src" -p1 < "$study/$patchfile" || exit 1
    python3 -B "$study/../evaluator/build_linux.py" "$arm-replay" "$src" || exit 1
done

arms=(--arm current "$base" "$base/python" 16)
for width in 1 2 4 8 16; do
    source_dir="$work/raceenv-replay"
    if [[ "$width" == 4 ]]; then source_dir="$work/race4-replay"; fi
    arms+=(--arm "w$width" "$source_dir" "$source_dir/python" "$width")
done
python3 -B "$study/screen.py" "${arms[@]}" --race-only --repeats 5 --out "$work/race-width-replay"
python3 -B "$study/screen.py" --arm current "$base" "$base/python" 16 \
    --arm w1 "$work/raceenv-replay" "$work/raceenv-replay/python" 1 \
    --arm w4 "$work/race4-replay" "$work/race4-replay/python" 4 \
    --repeats 4 --out "$work/race-full-replay"
```
