# SPDX-License-Identifier: MPL-2.0
"""Build two independent CPU neural prototypes and check all saved sibling outputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import struct
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BUILD = ROOT / "build/truncated_pass2"
WORK = BUILD / "kernels"
CMAKE = ROOT / "build/logs/linux_cpu/final-gcc"
TOOL = ROOT / "build/logs/linux_cpu/toolchain/usr"


def variant(source, arm):
    if arm == "direct":
        return source
    if arm == "tile":
        start = source.index("    std::memcpy(hiddens, saved_base", source.index("NeuralNetwork::forward_with_deltas"))
        end = source.index("    // Apply sigmoid to updated hidden sums", start)
        kernel = """
#if defined(BGBOT_USE_AVX2)
    int h = 0;
    for (; h + 31 < nh; h += 32) {
        __m256 a0 = _mm256_loadu_ps(saved_base + h);
        __m256 a1 = _mm256_loadu_ps(saved_base + h + 8);
        __m256 a2 = _mm256_loadu_ps(saved_base + h + 16);
        __m256 a3 = _mm256_loadu_ps(saved_base + h + 24);
        for (int d = 0; d < n_deltas; ++d) {
            const __m256 amount = _mm256_set1_ps(delta_vals[d]);
            const float* w = hidden_weights_T_.data() + delta_indices[d] * nh + h;
            a0 = _mm256_fmadd_ps(amount, _mm256_loadu_ps(w), a0);
            a1 = _mm256_fmadd_ps(amount, _mm256_loadu_ps(w + 8), a1);
            a2 = _mm256_fmadd_ps(amount, _mm256_loadu_ps(w + 16), a2);
            a3 = _mm256_fmadd_ps(amount, _mm256_loadu_ps(w + 24), a3);
        }
        _mm256_storeu_ps(hiddens + h, a0);
        _mm256_storeu_ps(hiddens + h + 8, a1);
        _mm256_storeu_ps(hiddens + h + 16, a2);
        _mm256_storeu_ps(hiddens + h + 24, a3);
    }
    for (; h + 7 < nh; h += 8) {
        __m256 a = _mm256_loadu_ps(saved_base + h);
        for (int d = 0; d < n_deltas; ++d) {
            const float* w = hidden_weights_T_.data() + delta_indices[d] * nh + h;
            a = _mm256_fmadd_ps(_mm256_set1_ps(delta_vals[d]), _mm256_loadu_ps(w), a);
        }
        _mm256_storeu_ps(hiddens + h, a);
    }
    for (; h < nh; ++h) {
        float a = saved_base[h];
        for (int d = 0; d < n_deltas; ++d)
            a += delta_vals[d] * hidden_weights_T_[delta_indices[d] * nh + h];
        hiddens[h] = a;
    }
#else
"""
        return source[:start] + kernel + source[start:end] + "#endif\n" + source[end:]
    if arm == "fuse":
        start = source.index("// Apply sigmoid to updated hidden sums")
        end = source.index("// ======================== Transposed Weights", start)
        body = source[start:end]
        _, body = body.split("// Output layer with SIMD")
        body = body.replace("__m256 z = _mm256_loadu_ps(&hiddens[h]);",
                            "__m256 z = sigmoid256_ps(_mm256_loadu_ps(&hiddens[h]));")
        body = body.replace("const float z = hiddens[h];", "const float z = fast_sigmoid(hiddens[h]);")
        return source[:start] + "// Activate hidden units as the output layer consumes them.\n" + body + source[end:]
    if arm != "exp":
        raise ValueError(arm)
    # Degree-five Chebyshev interpolant of 2**r over [-0.5, 0.5].
    theta = [math.pi * (k + .5) / 6 for k in range(6)]
    c = [sum(math.cos(j*t) * 2**(.5*math.cos(t)) for t in theta) * (1 if j == 0 else 2)/6
         for j in range(6)]
    p = [c[0]-c[2]+c[4], 2*(c[1]-3*c[3]+5*c[5]), 4*(2*c[2]-8*c[4]),
         8*(4*c[3]-20*c[5]), 128*c[4], 512*c[5]]
    poly = "\n".join(f"    const __m256 p{i} = _mm256_set1_ps({v:.12g}f);" for i, v in enumerate(p))
    begin = source.index("    // Cephes exp polynomial constants")
    end = source.index("\n}", begin)
    return source[:begin] + """
    const __m256 y = _mm256_min_ps(_mm256_set1_ps(126.0f), _mm256_max_ps(
        _mm256_set1_ps(-126.0f), _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f))));
    const __m256 n = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_sub_ps(y, n);
    const __m256 r2 = _mm256_mul_ps(r, r);
""" + poly + """
    const __m256 a = _mm256_fmadd_ps(p1, r, p0);
    const __m256 b = _mm256_fmadd_ps(p3, r, p2);
    const __m256 c = _mm256_fmadd_ps(p5, r, p4);
    const __m256 mantissa = _mm256_fmadd_ps(_mm256_fmadd_ps(c, r2, b), r2, a);
    const __m256i exponent = _mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_mul_ps(_mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23)), mantissa);
""" + source[end:]


def build(arms):
    WORK.mkdir(exist_ok=True)
    commands = json.loads((CMAKE / "compile_commands.json").read_bytes())
    command = next(row["command"] for row in commands if row["file"].endswith("/neural_net.cpp"))
    flags = shlex.split(command)
    flags = flags[:flags.index("-o")]
    source = (BUILD / "eval-direct/cpp/src/neural_net.cpp").read_text()
    # Link only the production objects needed by this standalone executable.
    for part in ["board", "encoding", "strategy"]:
        path = BUILD / "eval-direct/cpp/src" / (part + ".cpp")
        args = [*flags, "-ffunction-sections", "-fdata-sections", "-c", str(path), "-o", str(WORK / (part+".o"))]
        if not (WORK / (part+".o")).exists():
            subprocess.run(args, check=True)
    for arm in arms:
        file = WORK / (arm + ".cpp")
        file.write_text(variant(source, arm))
        subprocess.run([*flags, "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
                        str(HERE / "kernel.cpp"), str(file), *sorted(WORK.glob("*.o")),
                        "-o", str(WORK / arm), "-pthread"], check=True)


def run(arms):
    stats = {a: [] for a in arms}
    for rep in range(3):
        for arm in arms[rep:] + arms[:rep]:
            result = subprocess.check_output([
                str(WORK / arm), str(ROOT / "build/nn_trials/corpus.txt"),
                str(ROOT / "models/sl_s9_att_race.weights.best"), str(WORK / (arm+".bin"))], text=True)
            for row in result.splitlines():
                mode, _, elapsed, _ = row.split()
                stats[arm].append({"direct": bool(int(mode)), "nanoseconds": float(elapsed)})
    binary = (WORK / "direct.bin").read_bytes()
    unpack = lambda data: struct.unpack("<" + "f" * (len(data)//4), data)
    reference = unpack(binary)
    summary = {}
    for arm, measurements in stats.items():
        raw = (WORK / (arm + ".bin")).read_bytes()
        values = unpack(raw)
        half = len(values) // 2
        summary[arm] = dict(median_ns={str(b): statistics.median(m["nanoseconds"] for m in measurements
                                                                  if m["direct"] == b) for b in [False, True]},
                            floats=len(values), max_vs_control=max(abs(a-b) for a,b in zip(reference, values)),
                            max_between_delta_paths=max(abs(a-b) for a,b in zip(values[:half], values[half:])),
                            sha256=hashlib.sha256(raw).hexdigest(), measurements=measurements)
    (HERE / "kernels.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({a:{k:v for k,v in s.items() if k != "measurements"} for a,s in summary.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "run"])
    parser.add_argument("arms", nargs="+", choices=["direct", "fuse", "exp", "tile"])
    args = parser.parse_args()
    os.sched_setaffinity(0, [4])
    os.environ.update(PATH=str(TOOL / "bin") + ":/usr/bin:/bin",
                      LD_LIBRARY_PATH=str(TOOL / "lib/x86_64-linux-gnu"),
                      TMPDIR=str(ROOT / "build/logs/linux_cpu/tmp"))
    (build if args.action == "build" else run)(args.arms)
