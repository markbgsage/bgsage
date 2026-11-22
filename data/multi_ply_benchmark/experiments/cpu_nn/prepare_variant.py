# SPDX-License-Identifier: MPL-2.0
"""Prepare an isolated engine snapshot with one of the measured NN kernels."""
import argparse
from pathlib import Path
import shutil
import subprocess

here=Path(__file__).resolve().parent
root=here.parents[3]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("variant",choices=("base","scan","scan-compare","block","both"))
p.add_argument("destination",type=Path)
p.add_argument("--source",type=Path,default=root,help="Engine sources for all other components, including the encoder")
a=p.parse_args()
destination=a.destination.resolve()
if not destination.is_relative_to(root/"build") or destination.exists():
    raise ValueError("Destination must be a new directory below bgsage/build")
base=subprocess.check_output(["git","-C",str(root),"show",
    "139427d826dd67798226e7d23c003c90315306ba:cpp/src/neural_net.cpp"]).decode("utf-8")
start=base.index("std::array<float, NN_OUTPUTS> NeuralNetwork::forward_from_base(")
lo=base.index("    for (int i = 0; i < ni; ++i) {", start)
hi=base.index("    // Apply deltas using transposed",lo)
sparse='''    int i = 0;
#if defined(BGBOT_USE_AVX2)
    for (; i + 7 < ni && n_deltas < 64; i += 8) {
        const __m256 current = _mm256_loadu_ps(inputs + i);
        const __m256 previous = _mm256_loadu_ps(saved_inputs + i);
        const __m256 delta = _mm256_sub_ps(current, previous);
        unsigned mask = _mm256_movemask_ps(_mm256_cmp_ps(delta, _mm256_setzero_ps(), _CMP_NEQ_UQ));
        while (mask && n_deltas < 64) {
#if defined(_MSC_VER)
            unsigned long lane;
            _BitScanForward(&lane, mask);
#else
            const int lane = __builtin_ctz(mask);
#endif
            delta_indices[n_deltas] = i + lane;
            delta_vals[n_deltas++] = inputs[i + lane] - saved_inputs[i + lane];
            mask &= mask - 1;
        }
    }
#endif
    for (; i < ni && n_deltas < 64; ++i) {
        const float d = inputs[i] - saved_inputs[i];
        if (d != 0.0f) {
            delta_indices[n_deltas] = i;
            delta_vals[n_deltas++] = d;
        }
    }

'''
def fuse(s):
    old="    for (int d = 0; d < n_deltas; ++d) {"
    at=s.index(old,start)
    return s[:at]+'''    int d = 0;
#if defined(BGBOT_USE_AVX2)
    // Retain one hidden tile in registers across four successive deltas.
    // The per-hidden-unit FMA sequence stays in original input order.
    for (; d + 3 < n_deltas; d += 4) {
        const float* w0 = hidden_weights_T_.data() + delta_indices[d] * nh;
        const float* w1 = hidden_weights_T_.data() + delta_indices[d + 1] * nh;
        const float* w2 = hidden_weights_T_.data() + delta_indices[d + 2] * nh;
        const float* w3 = hidden_weights_T_.data() + delta_indices[d + 3] * nh;
        const __m256 v0 = _mm256_set1_ps(delta_vals[d]);
        const __m256 v1 = _mm256_set1_ps(delta_vals[d + 1]);
        const __m256 v2 = _mm256_set1_ps(delta_vals[d + 2]);
        const __m256 v3 = _mm256_set1_ps(delta_vals[d + 3]);
        int h = 0;
        for (; h + 7 < nh; h += 8) {
            __m256 z = _mm256_loadu_ps(hiddens + h);
            z = _mm256_fmadd_ps(v0, _mm256_loadu_ps(w0 + h), z);
            z = _mm256_fmadd_ps(v1, _mm256_loadu_ps(w1 + h), z);
            z = _mm256_fmadd_ps(v2, _mm256_loadu_ps(w2 + h), z);
            z = _mm256_fmadd_ps(v3, _mm256_loadu_ps(w3 + h), z);
            _mm256_storeu_ps(hiddens + h, z);
        }
        for (; h < nh; ++h) {
            hiddens[h] += delta_vals[d] * w0[h];
            hiddens[h] += delta_vals[d + 1] * w1[h];
            hiddens[h] += delta_vals[d + 2] * w2[h];
            hiddens[h] += delta_vals[d + 3] * w3[h];
        }
    }
#endif
    for (; d < n_deltas; ++d) {'''+s[at+len(old):]
scan=base[:lo]+sparse+base[hi:]

variants=dict(base=base,scan=scan,block=fuse(base),both=fuse(scan))
variants["scan-compare"]=scan.replace(
    "        const __m256 delta = _mm256_sub_ps(current, previous);\n", ""
).replace("_mm256_cmp_ps(delta, _mm256_setzero_ps(), _CMP_NEQ_UQ)", "_mm256_cmp_ps(current, previous, _CMP_NEQ_UQ)")
destination.mkdir(parents=True)
for name in ("src","include","pybind"):
    shutil.copytree(a.source/"cpp"/name,destination/"cpp"/name)
shutil.copy2(a.source/"cpp/CMakeLists_cpu.txt",destination/"cpp/CMakeLists_cpu.txt")
(destination/"cpp/src/neural_net.cpp").write_text(variants[a.variant],encoding="utf-8")
(destination/"CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.18)\nproject(bgsage LANGUAGES CXX)\ninclude("${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists_cpu.txt")\n')
