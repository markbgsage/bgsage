// SPDX-License-Identifier: MPL-2.0
// The harness generates the include from the tested engine's neural_net.cpp.
#include "engine_sigmoid.inc"

#include <cstdio>

__attribute__((noinline)) static void narrow(const float* x, float* result) {
    for (int i = 0; i < 16; i += 8) {
        _mm256_storeu_ps(result + i, bgbot::sigmoid256_ps(_mm256_loadu_ps(x + i)));
    }
}

__attribute__((noinline, target("avx512f,fma"))) static void wide(const float* x, float* result) {
    _mm512_storeu_ps(result, bgbot::sigmoid512_ps(_mm512_loadu_ps(x)));
}

int main() {
    if (!__builtin_cpu_supports("avx512f")) {
        std::fprintf(stderr, "This native comparison requires AVX-512F.\n");
        return 1;
    }
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> broad(-120.0f, 120.0f), central(-6.0f, 6.0f);
    int unequal = 0;
    float worst = 0;
    for (int j = 0; j < 65536; ++j) {
        alignas(64) float x[16], baseline[16], candidate[16];
        for (int i = 0; i < 16; ++i) x[i] = i % 2 ? broad(rng) : central(rng);
        narrow(x, baseline);
        wide(x, candidate);
        for (int i = 0; i < 16; ++i) {
            if (std::memcmp(baseline + i, candidate + i, sizeof(float)) != 0) ++unequal;
            worst = std::max(worst, std::abs(baseline[i] - candidate[i]));
        }
    }
    std::printf("{\"inputs\":1048576,\"unequal\":%d,\"max_absolute_difference\":%.10g}\n", unequal, worst);
}
