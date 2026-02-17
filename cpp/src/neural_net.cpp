#include "bgbot/neural_net.h"
#include "bgbot/board.h"
#include "bgbot/encoding.h"
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// Platform-specific SIMD headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #define BGBOT_USE_AVX2 1
#elif defined(__aarch64__) || defined(__arm64__)
  #include <arm_neon.h>
  #define BGBOT_USE_NEON 1
#endif

namespace bgbot {

// ======================== Utility ========================

// Fast sigmoid via lookup table with linear interpolation.
// Covers [-8, 8] with 4096 entries (512 per unit). Outside this range,
// sigmoid saturates to 0 or 1 (sigmoid(±8) ≈ 0.9997/0.0003).
// Max error with linear interpolation: < 1e-5 (negligible for play quality).
// Speed: ~3-5 cycles vs ~75 cycles for std::exp.
static constexpr int SIGMOID_TABLE_SIZE = 4096;
static constexpr float SIGMOID_RANGE = 8.0f;
static constexpr float SIGMOID_SCALE = SIGMOID_TABLE_SIZE / (2.0f * SIGMOID_RANGE);
static constexpr float SIGMOID_OFFSET = SIGMOID_RANGE;  // maps [-8,8] to [0, TABLE_SIZE]

struct SigmoidTable {
    float table[SIGMOID_TABLE_SIZE + 1];  // +1 for interpolation at upper bound

    SigmoidTable() {
        for (int i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
            float x = (static_cast<float>(i) / SIGMOID_SCALE) - SIGMOID_OFFSET;
            table[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }
};

static const SigmoidTable sigmoid_lut;

static inline float fast_sigmoid(float x) {
    if (x >= SIGMOID_RANGE) return 1.0f;
    if (x <= -SIGMOID_RANGE) return 0.0f;
    float idx_f = (x + SIGMOID_OFFSET) * SIGMOID_SCALE;
    int idx = static_cast<int>(idx_f);
    float frac = idx_f - static_cast<float>(idx);
    return sigmoid_lut.table[idx] + frac * (sigmoid_lut.table[idx + 1] - sigmoid_lut.table[idx]);
}

// ======================== AVX2 Vectorized Sigmoid ========================
// Based on Cephes polynomial approximation of exp().
// Computes sigmoid(x) = 1/(1+exp(-x)) for 8 floats simultaneously.
// Max error vs std::exp: ~1-2 ULP (better than the LUT approach).
// Used for the hidden layer sigmoid in the column-major forward pass.
#if defined(BGBOT_USE_AVX2)
static inline __m256 exp256_ps(__m256 x) {
    // Cephes exp polynomial constants
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 half   = _mm256_set1_ps(0.5f);
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2e  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1     = _mm256_set1_ps(0.693359375f);
    const __m256 c2     = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0     = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1     = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2     = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3     = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4     = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5     = _mm256_set1_ps(5.0000001201E-1f);

    // Clamp input
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    // exp(x) = 2^n * exp(f) where f = x - n*ln(2) and n = floor(x/ln(2) + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, log2e, half);
    fx = _mm256_floor_ps(fx);

    // Reduce: x = x - fx * ln(2) (two-step for precision)
    __m256 tmp = _mm256_mul_ps(fx, c1);
    __m256 z   = _mm256_mul_ps(fx, c2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x, x);

    // Horner polynomial evaluation: exp(f) ≈ 1 + f + f²*(p5 + f*(p4 + f*(p3 + f*(p2 + f*(p1 + f*p0)))))
    __m256 y = _mm256_fmadd_ps(p0, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    // Scale by 2^n: convert n to integer, shift into exponent bits
    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);

    return _mm256_mul_ps(y, pow2n);
}

static inline __m256 sigmoid256_ps(__m256 x) {
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    __m256 neg_x = _mm256_sub_ps(zero, x);
    __m256 exp_neg_x = exp256_ps(neg_x);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
}
#endif

// SIMD dot product: compute sum of a[i]*b[i] for i=0..n-1
// Both a and b must be at least n floats. No alignment required.
#ifdef _MSC_VER
__forceinline
#else
__attribute__((always_inline)) inline
#endif
float simd_dot(const float* __restrict a, const float* __restrict b, int n) {
#if defined(BGBOT_USE_AVX2)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    // Process 16 floats per iteration (2 x 8-wide FMA)
    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    // Process remaining 8-wide chunks
    for (; i + 7 < n; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }

    // Horizontal reduction: sum all 8 lanes
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float dot = _mm_cvtss_f32(s);

    // Handle remaining elements (< 8)
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#elif defined(BGBOT_USE_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 8 floats per iteration (2 x 4-wide FMA)
    for (; i + 7 < n; i += 8) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i),     vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    sum0 = vaddq_f32(sum0, sum1);
    // Process remaining 4-wide chunk
    for (; i + 3 < n; i += 4) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    // Horizontal reduction: sum all 4 lanes
    float dot = vaddvq_f32(sum0);

    // Handle remaining elements (< 4)
    for (; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#else
    // Scalar fallback
    float dot = 0.0f;
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
#endif
}

// Keep old name as alias for compatibility within this file
#define avx2_dot simd_dot

static double terminal_equity(GameResult result) {
    switch (result) {
        case GameResult::WIN_SINGLE:      return  1.0;
        case GameResult::WIN_GAMMON:       return  2.0;
        case GameResult::WIN_BACKGAMMON:   return  3.0;
        case GameResult::LOSS_SINGLE:      return -1.0;
        case GameResult::LOSS_GAMMON:      return -2.0;
        case GameResult::LOSS_BACKGAMMON:  return -3.0;
        default: return 0.0;
    }
}

// ======================== NeuralNetwork ========================

NeuralNetwork::NeuralNetwork(int n_hidden, int n_inputs, float eps, uint32_t seed)
    : n_hidden_(n_hidden), n_inputs_(n_inputs)
{
    hidden_weights_.resize(n_hidden_ * (n_inputs_ + 1));
    output_weights_.resize(NN_OUTPUTS * (n_hidden_ + 1));

    cached_inputs_.resize(n_inputs_);
    cached_hiddens_.resize(n_hidden_ + 1);
    cached_hprods_.resize(n_hidden_);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, eps);

    for (auto& w : hidden_weights_) w = dist(rng);
    for (auto& w : output_weights_) w = dist(rng);
}

// ======================== Forward pass (const) ========================

std::array<float, NN_OUTPUTS> NeuralNetwork::forward(const float* inputs) const {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;
    const int hid_stride = nh + 1;

    ensure_transposed_weights();

    // Stack buffer for hidden layer (512 covers all current architectures: max 400)
    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    for (int h = 0; h < nh; ++h) {
        hiddens[h] = hidden_weights_[h * inp_stride + ni];
    }

    for (int i = 0; i < ni; ++i) {
        const float vi = inputs[i];
        if (vi == 0.0f) continue;
        const float* col = &hidden_weights_T_[i * nh];
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        __m256 vvi = _mm256_set1_ps(vi);
        for (; h + 15 < nh; h += 16) {
            __m256 acc0 = _mm256_loadu_ps(&hiddens[h]);
            __m256 wt0  = _mm256_loadu_ps(&col[h]);
            acc0 = _mm256_fmadd_ps(vvi, wt0, acc0);
            _mm256_storeu_ps(&hiddens[h], acc0);
            __m256 acc1 = _mm256_loadu_ps(&hiddens[h + 8]);
            __m256 wt1  = _mm256_loadu_ps(&col[h + 8]);
            acc1 = _mm256_fmadd_ps(vvi, wt1, acc1);
            _mm256_storeu_ps(&hiddens[h + 8], acc1);
        }
        for (; h + 7 < nh; h += 8) {
            __m256 acc0 = _mm256_loadu_ps(&hiddens[h]);
            __m256 wt0  = _mm256_loadu_ps(&col[h]);
            acc0 = _mm256_fmadd_ps(vvi, wt0, acc0);
            _mm256_storeu_ps(&hiddens[h], acc0);
        }
#elif defined(BGBOT_USE_NEON)
        float32x4_t vvi = vdupq_n_f32(vi);
        for (; h + 7 < nh; h += 8) {
            float32x4_t acc0 = vld1q_f32(&hiddens[h]);
            float32x4_t wt0  = vld1q_f32(&col[h]);
            acc0 = vfmaq_f32(acc0, vvi, wt0);
            vst1q_f32(&hiddens[h], acc0);
            float32x4_t acc1 = vld1q_f32(&hiddens[h + 4]);
            float32x4_t wt1  = vld1q_f32(&col[h + 4]);
            acc1 = vfmaq_f32(acc1, vvi, wt1);
            vst1q_f32(&hiddens[h + 4], acc1);
        }
#endif
        for (; h < nh; ++h) {
            hiddens[h] += vi * col[h];
        }
    }

    for (int h = 0; h < nh; ++h) {
        hiddens[h] = fast_sigmoid(hiddens[h]);
    }

    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }

    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);

    return outputs;
}

// ======================== Batch forward (const) ========================

void NeuralNetwork::forward_batch(
    const float* inputs_array,
    std::array<float, NN_OUTPUTS>* outputs_array,
    int count) const
{
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;
    const int hid_stride = nh + 1;

    ensure_transposed_weights();

    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    for (int b = 0; b < count; ++b) {
        const float* inputs = inputs_array + b * ni;

        for (int h = 0; h < nh; ++h) {
            hiddens[h] = hidden_weights_[h * inp_stride + ni];
        }
        for (int i = 0; i < ni; ++i) {
            const float vi = inputs[i];
            if (vi == 0.0f) continue;
            const float* col = &hidden_weights_T_[i * nh];
            int h = 0;
#if defined(BGBOT_USE_AVX2)
            __m256 vvi = _mm256_set1_ps(vi);
            for (; h + 15 < nh; h += 16) {
                __m256 acc0 = _mm256_loadu_ps(&hiddens[h]);
                __m256 wt0  = _mm256_loadu_ps(&col[h]);
                acc0 = _mm256_fmadd_ps(vvi, wt0, acc0);
                _mm256_storeu_ps(&hiddens[h], acc0);
                __m256 acc1 = _mm256_loadu_ps(&hiddens[h + 8]);
                __m256 wt1  = _mm256_loadu_ps(&col[h + 8]);
                acc1 = _mm256_fmadd_ps(vvi, wt1, acc1);
                _mm256_storeu_ps(&hiddens[h + 8], acc1);
            }
            for (; h + 7 < nh; h += 8) {
                __m256 acc0 = _mm256_loadu_ps(&hiddens[h]);
                __m256 wt0  = _mm256_loadu_ps(&col[h]);
                acc0 = _mm256_fmadd_ps(vvi, wt0, acc0);
                _mm256_storeu_ps(&hiddens[h], acc0);
            }
#elif defined(BGBOT_USE_NEON)
            float32x4_t vvi = vdupq_n_f32(vi);
            for (; h + 7 < nh; h += 8) {
                float32x4_t acc0 = vld1q_f32(&hiddens[h]);
                float32x4_t wt0  = vld1q_f32(&col[h]);
                acc0 = vfmaq_f32(acc0, vvi, wt0);
                vst1q_f32(&hiddens[h], acc0);
                float32x4_t acc1 = vld1q_f32(&hiddens[h + 4]);
                float32x4_t wt1  = vld1q_f32(&col[h + 4]);
                acc1 = vfmaq_f32(acc1, vvi, wt1);
                vst1q_f32(&hiddens[h + 4], acc1);
            }
#endif
            for (; h < nh; ++h) {
                hiddens[h] += vi * col[h];
            }
        }
        for (int h = 0; h < nh; ++h) {
            hiddens[h] = fast_sigmoid(hiddens[h]);
        }

        auto& outputs = outputs_array[b];
        const float* w0 = &output_weights_[0 * hid_stride];
        const float* w1 = &output_weights_[1 * hid_stride];
        const float* w2 = &output_weights_[2 * hid_stride];
        const float* w3 = &output_weights_[3 * hid_stride];
        const float* w4 = &output_weights_[4 * hid_stride];

        float o0 = w0[nh];
        float o1 = w1[nh];
        float o2 = w2[nh];
        float o3 = w3[nh];
        float o4 = w4[nh];
        for (int h = 0; h < nh; ++h) {
            const float z = hiddens[h];
            o0 += w0[h] * z;
            o1 += w1[h] * z;
            o2 += w2[h] * z;
            o3 += w3[h] * z;
            o4 += w4[h] * z;
        }
        outputs[0] = fast_sigmoid(o0);
        outputs[1] = fast_sigmoid(o1);
        outputs[2] = fast_sigmoid(o2);
        outputs[3] = fast_sigmoid(o3);
        outputs[4] = fast_sigmoid(o4);
    }
}

// ======================== Incremental (delta) evaluation ========================

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_save_base(
    const float* inputs, float* saved_base, float* saved_inputs) const
{
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;
    const int hid_stride = nh + 1;

    ensure_transposed_weights();

    // Save input vector for later delta computation
    std::memcpy(saved_inputs, inputs, ni * sizeof(float));

    // Compute hidden layer pre-sigmoid sums and save them
    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    for (int h = 0; h < nh; ++h) {
        float sum = hidden_weights_[h * inp_stride + ni];
        saved_base[h] = sum;          // save pre-sigmoid sum
        hiddens[h] = sum;
    }
    for (int i = 0; i < ni; ++i) {
        const float vi = inputs[i];
        if (vi == 0.0f) continue;
        const float* col = &hidden_weights_T_[i * nh];
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        __m256 vvi = _mm256_set1_ps(vi);
        for (; h + 15 < nh; h += 16) {
            __m256 base0 = _mm256_loadu_ps(&hiddens[h]);
            __m256 wb0 = _mm256_loadu_ps(&col[h]);
            __m256 base1 = _mm256_loadu_ps(&hiddens[h + 8]);
            __m256 wb1 = _mm256_loadu_ps(&col[h + 8]);
            __m256 save0 = _mm256_loadu_ps(&saved_base[h]);
            __m256 save1 = _mm256_loadu_ps(&saved_base[h + 8]);
            __m256 prod0 = _mm256_mul_ps(vvi, wb0);
            __m256 prod1 = _mm256_mul_ps(vvi, wb1);
            base0 = _mm256_add_ps(base0, prod0);
            base1 = _mm256_add_ps(base1, prod1);
            save0 = _mm256_add_ps(save0, prod0);
            save1 = _mm256_add_ps(save1, prod1);
            _mm256_storeu_ps(&hiddens[h], base0);
            _mm256_storeu_ps(&hiddens[h + 8], base1);
            _mm256_storeu_ps(&saved_base[h], save0);
            _mm256_storeu_ps(&saved_base[h + 8], save1);
        }
        for (; h + 7 < nh; h += 8) {
            __m256 base = _mm256_loadu_ps(&hiddens[h]);
            __m256 wb   = _mm256_loadu_ps(&col[h]);
            __m256 save = _mm256_loadu_ps(&saved_base[h]);
            __m256 prod = _mm256_mul_ps(vvi, wb);
            base = _mm256_add_ps(base, prod);
            save = _mm256_add_ps(save, prod);
            _mm256_storeu_ps(&hiddens[h], base);
            _mm256_storeu_ps(&saved_base[h], save);
        }
#elif defined(BGBOT_USE_NEON)
        float32x4_t vvi = vdupq_n_f32(vi);
        for (; h + 7 < nh; h += 8) {
            float32x4_t base0 = vld1q_f32(&hiddens[h]);
            float32x4_t wb0  = vld1q_f32(&col[h]);
            float32x4_t wb1  = vld1q_f32(&col[h + 4]);
            float32x4_t base1 = vld1q_f32(&hiddens[h + 4]);
            float32x4_t save0 = vld1q_f32(&saved_base[h]);
            float32x4_t save1 = vld1q_f32(&saved_base[h + 4]);
            base0 = vfmaq_f32(base0, vvi, wb0);
            base1 = vfmaq_f32(base1, vvi, wb1);
            save0 = vfmaq_f32(save0, vvi, wb0);
            save1 = vfmaq_f32(save1, vvi, wb1);
            vst1q_f32(&hiddens[h], base0);
            vst1q_f32(&hiddens[h + 4], base1);
            vst1q_f32(&saved_base[h], save0);
            vst1q_f32(&saved_base[h + 4], save1);
        }
#endif
        for (; h < nh; ++h) {
            const float v = vi * col[h];
            hiddens[h] += v;
            saved_base[h] += v;
        }
    }
    for (int h = 0; h < nh; ++h) {
        hiddens[h] = fast_sigmoid(hiddens[h]);
    }

    // Output layer (same as normal forward)
    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }
    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);
    return outputs;
}

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_from_base(
    const float* inputs, const float* saved_base, const float* saved_inputs) const
{
    ensure_transposed_weights();

    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int hid_stride = nh + 1;

    // Copy saved pre-sigmoid hidden sums, then apply input deltas
    float hiddens_stack[512];
    std::vector<float> hiddens_heap;
    float* hiddens;
    if (nh <= 512) {
        hiddens = hiddens_stack;
    } else {
        hiddens_heap.resize(nh);
        hiddens = hiddens_heap.data();
    }

    std::memcpy(hiddens, saved_base, nh * sizeof(float));

    // Collect changed inputs into a compact list.
    // In backgammon, a move typically changes 4-12 of 196/244 inputs.
    int delta_indices[64];
    float delta_vals[64];
    int n_deltas = 0;

    for (int i = 0; i < ni; ++i) {
        float d = inputs[i] - saved_inputs[i];
        if (d != 0.0f && n_deltas < 64) {
            delta_indices[n_deltas] = i;
            delta_vals[n_deltas] = d;
            n_deltas++;
        }
    }

    // Apply deltas using transposed weight matrix for contiguous column access.
    // hidden_weights_T_[col * nh + h] gives the weight for hidden node h, input col.
    // This avoids the strided column extraction step entirely.
    for (int d = 0; d < n_deltas; ++d) {
        const int col = delta_indices[d];
        const float dv = delta_vals[d];
        const float* col_weights = &hidden_weights_T_[col * nh];

        // Apply: hiddens[h] += dv * col_weights[h] using SIMD FMA
        int h = 0;
#if defined(BGBOT_USE_AVX2)
        __m256 vdv = _mm256_set1_ps(dv);
        for (; h + 15 < nh; h += 16) {
            __m256 vh0 = _mm256_loadu_ps(&hiddens[h]);
            __m256 vc0 = _mm256_loadu_ps(&col_weights[h]);
            vh0 = _mm256_fmadd_ps(vdv, vc0, vh0);
            _mm256_storeu_ps(&hiddens[h], vh0);
            __m256 vh1 = _mm256_loadu_ps(&hiddens[h + 8]);
            __m256 vc1 = _mm256_loadu_ps(&col_weights[h + 8]);
            vh1 = _mm256_fmadd_ps(vdv, vc1, vh1);
            _mm256_storeu_ps(&hiddens[h + 8], vh1);
        }
        for (; h + 7 < nh; h += 8) {
            __m256 vh = _mm256_loadu_ps(&hiddens[h]);
            __m256 vc = _mm256_loadu_ps(&col_weights[h]);
            vh = _mm256_fmadd_ps(vdv, vc, vh);
            _mm256_storeu_ps(&hiddens[h], vh);
        }
#elif defined(BGBOT_USE_NEON)
        float32x4_t vdv = vdupq_n_f32(dv);
        for (; h + 7 < nh; h += 8) {
            float32x4_t vh0 = vld1q_f32(&hiddens[h]);
            float32x4_t vc0 = vld1q_f32(&col_weights[h]);
            vh0 = vfmaq_f32(vh0, vdv, vc0);
            vst1q_f32(&hiddens[h], vh0);
            float32x4_t vh1 = vld1q_f32(&hiddens[h + 4]);
            float32x4_t vc1 = vld1q_f32(&col_weights[h + 4]);
            vh1 = vfmaq_f32(vh1, vdv, vc1);
            vst1q_f32(&hiddens[h + 4], vh1);
        }
        for (; h + 3 < nh; h += 4) {
            float32x4_t vh = vld1q_f32(&hiddens[h]);
            float32x4_t vc = vld1q_f32(&col_weights[h]);
            vh = vfmaq_f32(vh, vdv, vc);
            vst1q_f32(&hiddens[h], vh);
        }
#endif
        for (; h < nh; ++h) {
            hiddens[h] += dv * col_weights[h];
        }
    }

    // Apply sigmoid to updated hidden sums
    {
        int h = 0;
        for (; h < nh; ++h) {
            hiddens[h] = fast_sigmoid(hiddens[h]);
        }
    }

    // Output layer (same as normal forward)
    std::array<float, NN_OUTPUTS> outputs;
    const float* w0 = &output_weights_[0 * hid_stride];
    const float* w1 = &output_weights_[1 * hid_stride];
    const float* w2 = &output_weights_[2 * hid_stride];
    const float* w3 = &output_weights_[3 * hid_stride];
    const float* w4 = &output_weights_[4 * hid_stride];

    float o0 = w0[nh];
    float o1 = w1[nh];
    float o2 = w2[nh];
    float o3 = w3[nh];
    float o4 = w4[nh];
    for (int h = 0; h < nh; ++h) {
        const float z = hiddens[h];
        o0 += w0[h] * z;
        o1 += w1[h] * z;
        o2 += w2[h] * z;
        o3 += w3[h] * z;
        o4 += w4[h] * z;
    }
    outputs[0] = fast_sigmoid(o0);
    outputs[1] = fast_sigmoid(o1);
    outputs[2] = fast_sigmoid(o2);
    outputs[3] = fast_sigmoid(o3);
    outputs[4] = fast_sigmoid(o4);
    return outputs;
}

// ======================== Transposed Weights ========================

void NeuralNetwork::build_transposed_weights() const {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;

    hidden_weights_T_.resize(ni * nh);
    for (int i = 0; i < ni; ++i) {
        for (int h = 0; h < nh; ++h) {
            hidden_weights_T_[i * nh + h] = hidden_weights_[h * inp_stride + i];
        }
    }
}

// ======================== Equity ========================

double NeuralNetwork::compute_equity(const std::array<float, NN_OUTPUTS>& outputs) {
    return 2.0 * outputs[0] - 1.0
         + outputs[1] - outputs[3]
         + outputs[2] - outputs[4];
}

// ======================== Forward with gradient caching ========================

std::array<float, NN_OUTPUTS> NeuralNetwork::forward_with_gradients(const float* inputs) {
    const int nh = n_hidden_;
    const int ni = n_inputs_;
    const int inp_stride = ni + 1;
    const int hid_stride = nh + 1;

    std::copy(inputs, inputs + ni, cached_inputs_.begin());

    for (int h = 0; h < nh; ++h) {
        const float* w = &hidden_weights_[h * inp_stride];
        float sum = w[ni];
        for (int i = 0; i < ni; ++i) {
            sum += w[i] * inputs[i];
        }
        cached_hiddens_[h] = fast_sigmoid(sum);
    }
    cached_hiddens_[nh] = 1.0f;

    for (int o = 0; o < NN_OUTPUTS; ++o) {
        const float* w = &output_weights_[o * hid_stride];
        float sum = w[nh];
        for (int h = 0; h < nh; ++h) {
            sum += w[h] * cached_hiddens_[h];
        }
        cached_outputs_[o] = fast_sigmoid(sum);
    }

    for (int o = 0; o < NN_OUTPUTS; ++o) {
        cached_prods_[o] = cached_outputs_[o] * (1.0f - cached_outputs_[o]);
    }
    for (int h = 0; h < nh; ++h) {
        cached_hprods_[h] = cached_hiddens_[h] * (1.0f - cached_hiddens_[h]);
    }

    return cached_outputs_;
}

// ======================== TD weight update ========================

void NeuralNetwork::td_update(
    const std::array<float, NN_OUTPUTS>& targets, float alpha)
{
    const int nh = n_hidden_;
    const int ni = n_inputs_;

    std::array<float, NN_OUTPUTS> delta_prod;
    for (int o = 0; o < NN_OUTPUTS; ++o) {
        delta_prod[o] = (targets[o] - cached_outputs_[o]) * cached_prods_[o];
    }

    // Step 1: Compute bp_error using PRE-update output weights
    float bp_error_stack[256];
    std::vector<float> bp_error_heap;
    float* bp_error;
    if (nh <= 256) {
        bp_error = bp_error_stack;
    } else {
        bp_error_heap.resize(nh);
        bp_error = bp_error_heap.data();
    }

    const int hid_stride = nh + 1;
    for (int h = 0; h < nh; ++h) {
        float sum = 0.0f;
        for (int o = 0; o < NN_OUTPUTS; ++o) {
            sum += delta_prod[o] * output_weights_[o * hid_stride + h];
        }
        bp_error[h] = sum;
    }

    // Step 2: Update output weights
    for (int o = 0; o < NN_OUTPUTS; ++o) {
        float scale = alpha * delta_prod[o];
        float* w = &output_weights_[o * hid_stride];
        for (int h = 0; h <= nh; ++h) {
            w[h] += scale * cached_hiddens_[h];
        }
    }

    // Step 3: Update hidden weights
    const int inp_stride = ni + 1;
    for (int h = 0; h < nh; ++h) {
        float scale = alpha * bp_error[h] * cached_hprods_[h];
        if (scale == 0.0f) continue;
        float* w = &hidden_weights_[h * inp_stride];
        for (int i = 0; i < ni; ++i) {
            w[i] += scale * cached_inputs_[i];
        }
        w[ni] += scale;
    }
}

// ======================== Persistence ========================

bool NeuralNetwork::save_weights(const std::string& filepath) const {
    std::ofstream f(filepath, std::ios::binary);
    if (!f) return false;

    int32_t header[3] = { static_cast<int32_t>(n_hidden_),
                          static_cast<int32_t>(n_inputs_),
                          static_cast<int32_t>(NN_OUTPUTS) };
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(hidden_weights_.data()),
            hidden_weights_.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(output_weights_.data()),
            output_weights_.size() * sizeof(float));

    return f.good();
}

bool NeuralNetwork::load_weights(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f) return false;

    int32_t header[3];
    f.read(reinterpret_cast<char*>(header), sizeof(header));

    if (header[0] != n_hidden_ || header[1] != n_inputs_ || header[2] != NN_OUTPUTS) {
        return false;
    }

    f.read(reinterpret_cast<char*>(hidden_weights_.data()),
           hidden_weights_.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(output_weights_.data()),
           output_weights_.size() * sizeof(float));

    if (f.good()) {
        build_transposed_weights();
        return true;
    }
    return false;
}

// ======================== NNStrategy ========================

NNStrategy::NNStrategy(std::shared_ptr<NeuralNetwork> nn)
    : nn_(std::move(nn))
{}

NNStrategy::NNStrategy(const std::string& weights_path, int n_hidden, int n_inputs)
    : nn_(std::make_shared<NeuralNetwork>(n_hidden, n_inputs))
{
    if (!nn_->load_weights(weights_path)) {
        throw std::runtime_error("Failed to load weights from: " + weights_path);
    }
}

double NNStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }

    if (nn_->n_inputs() == EXTENDED_CONTACT_INPUTS) {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn_->forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn_->forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> NNStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }

    if (nn_->n_inputs() == EXTENDED_CONTACT_INPUTS) {
        auto inputs = compute_extended_contact_inputs(board);
        return nn_->forward(inputs.data());
    } else {
        auto inputs = compute_tesauro_inputs(board);
        return nn_->forward(inputs);
    }
}

int NNStrategy::best_move_index(const std::vector<Board>& candidates,
                                bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    const int ni = nn_->n_inputs();
    const bool use_extended = (ni == EXTENDED_CONTACT_INPUTS);

    constexpr int STACK_MAX = 32;

    // Flat input buffer: ni floats per position
    std::vector<float> inputs_flat(n * ni);
    std::array<float, NN_OUTPUTS> outputs_stack[STACK_MAX];
    int eval_indices_stack[STACK_MAX];

    std::array<float, NN_OUTPUTS>* outputs_buf;
    int* eval_idx_buf;
    std::vector<std::array<float, NN_OUTPUTS>> outputs_heap;
    std::vector<int> eval_indices_heap;

    if (n <= STACK_MAX) {
        outputs_buf = outputs_stack;
        eval_idx_buf = eval_indices_stack;
    } else {
        outputs_heap.resize(n);
        eval_indices_heap.resize(n);
        outputs_buf = outputs_heap.data();
        eval_idx_buf = eval_indices_heap.data();
    }

    double best_val = -1e30;
    int best_idx = 0;
    int n_to_eval = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER) {
            double val = terminal_equity(result);
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        } else {
            float* dest = inputs_flat.data() + n_to_eval * ni;
            if (use_extended) {
                auto ext = compute_extended_contact_inputs(candidates[i]);
                std::memcpy(dest, ext.data(), ni * sizeof(float));
            } else {
                auto tes = compute_tesauro_inputs(candidates[i]);
                std::memcpy(dest, tes.data(), ni * sizeof(float));
            }
            eval_idx_buf[n_to_eval] = i;
            n_to_eval++;
        }
    }

    if (n_to_eval > 0) {
        nn_->forward_batch(inputs_flat.data(), outputs_buf, n_to_eval);

        for (int j = 0; j < n_to_eval; ++j) {
            double val = NeuralNetwork::compute_equity(outputs_buf[j]);
            if (val > best_val) {
                best_val = val;
                best_idx = eval_idx_buf[j];
            }
        }
    }

    return best_idx;
}

// ======================== MultiNNStrategy ========================

MultiNNStrategy::MultiNNStrategy(std::shared_ptr<NeuralNetwork> contact_nn,
                                 std::shared_ptr<NeuralNetwork> crashed_nn,
                                 std::shared_ptr<NeuralNetwork> race_nn)
    : contact_nn_(std::move(contact_nn))
    , crashed_nn_(std::move(crashed_nn))
    , race_nn_(std::move(race_nn))
{}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& race_weights,
                                 int n_hidden)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = nullptr;  // use contact_nn for crashed positions
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& crashed_weights,
                                 const std::string& race_weights,
                                 int n_hidden)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS);
    if (!crashed_nn_->load_weights(crashed_weights)) {
        throw std::runtime_error("Failed to load crashed weights from: " + crashed_weights);
    }
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

MultiNNStrategy::MultiNNStrategy(const std::string& contact_weights,
                                 const std::string& crashed_weights,
                                 const std::string& race_weights,
                                 int n_hidden_contact,
                                 int n_hidden_crashed,
                                 int n_hidden_race)
{
    contact_nn_ = std::make_shared<NeuralNetwork>(n_hidden_contact, EXTENDED_CONTACT_INPUTS);
    if (!contact_nn_->load_weights(contact_weights)) {
        throw std::runtime_error("Failed to load contact weights from: " + contact_weights);
    }
    crashed_nn_ = std::make_shared<NeuralNetwork>(n_hidden_crashed, EXTENDED_CONTACT_INPUTS);
    if (!crashed_nn_->load_weights(crashed_weights)) {
        throw std::runtime_error("Failed to load crashed weights from: " + crashed_weights);
    }
    race_nn_ = std::make_shared<NeuralNetwork>(n_hidden_race, TESAURO_INPUTS);
    if (!race_nn_->load_weights(race_weights)) {
        throw std::runtime_error("Failed to load race weights from: " + race_weights);
    }
}

const NeuralNetwork& MultiNNStrategy::select_nn(PosType ptype) const {
    switch (ptype) {
        case PosType::RACE:
            return *race_nn_;
        case PosType::CRASHED:
            return crashed_nn_ ? *crashed_nn_ : *contact_nn_;
        case PosType::CONTACT:
        default:
            return *contact_nn_;
    }
}

double MultiNNStrategy::evaluate_with_nn(const Board& board, PosType ptype) const {
    const auto& nn = select_nn(ptype);

    if (ptype == PosType::RACE) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> MultiNNStrategy::probs_with_nn(
    const Board& board, PosType ptype) const
{
    const auto& nn = select_nn(ptype);
    if (ptype == PosType::RACE) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double MultiNNStrategy::evaluate(const Board& board, bool pre_move_is_race) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }

    PosType ptype = classify_position(board);
    return evaluate_with_nn(board, ptype);
}

std::array<float, NN_OUTPUTS> MultiNNStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    PosType ptype = classify_position(board);
    return probs_with_nn(board, ptype);
}

int MultiNNStrategy::best_move_index(const std::vector<Board>& candidates,
                                     bool pre_move_is_race) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            PosType ptype = classify_position(candidates[i]);
            val = evaluate_with_nn(candidates[i], ptype);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

// ======================== GamePlanStrategy ========================

GamePlanStrategy::GamePlanStrategy(std::shared_ptr<NeuralNetwork> purerace_nn,
                                   std::shared_ptr<NeuralNetwork> racing_nn,
                                   std::shared_ptr<NeuralNetwork> attacking_nn,
                                   std::shared_ptr<NeuralNetwork> priming_nn,
                                   std::shared_ptr<NeuralNetwork> anchoring_nn)
    : purerace_nn_(std::move(purerace_nn))
    , racing_nn_(std::move(racing_nn))
    , attacking_nn_(std::move(attacking_nn))
    , priming_nn_(std::move(priming_nn))
    , anchoring_nn_(std::move(anchoring_nn))
{}

GamePlanStrategy::GamePlanStrategy(const std::string& purerace_weights,
                                   const std::string& racing_weights,
                                   const std::string& attacking_weights,
                                   const std::string& priming_weights,
                                   const std::string& anchoring_weights,
                                   int n_hidden_purerace,
                                   int n_hidden_racing,
                                   int n_hidden_attacking,
                                   int n_hidden_priming,
                                   int n_hidden_anchoring)
{
    purerace_nn_ = std::make_shared<NeuralNetwork>(n_hidden_purerace, TESAURO_INPUTS);
    if (!purerace_nn_->load_weights(purerace_weights))
        throw std::runtime_error("Failed to load purerace weights: " + purerace_weights);

    racing_nn_ = std::make_shared<NeuralNetwork>(n_hidden_racing, EXTENDED_CONTACT_INPUTS);
    if (!racing_nn_->load_weights(racing_weights))
        throw std::runtime_error("Failed to load racing weights: " + racing_weights);

    attacking_nn_ = std::make_shared<NeuralNetwork>(n_hidden_attacking, EXTENDED_CONTACT_INPUTS);
    if (!attacking_nn_->load_weights(attacking_weights))
        throw std::runtime_error("Failed to load attacking weights: " + attacking_weights);

    priming_nn_ = std::make_shared<NeuralNetwork>(n_hidden_priming, EXTENDED_CONTACT_INPUTS);
    if (!priming_nn_->load_weights(priming_weights))
        throw std::runtime_error("Failed to load priming weights: " + priming_weights);

    anchoring_nn_ = std::make_shared<NeuralNetwork>(n_hidden_anchoring, EXTENDED_CONTACT_INPUTS);
    if (!anchoring_nn_->load_weights(anchoring_weights))
        throw std::runtime_error("Failed to load anchoring weights: " + anchoring_weights);

    // Prebuild transposed weights to avoid races in concurrent delta evaluation
    // (forward_from_base builds this lazily and is not synchronized).
    purerace_nn_->ensure_transposed_weights();
    racing_nn_->ensure_transposed_weights();
    attacking_nn_->ensure_transposed_weights();
    priming_nn_->ensure_transposed_weights();
    anchoring_nn_->ensure_transposed_weights();
}

const NeuralNetwork& GamePlanStrategy::select_nn(GamePlan gp) const {
    switch (gp) {
        case GamePlan::PURERACE:  return *purerace_nn_;
        case GamePlan::RACING:    return *racing_nn_;
        case GamePlan::ATTACKING: return *attacking_nn_;
        case GamePlan::PRIMING:   return *priming_nn_;
        case GamePlan::ANCHORING: return *anchoring_nn_;
        default:                  return *attacking_nn_;
    }
}

double GamePlanStrategy::evaluate_with_nn(const Board& board, GamePlan gp) const {
    const auto& nn = select_nn(gp);
    if (gp == GamePlan::PURERACE) {
        auto inputs = compute_tesauro_inputs(board);
        auto outputs = nn.forward(inputs);
        return NeuralNetwork::compute_equity(outputs);
    } else {
        // RACING, ATTACKING, PRIMING, ANCHORING all use extended inputs
        auto inputs = compute_extended_contact_inputs(board);
        auto outputs = nn.forward(inputs.data());
        return NeuralNetwork::compute_equity(outputs);
    }
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::probs_with_nn(
    const Board& board, GamePlan gp) const
{
    const auto& nn = select_nn(gp);
    if (gp == GamePlan::PURERACE) {
        auto inputs = compute_tesauro_inputs(board);
        return nn.forward(inputs);
    } else {
        auto inputs = compute_extended_contact_inputs(board);
        return nn.forward(inputs.data());
    }
}

double GamePlanStrategy::evaluate(const Board& board, bool /*pre_move_is_race*/) const {
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_equity(result);
    }
    GamePlan gp = classify_game_plan(board);
    return evaluate_with_nn(board, gp);
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::evaluate_probs(
    const Board& board, bool /*pre_move_is_race*/) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    GamePlan gp = classify_game_plan(board);
    return probs_with_nn(board, gp);
}

std::array<float, NN_OUTPUTS> GamePlanStrategy::evaluate_probs(
    const Board& board, const Board& pre_move_board) const
{
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        return terminal_probs(result);
    }
    GamePlan gp = classify_game_plan(pre_move_board);
    return probs_with_nn(board, gp);
}

int GamePlanStrategy::best_move_index(const std::vector<Board>& candidates,
                                      bool pre_move_is_race) const {
    // Fallback: use PURERACE for race, ATTACKING for non-race
    GamePlan gp = pre_move_is_race ? GamePlan::PURERACE : GamePlan::ATTACKING;
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

int GamePlanStrategy::best_move_index(const std::vector<Board>& candidates,
                                      const Board& pre_move_board) const {
    const int n = static_cast<int>(candidates.size());
    if (n == 1) return 0;

    // Classify the PRE-MOVE board to determine which NN to use for all candidates
    GamePlan gp = classify_game_plan(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

int GamePlanStrategy::evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1) {
        GameResult result = check_game_over(candidates[0]);
        if (result != GameResult::NOT_OVER) {
            equities[0] = terminal_equity(result);
        } else {
            GamePlan gp = classify_game_plan(pre_move_board);
            equities[0] = evaluate_with_nn(candidates[0], gp);
        }
        return 0;
    }

    // Classify once for all candidates
    GamePlan gp = classify_game_plan(pre_move_board);

    double best_val = -1e30;
    int best_idx = 0;

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        double val;
        if (result != GameResult::NOT_OVER) {
            val = terminal_equity(result);
        } else {
            val = evaluate_with_nn(candidates[i], gp);
        }
        equities[i] = val;
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    return best_idx;
}

int GamePlanStrategy::batch_evaluate_candidates_equity(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1) {
        // Single candidate — just use the non-batch path
        GameResult result = check_game_over(candidates[0]);
        if (result != GameResult::NOT_OVER) {
            equities[0] = terminal_equity(result);
        } else {
            GamePlan gp = classify_game_plan(pre_move_board);
            equities[0] = evaluate_with_nn(candidates[0], gp);
        }
        return 0;
    }

    GamePlan gp = classify_game_plan(pre_move_board);
    const auto& nn = select_nn(gp);
    const int ni = nn.n_inputs();
    const int nh = nn.n_hidden();

    // Separate terminals from non-terminals
    // Use thread_local to avoid per-call allocation
    thread_local std::vector<int> nn_indices;    // indices of non-terminal candidates

    nn_indices.clear();
    nn_indices.reserve(n);

    double best_val = -1e30;
    int best_idx = 0;

    // First pass: handle terminals, collect non-terminals
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER) {
            equities[i] = terminal_equity(result);
            if (equities[i] > best_val) {
                best_val = equities[i];
                best_idx = i;
            }
        } else {
            nn_indices.push_back(i);
        }
    }

    if (nn_indices.empty()) return best_idx;

    const int nn_count = static_cast<int>(nn_indices.size());

    // Use incremental (delta) evaluation: encode first candidate with
    // forward_save_base, then use forward_from_base for the rest.
    // This exploits the fact that candidate moves from the same position
    // share most input features — only a few change per move.
    thread_local std::vector<float> saved_base;   // pre-sigmoid hidden sums
    thread_local std::vector<float> saved_inputs;  // first candidate's input vector
    thread_local std::vector<float> cur_inputs;    // current candidate's input vector

    saved_base.resize(nh);
    saved_inputs.resize(ni);
    cur_inputs.resize(ni);

    // Encode and evaluate first non-terminal candidate (full pass + save base)
    if (gp == GamePlan::PURERACE) {
        auto inp = compute_tesauro_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    } else {
        auto inp = compute_extended_contact_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    }
    auto out0 = nn.forward_save_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
    double val0 = NeuralNetwork::compute_equity(out0);
    equities[nn_indices[0]] = val0;
    if (val0 > best_val) {
        best_val = val0;
        best_idx = nn_indices[0];
    }

    // Evaluate remaining candidates using delta from the base
    for (int j = 1; j < nn_count; ++j) {
        if (gp == GamePlan::PURERACE) {
            auto inp = compute_tesauro_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.begin());
        } else {
            auto inp = compute_extended_contact_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.begin());
        }
        auto out = nn.forward_from_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
        double val = NeuralNetwork::compute_equity(out);
        equities[nn_indices[j]] = val;
        if (val > best_val) {
            best_val = val;
            best_idx = nn_indices[j];
        }
    }

    return best_idx;
}

int GamePlanStrategy::batch_evaluate_candidates_equity_probs(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* probs_out) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1) {
        GameResult result = check_game_over(candidates[0]);
        if (result != GameResult::NOT_OVER) {
            equities[0] = terminal_equity(result);
            probs_out[0] = terminal_probs(result);
        } else {
            GamePlan gp = classify_game_plan(pre_move_board);
            probs_out[0] = probs_with_nn(candidates[0], gp);
            equities[0] = NeuralNetwork::compute_equity(probs_out[0]);
        }
        return 0;
    }

    GamePlan gp = classify_game_plan(pre_move_board);
    const auto& nn = select_nn(gp);
    const int ni = nn.n_inputs();
    const int nh = nn.n_hidden();

    thread_local std::vector<int> nn_indices;

    nn_indices.clear();
    nn_indices.reserve(n);

    double best_val = -1e30;
    int best_idx = 0;

    // First pass: handle terminals, collect non-terminals
    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER) {
            probs_out[i] = terminal_probs(result);
            equities[i] = terminal_equity(result);
            if (equities[i] > best_val) {
                best_val = equities[i];
                best_idx = i;
            }
        } else {
            nn_indices.push_back(i);
        }
    }

    if (nn_indices.empty()) return best_idx;

    const int nn_count = static_cast<int>(nn_indices.size());

    // Use incremental (delta) evaluation for speed.
    thread_local std::vector<float> saved_base;
    thread_local std::vector<float> saved_inputs;
    thread_local std::vector<float> cur_inputs;

    saved_base.resize(nh);
    saved_inputs.resize(ni);
    cur_inputs.resize(ni);

    // Encode and evaluate first non-terminal candidate
    if (gp == GamePlan::PURERACE) {
        auto inp = compute_tesauro_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    } else {
        auto inp = compute_extended_contact_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    }
    auto out0 = nn.forward_save_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
    probs_out[nn_indices[0]] = out0;
    double val0 = NeuralNetwork::compute_equity(out0);
    equities[nn_indices[0]] = val0;
    if (val0 > best_val) {
        best_val = val0;
        best_idx = nn_indices[0];
    }

    // Evaluate remaining candidates using delta from the base
    for (int j = 1; j < nn_count; ++j) {
        if (gp == GamePlan::PURERACE) {
            auto inp = compute_tesauro_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.data());
        } else {
            auto inp = compute_extended_contact_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.data());
        }
        auto out = nn.forward_from_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
        probs_out[nn_indices[j]] = out;
        double val = NeuralNetwork::compute_equity(out);
        equities[nn_indices[j]] = val;
        if (val > best_val) {
            best_val = val;
            best_idx = nn_indices[j];
        }
    }

    return best_idx;
}

int GamePlanStrategy::batch_evaluate_candidates_best_prob(
    const std::vector<Board>& candidates,
    const Board& pre_move_board,
    double* equities,
    std::array<float, NUM_OUTPUTS>* best_probs_out) const
{
    const int n = static_cast<int>(candidates.size());
    if (n == 0) return -1;
    if (n == 1) {
        GameResult result = check_game_over(candidates[0]);
        if (result != GameResult::NOT_OVER) {
            if (equities) equities[0] = terminal_equity(result);
            if (best_probs_out) {
                *best_probs_out = terminal_probs(result);
            }
        } else {
            GamePlan gp = classify_game_plan(pre_move_board);
            std::array<float, NUM_OUTPUTS> probs = probs_with_nn(candidates[0], gp);
            if (best_probs_out) {
                *best_probs_out = probs;
            }
            if (equities) equities[0] = NeuralNetwork::compute_equity(probs);
        }
        return 0;
    }

    GamePlan gp = classify_game_plan(pre_move_board);
    const auto& nn = select_nn(gp);
    const int ni = nn.n_inputs();
    const int nh = nn.n_hidden();

    thread_local std::vector<int> nn_indices;
    nn_indices.clear();
    nn_indices.reserve(n);

    double best_val = -1e30;
    int best_idx = 0;
    std::array<float, NUM_OUTPUTS> best_probs{};

    for (int i = 0; i < n; ++i) {
        GameResult result = check_game_over(candidates[i]);
        if (result != GameResult::NOT_OVER) {
            double val = terminal_equity(result);
            if (equities) equities[i] = val;
            if (val > best_val) {
                best_val = val;
                best_idx = i;
                if (best_probs_out) {
                    best_probs = terminal_probs(result);
                }
            }
        } else {
            nn_indices.push_back(i);
        }
    }

    if (nn_indices.empty()) {
        if (best_probs_out) {
            *best_probs_out = best_probs;
        }
        return best_idx;
    }

    const int nn_count = static_cast<int>(nn_indices.size());

    thread_local std::vector<float> saved_base;
    thread_local std::vector<float> saved_inputs;
    thread_local std::vector<float> cur_inputs;
    saved_base.resize(nh);
    saved_inputs.resize(ni);
    cur_inputs.resize(ni);

    if (gp == GamePlan::PURERACE) {
        auto inp = compute_tesauro_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    } else {
        auto inp = compute_extended_contact_inputs(candidates[nn_indices[0]]);
        std::copy(inp.begin(), inp.end(), cur_inputs.begin());
    }

    auto out0 = nn.forward_save_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
    double val0 = NeuralNetwork::compute_equity(out0);
    if (equities) equities[nn_indices[0]] = val0;
    if (val0 > best_val) {
        best_val = val0;
        best_idx = nn_indices[0];
        if (best_probs_out) best_probs = out0;
    }

    for (int j = 1; j < nn_count; ++j) {
        if (gp == GamePlan::PURERACE) {
            auto inp = compute_tesauro_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.data());
        } else {
            auto inp = compute_extended_contact_inputs(candidates[nn_indices[j]]);
            std::copy(inp.begin(), inp.end(), cur_inputs.data());
        }

        auto out = nn.forward_from_base(cur_inputs.data(), saved_base.data(), saved_inputs.data());
        double val = NeuralNetwork::compute_equity(out);
        if (equities) equities[nn_indices[j]] = val;
        if (val > best_val) {
            best_val = val;
            best_idx = nn_indices[j];
            best_probs = out;
        }
    }

    if (best_probs_out) {
        *best_probs_out = best_probs;
    }

    return best_idx;
}

} // namespace bgbot
