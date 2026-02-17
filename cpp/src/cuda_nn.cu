#include "bgbot/cuda_nn.h"
#include "bgbot/neural_net.h"
#include "bgbot/benchmark.h"
#include "bgbot/encoding.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace bgbot {

// ======================== Error checking macros ========================

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        throw std::runtime_error(std::string("CUDA error: ") +         \
            cudaGetErrorString(err) + " at " + __FILE__ + ":" +        \
            std::to_string(__LINE__));                                  \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t status = (call);                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
        throw std::runtime_error(std::string("cuBLAS error: ") +       \
            std::to_string(status) + " at " + __FILE__ + ":" +         \
            std::to_string(__LINE__));                                  \
    }                                                                   \
} while(0)

// ======================== CUDA Kernels ========================

// Sigmoid activation: out[i] = 1 / (1 + exp(-in[i]))
__global__ void sigmoid_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        if (x > 15.0f) data[idx] = 1.0f;
        else if (x < -15.0f) data[idx] = 0.0f;
        else data[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// Add bias to each row of a matrix: out[b*cols + j] += bias[j]
// data: [batch x cols], bias: [cols]
__global__ void add_bias_kernel(float* data, const float* bias, int batch, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * cols) {
        int j = idx % cols;
        data[idx] += bias[j];
    }
}

// Compute MSE gradient through sigmoid: delta[i] = (pred[i] - target[i]) * pred[i] * (1 - pred[i])
// Actually for td_update style: delta[i] = (target[i] - pred[i]) * pred[i] * (1 - pred[i])
// But we want gradient descent, so: delta[i] = (pred[i] - target[i]) * pred[i] * (1 - pred[i])
// Wait - looking at td_update: delta_prod[o] = (targets[o] - cached_outputs_[o]) * cached_prods_[o]
// And the update is: w += alpha * delta_prod * input  (gradient ASCENT on target - pred)
// This is equivalent to gradient descent on MSE loss with the right sign.
// Let's match td_update: delta = (target - pred) * pred * (1 - pred)
__global__ void output_delta_kernel(float* delta, const float* pred, const float* target,
                                     int batch, int n_outputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * n_outputs) {
        float p = pred[idx];
        float t = target[idx];
        delta[idx] = (t - p) * p * (1.0f - p);
    }
}

// Scale output delta by per-sample weight: delta[b*no + o] *= sample_weight[b]
// delta: [batch x n_outputs], sample_weights: [batch]
__global__ void scale_delta_by_sample_weight_kernel(float* delta, const float* sample_weights,
                                                      int batch, int n_outputs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * n_outputs) {
        int b = idx / n_outputs;
        delta[idx] *= sample_weights[b];
    }
}

// Compute hidden layer delta: h_delta[i] = bp_error[i] * hidden[i] * (1 - hidden[i])
__global__ void hidden_delta_kernel(float* h_delta, const float* hidden, const float* bp_error,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float h = hidden[idx];
        h_delta[idx] = bp_error[idx] * h * (1.0f - h);
    }
}

// SGD weight update: w[i] += alpha * dw[i]
// where dw = delta^T * input (computed by cuBLAS sgemm)
// Actually, cuBLAS sgemm with alpha=learning_rate directly does the accumulation
// But we also need bias updates. Let's keep a separate kernel for bias.

// Bias update: bias[j] += alpha * sum(delta[:, j]) over batch
// delta: [batch x width], bias: [width]
__global__ void bias_update_kernel(float* bias, const float* delta, float alpha,
                                    int batch, int width) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width) {
        float sum = 0.0f;
        for (int b = 0; b < batch; ++b) {
            sum += delta[b * width + j];
        }
        bias[j] += alpha * sum;
    }
}

// ======================== GPU Memory Manager ========================

struct GpuBuffers {
    // Network weights (persistent on GPU)
    float* d_hidden_weights = nullptr;  // [n_hidden x n_inputs] (no bias col)
    float* d_hidden_bias    = nullptr;  // [n_hidden]
    float* d_output_weights = nullptr;  // [n_outputs x n_hidden] (no bias col)
    float* d_output_bias    = nullptr;  // [n_outputs]

    // Training data (persistent on GPU)
    float* d_all_inputs  = nullptr;     // [n_positions x n_inputs]
    float* d_all_targets = nullptr;     // [n_positions x n_outputs]
    float* d_all_sample_weights = nullptr; // [n_positions] (per-sample weight, null if uniform)

    // Batch-sized intermediate buffers
    float* d_batch_inputs   = nullptr;  // [batch_size x n_inputs]
    float* d_hidden_act     = nullptr;  // [batch_size x n_hidden]
    float* d_output_act     = nullptr;  // [batch_size x n_outputs]
    float* d_batch_targets  = nullptr;  // [batch_size x n_outputs]
    float* d_batch_sample_weights = nullptr; // [batch_size] (gathered per-sample weights)
    float* d_output_delta   = nullptr;  // [batch_size x n_outputs]
    float* d_bp_error       = nullptr;  // [batch_size x n_hidden]
    float* d_hidden_delta   = nullptr;  // [batch_size x n_hidden]

    // Shuffle index buffer
    int* d_indices = nullptr;           // [n_positions]

    int n_hidden  = 0;
    int n_inputs  = 0;
    int n_outputs = 5;
    int batch_size = 0;
    int n_positions = 0;
    bool has_sample_weights = false;

    void allocate(int nh, int ni, int no, int bs, int np, bool sample_weights) {
        n_hidden = nh; n_inputs = ni; n_outputs = no; batch_size = bs; n_positions = np;
        has_sample_weights = sample_weights;

        CUDA_CHECK(cudaMalloc(&d_hidden_weights, nh * ni * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_bias, nh * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_weights, no * nh * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_bias, no * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_all_inputs, np * ni * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_all_targets, np * no * sizeof(float)));
        if (sample_weights) {
            CUDA_CHECK(cudaMalloc(&d_all_sample_weights, np * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_batch_sample_weights, bs * sizeof(float)));
        }

        CUDA_CHECK(cudaMalloc(&d_batch_inputs, bs * ni * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_act, bs * nh * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_act, bs * no * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_batch_targets, bs * no * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_delta, bs * no * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bp_error, bs * nh * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_delta, bs * nh * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_indices, np * sizeof(int)));
    }

    void free() {
        if (d_hidden_weights) cudaFree(d_hidden_weights);
        if (d_hidden_bias) cudaFree(d_hidden_bias);
        if (d_output_weights) cudaFree(d_output_weights);
        if (d_output_bias) cudaFree(d_output_bias);
        if (d_all_inputs) cudaFree(d_all_inputs);
        if (d_all_targets) cudaFree(d_all_targets);
        if (d_all_sample_weights) cudaFree(d_all_sample_weights);
        if (d_batch_inputs) cudaFree(d_batch_inputs);
        if (d_hidden_act) cudaFree(d_hidden_act);
        if (d_output_act) cudaFree(d_output_act);
        if (d_batch_targets) cudaFree(d_batch_targets);
        if (d_batch_sample_weights) cudaFree(d_batch_sample_weights);
        if (d_output_delta) cudaFree(d_output_delta);
        if (d_bp_error) cudaFree(d_bp_error);
        if (d_hidden_delta) cudaFree(d_hidden_delta);
        if (d_indices) cudaFree(d_indices);
        d_hidden_weights = nullptr;
    }
};

// ======================== Weight conversion helpers ========================

// Extract weights from NeuralNetwork's row-major [n x (m+1)] format (bias in last col)
// into separate weight [n x m] and bias [n] arrays.
static void split_weights_and_bias(const std::vector<float>& flat,
                                    int n_rows, int n_cols_plus_bias,
                                    std::vector<float>& weights,
                                    std::vector<float>& bias) {
    int n_cols = n_cols_plus_bias - 1;
    weights.resize(n_rows * n_cols);
    bias.resize(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            weights[r * n_cols + c] = flat[r * n_cols_plus_bias + c];
        }
        bias[r] = flat[r * n_cols_plus_bias + n_cols];
    }
}

// Merge separate weight [n x m] and bias [n] back into [n x (m+1)] flat format
static void merge_weights_and_bias(std::vector<float>& flat,
                                    int n_rows, int n_cols_plus_bias,
                                    const std::vector<float>& weights,
                                    const std::vector<float>& bias) {
    int n_cols = n_cols_plus_bias - 1;
    flat.resize(n_rows * n_cols_plus_bias);
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
            flat[r * n_cols_plus_bias + c] = weights[r * n_cols + c];
        }
        flat[r * n_cols_plus_bias + n_cols] = bias[r];
    }
}

// ======================== Gather batch kernel ========================

// Copy rows from all_data to batch_data based on indices
// all_data: [N x cols], indices: [batch_size], batch_data: [batch_size x cols]
__global__ void gather_rows_kernel(float* batch_data, const float* all_data,
                                    const int* indices, int cols, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * cols) {
        int b = idx / cols;
        int c = idx % cols;
        int src_row = indices[b];
        batch_data[idx] = all_data[src_row * cols + c];
    }
}

// ======================== Main training function ========================

SupervisedTrainResult cuda_supervised_train(const SupervisedTrainConfig& config) {
    auto t_start = std::chrono::steady_clock::now();

    const int nh = config.n_hidden;
    const int ni = config.n_inputs;
    const int no = NN_OUTPUTS;  // 5
    const int np = config.n_positions;
    const int bs = config.batch_size;

    // Create CPU neural network (for benchmarking and weight I/O)
    auto nn = std::make_shared<NeuralNetwork>(nh, ni, 0.1f, config.seed);
    if (!config.starting_weights.empty()) {
        if (!nn->load_weights(config.starting_weights)) {
            throw std::runtime_error("Failed to load weights: " + config.starting_weights);
        }
    }

    // cuBLAS handle
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Allocate GPU memory
    GpuBuffers gpu;
    bool use_sample_weights = (config.sample_weights != nullptr);
    gpu.allocate(nh, ni, no, bs, np, use_sample_weights);

    // Upload training data to GPU
    CUDA_CHECK(cudaMemcpy(gpu.d_all_inputs, config.inputs,
                           np * ni * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_all_targets, config.targets,
                           np * no * sizeof(float), cudaMemcpyHostToDevice));
    if (use_sample_weights) {
        CUDA_CHECK(cudaMemcpy(gpu.d_all_sample_weights, config.sample_weights,
                               np * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Split CPU weights into weight + bias arrays and upload to GPU
    std::vector<float> h_weights, h_bias;
    split_weights_and_bias(nn->hidden_weights(), nh, ni + 1, h_weights, h_bias);
    CUDA_CHECK(cudaMemcpy(gpu.d_hidden_weights, h_weights.data(),
                           nh * ni * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_hidden_bias, h_bias.data(),
                           nh * sizeof(float), cudaMemcpyHostToDevice));

    split_weights_and_bias(nn->output_weights(), no, nh + 1, h_weights, h_bias);
    CUDA_CHECK(cudaMemcpy(gpu.d_output_weights, h_weights.data(),
                           no * nh * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_output_bias, h_bias.data(),
                           no * sizeof(float), cudaMemcpyHostToDevice));

    // Create shuffle indices on CPU
    std::vector<int> indices(np);
    for (int i = 0; i < np; ++i) indices[i] = i;
    std::mt19937 rng(config.seed);

    // Strategy for benchmarking (will update weights each epoch)
    // NNStrategy now handles both Tesauro (196) and extended (214) inputs
    // based on the NN's n_inputs()
    auto bench_strat = std::make_shared<NNStrategy>(nn);

    SupervisedTrainResult result;
    result.weights_path = config.save_path;
    result.best_weights_path = config.save_path.empty() ? "" : config.save_path + ".best";
    double best_score = 1e9;

    // Score history for moving average display
    std::vector<double> score_history;

    // Helper macro for computing CUDA grid size
    #define DIV_CEIL(n, d) (((n) + (d) - 1) / (d))

    float alpha = config.alpha;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        // Shuffle indices on CPU and upload
        std::shuffle(indices.begin(), indices.end(), rng);
        CUDA_CHECK(cudaMemcpy(gpu.d_indices, indices.data(),
                               np * sizeof(int), cudaMemcpyHostToDevice));

        // Process batches
        for (int start = 0; start < np; start += bs) {
            int cur_bs = std::min(bs, np - start);

            // Gather batch inputs and targets from shuffled indices
            {
                int n_elements = cur_bs * ni;
                int n_blocks = (n_elements + 255) / 256;
                gather_rows_kernel<<<n_blocks, 256>>>(
                    gpu.d_batch_inputs, gpu.d_all_inputs,
                    gpu.d_indices + start, ni, cur_bs);
            }
            {
                int n_elements = cur_bs * no;
                int n_blocks = (n_elements + 255) / 256;
                gather_rows_kernel<<<n_blocks, 256>>>(
                    gpu.d_batch_targets, gpu.d_all_targets,
                    gpu.d_indices + start, no, cur_bs);
            }
            if (use_sample_weights) {
                int n_blocks = (cur_bs + 255) / 256;
                gather_rows_kernel<<<n_blocks, 256>>>(
                    gpu.d_batch_sample_weights, gpu.d_all_sample_weights,
                    gpu.d_indices + start, 1, cur_bs);
            }

            // ---- Forward pass ----

            // Hidden = inputs x hidden_weights^T
            // inputs: [cur_bs x ni], hidden_weights: [nh x ni]
            // Result: hidden_act = [cur_bs x nh]
            // cuBLAS is column-major, so we compute: C = B^T * A^T, then C^T is what we want
            // Actually, easier: sgemm with transB=T gives:
            // C[m,n] = alpha * A[m,k] * B^T[k,n] where A=inputs, B=hidden_weights
            // In cuBLAS col-major: we store row-major matrices as transposed col-major
            // So A_row[m,k] in col-major is A_col[k,m] = A^T
            // sgemm(N, N, nh, cur_bs, ni, 1, hidden_weights, nh, inputs, ni, 0, hidden_act, nh)
            // This computes: C_col = hidden_weights_col * inputs_col
            // = hidden_weights_row^T * inputs_row^T  (since col-major store is transpose of row-major)
            // C_row = (C_col)^T = inputs_row * hidden_weights_row^T  -- not quite right
            //
            // Let me think more carefully:
            // We want: hidden_act[b][h] = sum_i inputs[b][i] * hidden_weights[h][i]
            // This is: H = X * W^T where X=[cur_bs x ni], W=[nh x ni], H=[cur_bs x nh]
            //
            // In cuBLAS column-major:
            // X_cm[ni x cur_bs], W_cm[ni x nh], H_cm[nh x cur_bs]
            // H_cm = W_cm^T * X_cm = sgemm('T', 'N', nh, cur_bs, ni, 1, W, ni, X, ni, 0, H, nh)
            {
                float one = 1.0f, zero = 0.0f;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    nh, cur_bs, ni,
                    &one,
                    gpu.d_hidden_weights, ni,   // W: [nh x ni] stored row-major = [ni x nh] col-major
                    gpu.d_batch_inputs, ni,      // X: [cur_bs x ni] stored row-major = [ni x cur_bs] col-major
                    &zero,
                    gpu.d_hidden_act, nh));       // H: [cur_bs x nh] stored row-major = [nh x cur_bs] col-major
            }

            // Add hidden bias + sigmoid
            {
                int n = cur_bs * nh;
                int n_blocks = (n + 255) / 256;
                add_bias_kernel<<<n_blocks, 256>>>(gpu.d_hidden_act, gpu.d_hidden_bias, cur_bs, nh);
                sigmoid_kernel<<<n_blocks, 256>>>(gpu.d_hidden_act, n);
            }

            // Output = hidden_act x output_weights^T
            // O = H * Wo^T where H=[cur_bs x nh], Wo=[no x nh], O=[cur_bs x no]
            // cuBLAS: O_cm = Wo_cm^T * H_cm = sgemm('T', 'N', no, cur_bs, nh, ...)
            {
                float one = 1.0f, zero = 0.0f;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    no, cur_bs, nh,
                    &one,
                    gpu.d_output_weights, nh,
                    gpu.d_hidden_act, nh,
                    &zero,
                    gpu.d_output_act, no));
            }

            // Add output bias + sigmoid
            {
                int n = cur_bs * no;
                int n_blocks = (n + 255) / 256;
                add_bias_kernel<<<n_blocks, 256>>>(gpu.d_output_act, gpu.d_output_bias, cur_bs, no);
                sigmoid_kernel<<<n_blocks, 256>>>(gpu.d_output_act, n);
            }

            // ---- Backward pass ----

            // Output delta: (target - pred) * pred * (1 - pred)
            {
                int n = cur_bs * no;
                int n_blocks = (n + 255) / 256;
                output_delta_kernel<<<n_blocks, 256>>>(
                    gpu.d_output_delta, gpu.d_output_act, gpu.d_batch_targets,
                    cur_bs, no);
            }

            // Scale output delta by per-sample weight (if provided)
            if (use_sample_weights) {
                int n = cur_bs * no;
                int n_blocks = (n + 255) / 256;
                scale_delta_by_sample_weight_kernel<<<n_blocks, 256>>>(
                    gpu.d_output_delta, gpu.d_batch_sample_weights, cur_bs, no);
            }

            // Backprop error to hidden: bp_error = output_delta * output_weights
            // bp_error[b][h] = sum_o output_delta[b][o] * output_weights[o][h]
            // bp_error = output_delta * Wo where output_delta=[cur_bs x no], Wo=[no x nh]
            // cuBLAS: bp_cm = Wo_cm * delta_cm = sgemm('N', 'N', nh, cur_bs, no, ...)
            {
                float one = 1.0f, zero = 0.0f;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    nh, cur_bs, no,
                    &one,
                    gpu.d_output_weights, nh,
                    gpu.d_output_delta, no,
                    &zero,
                    gpu.d_bp_error, nh));
            }

            // Update output weights: Wo += lr * output_delta^T * hidden_act
            // dWo[o][h] = sum_b output_delta[b][o] * hidden_act[b][h]
            // dWo = delta^T * H where delta=[cur_bs x no], H=[cur_bs x nh]
            // cuBLAS: dWo_cm = H_cm * delta_cm^T = sgemm('N', 'T', nh, no, cur_bs, lr, ...)
            // lr = alpha / batch_size (mean gradient, standard mini-batch SGD)
            {
                float lr = alpha / cur_bs;
                float one = 1.0f;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    nh, no, cur_bs,
                    &lr,
                    gpu.d_hidden_act, nh,
                    gpu.d_output_delta, no,
                    &one,
                    gpu.d_output_weights, nh));
            }

            // Update output bias
            {
                int n_blocks = (no + 255) / 256;
                float lr = alpha / cur_bs;
                bias_update_kernel<<<n_blocks, 256>>>(
                    gpu.d_output_bias, gpu.d_output_delta, lr, cur_bs, no);
            }

            // Hidden delta: bp_error * hidden * (1 - hidden)
            {
                int n = cur_bs * nh;
                int n_blocks = (n + 255) / 256;
                hidden_delta_kernel<<<n_blocks, 256>>>(
                    gpu.d_hidden_delta, gpu.d_hidden_act, gpu.d_bp_error, n);
            }

            // Update hidden weights: Wh += lr * hidden_delta^T * inputs
            // dWh[h][i] = sum_b hidden_delta[b][h] * inputs[b][i]
            // cuBLAS: dWh_cm = X_cm * delta_cm^T = sgemm('N', 'T', ni, nh, cur_bs, lr, ...)
            {
                float lr = alpha / cur_bs;
                float one = 1.0f;
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    ni, nh, cur_bs,
                    &lr,
                    gpu.d_batch_inputs, ni,
                    gpu.d_hidden_delta, nh,
                    &one,
                    gpu.d_hidden_weights, ni));
            }

            // Update hidden bias
            {
                int n_blocks = (nh + 255) / 256;
                float lr = alpha / cur_bs;
                bias_update_kernel<<<n_blocks, 256>>>(
                    gpu.d_hidden_bias, gpu.d_hidden_delta, lr, cur_bs, nh);
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy weights back to CPU for benchmarking
        std::vector<float> hw_flat, hb_flat, ow_flat, ob_flat;
        hw_flat.resize(nh * ni);
        hb_flat.resize(nh);
        ow_flat.resize(no * nh);
        ob_flat.resize(no);

        CUDA_CHECK(cudaMemcpy(hw_flat.data(), gpu.d_hidden_weights,
                               nh * ni * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb_flat.data(), gpu.d_hidden_bias,
                               nh * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ow_flat.data(), gpu.d_output_weights,
                               no * nh * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ob_flat.data(), gpu.d_output_bias,
                               no * sizeof(float), cudaMemcpyDeviceToHost));

        // Merge into NeuralNetwork format
        merge_weights_and_bias(nn->hidden_weights(), nh, ni + 1, hw_flat, hb_flat);
        merge_weights_and_bias(nn->output_weights(), no, nh + 1, ow_flat, ob_flat);

        // Benchmark
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();

        double bm_score = -1.0;
        if (config.benchmark_scenarios && !config.benchmark_scenarios->empty()) {
            BenchmarkResult bm = score_benchmarks(*bench_strat, *config.benchmark_scenarios, 0);
            bm_score = bm.score();
        }

        // Track score history for moving average
        if (bm_score >= 0) {
            score_history.push_back(bm_score);
        }

        if (epoch % config.print_interval == 0 || epoch == config.epochs - 1) {
            std::cout << "Epoch " << std::setw(4) << epoch
                      << "  score=" << std::fixed << std::setprecision(2) << bm_score
                      << "  time=" << std::setprecision(1) << elapsed << "s";

            // Print 10-epoch moving average and diff if we have enough history
            int n = (int)score_history.size();
            if (n >= 10) {
                double avg = 0;
                for (int i = n - 10; i < n; i++) avg += score_history[i];
                avg /= 10.0;
                std::cout << std::setprecision(2) << "  avg10=" << avg;

                if (n >= 20) {
                    double prev_avg = 0;
                    for (int i = n - 20; i < n - 10; i++) prev_avg += score_history[i];
                    prev_avg /= 10.0;
                    double diff = avg - prev_avg;
                    std::cout << " (" << std::showpos << diff << std::noshowpos << ")";
                }
            }
            std::cout << std::endl;
        }

        // Save best
        if (bm_score >= 0 && bm_score < best_score) {
            best_score = bm_score;
            result.best_epoch = epoch;
            if (!result.best_weights_path.empty()) {
                nn->save_weights(result.best_weights_path);
            }
        }

        // Save periodic
        if (!config.save_path.empty() &&
            (epoch % config.print_interval == 0 || epoch == config.epochs - 1)) {
            nn->save_weights(config.save_path);
        }

        result.epochs_completed = epoch + 1;
    }

    // Cleanup
    gpu.free();
    CUBLAS_CHECK(cublasDestroy(cublas));

    auto t_end = std::chrono::steady_clock::now();
    result.total_seconds = std::chrono::duration<double>(t_end - t_start).count();
    result.best_score = best_score;

    return result;
}

bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace bgbot
