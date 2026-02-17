#pragma once

#include "neural_net.h"
#include "benchmark.h"
#include <vector>
#include <string>
#include <cstdint>

namespace bgbot {

// Result from GPU supervised training
struct SupervisedTrainResult {
    double best_score    = 1e9;
    int best_epoch       = -1;
    int epochs_completed = 0;
    double total_seconds = 0.0;
    std::string weights_path;
    std::string best_weights_path;
};

// Configuration for GPU supervised training
struct SupervisedTrainConfig {
    // Network architecture
    int n_hidden = 120;
    int n_inputs = 196;

    // Training hyperparameters
    float alpha      = 1.0f;
    int epochs       = 100;
    int batch_size   = 128;
    uint32_t seed    = 42;

    // I/O
    std::string starting_weights;  // empty = random init
    std::string save_path;         // periodic save
    int print_interval = 1;        // print every N epochs

    // Pre-encoded training data (CPU arrays)
    const float* inputs  = nullptr;  // [n_positions x n_inputs] row-major
    const float* targets = nullptr;  // [n_positions x 5] row-major
    const float* sample_weights = nullptr;  // [n_positions] per-sample weight (nullptr = uniform)
    int n_positions = 0;

    // Benchmark during training (CPU-side, using NeuralNetwork)
    const std::vector<BenchmarkScenario>* benchmark_scenarios = nullptr;
};

// Run supervised learning on GPU using cuBLAS + custom CUDA kernels.
// All training (forward, backward, weight update) happens on GPU.
// Benchmarking is done on CPU by copying weights back each epoch.
SupervisedTrainResult cuda_supervised_train(const SupervisedTrainConfig& config);

// Check if CUDA is available and working
bool cuda_available();

} // namespace bgbot
