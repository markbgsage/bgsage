// Profile raw neural network inference: encoding + forward pass.
//
// Designed for profiling with VTune, perf, or wall-clock microbenchmarks.
// Measures: (1) forward() only, (2) encoding only, (3) full evaluate_probs.
//
// Usage:
//   profile_nn_inference [n_iters] [model_dir]
//   Defaults: n_iters=100000, model_dir=<exe>/../models

#include "bgbot/neural_net.h"
#include "bgbot/encoding.h"
#include "bgbot/board.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <array>
#include <numeric>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

// Diverse contact positions for realistic profiling
static const Board POSITIONS[] = {
    // Starting position
    {0,-2,0,0,0,0,5,0,3,0,0,0,-5,5,0,0,0,-3,0,-5,0,0,0,0,2,0},
    // Mid-game attacking
    {0,0,0,2,2,-2,3,2,2,0,0,0,-3,4,0,0,0,-3,0,-3,-2,-2,0,0,0,0},
    // Priming position
    {0,-2,0,0,0,3,4,3,0,0,0,0,-3,3,0,0,0,-3,0,-4,0,-1,2,0,0,0},
    // Anchoring / back game
    {1,2,2,0,0,0,3,0,3,0,0,0,-4,3,0,0,0,-2,0,-5,-2,0,-1,0,0,2},
    // Late mid-game
    {0,0,0,0,3,-1,4,0,3,0,0,0,-2,2,0,-2,0,0,-3,-4,0,-1,0,3,0,0},
    // Blitz in progress
    {0,-1,0,0,0,0,5,0,3,0,0,0,-3,5,0,0,0,-3,0,-3,-2,0,0,0,2,0},
    // Scattered checkers
    {0,1,-1,2,0,-2,3,1,0,1,-1,0,-2,2,0,-1,1,-1,0,-3,-1,0,1,3,0,0},
    // Heavy prime vs anchor
    {0,0,0,3,3,3,3,3,0,0,0,0,-2,0,0,0,0,0,0,-5,-3,-2,-2,-1,0,0},
};
static constexpr int N_POSITIONS = sizeof(POSITIONS) / sizeof(POSITIONS[0]);

// Prevent dead-code elimination
static volatile float sink = 0.0f;

int main(int argc, char* argv[]) {
    int n_iters = 100000;
    if (argc > 1) {
        int n = std::atoi(argv[1]);
        if (n > 0) n_iters = n;
    }

    std::filesystem::path model_dir;
    if (argc > 2) {
        model_dir = std::filesystem::path(argv[2]);
    } else {
        model_dir = std::filesystem::absolute(
            std::filesystem::path(argv[0])).parent_path().parent_path() / "models";
    }

    // Load a 400-hidden contact NN (Stage 8 attacking-racing pair NN)
    const auto weight_path = (model_dir / "sl_s8_att_race.weights.best").string();
    NeuralNetwork nn(400, EXTENDED_CONTACT_INPUTS);
    nn.load_weights(weight_path);

    // Also set up a full GamePlanStrategy for evaluate_probs profiling
    const auto pr = (model_dir / "sl_s5_purerace.weights.best").string();
    const auto rc = (model_dir / "sl_s5_racing.weights.best").string();
    const auto at = (model_dir / "sl_s5_attacking.weights.best").string();
    const auto pm = (model_dir / "sl_s5_priming.weights.best").string();
    const auto an = (model_dir / "sl_s5_anchoring.weights.best").string();

    init_escape_tables();

    GamePlanStrategy strategy(pr, rc, at, pm, an, 200, 400, 400, 400, 400);

    printf("=== NN Inference Profiling ===\n");
    printf("Contact NN: 244 inputs, 400 hidden, 5 outputs\n");
    printf("Iterations: %d per benchmark\n", n_iters);
    printf("Positions: %d diverse contact boards\n", N_POSITIONS);
    printf("Model: %s\n\n", weight_path.c_str());

    // Pre-encode all positions
    std::array<std::array<float, EXTENDED_CONTACT_INPUTS>, N_POSITIONS> encoded;
    for (int p = 0; p < N_POSITIONS; ++p) {
        encoded[p] = compute_extended_contact_inputs(POSITIONS[p]);
    }

    // Sparsity analysis
    printf("--- Input Sparsity Analysis ---\n");
    for (int p = 0; p < N_POSITIONS; ++p) {
        int n_zero = 0;
        for (int i = 0; i < EXTENDED_CONTACT_INPUTS; ++i) {
            if (encoded[p][i] == 0.0f) ++n_zero;
        }
        printf("  Pos %d: %d/244 zero (%.0f%% sparse)\n", p, n_zero, 100.0 * n_zero / 244);
    }
    printf("\n");

    // Verify output for the first position (reference values)
    auto ref = nn.forward(encoded[0].data());
    printf("Reference output (pos 0): %.6f %.6f %.6f %.6f %.6f\n\n",
           ref[0], ref[1], ref[2], ref[3], ref[4]);

    // ========== Benchmark 1: forward() only ==========
    {
        for (int i = 0; i < 1000; ++i) {
            auto out = nn.forward(encoded[i % N_POSITIONS].data());
            sink = out[0];
        }

        auto start = Clock::now();
        for (int i = 0; i < n_iters; ++i) {
            auto out = nn.forward(encoded[i % N_POSITIONS].data());
            sink = out[0];
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / n_iters) * 1e6;
        printf("forward() only:         %8.2f us/call  (%d calls in %.3f s)\n",
               per_call_us, n_iters, elapsed);
    }

    // ========== Benchmark 2: encoding only ==========
    {
        for (int i = 0; i < 1000; ++i) {
            auto enc = compute_extended_contact_inputs(POSITIONS[i % N_POSITIONS]);
            sink = enc[0];
        }

        auto start = Clock::now();
        for (int i = 0; i < n_iters; ++i) {
            auto enc = compute_extended_contact_inputs(POSITIONS[i % N_POSITIONS]);
            sink = enc[0];
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / n_iters) * 1e6;
        printf("encoding only:          %8.2f us/call  (%d calls in %.3f s)\n",
               per_call_us, n_iters, elapsed);
    }

    // ========== Benchmark 3: full evaluate_probs (classify + encode + forward) ==========
    {
        for (int i = 0; i < 1000; ++i) {
            auto out = strategy.evaluate_probs(POSITIONS[i % N_POSITIONS], false);
            sink = out[0];
        }

        auto start = Clock::now();
        for (int i = 0; i < n_iters; ++i) {
            auto out = strategy.evaluate_probs(POSITIONS[i % N_POSITIONS], false);
            sink = out[0];
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / n_iters) * 1e6;
        printf("evaluate_probs (full):  %8.2f us/call  (%d calls in %.3f s)\n",
               per_call_us, n_iters, elapsed);
    }

    // ========== Benchmark 4: batch forward (amortized overhead) ==========
    {
        constexpr int BATCH = 16;
        float batch_inputs[BATCH * EXTENDED_CONTACT_INPUTS];
        std::array<float, 5> batch_outputs[BATCH];
        for (int b = 0; b < BATCH; ++b) {
            std::copy(encoded[b % N_POSITIONS].begin(), encoded[b % N_POSITIONS].end(),
                      batch_inputs + b * EXTENDED_CONTACT_INPUTS);
        }

        int batch_iters = n_iters / BATCH;
        for (int i = 0; i < 100; ++i) {
            nn.forward_batch(batch_inputs, batch_outputs, BATCH);
        }

        auto start = Clock::now();
        for (int i = 0; i < batch_iters; ++i) {
            nn.forward_batch(batch_inputs, batch_outputs, BATCH);
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / (batch_iters * BATCH)) * 1e6;
        printf("forward_batch (x%d):    %8.2f us/call  (%d evals in %.3f s)\n",
               BATCH, per_call_us, batch_iters * BATCH, elapsed);
    }

    // ========== Benchmark 5: forward_save_base + forward_from_base (delta eval) ==========
    {
        float saved_base[512];
        float saved_inputs[EXTENDED_CONTACT_INPUTS];

        auto base_out = nn.forward_save_base(encoded[0].data(), saved_base, saved_inputs);

        for (int i = 0; i < 1000; ++i) {
            auto out = nn.forward_from_base(encoded[(i + 1) % N_POSITIONS].data(),
                                             saved_base, saved_inputs);
            sink = out[0];
        }

        auto start = Clock::now();
        for (int i = 0; i < n_iters; ++i) {
            auto out = nn.forward_from_base(encoded[(i + 1) % N_POSITIONS].data(),
                                             saved_base, saved_inputs);
            sink = out[0];
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / n_iters) * 1e6;
        printf("forward_from_base:      %8.2f us/call  (%d calls in %.3f s)\n",
               per_call_us, n_iters, elapsed);
    }

    // ========== Benchmark 6: classify_game_plan cost ==========
    {
        for (int i = 0; i < 1000; ++i) {
            auto gp = classify_game_plan(POSITIONS[i % N_POSITIONS]);
            sink = static_cast<float>(static_cast<int>(gp));
        }

        auto start = Clock::now();
        for (int i = 0; i < n_iters; ++i) {
            auto gp = classify_game_plan(POSITIONS[i % N_POSITIONS]);
            sink = static_cast<float>(static_cast<int>(gp));
        }
        auto end = Clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double per_call_us = (elapsed / n_iters) * 1e6;
        printf("classify_game_plan:     %8.2f us/call  (%d calls in %.3f s)\n",
               per_call_us, n_iters, elapsed);
    }

    printf("\nDone. sink=%.6f\n", (double)sink);
    return 0;
}
