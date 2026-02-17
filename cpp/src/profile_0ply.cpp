// Profile 0-ply evaluation: encode + forward pass for a contact position.
// Usage: bgbot_profile_0ply [iterations]
//   Default iterations auto-calibrated to ~1 second.
//
// Measures:
//   1. Full evaluate_probs (classify + encode + forward)
//   2. Encoding only (compute_extended_contact_inputs)
//   3. Forward pass only (nn.forward)
//   4. Game plan classification only

#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/board.h"
#include "bgbot/types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <array>

using namespace bgbot;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    // Test position: a contact position (non-race)
    Board board = {0, -2, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, -4, 3, -1, 0, 0, -3, 2, -4, -1, 0, 0, 0, 0, 0};

    // Initialize escape tables (one-time cost, not measured)
    init_escape_tables();

    // Classify game plan for this position
    GamePlan gp = classify_game_plan(board);
    printf("Position game plan: %s\n", game_plan_name(gp));

    // Load weights for the appropriate NN
    // For profiling, we use random weights (same compute cost as real weights)
    int n_hidden = 400;  // contact NNs use 400 hidden
    int n_inputs = EXTENDED_CONTACT_INPUTS;  // 244
    NeuralNetwork nn(n_hidden, n_inputs, 0.1f, 42);
    nn.ensure_transposed_weights();  // build transposed weights for column-major forward

    // Warm up
    for (int i = 0; i < 100; ++i) {
        auto inputs = compute_extended_contact_inputs(board);
        auto probs = nn.forward(inputs.data());
        (void)probs;
    }

    // --- Auto-calibrate iteration count ---
    int iterations = 0;
    if (argc > 1) {
        iterations = std::atoi(argv[1]);
    }
    if (iterations <= 0) {
        // Run for 0.1s to estimate per-iteration cost
        auto t0 = Clock::now();
        int cal = 0;
        while (true) {
            auto inputs = compute_extended_contact_inputs(board);
            auto probs = nn.forward(inputs.data());
            (void)probs;
            ++cal;
            if (cal % 100 == 0) {
                auto t1 = Clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();
                if (elapsed >= 0.1) {
                    double per_iter = elapsed / cal;
                    iterations = static_cast<int>(1.0 / per_iter);
                    if (iterations < 1000) iterations = 1000;
                    break;
                }
            }
        }
        printf("Auto-calibrated: %d iterations (targeting ~1 second)\n", iterations);
    }

    // Print reference output for correctness validation
    {
        auto inputs = compute_extended_contact_inputs(board);
        auto probs = nn.forward(inputs.data());
        printf("\nReference output (random weights):\n");
        printf("  P(win)=%.8f  P(gw)=%.8f  P(bw)=%.8f  P(gl)=%.8f  P(bl)=%.8f\n",
               probs[0], probs[1], probs[2], probs[3], probs[4]);
        double eq = NeuralNetwork::compute_equity(probs);
        printf("  Equity=%.8f\n", eq);
    }

    printf("\n=== Profiling %d iterations ===\n\n", iterations);

    // --- Benchmark 1: Full evaluation (classify + encode + forward) ---
    {
        auto t0 = Clock::now();
        std::array<float, 5> probs;
        for (int i = 0; i < iterations; ++i) {
            GamePlan g = classify_game_plan(board);
            (void)g;
            auto inputs = compute_extended_contact_inputs(board);
            probs = nn.forward(inputs.data());
        }
        auto t1 = Clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double per_iter_us = elapsed / iterations * 1e6;
        printf("[Full 0-ply eval]  Total: %.3f s  |  Per-call: %.2f us  |  Throughput: %.0f evals/s\n",
               elapsed, per_iter_us, iterations / elapsed);
        // Prevent optimization
        volatile float sink = probs[0];
        (void)sink;
    }

    // --- Benchmark 2: Encoding only ---
    {
        auto t0 = Clock::now();
        std::array<float, EXTENDED_CONTACT_INPUTS> inputs;
        for (int i = 0; i < iterations; ++i) {
            inputs = compute_extended_contact_inputs(board);
        }
        auto t1 = Clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double per_iter_us = elapsed / iterations * 1e6;
        printf("[Encoding only]    Total: %.3f s  |  Per-call: %.2f us  |  Throughput: %.0f evals/s\n",
               elapsed, per_iter_us, iterations / elapsed);
        volatile float sink = inputs[0];
        (void)sink;
    }

    // --- Benchmark 3: Forward pass only (pre-encoded inputs) ---
    {
        auto inputs = compute_extended_contact_inputs(board);
        auto t0 = Clock::now();
        std::array<float, 5> probs;
        for (int i = 0; i < iterations; ++i) {
            probs = nn.forward(inputs.data());
        }
        auto t1 = Clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double per_iter_us = elapsed / iterations * 1e6;
        printf("[Forward pass]     Total: %.3f s  |  Per-call: %.2f us  |  Throughput: %.0f evals/s\n",
               elapsed, per_iter_us, iterations / elapsed);
        volatile float sink = probs[0];
        (void)sink;
    }

    // --- Benchmark 4: Classification only ---
    {
        auto t0 = Clock::now();
        GamePlan g;
        for (int i = 0; i < iterations; ++i) {
            g = classify_game_plan(board);
        }
        auto t1 = Clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double per_iter_us = elapsed / iterations * 1e6;
        printf("[Classification]   Total: %.3f s  |  Per-call: %.2f us  |  Throughput: %.0f evals/s\n",
               elapsed, per_iter_us, iterations / elapsed);
        volatile int sink = static_cast<int>(g);
        (void)sink;
    }

    printf("\nDone.\n");
    return 0;
}
