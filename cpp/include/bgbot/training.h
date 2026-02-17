#pragma once

#include "neural_net.h"
#include "benchmark.h"
#include <vector>
#include <string>
#include <cstdint>

namespace bgbot {

// One row in the training progress log
struct TrainingHistoryEntry {
    int game_number;
    double contact_score;     // benchmark ER in millipips
    double elapsed_seconds;
};

// Configuration for td_train()
struct TDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    int n_hidden            = 120;
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 1000;        // benchmark every N games
    std::string model_name  = "td_test";   // saved as models/{name}.weights
    std::string models_dir  = "models";
    std::string resume_from = "";          // path to existing .weights file

    // Benchmark scenarios (pre-loaded by Python, passed in).
    // If nullptr, no benchmarking during training.
    const std::vector<BenchmarkScenario>* benchmark_scenarios = nullptr;
};

// Results returned by td_train()
struct TDTrainResult {
    int games_played       = 0;
    double total_seconds   = 0.0;
    std::vector<TrainingHistoryEntry> history;
};

// Run TD(0) self-play training with a single 196-input network.
TDTrainResult td_train(const TDTrainConfig& config);

// Configuration for multi-network TD training
struct MultiTDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    int n_hidden_contact    = 120;   // hidden nodes for contact NN
    int n_hidden_crashed    = 120;   // hidden nodes for crashed NN
    int n_hidden_race       = 80;    // hidden nodes for race NN
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 1000;
    std::string model_name  = "td_multi";
    std::string models_dir  = "models";

    // Resume paths for each network (empty = start from scratch)
    std::string resume_contact = "";
    std::string resume_crashed = "";
    std::string resume_race    = "";

    // Benchmark scenarios for progress tracking
    const std::vector<BenchmarkScenario>* contact_benchmark = nullptr;
};

// Run TD(0) self-play training with three separate networks:
// - Contact: 214 inputs (extended encoding), n_hidden_contact hidden nodes
// - Crashed: 214 inputs (extended encoding), n_hidden_crashed hidden nodes
// - Race: 196 inputs (Tesauro encoding), n_hidden_race hidden nodes
//
// During self-play, classifies each position and updates the appropriate network.
TDTrainResult td_train_multi(const MultiTDTrainConfig& config);


// Configuration for 5-network game plan TD training
struct GamePlanTDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    int n_hidden_purerace   = 80;
    int n_hidden_racing     = 120;
    int n_hidden_attacking  = 120;
    int n_hidden_priming    = 120;
    int n_hidden_anchoring  = 120;
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 1000;
    std::string model_name  = "td_gameplan";
    std::string models_dir  = "models";

    std::string resume_purerace  = "";
    std::string resume_racing    = "";
    std::string resume_attacking = "";
    std::string resume_priming   = "";
    std::string resume_anchoring = "";

    // Benchmark scenarios for each game plan type (pre-loaded by Python)
    const std::vector<BenchmarkScenario>* purerace_benchmark = nullptr;
    const std::vector<BenchmarkScenario>* attacking_benchmark = nullptr;
    const std::vector<BenchmarkScenario>* priming_benchmark   = nullptr;
    const std::vector<BenchmarkScenario>* anchoring_benchmark = nullptr;
    const std::vector<BenchmarkScenario>* race_benchmark      = nullptr;
};

// Run TD(0) self-play training with five game plan networks.
TDTrainResult td_train_gameplan(const GamePlanTDTrainConfig& config);

} // namespace bgbot
