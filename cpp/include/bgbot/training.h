// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
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

// One (position, target-equity) row of a back-game equity benchmark. The
// target is the cubeless post-move equity from the perspective of the player
// whose checkers are positive (as stored in the *-backgame-*-rollout files).
struct EquityBenchmarkEntry {
    Board board;
    float target_equity;
};

// Configuration for td_train_pasko(): a single extended-contact network
// (244 inputs) trained by self-play from an arbitrary fixed start position.
// Intended for the "Paskogammon" side project — games always start from
// start_board with the positive player on roll, and the opening roll may be
// doubles (both differ from standard backgammon). Progress is tracked with a
// back-game equity benchmark (mean |equity - target| * 1000) rather than the
// GNUbg .bm best-move benchmarks used by the other trainers.
struct PaskoTDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    int n_hidden            = 400;
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 10000;
    std::string model_name  = "td_pasko";
    std::string models_dir  = "models";
    std::string resume_from = "";

    // Fixed opening position; positive player is always on roll.
    Board start_board = STARTING_BOARD;

    // Back-game equity benchmark scored during training. If nullptr/empty, no
    // benchmarking is done.
    const std::vector<EquityBenchmarkEntry>* benchmark = nullptr;
};

// Run TD(0) self-play training with a single 244-input extended-contact
// network, starting every game from config.start_board (positive player on
// roll, opening doubles allowed). Saves {model_name}.weights, .weights.best
// (best benchmark ER), and .history.csv.
TDTrainResult td_train_pasko(const PaskoTDTrainConfig& config);

// Configuration for td_train_backgame_truncated(): one 244-input backgame
// category NN (Stage 11) trained by TRUNCATED TD self-play. Games start from
// the provided backgame reference positions (cycled) and are played with
// 1-ply decisions by the training NN, but a game's TD chain ends the moment a
// post-move position leaves every backgame category (backgame_category() ==
// NONE): the final update targets a frozen reference model's ref_plies-ply
// cubeless post-move evaluation of that position, standing in for the game
// outcome exactly as the terminal 0/1 targets do in ordinary TD. A game that
// genuinely ends while still in a backgame (the opponent bears off through
// the anchors) uses the real outcome as usual.
struct BackgameTDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    int n_hidden            = 400;
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 10000;
    std::string model_name  = "td_s11_bg";
    std::string models_dir  = "models";
    std::string resume_from = "";

    // Start positions, positive player to act; game i starts from [i % size].
    // With randomize_first_mover, a coin decides which side acts first (the
    // board is flipped when the other side does).
    std::vector<Board> start_boards;
    bool randomize_first_mover = true;

    // Safety valve: a path this long is force-truncated with the reference
    // target rather than looping (a real game ends far sooner).
    int max_half_moves = 2000;

    // Also train the EXIT position itself toward the reference eval (an extra
    // supervised update at each truncation, in the exit position's own frame).
    // Without this, exit positions are only ever READ (by the reference, to
    // supply the predecessor's target) and never appear as training inputs —
    // so the net's own valuation of boundary positions stays at its random
    // initialisation, which the move selection nevertheless consults whenever
    // an exit move is a candidate. Diagnostic/experimental; default off.
    bool anchor_boundary = false;

    // Frozen reference model for truncation targets (Stage 9: 19 paths).
    std::vector<std::string> ref_weight_paths;
    std::vector<int> ref_hidden_sizes;
    int ref_plies    = 3;
    bool ref_parallel = true;   // parallelize the reference eval's interior
    int ref_threads  = 0;       // 0 = auto

    // Same equity benchmark as td_train_pasko (board + target equity rows).
    const std::vector<EquityBenchmarkEntry>* benchmark = nullptr;
};

// Truncated TD(0) self-play for one Stage 11 backgame category NN. Saves
// {model_name}.weights, .weights.best (best benchmark ER) and .history.csv.
TDTrainResult td_train_backgame_truncated(const BackgameTDTrainConfig& config);

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

// Configuration for 17-network game plan pair TD training.
// Supports NN sharing: canonical_map[i] specifies which NN index to use for
// pair index i. Default is identity (each pair has its own NN). To share NNs,
// set multiple indices to the same canonical value (e.g., canonical_map[11] =
// canonical_map[15] = 12 to share (prim_prim) and (anch_prim) with (prim_anch)).
struct GamePlanPairTDTrainConfig {
    int n_games             = 5000;
    float alpha             = 0.1f;
    std::array<int, NUM_PAIR_NNS> hidden_sizes = {};  // [0]=purerace, [1-16]=contact pairs
    float weight_init_eps   = 0.1f;
    uint32_t seed           = 42;
    int benchmark_interval  = 1000;
    std::string model_name  = "td_s7";
    std::string models_dir  = "models";
    std::array<std::string, NUM_PAIR_NNS> resume_paths = {};
    std::array<const std::vector<BenchmarkScenario>*, NUM_PAIR_NNS> benchmarks = {};

    // Canonical index mapping: canonical_map[i] = physical NN to use for pair i.
    // Default: identity (each pair has its own NN). Shared pairs point to the
    // same canonical index.
    std::array<int, NUM_PAIR_NNS> canonical_map = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
};

// Run TD(0) self-play training with 17 game plan pair networks.
TDTrainResult td_train_gameplan_pair(const GamePlanPairTDTrainConfig& config);

} // namespace bgbot
