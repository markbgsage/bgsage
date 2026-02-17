#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>

#include "bgbot/types.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/strategy.h"
#include "bgbot/pubeval.h"
#include "bgbot/benchmark.h"
#include "bgbot/game.h"
#include "bgbot/encoding.h"
#include "bgbot/neural_net.h"
#include "bgbot/training.h"
#include "bgbot/multipy.h"
#include "bgbot/rollout.h"
#include "bgbot/cube.h"

#include "bgbot/cuda_nn.h"

namespace py = pybind11;
using namespace bgbot;

static BenchmarkResult run_score_benchmarks(const Strategy& strategy,
                                             const std::vector<BenchmarkScenario>& scenarios,
                                             int n_threads) {
    return score_benchmarks(strategy, scenarios, n_threads);
}

// Convert a Python list of 26 ints to a Board
static Board list_to_board(const std::vector<int>& v) {
    if (v.size() != 26) {
        throw std::runtime_error("Board must have exactly 26 elements");
    }
    Board b;
    std::copy(v.begin(), v.end(), b.begin());
    return b;
}

// C++ container for benchmark scenarios — avoids pybind11 vector copy issues
// with large datasets. Scenarios are built and stored entirely in C++.
struct ScenarioSet {
    std::vector<BenchmarkScenario> scenarios;

    void reserve(int n) { scenarios.reserve(n); }
    int size() const { return static_cast<int>(scenarios.size()); }

    void add(const std::vector<int>& start_board,
             int die1, int die2,
             const std::vector<std::vector<int>>& ranked_boards,
             const std::vector<double>& ranked_errors) {
        BenchmarkScenario s;
        s.start_board = list_to_board(start_board);
        s.die1 = die1;
        s.die2 = die2;
        s.ranked_boards.reserve(ranked_boards.size());
        for (const auto& rb : ranked_boards) {
            s.ranked_boards.push_back(list_to_board(rb));
        }
        s.ranked_errors = ranked_errors;
        scenarios.push_back(std::move(s));
    }
};

PYBIND11_MODULE(bgbot_cpp, m) {
    m.doc() = "Backgammon bot C++ engine";

    // --- ScenarioSet (preferred for large benchmark sets) ---
    py::class_<ScenarioSet, std::shared_ptr<ScenarioSet>>(m, "ScenarioSet")
        .def(py::init<>())
        .def("reserve", &ScenarioSet::reserve)
        .def("size", &ScenarioSet::size)
        .def("add", &ScenarioSet::add,
             py::arg("start_board"), py::arg("die1"), py::arg("die2"),
             py::arg("ranked_boards"), py::arg("ranked_errors"))
        .def("__len__", &ScenarioSet::size);

    // --- BenchmarkScenario (kept for backward compat) ---
    py::class_<BenchmarkScenario>(m, "BenchmarkScenario")
        .def(py::init<>())
        .def_readwrite("die1", &BenchmarkScenario::die1)
        .def_readwrite("die2", &BenchmarkScenario::die2);

    // Factory function to build a BenchmarkScenario from Python data
    m.def("make_scenario", [](const std::vector<int>& start_board,
                               int die1, int die2,
                               const std::vector<std::vector<int>>& ranked_boards,
                               const std::vector<double>& ranked_errors) {
        BenchmarkScenario s;
        s.start_board = list_to_board(start_board);
        s.die1 = die1;
        s.die2 = die2;
        s.ranked_boards.reserve(ranked_boards.size());
        for (const auto& rb : ranked_boards) {
            s.ranked_boards.push_back(list_to_board(rb));
        }
        s.ranked_errors = ranked_errors;
        return s;
    }, "Create a BenchmarkScenario from Python data");

    // --- BenchmarkResult ---
    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def(py::init<>())
        .def_readonly("total_error", &BenchmarkResult::total_error)
        .def_readonly("count", &BenchmarkResult::count)
        .def("score", &BenchmarkResult::score);

    // --- PubEval ---
    py::enum_<PubEval::WeightSource>(m, "PubEvalWeights")
        .value("TESAURO", PubEval::WeightSource::TESAURO)
        .value("LISTNET", PubEval::WeightSource::LISTNET);

    py::class_<PubEval>(m, "PubEval")
        .def(py::init<PubEval::WeightSource>(),
             py::arg("weights") = PubEval::WeightSource::TESAURO);

    // --- Benchmark scoring ---
    m.def("score_benchmarks_pubeval", [](const ScenarioSet& ss,
                                          PubEval::WeightSource weights,
                                          int n_threads) {
        PubEval strat(weights);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmark scenarios using PubEval strategy",
       py::arg("scenarios"),
       py::arg("weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    m.def("score_benchmarks_pubeval", [](const std::vector<BenchmarkScenario>& scenarios,
                                          PubEval::WeightSource weights,
                                          int n_threads) {
        PubEval strat(weights);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, scenarios, n_threads);
    }, "Score benchmark scenarios using PubEval strategy (list)",
       py::arg("scenarios"),
       py::arg("weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- Game simulation ---
    py::class_<GameStats>(m, "GameStats")
        .def_readonly("n_games", &GameStats::n_games)
        .def_readonly("p1_wins", &GameStats::p1_wins)
        .def_readonly("p1_gammons", &GameStats::p1_gammons)
        .def_readonly("p1_backgammons", &GameStats::p1_backgammons)
        .def_readonly("p2_wins", &GameStats::p2_wins)
        .def_readonly("p2_gammons", &GameStats::p2_gammons)
        .def_readonly("p2_backgammons", &GameStats::p2_backgammons)
        .def("avg_ppg", &GameStats::avg_ppg);

    m.def("play_games_pubeval", [](int n_games, uint32_t seed,
                                    PubEval::WeightSource w1,
                                    PubEval::WeightSource w2,
                                    int n_threads) {
        PubEval strat1(w1);
        PubEval strat2(w2);
        py::gil_scoped_release release;
        return play_games_parallel(strat1, strat2, n_games, seed, n_threads);
    }, "Play games between two PubEval strategies",
       py::arg("n_games"), py::arg("seed") = 42,
       py::arg("weights1") = PubEval::WeightSource::TESAURO,
       py::arg("weights2") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- NeuralNetwork ---
    py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def(py::init<int, int, float, uint32_t>(),
             py::arg("n_hidden") = 120,
             py::arg("n_inputs") = NN_INPUTS,
             py::arg("eps") = 0.1f,
             py::arg("seed") = 42)
        .def("save_weights", &NeuralNetwork::save_weights)
        .def("load_weights", &NeuralNetwork::load_weights)
        .def("n_hidden", &NeuralNetwork::n_hidden)
        .def("n_inputs", &NeuralNetwork::n_inputs)
        .def_static("compute_equity", &NeuralNetwork::compute_equity);

    // --- NNStrategy ---
    py::class_<NNStrategy>(m, "NNStrategy")
        .def(py::init<std::shared_ptr<NeuralNetwork>>())
        .def(py::init<const std::string&, int>(),
             py::arg("weights_path"),
             py::arg("n_hidden") = 120)
        .def("evaluate_board", [](NNStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            auto b = list_to_board(board);
            auto pmb = list_to_board(pre_move_board);
            auto probs = self.evaluate_probs(b, is_race(pmb));
            double eq = NeuralNetwork::compute_equity(probs);
            py::dict result;
            result["probs"] = probs;
            result["equity"] = eq;
            return result;
        }, "Evaluate board, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- MultiNNStrategy ---
    py::class_<MultiNNStrategy>(m, "MultiNNStrategy")
        .def(py::init<const std::string&, const std::string&, int>(),
             py::arg("contact_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden") = 120)
        .def(py::init<const std::string&, const std::string&, const std::string&, int>(),
             py::arg("contact_weights"),
             py::arg("crashed_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden") = 120)
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int, int>(),
             py::arg("contact_weights"),
             py::arg("crashed_weights"),
             py::arg("race_weights"),
             py::arg("n_hidden_contact"),
             py::arg("n_hidden_crashed"),
             py::arg("n_hidden_race"))
        .def("evaluate_board", [](MultiNNStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            auto b = list_to_board(board);
            auto pmb = list_to_board(pre_move_board);
            auto probs = self.evaluate_probs(b, is_race(pmb));
            double eq = NeuralNetwork::compute_equity(probs);
            py::dict result;
            result["probs"] = probs;
            result["equity"] = eq;
            return result;
        }, "Evaluate board, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- Benchmark scoring with NNStrategy ---
    m.def("score_benchmarks_nn", [](const ScenarioSet& ss,
                                     const std::string& weights_path,
                                     int n_hidden,
                                     int n_inputs,
                                     int n_threads) {
        NNStrategy strat(weights_path, n_hidden, n_inputs);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmark scenarios using NN strategy",
       py::arg("scenarios"),
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_threads") = 0);

    m.def("score_benchmarks_nn", [](const std::vector<BenchmarkScenario>& scenarios,
                                     const std::string& weights_path,
                                     int n_hidden,
                                     int n_inputs,
                                     int n_threads) {
        NNStrategy strat(weights_path, n_hidden, n_inputs);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, scenarios, n_threads);
    }, "Score benchmark scenarios using NN strategy (list)",
       py::arg("scenarios"),
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_threads") = 0);

    // --- Benchmark scoring with MultiNNStrategy (2-NN: contact+race) ---
    m.def("score_benchmarks_multi_nn", [](const ScenarioSet& ss,
                                           const std::string& contact_weights,
                                           const std::string& race_weights,
                                           int n_hidden,
                                           int n_threads) {
        MultiNNStrategy strat(contact_weights, race_weights, n_hidden);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using MultiNNStrategy (contact+race)",
       py::arg("scenarios"),
       py::arg("contact_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden") = 120,
       py::arg("n_threads") = 0);

    // --- Benchmark scoring with MultiNNStrategy (3-NN: contact+crashed+race, separate hidden sizes) ---
    m.def("score_benchmarks_3nn", [](const ScenarioSet& ss,
                                      const std::string& contact_weights,
                                      const std::string& crashed_weights,
                                      const std::string& race_weights,
                                      int n_hidden_contact,
                                      int n_hidden_crashed,
                                      int n_hidden_race,
                                      int n_threads) {
        MultiNNStrategy strat(contact_weights, crashed_weights, race_weights,
                              n_hidden_contact, n_hidden_crashed, n_hidden_race);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 3-NN MultiNNStrategy (separate hidden sizes)",
       py::arg("scenarios"),
       py::arg("contact_weights"),
       py::arg("crashed_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("n_threads") = 0);

    // --- TD Training ---
    py::class_<TrainingHistoryEntry>(m, "TrainingHistoryEntry")
        .def_readonly("game_number", &TrainingHistoryEntry::game_number)
        .def_readonly("contact_score", &TrainingHistoryEntry::contact_score)
        .def_readonly("elapsed_seconds", &TrainingHistoryEntry::elapsed_seconds);

    py::class_<TDTrainResult>(m, "TDTrainResult")
        .def_readonly("games_played", &TDTrainResult::games_played)
        .def_readonly("total_seconds", &TDTrainResult::total_seconds)
        .def_readonly("history", &TDTrainResult::history);

    m.def("td_train", [](int n_games, float alpha, int n_hidden, float eps,
                          uint32_t seed, int benchmark_interval,
                          const std::string& model_name,
                          const std::string& models_dir,
                          const std::string& resume_from,
                          std::shared_ptr<ScenarioSet> ss) {
        TDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden = n_hidden;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_from = resume_from;
        if (ss && !ss->scenarios.empty()) {
            config.benchmark_scenarios = &ss->scenarios;
        }
        return td_train(config);
    }, "Run TD(0) self-play training",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden") = 120,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_test",
       py::arg("models_dir") = "models",
       py::arg("resume_from") = "",
       py::arg("scenarios") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Multi-Network TD Training ---
    m.def("td_train_multi", [](int n_games, float alpha,
                                int n_hidden_contact, int n_hidden_crashed, int n_hidden_race,
                                float eps,
                                uint32_t seed, int benchmark_interval,
                                const std::string& model_name,
                                const std::string& models_dir,
                                const std::string& resume_contact,
                                const std::string& resume_crashed,
                                const std::string& resume_race,
                                std::shared_ptr<ScenarioSet> contact_ss) {
        MultiTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden_contact = n_hidden_contact;
        config.n_hidden_crashed = n_hidden_crashed;
        config.n_hidden_race = n_hidden_race;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_contact = resume_contact;
        config.resume_crashed = resume_crashed;
        config.resume_race = resume_race;
        if (contact_ss && !contact_ss->scenarios.empty()) {
            config.contact_benchmark = &contact_ss->scenarios;
        }
        py::gil_scoped_release release;
        return td_train_multi(config);
    }, "Run multi-network TD(0) self-play training (contact+crashed 214-input, race 196-input)",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_multi",
       py::arg("models_dir") = "models",
       py::arg("resume_contact") = "",
       py::arg("resume_crashed") = "",
       py::arg("resume_race") = "",
       py::arg("contact_benchmark") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Supervised Learning ---
    m.def("supervised_train", [](py::array_t<int32_t> boards_np,
                                  py::array_t<float> targets_np,
                                  const std::string& weights_path,
                                  int n_hidden,
                                  int n_inputs,
                                  float alpha,
                                  int epochs,
                                  int batch_size,
                                  uint32_t seed,
                                  int print_interval,
                                  const std::string& save_path,
                                  std::shared_ptr<ScenarioSet> benchmark_ss) {
        // Validate input shapes
        auto boards_info = boards_np.request();
        auto targets_info = targets_np.request();
        if (boards_info.ndim != 2 || boards_info.shape[1] != 26) {
            throw std::runtime_error("boards must be shape [N, 26]");
        }
        if (targets_info.ndim != 2 || targets_info.shape[1] != 5) {
            throw std::runtime_error("targets must be shape [N, 5]");
        }
        int n_positions = static_cast<int>(boards_info.shape[0]);
        if (targets_info.shape[0] != n_positions) {
            throw std::runtime_error("boards and targets must have same number of rows");
        }

        const int32_t* boards_ptr = static_cast<const int32_t*>(boards_info.ptr);
        const float* targets_ptr = static_cast<const float*>(targets_info.ptr);

        // Create or load neural network
        auto nn = std::make_shared<NeuralNetwork>(n_hidden, n_inputs, 0.1f, seed);
        if (!weights_path.empty()) {
            if (!nn->load_weights(weights_path)) {
                throw std::runtime_error("Failed to load weights: " + weights_path);
            }
        }

        // Pre-encode all positions
        std::vector<float> all_inputs(n_positions * n_inputs);
        for (int i = 0; i < n_positions; ++i) {
            Board b;
            for (int j = 0; j < 26; ++j) {
                b[j] = boards_ptr[i * 26 + j];
            }
            if (n_inputs == EXTENDED_CONTACT_INPUTS) {
                auto inp = compute_extended_contact_inputs(b);
                std::copy(inp.begin(), inp.end(), &all_inputs[i * n_inputs]);
            } else {
                auto inp = compute_tesauro_inputs(b);
                std::copy(inp.begin(), inp.end(), &all_inputs[i * n_inputs]);
            }
        }

        // Create strategy for benchmarking
        // For 214-input networks, use MultiNNStrategy so it encodes correctly
        std::shared_ptr<Strategy> strat_ptr;
        if (n_inputs == EXTENDED_CONTACT_INPUTS) {
            // Contact network: wrap in MultiNNStrategy with a dummy race NN
            auto dummy_race = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS, 0.1f, seed);
            strat_ptr = std::make_shared<MultiNNStrategy>(nn, nullptr, dummy_race);
        } else {
            strat_ptr = std::make_shared<NNStrategy>(nn);
        }
        const Strategy& strat = *strat_ptr;
        const std::vector<BenchmarkScenario>* bm_scenarios = nullptr;
        if (benchmark_ss && !benchmark_ss->scenarios.empty()) {
            bm_scenarios = &benchmark_ss->scenarios;
        }

        // Shuffle indices
        std::vector<int> indices(n_positions);
        for (int i = 0; i < n_positions; ++i) indices[i] = i;
        std::mt19937 rng(seed);

        py::gil_scoped_release release;

        auto t_start = std::chrono::steady_clock::now();
        double best_score = 1e9;
        std::string best_weights_path = save_path.empty() ? "" : save_path + ".best";

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle
            std::shuffle(indices.begin(), indices.end(), rng);

            // Mini-batch SGD
            double epoch_loss = 0.0;
            for (int start = 0; start < n_positions; start += batch_size) {
                int end = std::min(start + batch_size, n_positions);
                for (int b = start; b < end; ++b) {
                    int idx = indices[b];
                    const float* inp = &all_inputs[idx * n_inputs];
                    std::array<float, NN_OUTPUTS> target;
                    for (int k = 0; k < 5; ++k) {
                        target[k] = targets_ptr[idx * 5 + k];
                    }
                    auto pred = nn->forward_with_gradients(inp);
                    nn->td_update(target, alpha);

                    // Accumulate loss for monitoring
                    for (int k = 0; k < 5; ++k) {
                        float d = pred[k] - target[k];
                        epoch_loss += d * d;
                    }
                }
            }
            epoch_loss /= n_positions;

            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            // Benchmark
            double bm_score = -1.0;
            if (bm_scenarios && !bm_scenarios->empty()) {
                BenchmarkResult bm = score_benchmarks(strat, *bm_scenarios, 0);
                bm_score = bm.score();
            }

            if (epoch % print_interval == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << std::setw(4) << epoch
                          << "  loss=" << std::fixed << std::setprecision(6) << epoch_loss
                          << "  score=" << std::setprecision(2) << bm_score
                          << "  time=" << std::setprecision(1) << elapsed << "s"
                          << std::endl;
            }

            // Save best
            if (bm_score >= 0 && bm_score < best_score) {
                best_score = bm_score;
                if (!best_weights_path.empty()) {
                    nn->save_weights(best_weights_path);
                }
            }

            // Save periodic
            if (!save_path.empty() && (epoch % print_interval == 0 || epoch == epochs - 1)) {
                nn->save_weights(save_path);
            }
        }

        // Return results as dict
        py::gil_scoped_acquire acquire;
        py::dict result;
        result["best_score"] = best_score;
        result["epochs"] = epochs;
        auto t_end = std::chrono::steady_clock::now();
        result["total_seconds"] = std::chrono::duration<double>(t_end - t_start).count();
        if (!save_path.empty()) {
            result["weights_path"] = save_path;
            result["best_weights_path"] = best_weights_path;
        }
        return result;
    }, "Run supervised learning training",
       py::arg("boards"),
       py::arg("targets"),
       py::arg("weights_path") = "",
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = NN_INPUTS,
       py::arg("alpha") = 0.01f,
       py::arg("epochs") = 100,
       py::arg("batch_size") = 512,
       py::arg("seed") = 42,
       py::arg("print_interval") = 1,
       py::arg("save_path") = "",
       py::arg("benchmark_scenarios") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- NN vs PubEval game simulation (parallelized) ---
    m.def("play_games_nn_vs_pubeval", [](const std::string& weights_path,
                                          int n_hidden,
                                          int n_games,
                                          uint32_t seed,
                                          PubEval::WeightSource pe_weights,
                                          int n_threads) {
        auto nn_ptr = std::make_shared<NeuralNetwork>(n_hidden);
        nn_ptr->load_weights(weights_path);
        NNStrategy nn_strat(nn_ptr);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: NN (p1) vs PubEval (p2)",
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- MultiNN vs PubEval (2-NN) ---
    m.def("play_games_multi_nn_vs_pubeval", [](const std::string& contact_weights,
                                                const std::string& race_weights,
                                                int n_hidden,
                                                int n_games,
                                                uint32_t seed,
                                                PubEval::WeightSource pe_weights,
                                                int n_threads) {
        MultiNNStrategy nn_strat(contact_weights, race_weights, n_hidden);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: MultiNN (p1) vs PubEval (p2)",
       py::arg("contact_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- 3-NN vs PubEval (separate hidden sizes) ---
    m.def("play_games_3nn_vs_pubeval", [](const std::string& contact_weights,
                                           const std::string& crashed_weights,
                                           const std::string& race_weights,
                                           int n_hidden_contact,
                                           int n_hidden_crashed,
                                           int n_hidden_race,
                                           int n_games,
                                           uint32_t seed,
                                           PubEval::WeightSource pe_weights,
                                           int n_threads) {
        MultiNNStrategy nn_strat(contact_weights, crashed_weights, race_weights,
                                 n_hidden_contact, n_hidden_crashed, n_hidden_race);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: 3-NN (p1) vs PubEval (p2), separate hidden sizes",
       py::arg("contact_weights"),
       py::arg("crashed_weights"),
       py::arg("race_weights"),
       py::arg("n_hidden_contact") = 120,
       py::arg("n_hidden_crashed") = 120,
       py::arg("n_hidden_race") = 80,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // --- Utility: weight transfer from 196-input to 214-input network ---
    m.def("create_extended_from_tesauro", [](const std::string& src_weights,
                                              const std::string& dst_weights,
                                              int n_hidden,
                                              uint32_t seed) {
        // Load 196-input network
        auto src_nn = std::make_shared<NeuralNetwork>(n_hidden, TESAURO_INPUTS);
        if (!src_nn->load_weights(src_weights)) {
            throw std::runtime_error("Failed to load source weights: " + src_weights);
        }

        // Create 214-input network with small random weights
        auto dst_nn = std::make_shared<NeuralNetwork>(n_hidden, EXTENDED_CONTACT_INPUTS, 0.01f, seed);

        // Copy output weights (identical: n_hidden -> 5)
        dst_nn->output_weights() = src_nn->output_weights();

        // Copy hidden weights: map Tesauro indices to extended indices
        // Tesauro layout:  [0-95] player points, [96] bar, [97] borne-off, [98-193] opp points, [194] bar, [195] borne-off
        // Extended layout:  [0-95] player points, [96] bar, [97-99] 3 borne-off buckets, [100-106] features,
        //                   [107-202] opp points, [203] bar, [204-206] 3 borne-off buckets, [207-213] features
        const int src_stride = TESAURO_INPUTS + 1;  // 197
        const int dst_stride = EXTENDED_CONTACT_INPUTS + 1;  // 215

        for (int h = 0; h < n_hidden; ++h) {
            // Player point weights: src[0-95] -> dst[0-95]
            for (int i = 0; i < 96; ++i) {
                dst_nn->hidden_weights()[h * dst_stride + i] = src_nn->hidden_weights()[h * src_stride + i];
            }
            // Player bar: src[96] -> dst[96]
            dst_nn->hidden_weights()[h * dst_stride + 96] = src_nn->hidden_weights()[h * src_stride + 96];
            // Player borne-off: src[97] -> dst[97] (first bucket), others stay random small
            dst_nn->hidden_weights()[h * dst_stride + 97] = src_nn->hidden_weights()[h * src_stride + 97];
            // New features [98-106] stay random (small init)

            // Opponent point weights: src[98-193] -> dst[107-202]
            for (int i = 0; i < 96; ++i) {
                dst_nn->hidden_weights()[h * dst_stride + 107 + i] = src_nn->hidden_weights()[h * src_stride + 98 + i];
            }
            // Opponent bar: src[194] -> dst[203]
            dst_nn->hidden_weights()[h * dst_stride + 203] = src_nn->hidden_weights()[h * src_stride + 194];
            // Opponent borne-off: src[195] -> dst[204] (first bucket)
            dst_nn->hidden_weights()[h * dst_stride + 204] = src_nn->hidden_weights()[h * src_stride + 195];
            // New features [205-213] stay random (small init)

            // Bias: src[196] -> dst[214]
            dst_nn->hidden_weights()[h * dst_stride + EXTENDED_CONTACT_INPUTS] =
                src_nn->hidden_weights()[h * src_stride + TESAURO_INPUTS];
        }

        // Save
        if (!dst_nn->save_weights(dst_weights)) {
            throw std::runtime_error("Failed to save extended weights: " + dst_weights);
        }
        return true;
    }, "Create 214-input network weights from 196-input Tesauro weights",
       py::arg("src_weights"),
       py::arg("dst_weights"),
       py::arg("n_hidden") = 120,
       py::arg("seed") = 42);

    // --- GamePlanStrategy ---
    py::class_<GamePlanStrategy>(m, "GamePlanStrategy")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, const std::string&,
                      const std::string&,
                      int, int, int, int, int>(),
             py::arg("purerace_weights"),
             py::arg("racing_weights"),
             py::arg("attacking_weights"),
             py::arg("priming_weights"),
             py::arg("anchoring_weights"),
             py::arg("n_hidden_purerace") = 80,
             py::arg("n_hidden_racing") = 120,
             py::arg("n_hidden_attacking") = 120,
             py::arg("n_hidden_priming") = 120,
             py::arg("n_hidden_anchoring") = 120)
        .def("evaluate_board", [](GamePlanStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            auto b = list_to_board(board);
            auto pmb = list_to_board(pre_move_board);
            auto probs = self.evaluate_probs(b, pmb);
            double eq = NeuralNetwork::compute_equity(probs);
            py::dict result;
            result["probs"] = probs;
            result["equity"] = eq;
            return result;
        }, "Evaluate board at 0-ply, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    // --- Benchmark scoring with GamePlanStrategy (5-NN) ---
    m.def("score_benchmarks_5nn", [](const ScenarioSet& ss,
                                      const std::string& purerace_w,
                                      const std::string& racing_w,
                                      const std::string& attacking_w,
                                      const std::string& priming_w,
                                      const std::string& anchoring_w,
                                      int n_h_purerace,
                                      int n_h_racing,
                                      int n_h_attacking,
                                      int n_h_priming,
                                      int n_h_anchoring,
                                      int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        py::gil_scoped_release release;
        return run_score_benchmarks(strat, ss.scenarios, n_threads);
    }, "Score benchmarks using 5-NN GamePlanStrategy",
       py::arg("scenarios"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_threads") = 0);

    // --- 5-NN vs PubEval ---
    m.def("play_games_5nn_vs_pubeval", [](const std::string& purerace_w,
                                           const std::string& racing_w,
                                           const std::string& attacking_w,
                                           const std::string& priming_w,
                                           const std::string& anchoring_w,
                                           int n_h_purerace,
                                           int n_h_racing,
                                           int n_h_attacking,
                                           int n_h_priming,
                                           int n_h_anchoring,
                                           int n_games,
                                           uint32_t seed,
                                           PubEval::WeightSource pe_weights,
                                           int n_threads) {
        GamePlanStrategy nn_strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                                  n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(nn_strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: 5-NN GamePlan (p1) vs PubEval (p2)",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // 5-NN self-play (same strategy vs itself)
    m.def("play_games_5nn_vs_self", [](const std::string& purerace_w,
                                       const std::string& racing_w,
                                       const std::string& attacking_w,
                                       const std::string& priming_w,
                                       const std::string& anchoring_w,
                                       int n_h_purerace,
                                       int n_h_racing,
                                       int n_h_attacking,
                                       int n_h_priming,
                                       int n_h_anchoring,
                                       int n_games,
                                       uint32_t seed,
                                       int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        py::gil_scoped_release release;
        return play_games_parallel(strat, strat, n_games, seed, n_threads);
    }, "Play games: 5-NN GamePlan vs itself (self-play outcome distribution)",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("n_threads") = 0);

    // --- Game Plan TD Training ---
    m.def("td_train_gameplan", [](int n_games, float alpha,
                                   int n_hidden_purerace,
                                   int n_hidden_racing, int n_hidden_attacking,
                                   int n_hidden_priming, int n_hidden_anchoring,
                                   float eps, uint32_t seed, int benchmark_interval,
                                   const std::string& model_name,
                                   const std::string& models_dir,
                                   const std::string& resume_purerace,
                                   const std::string& resume_racing,
                                   const std::string& resume_attacking,
                                   const std::string& resume_priming,
                                   const std::string& resume_anchoring,
                                   std::shared_ptr<ScenarioSet> purerace_ss,
                                   std::shared_ptr<ScenarioSet> attacking_ss,
                                   std::shared_ptr<ScenarioSet> priming_ss,
                                   std::shared_ptr<ScenarioSet> anchoring_ss,
                                   std::shared_ptr<ScenarioSet> race_ss) {
        GamePlanTDTrainConfig config;
        config.n_games = n_games;
        config.alpha = alpha;
        config.n_hidden_purerace = n_hidden_purerace;
        config.n_hidden_racing = n_hidden_racing;
        config.n_hidden_attacking = n_hidden_attacking;
        config.n_hidden_priming = n_hidden_priming;
        config.n_hidden_anchoring = n_hidden_anchoring;
        config.weight_init_eps = eps;
        config.seed = seed;
        config.benchmark_interval = benchmark_interval;
        config.model_name = model_name;
        config.models_dir = models_dir;
        config.resume_purerace = resume_purerace;
        config.resume_racing = resume_racing;
        config.resume_attacking = resume_attacking;
        config.resume_priming = resume_priming;
        config.resume_anchoring = resume_anchoring;
        if (purerace_ss && !purerace_ss->scenarios.empty())
            config.purerace_benchmark = &purerace_ss->scenarios;
        if (attacking_ss && !attacking_ss->scenarios.empty())
            config.attacking_benchmark = &attacking_ss->scenarios;
        if (priming_ss && !priming_ss->scenarios.empty())
            config.priming_benchmark = &priming_ss->scenarios;
        if (anchoring_ss && !anchoring_ss->scenarios.empty())
            config.anchoring_benchmark = &anchoring_ss->scenarios;
        if (race_ss && !race_ss->scenarios.empty())
            config.race_benchmark = &race_ss->scenarios;
        py::gil_scoped_release release;
        return td_train_gameplan(config);
    }, "Run 5-network game plan TD(0) self-play training",
       py::arg("n_games") = 5000,
       py::arg("alpha") = 0.1f,
       py::arg("n_hidden_purerace") = 80,
       py::arg("n_hidden_racing") = 120,
       py::arg("n_hidden_attacking") = 120,
       py::arg("n_hidden_priming") = 120,
       py::arg("n_hidden_anchoring") = 120,
       py::arg("eps") = 0.1f,
       py::arg("seed") = 42,
       py::arg("benchmark_interval") = 1000,
       py::arg("model_name") = "td_gameplan",
       py::arg("models_dir") = "models",
       py::arg("resume_purerace") = "",
       py::arg("resume_racing") = "",
       py::arg("resume_attacking") = "",
       py::arg("resume_priming") = "",
       py::arg("resume_anchoring") = "",
       py::arg("purerace_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("attacking_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("priming_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("anchoring_benchmark") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("race_benchmark") = std::shared_ptr<ScenarioSet>(nullptr));

    // --- Encoding utilities for testing/validation ---
    m.def("is_race", [](const std::vector<int>& board) {
        return is_race(list_to_board(board));
    });
    m.def("is_crashed", [](const std::vector<int>& board) {
        return is_crashed(list_to_board(board));
    });
    m.def("classify_position", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        PosType pt = classify_position(b);
        switch (pt) {
            case PosType::CONTACT: return std::string("contact");
            case PosType::CRASHED: return std::string("crashed");
            case PosType::RACE: return std::string("race");
            default: return std::string("unknown");
        }
    });
    m.def("classify_game_plan", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        GamePlan gp = classify_game_plan(b);
        return std::string(game_plan_name(gp));
    });
    m.def("classify_game_plans_batch", [](py::array_t<int32_t> boards_np) {
        auto info = boards_np.request();
        if (info.ndim != 2 || info.shape[1] != 26) {
            throw std::runtime_error("boards must be shape [N, 26]");
        }
        int n = static_cast<int>(info.shape[0]);
        const int32_t* ptr = static_cast<const int32_t*>(info.ptr);
        py::array_t<int32_t> result(n);
        auto result_ptr = static_cast<int32_t*>(result.request().ptr);
        for (int i = 0; i < n; ++i) {
            Board b;
            for (int j = 0; j < 26; ++j) b[j] = ptr[i * 26 + j];
            result_ptr[i] = static_cast<int32_t>(classify_game_plan(b));
        }
        return result;
    }, "Classify game plan for a batch of boards. Returns int array: 0=purerace, 1=racing, 2=attacking, 3=priming, 4=anchoring",
       py::arg("boards"));
    m.def("compute_extended_contact_inputs", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto inp = compute_extended_contact_inputs(b);
        py::array_t<float> result(EXTENDED_CONTACT_INPUTS);
        std::copy(inp.begin(), inp.end(), result.mutable_data());
        return result;
    });
    m.def("compute_tesauro_inputs", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto inp = compute_tesauro_inputs(b);
        py::array_t<float> result(196);
        std::copy(inp.begin(), inp.end(), result.mutable_data());
        return result;
    });

    // Expose helper functions for validation
    m.def("hitting_shots", [](const std::vector<int>& board) {
        return hitting_shots(list_to_board(board));
    });
    m.def("double_hitting_shots", [](const std::vector<int>& board) {
        return double_hitting_shots(list_to_board(board));
    });
    m.def("back_escapes", [](const std::vector<int>& board) {
        return back_escapes(list_to_board(board));
    });
    m.def("max_point", [](const std::vector<int>& board) {
        return max_point(list_to_board(board));
    });
    m.def("max_anchor_point", [](const std::vector<int>& board) {
        return max_anchor_point(list_to_board(board));
    });
    m.def("prob_no_enter_from_bar", [](const std::vector<int>& board) {
        auto r = prob_no_enter_from_bar(list_to_board(board));
        return py::make_tuple(r.player, r.opponent);
    });
    m.def("forward_anchor_points", [](const std::vector<int>& board) {
        auto r = forward_anchor_points(list_to_board(board));
        return py::make_tuple(r.player, r.opponent);
    });

    // New GNUbg feature helpers for testing/validation
    m.def("break_contact", [](const std::vector<int>& board) {
        return break_contact(list_to_board(board));
    });
    m.def("free_pip", [](const std::vector<int>& board) {
        return free_pip(list_to_board(board));
    });
    m.def("compute_piploss", [](const std::vector<int>& board) {
        return compute_piploss(list_to_board(board));
    });
    m.def("timing_feature", [](const std::vector<int>& board) {
        return timing(list_to_board(board));
    });
    m.def("backbone_feature", [](const std::vector<int>& board) {
        return backbone(list_to_board(board));
    });
    m.def("backg_feature", [](const std::vector<int>& board) {
        return backg(list_to_board(board));
    });
    m.def("backg1_feature", [](const std::vector<int>& board) {
        return backg1(list_to_board(board));
    });
    m.def("enter_loss", [](const std::vector<int>& board) {
        return enter_loss(list_to_board(board));
    });
    m.def("containment_feature", [](const std::vector<int>& board) {
        return containment(list_to_board(board));
    });
    m.def("acontainment_feature", [](const std::vector<int>& board) {
        return acontainment(list_to_board(board));
    });
    m.def("mobility_feature", [](const std::vector<int>& board) {
        return mobility(list_to_board(board));
    });
    m.def("moment2_feature", [](const std::vector<int>& board) {
        return moment2(list_to_board(board));
    });
    m.def("back_rescue_escapes", [](const std::vector<int>& board) {
        return back_rescue_escapes(list_to_board(board));
    });
    m.def("init_escape_tables", &init_escape_tables,
          "Initialize the escape lookup tables (called automatically on first use)");

    // --- CUDA GPU Supervised Learning ---
    m.def("cuda_available", &cuda_available,
          "Check if CUDA GPU is available");

    m.def("cuda_supervised_train", [](py::array_t<int32_t> boards_np,
                                       py::array_t<float> targets_np,
                                       const std::string& weights_path,
                                       int n_hidden,
                                       int n_inputs,
                                       float alpha,
                                       int epochs,
                                       int batch_size,
                                       uint32_t seed,
                                       int print_interval,
                                       const std::string& save_path,
                                       std::shared_ptr<ScenarioSet> benchmark_ss,
                                       py::object sample_weights_obj,
                                       const std::string& label) {
        // Validate input shapes
        auto boards_info = boards_np.request();
        auto targets_info = targets_np.request();
        if (boards_info.ndim != 2 || boards_info.shape[1] != 26) {
            throw std::runtime_error("boards must be shape [N, 26]");
        }
        if (targets_info.ndim != 2 || targets_info.shape[1] != 5) {
            throw std::runtime_error("targets must be shape [N, 5]");
        }
        int n_positions = static_cast<int>(boards_info.shape[0]);
        if (targets_info.shape[0] != n_positions) {
            throw std::runtime_error("boards and targets must have same number of rows");
        }

        // Handle optional sample_weights
        const float* sample_weights_ptr = nullptr;
        py::array_t<float> sample_weights_np;
        if (!sample_weights_obj.is_none()) {
            sample_weights_np = sample_weights_obj.cast<py::array_t<float>>();
            auto sw_info = sample_weights_np.request();
            if (sw_info.ndim != 1 || sw_info.shape[0] != n_positions) {
                throw std::runtime_error("sample_weights must be shape [N]");
            }
            sample_weights_ptr = static_cast<const float*>(sw_info.ptr);
        }

        const int32_t* boards_ptr = static_cast<const int32_t*>(boards_info.ptr);
        const float* targets_ptr = static_cast<const float*>(targets_info.ptr);

        // Pre-encode all positions on CPU
        std::vector<float> all_inputs(n_positions * n_inputs);
        for (int i = 0; i < n_positions; ++i) {
            Board b;
            for (int j = 0; j < 26; ++j) {
                b[j] = boards_ptr[i * 26 + j];
            }
            if (n_inputs == EXTENDED_CONTACT_INPUTS) {
                auto inp = compute_extended_contact_inputs(b);
                std::copy(inp.begin(), inp.end(), &all_inputs[i * n_inputs]);
            } else {
                auto inp = compute_tesauro_inputs(b);
                std::copy(inp.begin(), inp.end(), &all_inputs[i * n_inputs]);
            }
        }

        std::cout << "Pre-encoded " << n_positions << " positions (" << n_inputs << " inputs each)"
                  << "  hidden=" << n_hidden << "  alpha=" << alpha << "  batch=" << batch_size << "  seed=" << seed;
        if (!label.empty()) {
            std::cout << "  [" << label << "]";
        }
        if (sample_weights_ptr) {
            std::cout << "  (sample weights enabled)";
        }
        std::cout << std::endl;

        // Build config
        SupervisedTrainConfig config;
        config.n_hidden = n_hidden;
        config.n_inputs = n_inputs;
        config.alpha = alpha;
        config.epochs = epochs;
        config.batch_size = batch_size;
        config.seed = seed;
        config.starting_weights = weights_path;
        config.save_path = save_path;
        config.print_interval = print_interval;
        config.inputs = all_inputs.data();
        config.targets = targets_ptr;
        config.sample_weights = sample_weights_ptr;
        config.n_positions = n_positions;
        if (benchmark_ss && !benchmark_ss->scenarios.empty()) {
            config.benchmark_scenarios = &benchmark_ss->scenarios;
        }

        py::gil_scoped_release release;
        SupervisedTrainResult train_result = cuda_supervised_train(config);
        py::gil_scoped_acquire acquire;

        // Return results as dict
        py::dict result;
        result["best_score"] = train_result.best_score;
        result["best_epoch"] = train_result.best_epoch;
        result["epochs_completed"] = train_result.epochs_completed;
        result["total_seconds"] = train_result.total_seconds;
        result["weights_path"] = train_result.weights_path;
        result["best_weights_path"] = train_result.best_weights_path;
        return result;
    }, "Run GPU-accelerated supervised learning training",
       py::arg("boards"),
       py::arg("targets"),
       py::arg("weights_path") = "",
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = NN_INPUTS,
       py::arg("alpha") = 1.0f,
       py::arg("epochs") = 100,
       py::arg("batch_size") = 4096,
       py::arg("seed") = 42,
       py::arg("print_interval") = 1,
       py::arg("save_path") = "",
       py::arg("benchmark_scenarios") = std::shared_ptr<ScenarioSet>(nullptr),
       py::arg("sample_weights") = py::none(),
       py::arg("label") = "");

    // --- Utility functions for debugging ---
    m.def("flip_board", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        auto f = flip(b);
        std::vector<int> result(f.begin(), f.end());
        return result;
    }, "Flip board perspective");

    m.def("possible_moves", [](const std::vector<int>& board, int d1, int d2) {
        auto b = list_to_board(board);
        std::vector<Board> results;
        possible_boards(b, d1, d2, results);
        std::vector<std::vector<int>> out;
        for (const auto& r : results) {
            out.emplace_back(r.begin(), r.end());
        }
        return out;
    }, "Generate all legal post-move boards");

    m.def("possible_single_die_moves", [](const std::vector<int>& board, int die) {
        auto b = list_to_board(board);
        std::vector<Board> results;
        possible_boards_one_die(b, die, results);

        // Build list of {from, to, board} dicts by comparing each result to input
        py::list out;
        for (const auto& nb : results) {
            int from_pt = -1, to_pt = -1;

            // Find which point lost a checker (from)
            if (b[25] > 0 && nb[25] < b[25]) {
                from_pt = 25; // entered from bar
            } else {
                for (int i = 1; i <= 24; ++i) {
                    if (nb[i] < b[i] && b[i] > 0) {
                        from_pt = i;
                        break;
                    }
                }
            }

            // Find which point gained a checker (to), or 0 for bear-off
            for (int i = 1; i <= 24; ++i) {
                // Check if player 1 gained a checker here
                int p1_before = b[i] > 0 ? b[i] : 0;
                int p1_after = nb[i] > 0 ? nb[i] : 0;
                // Also handle hitting: b[i] == -1, nb[i] == 1
                if (b[i] < 0 && nb[i] > 0) p1_after = nb[i];
                if (p1_after > p1_before) {
                    to_pt = i;
                    break;
                }
            }
            if (to_pt == -1) {
                // Must be bearing off
                to_pt = 0;
            }

            py::dict d;
            d["from"] = from_pt;
            d["to"] = to_pt;
            d["board"] = std::vector<int>(nb.begin(), nb.end());
            out.append(d);
        }
        return out;
    }, "Generate all legal single-die moves with from/to info",
       py::arg("board"), py::arg("die"));

    m.def("check_game_over", [](const std::vector<int>& board) {
        auto b = list_to_board(board);
        return static_cast<int>(check_game_over(b));
    }, "Check if game is over. Returns 0=not over, 1/2/3=win single/gammon/bg, -1/-2/-3=loss",
       py::arg("board"));

    m.def("invert_probs_py", [](const std::array<float, NUM_OUTPUTS>& p) {
        return invert_probs(p);
    }, "Invert probabilities from one perspective to the other");

    // --- Multi-Ply Strategy ---
    py::class_<MultiPlyStrategy, std::shared_ptr<MultiPlyStrategy>>(m, "MultiPlyStrategy")
        .def("n_plies", &MultiPlyStrategy::n_plies)
        .def("clear_cache", &MultiPlyStrategy::clear_cache)
        .def("cache_size", &MultiPlyStrategy::cache_size)
        .def("cache_hits", &MultiPlyStrategy::cache_hits)
        .def("cache_misses", &MultiPlyStrategy::cache_misses)
        .def("evaluate_board", [](MultiPlyStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            std::array<float, NUM_OUTPUTS> probs;
            double eq;
            {
                py::gil_scoped_release release;
                auto b = list_to_board(board);
                auto pmb = list_to_board(pre_move_board);
                probs = self.evaluate_probs(b, pmb);
                eq = NeuralNetwork::compute_equity(probs);
            }
            py::dict result;
            result["probs"] = probs;
            result["equity"] = eq;
            return result;
        }, "Evaluate board at N-ply depth, returns probs and equity",
           py::arg("board"), py::arg("pre_move_board"));

    py::class_<MoveFilter>(m, "MoveFilter")
        .def(py::init<>())
        .def(py::init<int, float>(),
             py::arg("max_moves"), py::arg("threshold"))
        .def_readwrite("max_moves", &MoveFilter::max_moves)
        .def_readwrite("threshold", &MoveFilter::threshold);

    m.attr("MOVE_FILTER_TINY")   = MoveFilters::TINY;
    m.attr("MOVE_FILTER_NARROW") = MoveFilters::NARROW;
    m.attr("MOVE_FILTER_NORMAL") = MoveFilters::NORMAL;
    m.attr("MOVE_FILTER_LARGE")  = MoveFilters::LARGE;
    m.attr("MOVE_FILTER_HUGE")   = MoveFilters::HUGE_;

    // Create N-ply strategy wrapping a 5-NN GamePlanStrategy
    m.def("create_multipy_5nn", [](const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int n_plies,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    bool full_depth_opponent,
                                    bool parallel_evaluate,
                                    int parallel_threads) {
        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        MoveFilter filter{filter_max_moves, filter_threshold};
        return std::make_shared<MultiPlyStrategy>(
            base, n_plies, filter, full_depth_opponent,
            parallel_evaluate, parallel_threads);
    }, "Create N-ply strategy wrapping 5-NN GamePlanStrategy",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 120,
       py::arg("n_hidden_racing") = 250,
       py::arg("n_hidden_attacking") = 250,
       py::arg("n_hidden_priming") = 250,
       py::arg("n_hidden_anchoring") = 250,
       py::arg("n_plies") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    // Create N-ply strategy wrapping a single NNStrategy
    m.def("create_multipy_nn", [](const std::string& weights_path,
                                   int n_hidden,
                                   int n_inputs,
                                   int n_plies,
                                   int filter_max_moves,
                                   float filter_threshold,
                                   bool full_depth_opponent,
                                   bool parallel_evaluate,
                                   int parallel_threads) {
        auto base = std::make_shared<NNStrategy>(weights_path, n_hidden, n_inputs);
        MoveFilter filter{filter_max_moves, filter_threshold};
        return std::make_shared<MultiPlyStrategy>(
            base, n_plies, filter, full_depth_opponent,
            parallel_evaluate, parallel_threads);
    }, "Create N-ply strategy wrapping a single NNStrategy",
       py::arg("weights_path"),
       py::arg("n_hidden") = 120,
       py::arg("n_inputs") = 196,
       py::arg("n_plies") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("full_depth_opponent") = false,
       py::arg("parallel_evaluate") = false,
       py::arg("parallel_threads") = 0);

    // Benchmark scoring with MultiPlyStrategy
    m.def("score_benchmarks_multipy", [](const ScenarioSet& ss,
                                          std::shared_ptr<MultiPlyStrategy> strat,
                                          int n_threads) {
        strat->clear_cache();
        py::gil_scoped_release release;
        return run_score_benchmarks(*strat, ss.scenarios, n_threads);
    }, "Score benchmarks using multi-ply strategy",
       py::arg("scenarios"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);

    // Multi-ply vs PubEval game simulation
    m.def("play_games_multipy_vs_pubeval", [](std::shared_ptr<MultiPlyStrategy> strat,
                                               int n_games,
                                               uint32_t seed,
                                               PubEval::WeightSource pe_weights,
                                               int n_threads) {
        strat->clear_cache();
        PubEval pe_strat(pe_weights);
        py::gil_scoped_release release;
        return play_games_parallel(*strat, pe_strat, n_games, seed, n_threads);
    }, "Play games: multi-ply strategy (p1) vs PubEval (p2)",
       py::arg("strategy"),
       py::arg("n_games") = 1000,
       py::arg("seed") = 42,
       py::arg("pe_weights") = PubEval::WeightSource::TESAURO,
       py::arg("n_threads") = 0);

    // ======================== Per-scenario benchmark scoring ========================

    m.def("score_benchmarks_per_scenario_5nn", [](const ScenarioSet& ss,
                                                   const std::string& purerace_w,
                                                   const std::string& racing_w,
                                                   const std::string& attacking_w,
                                                   const std::string& priming_w,
                                                   const std::string& anchoring_w,
                                                   int n_h_purerace,
                                                   int n_h_racing,
                                                   int n_h_attacking,
                                                   int n_h_priming,
                                                   int n_h_anchoring,
                                                   int n_threads) {
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        const int n = static_cast<int>(ss.scenarios.size());
        std::vector<double> errors(n);
        {
            py::gil_scoped_release release;
            score_benchmarks_per_scenario(strat, ss.scenarios, errors.data(), n_threads);
        }
        return errors;
    }, "Score benchmarks per-scenario using 5-NN, returns list of errors",
       py::arg("scenarios"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 120,
       py::arg("n_hidden_racing") = 250,
       py::arg("n_hidden_attacking") = 250,
       py::arg("n_hidden_priming") = 250,
       py::arg("n_hidden_anchoring") = 250,
       py::arg("n_threads") = 0);

    // ======================== Benchmark PR scoring ========================

    // Score a 0-ply strategy against benchmark PR data.
    // decisions: list of dicts with 'board', 'dice', 'candidates', 'rollout_equities'
    // Returns a dict with overall PR, per-plan PR, per-plan counts, etc.
    m.def("score_benchmark_pr_0ply", [](py::list decisions_py,
                                         const std::string& purerace_w,
                                         const std::string& racing_w,
                                         const std::string& attacking_w,
                                         const std::string& priming_w,
                                         const std::string& anchoring_w,
                                         int n_h_purerace,
                                         int n_h_racing,
                                         int n_h_attacking,
                                         int n_h_priming,
                                         int n_h_anchoring,
                                         int n_threads) {
        // Parse decisions from Python
        std::vector<PRDecision> decisions;
        decisions.reserve(decisions_py.size());
        for (auto item : decisions_py) {
            PRDecision dec;
            auto d = item.cast<py::dict>();
            dec.board = list_to_board(d["board"].cast<std::vector<int>>());
            auto dice = d["dice"].cast<std::vector<int>>();
            dec.die1 = dice[0];
            dec.die2 = dice[1];
            auto cands = d["candidates"].cast<std::vector<std::vector<int>>>();
            for (auto& c : cands) {
                dec.candidates.push_back(list_to_board(c));
            }
            dec.rollout_equities = d["rollout_equities"].cast<std::vector<double>>();
            decisions.push_back(std::move(dec));
        }

        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        MoveFilter filter{5, 0.08f};  // TINY (unused for 0-ply but needed by signature)
        PRResult result;
        {
            py::gil_scoped_release release;
            result = score_benchmark_pr(strat, nullptr, decisions, filter, n_threads);
        }

        py::dict out;
        out["overall_pr"] = result.overall_pr();
        out["total_decisions"] = result.total_decisions();
        out["total_error"] = result.total_error();
        out["n_outside"] = result.n_outside;
        out["n_skipped"] = result.n_skipped;
        py::dict gp_pr, gp_n, gp_n_err;
        const char* gp_names[] = {"purerace", "racing", "attacking", "priming", "anchoring"};
        for (int i = 0; i < 5; ++i) {
            gp_pr[gp_names[i]] = result.gp_pr(i);
            gp_n[gp_names[i]] = result.gp_n_decisions[i];
            gp_n_err[gp_names[i]] = result.gp_n_with_error[i];
        }
        out["per_plan_pr"] = gp_pr;
        out["per_plan_n"] = gp_n;
        out["per_plan_n_with_error"] = gp_n_err;
        return out;
    }, "Score 0-ply strategy against benchmark PR data (parallel)",
       py::arg("decisions"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("n_threads") = 0);

    // Score an N-ply strategy against benchmark PR data.
    // Uses 0-ply pre-filter + full-depth scoring on survivors.
    m.def("score_benchmark_pr_nply", [](py::list decisions_py,
                                         const std::string& purerace_w,
                                         const std::string& racing_w,
                                         const std::string& attacking_w,
                                         const std::string& priming_w,
                                         const std::string& anchoring_w,
                                         int n_h_purerace,
                                         int n_h_racing,
                                         int n_h_attacking,
                                         int n_h_priming,
                                         int n_h_anchoring,
                                         int n_plies,
                                         int filter_max_moves,
                                         float filter_threshold,
                                         int n_threads) {
        // Parse decisions from Python
        std::vector<PRDecision> decisions;
        decisions.reserve(decisions_py.size());
        for (auto item : decisions_py) {
            PRDecision dec;
            auto d = item.cast<py::dict>();
            dec.board = list_to_board(d["board"].cast<std::vector<int>>());
            auto dice = d["dice"].cast<std::vector<int>>();
            dec.die1 = dice[0];
            dec.die2 = dice[1];
            auto cands = d["candidates"].cast<std::vector<std::vector<int>>>();
            for (auto& c : cands) {
                dec.candidates.push_back(list_to_board(c));
            }
            dec.rollout_equities = d["rollout_equities"].cast<std::vector<double>>();
            decisions.push_back(std::move(dec));
        }

        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        MoveFilter filter{filter_max_moves, filter_threshold};
        MultiPlyStrategy strat(base, n_plies, filter);
        strat.clear_cache();

        PRResult result;
        {
            py::gil_scoped_release release;
            result = score_benchmark_pr(strat, base.get(), decisions, filter, n_threads);
        }

        py::dict out;
        out["overall_pr"] = result.overall_pr();
        out["total_decisions"] = result.total_decisions();
        out["total_error"] = result.total_error();
        out["n_outside"] = result.n_outside;
        out["n_skipped"] = result.n_skipped;
        py::dict gp_pr, gp_n, gp_n_err;
        const char* gp_names[] = {"purerace", "racing", "attacking", "priming", "anchoring"};
        for (int i = 0; i < 5; ++i) {
            gp_pr[gp_names[i]] = result.gp_pr(i);
            gp_n[gp_names[i]] = result.gp_n_decisions[i];
            gp_n_err[gp_names[i]] = result.gp_n_with_error[i];
        }
        out["per_plan_pr"] = gp_pr;
        out["per_plan_n"] = gp_n;
        out["per_plan_n_with_error"] = gp_n_err;
        return out;
    }, "Score N-ply strategy against benchmark PR data (parallel)",
       py::arg("decisions"),
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("n_plies") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0);

    // ======================== Rollout Strategy ========================

    py::class_<RolloutConfig>(m, "RolloutConfig")
        .def(py::init<>())
        .def_readwrite("n_trials", &RolloutConfig::n_trials)
        .def_readwrite("truncation_depth", &RolloutConfig::truncation_depth)
        .def_readwrite("decision_ply", &RolloutConfig::decision_ply)
        .def_readwrite("vr_ply", &RolloutConfig::vr_ply)
        .def_readwrite("filter", &RolloutConfig::filter)
        .def_readwrite("n_threads", &RolloutConfig::n_threads)
        .def_readwrite("seed", &RolloutConfig::seed);

    py::class_<RolloutResult>(m, "RolloutResult")
        .def_readonly("equity", &RolloutResult::equity)
        .def_readonly("std_error", &RolloutResult::std_error)
        .def_readonly("mean_probs", &RolloutResult::mean_probs)
        .def_readonly("prob_std_errors", &RolloutResult::prob_std_errors)
        .def_readonly("scalar_vr_equity", &RolloutResult::scalar_vr_equity)
        .def_readonly("scalar_vr_se", &RolloutResult::scalar_vr_se);

    py::class_<RolloutStrategy, std::shared_ptr<RolloutStrategy>>(m, "RolloutStrategy")
        .def("config", &RolloutStrategy::config, py::return_value_policy::reference_internal)
        .def("rollout_position", [](const RolloutStrategy& self,
                                     const std::vector<int>& board_vec) {
            Board board = list_to_board(board_vec);
            py::gil_scoped_release release;
            return self.rollout_position(board, board);
        }, "Rollout a single post-move position",
           py::arg("board"))
        .def("evaluate_board", [](RolloutStrategy& self,
                                   const std::vector<int>& board,
                                   const std::vector<int>& pre_move_board) {
            auto b = list_to_board(board);
            auto pmb = list_to_board(pre_move_board);
            py::gil_scoped_release release;
            auto r = self.rollout_position(b, pmb);
            py::gil_scoped_acquire acquire;
            py::dict result;
            result["probs"] = r.mean_probs;
            result["equity"] = r.equity;
            result["std_error"] = r.std_error;
            result["prob_std_errors"] = r.prob_std_errors;
            result["scalar_vr_equity"] = r.scalar_vr_equity;
            result["scalar_vr_se"] = r.scalar_vr_se;
            return result;
        }, "Evaluate board via rollout, returns probs, equity, std_error, prob_std_errors, scalar_vr_equity/se",
           py::arg("board"), py::arg("pre_move_board"));

    // Create rollout strategy wrapping a 5-NN GamePlanStrategy
    m.def("create_rollout_5nn", [](const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int n_trials,
                                    int truncation_depth,
                                    int decision_ply,
                                    int vr_ply,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    int n_threads,
                                    uint32_t seed,
                                    int late_ply,
                                    int late_threshold) {
        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        RolloutConfig config;
        config.n_trials = n_trials;
        config.truncation_depth = truncation_depth;
        config.decision_ply = decision_ply;
        config.vr_ply = vr_ply;
        config.filter = {filter_max_moves, filter_threshold};
        config.n_threads = n_threads;
        config.seed = seed;
        config.late_ply = late_ply;
        config.late_threshold = late_threshold;
        return std::make_shared<RolloutStrategy>(base, config);
    }, "Create rollout strategy wrapping 5-NN GamePlanStrategy",
       py::arg("purerace_weights"),
       py::arg("racing_weights"),
       py::arg("attacking_weights"),
       py::arg("priming_weights"),
       py::arg("anchoring_weights"),
       py::arg("n_hidden_purerace") = 120,
       py::arg("n_hidden_racing") = 250,
       py::arg("n_hidden_attacking") = 250,
       py::arg("n_hidden_priming") = 250,
       py::arg("n_hidden_anchoring") = 250,
       py::arg("n_trials") = 36,
       py::arg("truncation_depth") = 7,
       py::arg("decision_ply") = 0,
       py::arg("vr_ply") = 0,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20);

    // Score benchmarks using rollout strategy
    // n_threads=1 for outer loop because parallelism is within each scenario
    m.def("score_benchmarks_rollout", [](const ScenarioSet& ss,
                                          std::shared_ptr<RolloutStrategy> strat,
                                          int n_threads) {
        py::gil_scoped_release release;
        return run_score_benchmarks(*strat, ss.scenarios, n_threads);
    }, "Score benchmarks using rollout strategy",
       py::arg("scenarios"),
       py::arg("strategy"),
       py::arg("n_threads") = 1);

    // ======================== Doubling Cube (Janowski) ========================

    py::enum_<CubeOwner>(m, "CubeOwner")
        .value("CENTERED", CubeOwner::CENTERED)
        .value("PLAYER", CubeOwner::PLAYER)
        .value("OPPONENT", CubeOwner::OPPONENT);

    py::class_<CubeInfo>(m, "CubeInfo")
        .def(py::init<>())
        .def(py::init([](int value, CubeOwner owner) {
            return CubeInfo{value, owner};
        }), py::arg("cube_value") = 1, py::arg("owner") = CubeOwner::CENTERED)
        .def_readwrite("cube_value", &CubeInfo::cube_value)
        .def_readwrite("owner", &CubeInfo::owner);

    py::class_<CubeDecision>(m, "CubeDecision")
        .def_readonly("equity_nd", &CubeDecision::equity_nd)
        .def_readonly("equity_dt", &CubeDecision::equity_dt)
        .def_readonly("equity_dp", &CubeDecision::equity_dp)
        .def_readonly("should_double", &CubeDecision::should_double)
        .def_readonly("should_take", &CubeDecision::should_take)
        .def_readonly("optimal_equity", &CubeDecision::optimal_equity);

    m.def("cubeless_equity", [](const std::array<float, NUM_OUTPUTS>& probs) {
        return cubeless_equity(probs);
    }, "Compute cubeless equity from 5 probabilities",
       py::arg("probs"));

    m.def("cl2cf_money", [](const std::array<float, NUM_OUTPUTS>& probs,
                             CubeOwner owner, float cube_x) {
        return cl2cf_money(probs, owner, cube_x);
    }, "Cubeless-to-cubeful conversion (Janowski). Returns cubeful equity normalized to cube=1.",
       py::arg("probs"), py::arg("owner"), py::arg("cube_x"));

    m.def("cubeful_equity_nply", [](const std::vector<int>& board_vec,
                                     CubeOwner owner,
                                     GamePlanStrategy& strategy,
                                     int n_plies,
                                     int filter_max_moves,
                                     float filter_threshold,
                                     int n_threads) {
        Board board = list_to_board(board_vec);
        MoveFilter filter{filter_max_moves, filter_threshold};
        py::gil_scoped_release release;
        float eq = cubeful_equity_nply(board, owner, strategy, n_plies, filter, n_threads);
        return eq;
    }, "Compute cubeful equity for a pre-roll position at N-ply depth.\n"
       "Uses recursive cube decision modeling (Janowski only at 0-ply leaves).\n"
       "Returns cubeful equity normalized to cube value 1.",
       py::arg("board"), py::arg("owner"),
       py::arg("strategy"),
       py::arg("n_plies") = 1,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1);

    m.def("cube_efficiency", [](const std::vector<int>& board, bool is_race_pos) {
        return cube_efficiency(list_to_board(board), is_race_pos);
    }, "Cube efficiency for a position (0.68 contact, pip-dependent race)",
       py::arg("board"), py::arg("is_race"));

    m.def("cube_decision_0ply", [](const std::array<float, NUM_OUTPUTS>& probs,
                                    int cube_value, CubeOwner owner, float cube_x) {
        CubeInfo ci{cube_value, owner};
        return cube_decision_0ply(probs, ci, cube_x);
    }, "Compute 0-ply cube decision from cubeless pre-roll probs (Janowski).\n"
       "Returns CubeDecision with equity_nd, equity_dt, equity_dp, should_double, should_take, optimal_equity.",
       py::arg("probs"), py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED, py::arg("cube_x") = 0.68f);

    // Convenience: evaluate pre-roll probs + cube decision in one call
    m.def("evaluate_cube_decision", [](const std::vector<int>& checkers,
                                        int cube_value, CubeOwner owner,
                                        const std::string& purerace_w,
                                        const std::string& racing_w,
                                        const std::string& attacking_w,
                                        const std::string& priming_w,
                                        const std::string& anchoring_w,
                                        int n_h_purerace,
                                        int n_h_racing,
                                        int n_h_attacking,
                                        int n_h_priming,
                                        int n_h_anchoring) {
        Board board = list_to_board(checkers);
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);

        // Pre-roll probs: flip → evaluate → invert
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_move_probs = strat.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_move_probs);

        // Cube efficiency
        float x = cube_efficiency(board, race);

        // Cube decision
        CubeInfo ci{cube_value, owner};
        auto cd = cube_decision_0ply(pre_roll_probs, ci, x);

        // Cubeless equity
        float cl_eq = cubeless_equity(pre_roll_probs);

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["cubeless_equity"] = cl_eq;
        result["cube_x"] = x;
        result["equity_nd"] = cd.equity_nd;
        result["equity_dt"] = cd.equity_dt;
        result["equity_dp"] = cd.equity_dp;
        result["should_double"] = cd.should_double;
        result["should_take"] = cd.should_take;
        result["optimal_equity"] = cd.optimal_equity;
        result["is_race"] = race;
        return result;
    }, "Evaluate pre-roll probs and cube decision for a position.\n"
       "Returns dict with probs, cubeless_equity, cube_x, equity_nd/dt/dp, should_double, should_take, optimal_equity.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("purerace_weights") = "",
       py::arg("racing_weights") = "",
       py::arg("attacking_weights") = "",
       py::arg("priming_weights") = "",
       py::arg("anchoring_weights") = "",
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400);

    // N-ply cube decision (standalone — creates its own strategy, serial by default)
    m.def("cube_decision_nply", [](const std::vector<int>& checkers,
                                    int cube_value, CubeOwner owner,
                                    int n_plies,
                                    const std::string& purerace_w,
                                    const std::string& racing_w,
                                    const std::string& attacking_w,
                                    const std::string& priming_w,
                                    const std::string& anchoring_w,
                                    int n_h_purerace,
                                    int n_h_racing,
                                    int n_h_attacking,
                                    int n_h_priming,
                                    int n_h_anchoring,
                                    int filter_max_moves,
                                    float filter_threshold,
                                    int n_threads) {
        Board board = list_to_board(checkers);
        GamePlanStrategy strat(purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
                               n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        CubeInfo ci{cube_value, owner};
        MoveFilter filter{filter_max_moves, filter_threshold};

        CubeDecision cd;
        {
            py::gil_scoped_release release;
            cd = cube_decision_nply(board, ci, strat, n_plies, filter, n_threads);
        }

        // Also get cubeless pre-roll probs for display
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = strat.evaluate_probs(flipped, race);
        auto pre_roll_probs = invert_probs(post_probs);
        float cl_eq = cubeless_equity(pre_roll_probs);

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["cubeless_equity"] = cl_eq;
        result["equity_nd"] = cd.equity_nd;
        result["equity_dt"] = cd.equity_dt;
        result["equity_dp"] = cd.equity_dp;
        result["should_double"] = cd.should_double;
        result["should_take"] = cd.should_take;
        result["optimal_equity"] = cd.optimal_equity;
        result["n_plies"] = n_plies;
        return result;
    }, "N-ply cube decision for a pre-roll position.\n"
       "Returns dict with probs, cubeless_equity, equity_nd/dt/dp, should_double, should_take, optimal_equity.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("n_plies") = 1,
       py::arg("purerace_weights") = "",
       py::arg("racing_weights") = "",
       py::arg("attacking_weights") = "",
       py::arg("priming_weights") = "",
       py::arg("anchoring_weights") = "",
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 1);

    // Cubeful rollout: run cubeless rollout, then apply Janowski to the mean probs
    m.def("cube_decision_rollout", [](const std::vector<int>& checkers,
                                       int cube_value, CubeOwner owner,
                                       const std::string& purerace_w,
                                       const std::string& racing_w,
                                       const std::string& attacking_w,
                                       const std::string& priming_w,
                                       const std::string& anchoring_w,
                                       int n_h_purerace,
                                       int n_h_racing,
                                       int n_h_attacking,
                                       int n_h_priming,
                                       int n_h_anchoring,
                                       int n_trials,
                                       int truncation_depth,
                                       int decision_ply,
                                       int vr_ply,
                                       int filter_max_moves,
                                       float filter_threshold,
                                       int n_threads,
                                       uint32_t seed,
                                       int late_ply,
                                       int late_threshold) {
        Board board = list_to_board(checkers);
        bool race = is_race(board);

        auto base = std::make_shared<GamePlanStrategy>(
            purerace_w, racing_w, attacking_w, priming_w, anchoring_w,
            n_h_purerace, n_h_racing, n_h_attacking, n_h_priming, n_h_anchoring);
        RolloutConfig config;
        config.n_trials = n_trials;
        config.truncation_depth = truncation_depth;
        config.decision_ply = decision_ply;
        config.vr_ply = vr_ply;
        config.filter = {filter_max_moves, filter_threshold};
        config.n_threads = n_threads;
        config.seed = seed;
        config.late_ply = late_ply;
        config.late_threshold = late_threshold;
        auto rollout = std::make_shared<RolloutStrategy>(base, config);

        // Run cubeless rollout on the flipped board (post-move semantics)
        // to get pre-roll cubeless probs
        Board flipped = flip(board);
        RolloutResult rr;
        {
            py::gil_scoped_release release;
            rr = rollout->rollout_position(flipped, flipped);
        }
        // The rollout gives us post-move probs from flipped perspective.
        // Invert to get pre-roll probs from player's perspective.
        auto pre_roll_probs = invert_probs(rr.mean_probs);
        float cl_eq = cubeless_equity(pre_roll_probs);

        // Apply Janowski to get cubeful equities
        CubeInfo ci{cube_value, owner};
        auto cd = cube_decision_from_probs(pre_roll_probs, ci, board, race);

        py::dict result;
        result["probs"] = pre_roll_probs;
        result["cubeless_equity"] = cl_eq;
        result["cubeless_se"] = rr.std_error;
        result["equity_nd"] = cd.equity_nd;
        result["equity_dt"] = cd.equity_dt;
        result["equity_dp"] = cd.equity_dp;
        result["should_double"] = cd.should_double;
        result["should_take"] = cd.should_take;
        result["optimal_equity"] = cd.optimal_equity;
        result["is_race"] = race;
        return result;
    }, "Cubeful rollout: cubeless rollout + Janowski conversion.\n"
       "Runs cubeless rollout to get mean probs, then applies Janowski for cube decision.",
       py::arg("checkers"),
       py::arg("cube_value") = 1,
       py::arg("owner") = CubeOwner::CENTERED,
       py::arg("purerace_weights") = "",
       py::arg("racing_weights") = "",
       py::arg("attacking_weights") = "",
       py::arg("priming_weights") = "",
       py::arg("anchoring_weights") = "",
       py::arg("n_hidden_purerace") = 200,
       py::arg("n_hidden_racing") = 400,
       py::arg("n_hidden_attacking") = 400,
       py::arg("n_hidden_priming") = 400,
       py::arg("n_hidden_anchoring") = 400,
       py::arg("n_trials") = 1296,
       py::arg("truncation_depth") = 0,
       py::arg("decision_ply") = 0,
       py::arg("vr_ply") = 0,
       py::arg("filter_max_moves") = 5,
       py::arg("filter_threshold") = 0.08f,
       py::arg("n_threads") = 0,
       py::arg("seed") = 42,
       py::arg("late_ply") = -1,
       py::arg("late_threshold") = 20);

    // ======================== Batch position evaluation ========================

    // Evaluate a batch of pre-roll positions in parallel.
    // Each position is a (board, cube_value, cube_owner) tuple.
    // Uses a MultiPlyStrategy (or GamePlanStrategy for 0-ply) shared across
    // threads. Individual N-ply evaluations are serial; parallelism is across
    // positions. Each thread gets its own thread-local position cache.
    //
    // Returns a list of dicts, one per position:
    //   probs: [5 floats] cubeless pre-roll probabilities
    //   cubeless_equity: float
    //   cubeful_equity: float (Janowski conversion with auto cube efficiency)
    //   equity_nd: float (No Double cubeful equity)
    //   equity_dt: float (Double/Take cubeful equity)
    //   equity_dp: float (Double/Pass = +1.0 for money)
    //   should_double: bool
    //   should_take: bool
    //   optimal_action: str ("No Double", "Double/Take", "Double/Pass")
    m.def("batch_evaluate_positions", [](
            py::list positions,
            std::shared_ptr<MultiPlyStrategy> strategy,
            int n_threads) {
        // Parse positions into C++ structs
        struct PosInput {
            Board board;
            int cube_value;
            CubeOwner owner;
        };
        const int n = static_cast<int>(py::len(positions));
        std::vector<PosInput> inputs(n);
        for (int i = 0; i < n; ++i) {
            py::tuple pos = positions[i].cast<py::tuple>();
            auto checkers = pos[0].cast<std::vector<int>>();
            inputs[i].board = list_to_board(checkers);
            inputs[i].cube_value = pos[1].cast<int>();
            inputs[i].owner = pos[2].cast<CubeOwner>();
        }

        // Result storage
        struct PosResult {
            std::array<float, NUM_OUTPUTS> probs;
            float cubeless_equity;
            float cubeful_equity;
            CubeDecision cube_decision;
        };
        std::vector<PosResult> results(n);

        {
            py::gil_scoped_release release;

            if (n_threads <= 0) {
                n_threads = static_cast<int>(std::thread::hardware_concurrency());
                if (n_threads <= 0) n_threads = 1;
            }
            n_threads = std::min(n_threads, n);

            auto evaluate_position = [&](int i) {
                const auto& inp = inputs[i];
                auto& out = results[i];

                // Pre-roll probs: flip → evaluate → invert
                Board flipped = flip(inp.board);
                bool race = is_race(inp.board);
                auto post_probs = strategy->evaluate_probs(flipped, race);
                out.probs = invert_probs(post_probs);
                out.cubeless_equity = cubeless_equity(out.probs);

                // Cubeful equity via Janowski
                float x = cube_efficiency(inp.board, race);
                out.cubeful_equity = cl2cf_money(out.probs, inp.owner, x);

                // Cube decision
                CubeInfo ci{inp.cube_value, inp.owner};
                out.cube_decision = cube_decision_0ply(out.probs, ci, x);
            };

            if (n_threads <= 1) {
                for (int i = 0; i < n; ++i) {
                    evaluate_position(i);
                }
            } else {
                // Parallel across positions using std::thread
                // (don't use multipy_parallel_for — those threads may be
                //  needed by parallel_evaluate inside the strategy)
                std::vector<std::thread> threads;
                threads.reserve(n_threads);
                std::atomic<int> next_pos{0};

                for (int t = 0; t < n_threads; ++t) {
                    threads.emplace_back([&]() {
                        while (true) {
                            int i = next_pos.fetch_add(1);
                            if (i >= n) break;
                            evaluate_position(i);
                        }
                    });
                }
                for (auto& th : threads) {
                    th.join();
                }
            }

            strategy->clear_cache();
        }

        // Convert results to Python
        py::list out;
        for (int i = 0; i < n; ++i) {
            const auto& r = results[i];
            const auto& cd = r.cube_decision;
            py::dict d;
            d["probs"] = r.probs;
            d["cubeless_equity"] = r.cubeless_equity;
            d["cubeful_equity"] = r.cubeful_equity;
            d["equity_nd"] = cd.equity_nd;
            d["equity_dt"] = cd.equity_dt;
            d["equity_dp"] = cd.equity_dp;
            d["should_double"] = cd.should_double;
            d["should_take"] = cd.should_take;
            const char* action = cd.should_double
                ? (cd.should_take ? "Double/Take" : "Double/Pass")
                : "No Double";
            d["optimal_action"] = action;
            out.append(d);
        }
        return out;
    }, "Evaluate a batch of pre-roll positions in parallel.\n"
       "positions: list of (board, cube_value, CubeOwner) tuples.\n"
       "strategy: MultiPlyStrategy (0-ply uses n_plies=0 wrapper).\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity, cube decision fields.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);

    // Overload that takes a GamePlanStrategy (0-ply) directly
    m.def("batch_evaluate_positions", [](
            py::list positions,
            GamePlanStrategy& strategy,
            int n_threads) {
        struct PosInput {
            Board board;
            int cube_value;
            CubeOwner owner;
        };
        const int n = static_cast<int>(py::len(positions));
        std::vector<PosInput> inputs(n);
        for (int i = 0; i < n; ++i) {
            py::tuple pos = positions[i].cast<py::tuple>();
            auto checkers = pos[0].cast<std::vector<int>>();
            inputs[i].board = list_to_board(checkers);
            inputs[i].cube_value = pos[1].cast<int>();
            inputs[i].owner = pos[2].cast<CubeOwner>();
        }

        struct PosResult {
            std::array<float, NUM_OUTPUTS> probs;
            float cubeless_equity;
            float cubeful_equity;
            CubeDecision cube_decision;
        };
        std::vector<PosResult> results(n);

        {
            py::gil_scoped_release release;

            if (n_threads <= 0) {
                n_threads = static_cast<int>(std::thread::hardware_concurrency());
                if (n_threads <= 0) n_threads = 1;
            }
            n_threads = std::min(n_threads, n);

            auto evaluate_position = [&](int i) {
                const auto& inp = inputs[i];
                auto& out = results[i];

                Board flipped = flip(inp.board);
                bool race = is_race(inp.board);
                auto post_probs = strategy.evaluate_probs(flipped, race);
                out.probs = invert_probs(post_probs);
                out.cubeless_equity = cubeless_equity(out.probs);

                float x = cube_efficiency(inp.board, race);
                out.cubeful_equity = cl2cf_money(out.probs, inp.owner, x);

                CubeInfo ci{inp.cube_value, inp.owner};
                out.cube_decision = cube_decision_0ply(out.probs, ci, x);
            };

            if (n_threads <= 1) {
                for (int i = 0; i < n; ++i) {
                    evaluate_position(i);
                }
            } else {
                std::vector<std::thread> threads;
                threads.reserve(n_threads);
                std::atomic<int> next_pos{0};

                for (int t = 0; t < n_threads; ++t) {
                    threads.emplace_back([&]() {
                        while (true) {
                            int i = next_pos.fetch_add(1);
                            if (i >= n) break;
                            evaluate_position(i);
                        }
                    });
                }
                for (auto& th : threads) {
                    th.join();
                }
            }
        }

        py::list out;
        for (int i = 0; i < n; ++i) {
            const auto& r = results[i];
            const auto& cd = r.cube_decision;
            py::dict d;
            d["probs"] = r.probs;
            d["cubeless_equity"] = r.cubeless_equity;
            d["cubeful_equity"] = r.cubeful_equity;
            d["equity_nd"] = cd.equity_nd;
            d["equity_dt"] = cd.equity_dt;
            d["equity_dp"] = cd.equity_dp;
            d["should_double"] = cd.should_double;
            d["should_take"] = cd.should_take;
            const char* action = cd.should_double
                ? (cd.should_take ? "Double/Take" : "Double/Pass")
                : "No Double";
            d["optimal_action"] = action;
            out.append(d);
        }
        return out;
    }, "Evaluate a batch of pre-roll positions at 0-ply in parallel.\n"
       "positions: list of (board, cube_value, CubeOwner) tuples.\n"
       "strategy: GamePlanStrategy (0-ply).\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity, cube decision fields.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);

    // ======================== Batch post-move position evaluation ========================

    // Evaluate a batch of post-move positions in parallel.
    // Each position is a (board, cube_owner) tuple.
    // "Post-move" means the board is from the perspective of the player who just moved,
    // right before the opponent rolls. The NN is evaluated directly (no flip/invert).
    // Returns a list of dicts: probs, cubeless_equity, cubeful_equity.
    m.def("batch_evaluate_post_move", [](
            py::list positions,
            GamePlanStrategy& strategy,
            int n_threads) {
        struct PosInput {
            Board board;
            CubeOwner owner;
        };
        const int n = static_cast<int>(py::len(positions));
        std::vector<PosInput> inputs(n);
        for (int i = 0; i < n; ++i) {
            py::tuple pos = positions[i].cast<py::tuple>();
            auto checkers = pos[0].cast<std::vector<int>>();
            inputs[i].board = list_to_board(checkers);
            inputs[i].owner = pos[1].cast<CubeOwner>();
        }

        struct PosResult {
            std::array<float, NUM_OUTPUTS> probs;
            float cubeless_equity;
            float cubeful_equity;
        };
        std::vector<PosResult> results(n);

        {
            py::gil_scoped_release release;

            if (n_threads <= 0) {
                n_threads = static_cast<int>(std::thread::hardware_concurrency());
                if (n_threads <= 0) n_threads = 1;
            }
            n_threads = std::min(n_threads, n);

            auto evaluate_position = [&](int i) {
                const auto& inp = inputs[i];
                auto& out = results[i];

                // Post-move: evaluate directly (NN returns mover's perspective probs)
                bool race = is_race(inp.board);
                out.probs = strategy.evaluate_probs(inp.board, race);
                out.cubeless_equity = NeuralNetwork::compute_equity(out.probs);

                // Cubeful equity via Janowski
                float x = cube_efficiency(inp.board, race);
                out.cubeful_equity = cl2cf_money(out.probs, inp.owner, x);
            };

            if (n_threads <= 1) {
                for (int i = 0; i < n; ++i) {
                    evaluate_position(i);
                }
            } else {
                std::vector<std::thread> threads;
                threads.reserve(n_threads);
                std::atomic<int> next_pos{0};

                for (int t = 0; t < n_threads; ++t) {
                    threads.emplace_back([&]() {
                        while (true) {
                            int i = next_pos.fetch_add(1);
                            if (i >= n) break;
                            evaluate_position(i);
                        }
                    });
                }
                for (auto& th : threads) {
                    th.join();
                }
            }
        }

        // Convert results to Python
        py::list out;
        for (int i = 0; i < n; ++i) {
            const auto& r = results[i];
            py::dict d;
            d["probs"] = r.probs;
            d["cubeless_equity"] = r.cubeless_equity;
            d["cubeful_equity"] = r.cubeful_equity;
            out.append(d);
        }
        return out;
    }, "Evaluate a batch of post-move positions at 0-ply in parallel.\n"
       "positions: list of (board, CubeOwner) tuples.\n"
       "strategy: GamePlanStrategy (0-ply).\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);

    // Overload for MultiPlyStrategy
    m.def("batch_evaluate_post_move", [](
            py::list positions,
            std::shared_ptr<MultiPlyStrategy> strategy,
            int n_threads) {
        struct PosInput {
            Board board;
            CubeOwner owner;
        };
        const int n = static_cast<int>(py::len(positions));
        std::vector<PosInput> inputs(n);
        for (int i = 0; i < n; ++i) {
            py::tuple pos = positions[i].cast<py::tuple>();
            auto checkers = pos[0].cast<std::vector<int>>();
            inputs[i].board = list_to_board(checkers);
            inputs[i].owner = pos[1].cast<CubeOwner>();
        }

        struct PosResult {
            std::array<float, NUM_OUTPUTS> probs;
            float cubeless_equity;
            float cubeful_equity;
        };
        std::vector<PosResult> results(n);

        {
            py::gil_scoped_release release;

            if (n_threads <= 0) {
                n_threads = static_cast<int>(std::thread::hardware_concurrency());
                if (n_threads <= 0) n_threads = 1;
            }
            n_threads = std::min(n_threads, n);

            auto evaluate_position = [&](int i) {
                const auto& inp = inputs[i];
                auto& out = results[i];

                bool race = is_race(inp.board);
                out.probs = strategy->evaluate_probs(inp.board, race);
                out.cubeless_equity = NeuralNetwork::compute_equity(out.probs);

                float x = cube_efficiency(inp.board, race);
                out.cubeful_equity = cl2cf_money(out.probs, inp.owner, x);
            };

            if (n_threads <= 1) {
                for (int i = 0; i < n; ++i) {
                    evaluate_position(i);
                }
            } else {
                std::vector<std::thread> threads;
                threads.reserve(n_threads);
                std::atomic<int> next_pos{0};

                for (int t = 0; t < n_threads; ++t) {
                    threads.emplace_back([&]() {
                        while (true) {
                            int i = next_pos.fetch_add(1);
                            if (i >= n) break;
                            evaluate_position(i);
                        }
                    });
                }
                for (auto& th : threads) {
                    th.join();
                }
            }

            strategy->clear_cache();
        }

        py::list out;
        for (int i = 0; i < n; ++i) {
            const auto& r = results[i];
            py::dict d;
            d["probs"] = r.probs;
            d["cubeless_equity"] = r.cubeless_equity;
            d["cubeful_equity"] = r.cubeful_equity;
            out.append(d);
        }
        return out;
    }, "Evaluate a batch of post-move positions at N-ply in parallel.\n"
       "positions: list of (board, CubeOwner) tuples.\n"
       "strategy: MultiPlyStrategy.\n"
       "Returns list of dicts with probs, cubeless_equity, cubeful_equity.",
       py::arg("positions"),
       py::arg("strategy"),
       py::arg("n_threads") = 0);
}
