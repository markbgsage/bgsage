#include "bgbot/training.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace bgbot {

// ======================== Helper Functions ========================

static std::array<float, NN_OUTPUTS> flip_outputs(
    const std::array<float, NN_OUTPUTS>& outputs)
{
    return {
        1.0f - outputs[0],
        outputs[3],
        outputs[4],
        outputs[1],
        outputs[2]
    };
}

static std::array<float, NN_OUTPUTS> terminal_targets_flipped(GameResult result)
{
    switch (result) {
        case GameResult::WIN_SINGLE:      return {0, 0, 0, 0, 0};
        case GameResult::WIN_GAMMON:      return {0, 0, 0, 1, 0};
        case GameResult::WIN_BACKGAMMON:  return {0, 0, 0, 1, 1};
        case GameResult::LOSS_SINGLE:     return {1, 0, 0, 0, 0};
        case GameResult::LOSS_GAMMON:     return {1, 1, 0, 0, 0};
        case GameResult::LOSS_BACKGAMMON: return {1, 1, 1, 0, 0};
        default:                          return {0.5f, 0, 0, 0, 0};
    }
}

static void save_history_csv(const std::vector<TrainingHistoryEntry>& history,
                             const std::string& filepath)
{
    std::ofstream f(filepath);
    f << "game,contact_score,elapsed_seconds\n";
    for (const auto& e : history) {
        f << e.game_number << ","
          << e.contact_score << ","
          << e.elapsed_seconds << "\n";
    }
}

// ======================== TD Training ========================

TDTrainResult td_train(const TDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    auto nn = std::make_shared<NeuralNetwork>(
        config.n_hidden, NN_INPUTS, config.weight_init_eps, config.seed);

    if (!config.resume_from.empty()) {
        if (!nn->load_weights(config.resume_from)) {
            throw std::runtime_error("Failed to load weights: " + config.resume_from);
        }
    }

    NNStrategy strat(nn);

    std::mt19937 rng(config.seed);
    std::uniform_int_distribution<int> die(1, 6);

    std::string weights_path = config.models_dir + "/" + config.model_name + ".weights";
    std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    TDTrainResult train_result;

    // Pre-allocate reusable buffers for the hot loop
    std::vector<Board> candidates;
    candidates.reserve(32);
    std::array<float, NN_INPUTS> flipped_inputs;
    std::array<float, NN_INPUTS> post_inputs;

    // Track game outcome types in the current interval
    int interval_singles = 0, interval_gammons = 0, interval_backgammons = 0;
    int interval_games = 0;

    // ======== Main training loop ========
    for (int game_idx = 0; game_idx < config.n_games; ++game_idx) {

        // ---- Periodic benchmark and save ----
        if (game_idx % config.benchmark_interval == 0) {
            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            double bm_score = -1.0;
            if (config.benchmark_scenarios != nullptr &&
                !config.benchmark_scenarios->empty())
            {
                BenchmarkResult bm = score_benchmarks(
                    strat, *config.benchmark_scenarios, 0);
                bm_score = bm.score();
            }

            train_result.history.push_back({game_idx, bm_score, elapsed});

            std::cout << "Game " << std::setw(6) << game_idx
                      << "  contact=" << std::fixed << std::setprecision(2) << bm_score;

            if (interval_games > 0) {
                std::cout << std::setprecision(1)
                          << "  out: s=" << (100.0 * interval_singles / interval_games)
                          << "% g=" << (100.0 * interval_gammons / interval_games)
                          << "% b=" << (100.0 * interval_backgammons / interval_games) << "%";
            }

            std::cout << "  time=" << std::setprecision(1) << elapsed << "s"
                      << std::endl;

            interval_singles = interval_gammons = interval_backgammons = interval_games = 0;

            nn->save_weights(weights_path);
            save_history_csv(train_result.history, history_path);
        }

        // ======== Play one game with TD learning ========

        Board board = STARTING_BOARD;

        int d1, d2;
        do {
            d1 = die(rng);
            d2 = die(rng);
        } while (d1 == d2);

        if (d2 > d1) {
            board = flip(board);
        }

        // Game loop — unified for all moves (opening and subsequent)
        bool first_move = true;
        for (;;) {
            if (!first_move) {
                d1 = die(rng);
                d2 = die(rng);
            }
            first_move = false;

            // Step A: pre-roll evaluation from opponent's perspective
            Board flipped = flip(board);
            flipped_inputs = compute_tesauro_inputs(flipped);
            nn->forward_with_gradients(flipped_inputs);

            // Step B: generate moves and pick best (reusing candidates vector)
            possible_boards(board, d1, d2, candidates);
            if (candidates.size() == 1) {
                board = candidates[0];
            } else {
                int idx = strat.best_move_index(candidates, board);
                board = candidates[idx];
            }

            // Step C: check game over and compute TD target
            GameResult result = check_game_over(board);

            if (result != GameResult::NOT_OVER) {
                nn->td_update(terminal_targets_flipped(result), config.alpha);

                // Track game outcome type
                interval_games++;
                switch (result) {
                    case GameResult::WIN_SINGLE:
                    case GameResult::LOSS_SINGLE:
                        interval_singles++; break;
                    case GameResult::WIN_GAMMON:
                    case GameResult::LOSS_GAMMON:
                        interval_gammons++; break;
                    case GameResult::WIN_BACKGAMMON:
                    case GameResult::LOSS_BACKGAMMON:
                        interval_backgammons++; break;
                    default: break;
                }

                break;  // game over
            }

            // Non-terminal: evaluate post-move board, flip for TD target
            post_inputs = compute_tesauro_inputs(board);
            auto post_outputs = nn->forward(post_inputs);
            nn->td_update(flip_outputs(post_outputs), config.alpha);

            // Step E: flip for next player
            board = flip(board);
        }
    }

    // ---- Final benchmark + save ----
    {
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();

        double bm_score = -1.0;
        if (config.benchmark_scenarios != nullptr &&
            !config.benchmark_scenarios->empty())
        {
            BenchmarkResult bm = score_benchmarks(
                strat, *config.benchmark_scenarios, 0);
            bm_score = bm.score();
        }

        train_result.history.push_back({config.n_games, bm_score, elapsed});

        std::cout << "Game " << std::setw(6) << config.n_games
                  << "  contact=" << std::fixed << std::setprecision(2) << bm_score;

        if (interval_games > 0) {
            std::cout << std::setprecision(1)
                      << "  out: s=" << (100.0 * interval_singles / interval_games)
                      << "% g=" << (100.0 * interval_gammons / interval_games)
                      << "% b=" << (100.0 * interval_backgammons / interval_games) << "%";
        }

        std::cout << "  time=" << std::setprecision(1) << elapsed << "s"
                  << "  (final)" << std::endl;

        nn->save_weights(weights_path);
        save_history_csv(train_result.history, history_path);
    }

    auto t_end = std::chrono::steady_clock::now();
    train_result.games_played = config.n_games;
    train_result.total_seconds = std::chrono::duration<double>(t_end - t_start).count();

    return train_result;
}

// ======================== Multi-Network TD Training ========================

TDTrainResult td_train_multi(const MultiTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    // Print training configuration
    std::cout << "=== Multi-Network TD Training ===" << std::endl;
    std::cout << "  Contact: " << config.n_hidden_contact << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Crashed: " << config.n_hidden_crashed << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Race:    " << config.n_hidden_race    << " hidden, " << TESAURO_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:   " << config.alpha << std::endl;
    std::cout << "  Games:   " << config.n_games << std::endl;
    std::cout << "  Seed:    " << config.seed << std::endl;
    std::cout << std::endl;

    // Create three networks with separate hidden sizes
    auto contact_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_contact, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed);
    auto crashed_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_crashed, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 3);
    auto race_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_race, TESAURO_INPUTS, config.weight_init_eps, config.seed + 7);

    // Optionally resume from saved weights
    if (!config.resume_contact.empty()) {
        if (!contact_nn->load_weights(config.resume_contact)) {
            throw std::runtime_error("Failed to load contact weights: " + config.resume_contact);
        }
        std::cout << "  Resumed contact from: " << config.resume_contact << std::endl;
    }
    if (!config.resume_crashed.empty()) {
        if (!crashed_nn->load_weights(config.resume_crashed)) {
            throw std::runtime_error("Failed to load crashed weights: " + config.resume_crashed);
        }
        std::cout << "  Resumed crashed from: " << config.resume_crashed << std::endl;
    }
    if (!config.resume_race.empty()) {
        if (!race_nn->load_weights(config.resume_race)) {
            throw std::runtime_error("Failed to load race weights: " + config.resume_race);
        }
        std::cout << "  Resumed race from: " << config.resume_race << std::endl;
    }

    // All three networks are separate
    MultiNNStrategy strat(contact_nn, crashed_nn, race_nn);

    std::mt19937 rng(config.seed);
    std::uniform_int_distribution<int> die(1, 6);

    std::string contact_path = config.models_dir + "/" + config.model_name + "_contact.weights";
    std::string crashed_path = config.models_dir + "/" + config.model_name + "_crashed.weights";
    std::string race_path    = config.models_dir + "/" + config.model_name + "_race.weights";
    std::string history_path = config.models_dir + "/" + config.model_name + ".history.csv";

    TDTrainResult train_result;

    // Pre-allocate reusable buffers
    std::vector<Board> candidates;
    candidates.reserve(32);
    std::array<float, EXTENDED_CONTACT_INPUTS> contact_buf;  // large enough for either encoding
    std::array<float, EXTENDED_CONTACT_INPUTS> post_contact_buf;

    // Helper: encode a board with the appropriate encoding for the given position type
    auto encode = [&](const Board& board, PosType ptype, float* out) {
        if (ptype == PosType::RACE) {
            auto inputs = compute_tesauro_inputs(board);
            std::copy(inputs.begin(), inputs.end(), out);
        } else {
            auto inputs = compute_extended_contact_inputs(board);
            std::copy(inputs.begin(), inputs.end(), out);
        }
    };

    // Helper: select the right NN for a position type
    auto select_nn = [&](PosType ptype) -> NeuralNetwork& {
        switch (ptype) {
            case PosType::RACE:    return *race_nn;
            case PosType::CRASHED: return *crashed_nn;
            case PosType::CONTACT:
            default:               return *contact_nn;
        }
    };

    // Track position type counts for diagnostics (cumulative)
    int contact_updates = 0, crashed_updates = 0, race_updates = 0;

    // Track game outcome types in the current interval
    int interval_singles = 0, interval_gammons = 0, interval_backgammons = 0;
    int interval_games = 0;

    // ======== Main training loop ========
    for (int game_idx = 0; game_idx < config.n_games; ++game_idx) {

        // ---- Periodic benchmark and save ----
        if (game_idx % config.benchmark_interval == 0) {
            auto t_now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();

            double bm_score = -1.0;
            if (config.contact_benchmark != nullptr &&
                !config.contact_benchmark->empty())
            {
                BenchmarkResult bm = score_benchmarks(
                    strat, *config.contact_benchmark, 0);
                bm_score = bm.score();
            }

            train_result.history.push_back({game_idx, bm_score, elapsed});

            // Compute update percentages
            int total_updates = contact_updates + crashed_updates + race_updates;
            double c_pct = total_updates > 0 ? 100.0 * contact_updates / total_updates : 0;
            double x_pct = total_updates > 0 ? 100.0 * crashed_updates / total_updates : 0;
            double r_pct = total_updates > 0 ? 100.0 * race_updates / total_updates : 0;

            std::cout << "Game " << std::setw(6) << game_idx
                      << "  contact=" << std::fixed << std::setprecision(2) << bm_score
                      << "  upd%: c=" << std::setprecision(0) << c_pct
                      << " x=" << x_pct
                      << " r=" << r_pct;

            // Print game outcome fractions (if we have interval data)
            if (interval_games > 0) {
                double s_frac = 1.0 * interval_singles / interval_games;
                double g_frac = 1.0 * interval_gammons / interval_games;
                double b_frac = 1.0 * interval_backgammons / interval_games;
                std::cout << std::setprecision(1)
                          << "  out: s=" << (100.0 * s_frac)
                          << "% g=" << (100.0 * g_frac)
                          << "% b=" << (100.0 * b_frac) << "%";
            }

            std::cout << "  time=" << std::setprecision(1) << elapsed << "s"
                      << std::endl;

            // Reset interval counters
            interval_singles = 0;
            interval_gammons = 0;
            interval_backgammons = 0;
            interval_games = 0;

            contact_nn->save_weights(contact_path);
            crashed_nn->save_weights(crashed_path);
            race_nn->save_weights(race_path);
            save_history_csv(train_result.history, history_path);
        }

        // ======== Play one game with TD learning ========

        Board board = STARTING_BOARD;

        int d1, d2;
        do {
            d1 = die(rng);
            d2 = die(rng);
        } while (d1 == d2);

        if (d2 > d1) {
            board = flip(board);
        }

        // Game loop
        bool first_move = true;
        for (;;) {
            if (!first_move) {
                d1 = die(rng);
                d2 = die(rng);
            }
            first_move = false;

            // Step A: Pre-move evaluation from opponent's perspective
            Board flipped = flip(board);
            PosType pre_ptype = classify_position(flipped);
            NeuralNetwork& pre_nn = select_nn(pre_ptype);
            encode(flipped, pre_ptype, contact_buf.data());
            pre_nn.forward_with_gradients(contact_buf.data());

            // Step B: Generate moves and pick best using MultiNNStrategy
            possible_boards(board, d1, d2, candidates);
            if (candidates.size() == 1) {
                board = candidates[0];
            } else {
                int idx = strat.best_move_index(candidates, board);
                board = candidates[idx];
            }

            // Step C: Check game over
            GameResult result = check_game_over(board);

            if (result != GameResult::NOT_OVER) {
                // Terminal: TD update on the pre-move network
                pre_nn.td_update(terminal_targets_flipped(result), config.alpha);
                switch (pre_ptype) {
                    case PosType::RACE:    race_updates++; break;
                    case PosType::CRASHED: crashed_updates++; break;
                    default:               contact_updates++; break;
                }

                // Track game outcome type
                interval_games++;
                switch (result) {
                    case GameResult::WIN_SINGLE:
                    case GameResult::LOSS_SINGLE:
                        interval_singles++; break;
                    case GameResult::WIN_GAMMON:
                    case GameResult::LOSS_GAMMON:
                        interval_gammons++; break;
                    case GameResult::WIN_BACKGAMMON:
                    case GameResult::LOSS_BACKGAMMON:
                        interval_backgammons++; break;
                    default: break;
                }

                break;
            }

            // Step D: Non-terminal TD update
            PosType post_ptype = classify_position(board);
            NeuralNetwork& post_nn = select_nn(post_ptype);
            encode(board, post_ptype, post_contact_buf.data());
            auto post_outputs = post_nn.forward(post_contact_buf.data());

            // TD update on the PRE-MOVE network
            pre_nn.td_update(flip_outputs(post_outputs), config.alpha);
            switch (pre_ptype) {
                case PosType::RACE:    race_updates++; break;
                case PosType::CRASHED: crashed_updates++; break;
                default:               contact_updates++; break;
            }

            // Step E: flip for next player
            board = flip(board);
        }
    }

    // ---- Final benchmark + save ----
    {
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();

        double bm_score = -1.0;
        if (config.contact_benchmark != nullptr &&
            !config.contact_benchmark->empty())
        {
            BenchmarkResult bm = score_benchmarks(
                strat, *config.contact_benchmark, 0);
            bm_score = bm.score();
        }

        train_result.history.push_back({config.n_games, bm_score, elapsed});

        // Compute update percentages
        int total_updates = contact_updates + crashed_updates + race_updates;
        double c_pct = total_updates > 0 ? 100.0 * contact_updates / total_updates : 0;
        double x_pct = total_updates > 0 ? 100.0 * crashed_updates / total_updates : 0;
        double r_pct = total_updates > 0 ? 100.0 * race_updates / total_updates : 0;

        std::cout << "Game " << std::setw(6) << config.n_games
                  << "  contact=" << std::fixed << std::setprecision(2) << bm_score
                  << "  upd%: c=" << std::setprecision(0) << c_pct
                  << " x=" << x_pct
                  << " r=" << r_pct;

        if (interval_games > 0) {
            double s_frac = 1.0 * interval_singles / interval_games;
            double g_frac = 1.0 * interval_gammons / interval_games;
            double b_frac = 1.0 * interval_backgammons / interval_games;
            std::cout << std::setprecision(1)
                      << "  out: s=" << (100.0 * s_frac)
                      << "% g=" << (100.0 * g_frac)
                      << "% b=" << (100.0 * b_frac) << "%";
        }

        std::cout << "  time=" << std::setprecision(1) << elapsed << "s"
                  << "  (final)" << std::endl;

        contact_nn->save_weights(contact_path);
        crashed_nn->save_weights(crashed_path);
        race_nn->save_weights(race_path);
        save_history_csv(train_result.history, history_path);
    }

    auto t_end = std::chrono::steady_clock::now();
    train_result.games_played = config.n_games;
    train_result.total_seconds = std::chrono::duration<double>(t_end - t_start).count();

    return train_result;
}

// ======================== Game Plan TD Training (5-NN) ========================

TDTrainResult td_train_gameplan(const GamePlanTDTrainConfig& config)
{
    auto t_start = std::chrono::steady_clock::now();

    // Reset cout formatting (may be left in std::fixed from prior calls)
    std::cout << std::defaultfloat;

    std::cout << "=== Game Plan TD Training (5-NN) ===" << std::endl;
    std::cout << "  PureRace:  " << config.n_hidden_purerace  << " hidden, " << TESAURO_INPUTS << " inputs" << std::endl;
    std::cout << "  Racing:    " << config.n_hidden_racing     << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Attacking: " << config.n_hidden_attacking  << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Priming:   " << config.n_hidden_priming    << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Anchoring: " << config.n_hidden_anchoring  << " hidden, " << EXTENDED_CONTACT_INPUTS << " inputs" << std::endl;
    std::cout << "  Alpha:     " << config.alpha << std::endl;
    std::cout << "  Games:     " << config.n_games << std::endl;
    std::cout << "  Seed:      " << config.seed << std::endl;
    std::cout << std::endl;

    auto purerace_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_purerace, TESAURO_INPUTS, config.weight_init_eps, config.seed);
    auto racing_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_racing, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 1);
    auto attacking_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_attacking, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 2);
    auto priming_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_priming, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 3);
    auto anchoring_nn = std::make_shared<NeuralNetwork>(
        config.n_hidden_anchoring, EXTENDED_CONTACT_INPUTS, config.weight_init_eps, config.seed + 7);

    if (!config.resume_purerace.empty()) {
        if (!purerace_nn->load_weights(config.resume_purerace))
            throw std::runtime_error("Failed to load purerace weights: " + config.resume_purerace);
        std::cout << "  Resumed purerace from: " << config.resume_purerace << std::endl;
    }
    if (!config.resume_racing.empty()) {
        if (!racing_nn->load_weights(config.resume_racing))
            throw std::runtime_error("Failed to load racing weights: " + config.resume_racing);
        std::cout << "  Resumed racing from: " << config.resume_racing << std::endl;
    }
    if (!config.resume_attacking.empty()) {
        if (!attacking_nn->load_weights(config.resume_attacking))
            throw std::runtime_error("Failed to load attacking weights: " + config.resume_attacking);
        std::cout << "  Resumed attacking from: " << config.resume_attacking << std::endl;
    }
    if (!config.resume_priming.empty()) {
        if (!priming_nn->load_weights(config.resume_priming))
            throw std::runtime_error("Failed to load priming weights: " + config.resume_priming);
        std::cout << "  Resumed priming from: " << config.resume_priming << std::endl;
    }
    if (!config.resume_anchoring.empty()) {
        if (!anchoring_nn->load_weights(config.resume_anchoring))
            throw std::runtime_error("Failed to load anchoring weights: " + config.resume_anchoring);
        std::cout << "  Resumed anchoring from: " << config.resume_anchoring << std::endl;
    }

    GamePlanStrategy strat(purerace_nn, racing_nn, attacking_nn, priming_nn, anchoring_nn);

    std::mt19937 rng(config.seed);
    std::uniform_int_distribution<int> die(1, 6);

    std::string purerace_path  = config.models_dir + "/" + config.model_name + "_purerace.weights";
    std::string racing_path    = config.models_dir + "/" + config.model_name + "_racing.weights";
    std::string attacking_path = config.models_dir + "/" + config.model_name + "_attacking.weights";
    std::string priming_path   = config.models_dir + "/" + config.model_name + "_priming.weights";
    std::string anchoring_path = config.models_dir + "/" + config.model_name + "_anchoring.weights";
    std::string purerace_best  = config.models_dir + "/" + config.model_name + "_purerace.weights.best";
    std::string racing_best    = config.models_dir + "/" + config.model_name + "_racing.weights.best";
    std::string attacking_best = config.models_dir + "/" + config.model_name + "_attacking.weights.best";
    std::string priming_best   = config.models_dir + "/" + config.model_name + "_priming.weights.best";
    std::string anchoring_best = config.models_dir + "/" + config.model_name + "_anchoring.weights.best";
    std::string history_path   = config.models_dir + "/" + config.model_name + ".history.csv";

    double best_atk_score = 1e9;  // lower is better

    TDTrainResult train_result;

    std::vector<Board> candidates;
    candidates.reserve(32);
    std::array<float, EXTENDED_CONTACT_INPUTS> pre_buf;
    std::array<float, EXTENDED_CONTACT_INPUTS> post_buf;

    auto encode_gp = [&](const Board& board, GamePlan gp, float* out) {
        if (gp == GamePlan::PURERACE) {
            auto inputs = compute_tesauro_inputs(board);
            std::copy(inputs.begin(), inputs.end(), out);
        } else {
            // RACING, ATTACKING, PRIMING, ANCHORING all use extended inputs
            auto inputs = compute_extended_contact_inputs(board);
            std::copy(inputs.begin(), inputs.end(), out);
        }
    };

    auto select_nn_gp = [&](GamePlan gp) -> NeuralNetwork& {
        switch (gp) {
            case GamePlan::PURERACE:  return *purerace_nn;
            case GamePlan::RACING:    return *racing_nn;
            case GamePlan::ATTACKING: return *attacking_nn;
            case GamePlan::PRIMING:   return *priming_nn;
            case GamePlan::ANCHORING: return *anchoring_nn;
            default:                  return *attacking_nn;
        }
    };

    int purerace_updates = 0, racing_updates = 0, attacking_updates = 0, priming_updates = 0, anchoring_updates = 0;
    int interval_singles = 0, interval_gammons = 0, interval_backgammons = 0;
    int interval_games = 0;

    // Helper lambda for benchmark + print
    auto run_benchmark_and_print = [&](int game_idx, bool is_final) {
        auto t_now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();

        double pr_score = -1.0, atk_score = -1.0, prm_score = -1.0, anc_score = -1.0, race_score = -1.0;
        if (config.purerace_benchmark && !config.purerace_benchmark->empty()) {
            pr_score = score_benchmarks(strat, *config.purerace_benchmark, 0).score();
        }
        if (config.attacking_benchmark && !config.attacking_benchmark->empty()) {
            atk_score = score_benchmarks(strat, *config.attacking_benchmark, 0).score();
        }
        if (config.priming_benchmark && !config.priming_benchmark->empty()) {
            prm_score = score_benchmarks(strat, *config.priming_benchmark, 0).score();
        }
        if (config.anchoring_benchmark && !config.anchoring_benchmark->empty()) {
            anc_score = score_benchmarks(strat, *config.anchoring_benchmark, 0).score();
        }
        if (config.race_benchmark && !config.race_benchmark->empty()) {
            race_score = score_benchmarks(strat, *config.race_benchmark, 0).score();
        }

        train_result.history.push_back({game_idx, atk_score, elapsed});

        int total_upd = purerace_updates + racing_updates + attacking_updates + priming_updates + anchoring_updates;
        double pr_pct = total_upd > 0 ? 100.0 * purerace_updates / total_upd : 0;
        double r_pct = total_upd > 0 ? 100.0 * racing_updates / total_upd : 0;
        double a_pct = total_upd > 0 ? 100.0 * attacking_updates / total_upd : 0;
        double p_pct = total_upd > 0 ? 100.0 * priming_updates / total_upd : 0;
        double n_pct = total_upd > 0 ? 100.0 * anchoring_updates / total_upd : 0;

        std::cout << "Game " << std::setw(7) << game_idx
                  << std::fixed << std::setprecision(2)
                  << "  pr=" << pr_score
                  << "  atk=" << atk_score
                  << "  prm=" << prm_score
                  << "  anc=" << anc_score
                  << "  race=" << race_score
                  << std::setprecision(0)
                  << "  upd%: pr=" << pr_pct << " r=" << r_pct << " a=" << a_pct
                  << " p=" << p_pct << " n=" << n_pct;

        if (interval_games > 0) {
            std::cout << std::setprecision(1)
                      << "  out: s=" << (100.0 * interval_singles / interval_games)
                      << "% g=" << (100.0 * interval_gammons / interval_games)
                      << "% b=" << (100.0 * interval_backgammons / interval_games) << "%";
        }

        std::cout << "  time=" << std::setprecision(1) << elapsed << "s";
        if (is_final) std::cout << "  (final)";
        std::cout << std::endl;

        interval_singles = interval_gammons = interval_backgammons = interval_games = 0;

        purerace_nn->save_weights(purerace_path);
        racing_nn->save_weights(racing_path);
        attacking_nn->save_weights(attacking_path);
        priming_nn->save_weights(priming_path);
        anchoring_nn->save_weights(anchoring_path);

        // Save best weights based on attacking benchmark score
        if (atk_score >= 0 && atk_score < best_atk_score && game_idx > 0) {
            best_atk_score = atk_score;
            purerace_nn->save_weights(purerace_best);
            racing_nn->save_weights(racing_best);
            attacking_nn->save_weights(attacking_best);
            priming_nn->save_weights(priming_best);
            anchoring_nn->save_weights(anchoring_best);
            std::cout << "  ** New best (atk=" << atk_score << ")" << std::endl;
        }

        save_history_csv(train_result.history, history_path);
    };

    for (int game_idx = 0; game_idx < config.n_games; ++game_idx) {
        if (game_idx % config.benchmark_interval == 0) {
            run_benchmark_and_print(game_idx, false);
        }

        Board board = STARTING_BOARD;

        int d1, d2;
        do { d1 = die(rng); d2 = die(rng); } while (d1 == d2);
        if (d2 > d1) board = flip(board);

        bool first_move = true;
        for (;;) {
            if (!first_move) { d1 = die(rng); d2 = die(rng); }
            first_move = false;

            Board flipped = flip(board);
            GamePlan pre_gp = classify_game_plan(flipped);
            NeuralNetwork& pre_nn = select_nn_gp(pre_gp);
            encode_gp(flipped, pre_gp, pre_buf.data());
            pre_nn.forward_with_gradients(pre_buf.data());

            possible_boards(board, d1, d2, candidates);
            if (candidates.size() == 1) {
                board = candidates[0];
            } else {
                int idx = strat.best_move_index(candidates, board);
                board = candidates[idx];
            }

            GameResult result = check_game_over(board);

            if (result != GameResult::NOT_OVER) {
                pre_nn.td_update(terminal_targets_flipped(result), config.alpha);
                switch (pre_gp) {
                    case GamePlan::PURERACE:  purerace_updates++; break;
                    case GamePlan::RACING:    racing_updates++; break;
                    case GamePlan::ATTACKING: attacking_updates++; break;
                    case GamePlan::PRIMING:   priming_updates++; break;
                    case GamePlan::ANCHORING: anchoring_updates++; break;
                }

                interval_games++;
                switch (result) {
                    case GameResult::WIN_SINGLE:
                    case GameResult::LOSS_SINGLE:      interval_singles++; break;
                    case GameResult::WIN_GAMMON:
                    case GameResult::LOSS_GAMMON:       interval_gammons++; break;
                    case GameResult::WIN_BACKGAMMON:
                    case GameResult::LOSS_BACKGAMMON:   interval_backgammons++; break;
                    default: break;
                }
                break;
            }

            GamePlan post_gp = classify_game_plan(board);
            NeuralNetwork& post_nn = select_nn_gp(post_gp);
            encode_gp(board, post_gp, post_buf.data());
            auto post_outputs = post_nn.forward(post_buf.data());

            pre_nn.td_update(flip_outputs(post_outputs), config.alpha);
            switch (pre_gp) {
                case GamePlan::PURERACE:  purerace_updates++; break;
                case GamePlan::RACING:    racing_updates++; break;
                case GamePlan::ATTACKING: attacking_updates++; break;
                case GamePlan::PRIMING:   priming_updates++; break;
                case GamePlan::ANCHORING: anchoring_updates++; break;
            }

            board = flip(board);
        }
    }

    run_benchmark_and_print(config.n_games, true);

    auto t_end = std::chrono::steady_clock::now();
    train_result.games_played = config.n_games;
    train_result.total_seconds = std::chrono::duration<double>(t_end - t_start).count();

    return train_result;
}

} // namespace bgbot
