#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/cube.h"
#include "bgbot/encoding.h"
#include "bgbot/multipy.h"
#include "bgbot/neural_net.h"
#include "bgbot/rollout.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bgbot;

namespace {

constexpr int N_THREADS = 16;
constexpr const char* PRODUCTION_MODEL = "stage8";

struct ModelSpec {
    bool is_pair = false;
    std::vector<int> hidden_sizes;
    std::string pattern;
    std::vector<int> canonical_map;
};

const std::unordered_map<std::string, ModelSpec>& model_registry() {
    static const std::unordered_map<std::string, ModelSpec> registry = [] {
        std::unordered_map<std::string, ModelSpec> m;
        m.emplace("stage8", ModelSpec{
            true,
            std::vector<int>{100, 400, 400, 400, 400, 400, 400, 400, 400,
                             400, 400, 400, 400, 400, 400, 400, 400},
            "sl_s8_{plan}.weights.best",
            std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 12, 13, 14, 12, 12},
        });
        m.emplace("stage7", ModelSpec{
            true,
            std::vector<int>{100, 300, 300, 300, 300, 300, 300, 300, 300,
                             300, 300, 300, 300, 300, 300, 300, 300},
            "sl_s7_{plan}.weights.best",
            std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 12, 13, 14, 12, 12},
        });
        m.emplace("stage6", ModelSpec{
            false,
            std::vector<int>{100, 300, 300, 300, 300},
            "sl_s6_{plan}.weights.best",
            {},
        });
        m.emplace("stage5", ModelSpec{
            false,
            std::vector<int>{200, 400, 400, 400, 400},
            "sl_s5_{plan}.weights.best",
            {},
        });
        m.emplace("stage5small", ModelSpec{
            false,
            std::vector<int>{100, 200, 200, 200, 200},
            "sl_s5s_{plan}.weights.best",
            {},
        });
        m.emplace("stage4", ModelSpec{
            false,
            std::vector<int>{120, 250, 250, 250, 250},
            "sl_s4_{plan}.weights.best",
            {},
        });
        m.emplace("stage3", ModelSpec{
            false,
            std::vector<int>{120, 250, 250, 250, 250},
            "sl_{plan}.weights.best",
            {},
        });
        return m;
    }();
    return registry;
}

Board make_board(std::initializer_list<int> values) {
    if (values.size() != 26) {
        throw std::runtime_error("board initializer must contain 26 values");
    }
    Board board{};
    std::size_t i = 0;
    for (int v : values) {
        board[i++] = v;
    }
    return board;
}

struct PositionSpec {
    std::string label;
    Board board;
    CubeInfo cube;
};

const std::vector<PositionSpec>& benchmark_positions() {
    static const std::vector<PositionSpec> positions = [] {
        std::vector<PositionSpec> v;

        {
            CubeInfo cube;
            cube.cube_value = 1;
            cube.owner = CubeOwner::CENTERED;
            cube.jacoby = true;
            cube.beaver = true;
            v.push_back({
                "Bearoff race, centered, money",
                make_board({0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-3,-1,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 1;
            cube.owner = CubeOwner::CENTERED;
            cube.match.away1 = 5;
            cube.match.away2 = 5;
            v.push_back({
                "Bearoff race, centered, 5pt match 0-0",
                make_board({0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,-1,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 2;
            cube.owner = CubeOwner::PLAYER;
            cube.jacoby = true;
            cube.beaver = true;
            v.push_back({
                "Complex contact, player cube=2, money",
                make_board({0,0,0,0,0,-3,4,2,3,0,0,0,-4,4,-1,-1,2,-2,0,-4,0,0,0,0,0,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 4;
            cube.owner = CubeOwner::PLAYER;
            cube.match.away1 = 7;
            cube.match.away2 = 5;
            v.push_back({
                "Complex contact, player cube=4, 7pt match 0-2",
                make_board({0,0,0,2,2,-2,4,0,2,0,0,0,-4,3,0,0,0,-2,0,-3,0,2,-2,-2,0,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 1;
            cube.owner = CubeOwner::CENTERED;
            cube.jacoby = true;
            cube.beaver = true;
            v.push_back({
                "Mixed contact, centered, money",
                make_board({0,4,1,0,0,3,3,0,1,2,0,0,0,0,1,-2,0,-3,0,-6,-2,0,-1,-1,0,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 1;
            cube.owner = CubeOwner::CENTERED;
            cube.jacoby = true;
            cube.beaver = true;
            v.push_back({
                "Late race, centered, money",
                make_board({1,2,2,2,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,-3,-4,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 2;
            cube.owner = CubeOwner::PLAYER;
            cube.match.away1 = 4;
            cube.match.away2 = 3;
            v.push_back({
                "Contact, player cube=2, 4pt match 0-1",
                make_board({0,0,2,2,2,3,3,0,1,0,0,0,-4,0,0,0,0,-3,0,-3,1,-2,-2,-1,0,0}),
                cube,
            });
        }
        {
            CubeInfo cube;
            cube.cube_value = 2;
            cube.owner = CubeOwner::PLAYER;
            cube.jacoby = true;
            cube.beaver = true;
            v.push_back({
                "Contact, player cube=2, money",
                make_board({0,0,0,0,0,0,4,1,3,0,0,0,-4,3,-2,-1,0,0,0,-3,-2,-3,2,0,2,0}),
                cube,
            });
        }

        return v;
    }();
    return positions;
}

struct TruncatedPreset {
    int n_trials;
    int truncation_depth;
    int decision_ply;
    int late_ply;
    int late_threshold;
};

const std::unordered_map<std::string, TruncatedPreset>& truncated_presets() {
    static const std::unordered_map<std::string, TruncatedPreset> presets = {
        {"1T", {42, 5, 1, -1, 20}},
        {"2T", {360, 7, 2, 1, 2}},
        {"3T", {360, 5, 3, 2, 2}},
    };
    return presets;
}

bool is_supported_level(const std::string& level) {
    return level == "1ply" || level == "2ply" || level == "3ply" ||
           level == "4ply" || level == "1T" || level == "2T" || level == "3T";
}

std::string optimal_action(bool should_double, bool should_take, bool is_beaver) {
    if (!should_double) {
        return "No Double";
    }
    if (is_beaver) {
        return "Double/Beaver";
    }
    return should_take ? "Double/Take" : "Double/Pass";
}

bool is_repo_root(const std::filesystem::path& path) {
    return std::filesystem::exists(path / "models") &&
           std::filesystem::exists(path / "data") &&
           std::filesystem::exists(path / "cpp");
}

std::optional<std::filesystem::path> search_up_for_repo_root(std::filesystem::path start) {
    if (start.empty()) {
        return std::nullopt;
    }
    start = std::filesystem::weakly_canonical(start);
    for (auto cur = start; !cur.empty(); cur = cur.parent_path()) {
        if (is_repo_root(cur)) {
            return cur;
        }
        if (cur == cur.root_path()) {
            break;
        }
    }
    return std::nullopt;
}

std::filesystem::path find_repo_root(const char* argv0, const std::optional<std::filesystem::path>& override_root) {
    if (override_root.has_value()) {
        return std::filesystem::weakly_canonical(*override_root);
    }

    std::vector<std::filesystem::path> starts = {
        std::filesystem::current_path(),
        std::filesystem::absolute(std::filesystem::path(argv0)).parent_path(),
    };

    for (const auto& start : starts) {
        auto found = search_up_for_repo_root(start);
        if (found.has_value()) {
            return *found;
        }
    }

    throw std::runtime_error("unable to locate repo root; pass --repo-root");
}

std::string replace_plan_token(std::string pattern, const std::string& plan) {
    const std::string token = "{plan}";
    auto pos = pattern.find(token);
    if (pos == std::string::npos) {
        return pattern;
    }
    pattern.replace(pos, token.size(), plan);
    return pattern;
}

std::shared_ptr<Strategy> load_base_strategy(
    const std::string& model_name,
    const std::filesystem::path& models_dir)
{
    const auto& registry = model_registry();
    const auto it = registry.find(model_name);
    if (it == registry.end()) {
        throw std::runtime_error("unknown model: " + model_name);
    }
    const ModelSpec& spec = it->second;

    if (spec.is_pair) {
        const auto& pair_names = game_plan_pair_names();
        std::vector<int> canonical_map = spec.canonical_map;
        if (canonical_map.empty()) {
            canonical_map.resize(pair_names.size());
            for (std::size_t i = 0; i < canonical_map.size(); ++i) {
                canonical_map[i] = static_cast<int>(i);
            }
        }
        std::vector<std::string> paths;
        paths.reserve(pair_names.size());
        for (std::size_t i = 0; i < pair_names.size(); ++i) {
            const int canonical = canonical_map.at(i);
            const auto filename = replace_plan_token(spec.pattern, pair_names.at(canonical));
            const auto full_path = models_dir / filename;
            if (!std::filesystem::exists(full_path)) {
                throw std::runtime_error("missing weight file: " + full_path.string());
            }
            paths.push_back(full_path.string());
        }
        return std::make_shared<GamePlanPairStrategy>(paths, spec.hidden_sizes);
    }

    const std::array<std::string, 5> plans = {
        "purerace", "racing", "attacking", "priming", "anchoring"
    };
    std::array<std::string, 5> paths{};
    for (std::size_t i = 0; i < plans.size(); ++i) {
        const auto full_path = models_dir / replace_plan_token(spec.pattern, plans[i]);
        if (!std::filesystem::exists(full_path)) {
            throw std::runtime_error("missing weight file: " + full_path.string());
        }
        paths[i] = full_path.string();
    }
    return std::make_shared<GamePlanStrategy>(
        paths[0], paths[1], paths[2], paths[3], paths[4],
        spec.hidden_sizes.at(0), spec.hidden_sizes.at(1), spec.hidden_sizes.at(2),
        spec.hidden_sizes.at(3), spec.hidden_sizes.at(4));
}

struct CubeBenchmarkResult {
    std::array<float, NUM_OUTPUTS> probs{};
    std::optional<std::array<float, NUM_OUTPUTS>> prob_std_errors;
    float cubeless_equity = 0.0f;
    float equity_nd = 0.0f;
    float equity_dt = 0.0f;
    float equity_dp = 0.0f;
    bool should_double = false;
    bool should_take = false;
    float optimal_equity = 0.0f;
    bool is_beaver = false;
    std::optional<double> cubeless_se;
    std::optional<double> equity_nd_se;
    std::optional<double> equity_dt_se;
};

struct EvalContext {
    std::string level;
    std::shared_ptr<Strategy> base;
    std::shared_ptr<MultiPlyStrategy> nply;
    std::shared_ptr<RolloutStrategy> rollout;
    std::unique_ptr<BearoffDB> bearoff_db;
};

CubeBenchmarkResult evaluate_1ply(
    const EvalContext& ctx,
    const PositionSpec& pos)
{
    CubeBenchmarkResult out;
    if (ctx.bearoff_db && ctx.bearoff_db->is_bearoff(pos.board)) {
        out.probs = ctx.bearoff_db->lookup_probs(pos.board, false);
    } else {
        out.probs = invert_probs(ctx.base->evaluate_probs(flip(pos.board), pos.board));
    }

    out.cubeless_equity = cubeless_equity(out.probs);
    CubeDecision dec = cube_decision_1ply(out.probs, pos.cube, pos.board, is_race(pos.board));
    out.equity_nd = dec.equity_nd;
    out.equity_dt = dec.equity_dt;
    out.equity_dp = dec.equity_dp;
    out.should_double = dec.should_double;
    out.should_take = dec.should_take;
    out.optimal_equity = dec.optimal_equity;
    out.is_beaver = dec.is_beaver;
    return out;
}

CubeBenchmarkResult evaluate_nply(
    const EvalContext& ctx,
    const PositionSpec& pos,
    int n_plies)
{
    CubeBenchmarkResult out;
    CubeDecision dec = cube_decision_nply(
        pos.board, pos.cube, *ctx.base, n_plies, MoveFilters::TINY, N_THREADS);
    out.equity_nd = dec.equity_nd;
    out.equity_dt = dec.equity_dt;
    out.equity_dp = dec.equity_dp;
    out.should_double = dec.should_double;
    out.should_take = dec.should_take;
    out.optimal_equity = dec.optimal_equity;
    out.is_beaver = dec.is_beaver;

    if (ctx.bearoff_db && ctx.bearoff_db->is_bearoff(pos.board)) {
        out.probs = ctx.bearoff_db->lookup_probs(pos.board, false);
    } else {
        out.probs = invert_probs(ctx.nply->evaluate_probs(flip(pos.board), pos.board));
    }
    out.cubeless_equity = cubeless_equity(out.probs);
    ctx.nply->clear_cache();
    return out;
}

CubeBenchmarkResult evaluate_rollout(
    const EvalContext& ctx,
    const PositionSpec& pos)
{
    CubeBenchmarkResult out;
    auto cfr = ctx.rollout->cubeful_cube_decision(pos.board, pos.cube);

    out.probs = cfr.cubeless.mean_probs;
    out.prob_std_errors = cfr.cubeless.prob_std_errors;
    out.cubeless_equity = static_cast<float>(cfr.cubeless.equity);
    out.cubeless_se = cfr.cubeless.std_error;
    out.equity_nd = static_cast<float>(cfr.nd_equity);
    out.equity_nd_se = cfr.nd_se;

    if (ctx.bearoff_db && ctx.bearoff_db->is_bearoff(pos.board)) {
        out.probs = ctx.bearoff_db->lookup_probs(pos.board, false);
        out.cubeless_equity = cubeless_equity(out.probs);
    }

    if (!pos.cube.is_money()) {
        const int a1 = pos.cube.match.away1;
        const int a2 = pos.cube.match.away2;
        const int cv = pos.cube.cube_value;
        const bool craw = pos.cube.match.is_crawford;

        const float nd_m = static_cast<float>(cfr.nd_equity);
        const float dt_m = static_cast<float>(cfr.dt_equity);
        const float dp_m = dp_mwc(a1, a2, cv, craw);

        const bool auto_double = (!craw && a1 > 1 && a2 == 1);
        out.should_take = (dt_m <= dp_m);
        if (auto_double) {
            out.should_double = true;
        } else {
            out.should_double = (std::min(dt_m, dp_m) > nd_m);
        }

        out.equity_nd = mwc2eq(nd_m, a1, a2, cv, craw);
        out.equity_dt = mwc2eq(dt_m, a1, a2, cv, craw);
        out.equity_dp = mwc2eq(dp_m, a1, a2, cv, craw);
        out.optimal_equity = out.should_double ? std::min(out.equity_dt, out.equity_dp)
                                               : out.equity_nd;
    } else {
        out.equity_dp = 1.0f;
        const float actual_dt = static_cast<float>(cfr.dt_equity);
        out.is_beaver = (pos.cube.beaver && actual_dt < 0.0f);
        out.equity_dt = out.is_beaver ? 2.0f * actual_dt : actual_dt;
        out.equity_dt_se = cfr.dt_se;
        out.should_take = (out.equity_dt <= out.equity_dp);
        out.should_double = (std::min(out.equity_dt, out.equity_dp) > out.equity_nd);
        out.optimal_equity = out.should_double ? std::min(out.equity_dt, out.equity_dp)
                                               : out.equity_nd;
    }

    if (!out.equity_dt_se.has_value()) {
        out.equity_dt_se = cfr.dt_se;
    }
    return out;
}

CubeBenchmarkResult evaluate_position(const EvalContext& ctx, const PositionSpec& pos) {
    if (ctx.level == "1ply") {
        return evaluate_1ply(ctx, pos);
    }
    if (ctx.level == "2ply" || ctx.level == "3ply" || ctx.level == "4ply") {
        return evaluate_nply(ctx, pos, ctx.level[0] - '0');
    }
    return evaluate_rollout(ctx, pos);
}

void print_probs(const std::array<float, NUM_OUTPUTS>& probs) {
    std::cout << "    Probs: W=" << std::fixed << std::setprecision(4) << probs[0]
              << " GW=" << probs[1]
              << " BW=" << probs[2]
              << " GL=" << probs[3]
              << " BL=" << probs[4]
              << "\n";
}

void print_prob_std_errors(const std::array<float, NUM_OUTPUTS>& prob_ses) {
    std::cout << "    Prob SEs: W=" << std::fixed << std::setprecision(4) << prob_ses[0]
              << " GW=" << prob_ses[1]
              << " BW=" << prob_ses[2]
              << " GL=" << prob_ses[3]
              << " BL=" << prob_ses[4]
              << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0]
                      << " <1ply|2ply|3ply|4ply|1T|2T|3T> [--model NAME] [--repo-root PATH]\n";
            return 1;
        }

        std::string level = argv[1];
        if (!is_supported_level(level)) {
            throw std::runtime_error("unsupported level: " + level);
        }

        std::string model_name = PRODUCTION_MODEL;
        std::optional<std::filesystem::path> repo_root_override;
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--model") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--model requires a value");
                }
                model_name = argv[++i];
            } else if (arg == "--repo-root") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--repo-root requires a value");
                }
                repo_root_override = std::filesystem::path(argv[++i]);
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        init_escape_tables();

        const std::filesystem::path repo_root = find_repo_root(argv[0], repo_root_override);
        const std::filesystem::path models_dir = repo_root / "models";
        const std::filesystem::path data_dir = repo_root / "data";

        EvalContext ctx;
        ctx.level = level;
        ctx.base = load_base_strategy(model_name, models_dir);

        const auto bearoff_path = data_dir / "bearoff_1sided.db";
        if (std::filesystem::exists(bearoff_path)) {
            ctx.bearoff_db = std::make_unique<BearoffDB>();
            if (!ctx.bearoff_db->load(bearoff_path.string())) {
                throw std::runtime_error("failed to load bearoff db: " + bearoff_path.string());
            }
        }

        const bool is_rollout_level = !level.empty() && level.back() == 'T';
        if (is_rollout_level) {
            const auto preset_it = truncated_presets().find(level);
            const TruncatedPreset& preset = preset_it->second;

            RolloutConfig config;
            config.n_trials = preset.n_trials;
            config.truncation_depth = preset.truncation_depth;
            config.decision_ply = preset.decision_ply;
            config.filter = MoveFilters::TINY;
            config.n_threads = N_THREADS;
            config.seed = 42;
            config.late_ply = preset.late_ply;
            config.late_threshold = preset.late_threshold;
            config.enable_vr = true;
            config.parallelize_trials = true;

            ctx.rollout = std::make_shared<RolloutStrategy>(ctx.base, config);
            if (ctx.bearoff_db) {
                ctx.rollout->set_bearoff_db(ctx.bearoff_db.get());
            }
        } else if (level != "1ply") {
            const int n_plies = level[0] - '0';
            ctx.nply = std::make_shared<MultiPlyStrategy>(
                ctx.base, n_plies, MoveFilters::TINY, false, true, N_THREADS);
            if (ctx.bearoff_db) {
                ctx.nply->set_bearoff_db(ctx.bearoff_db.get());
            }
        }

        std::cout << "Cube Analysis Profiling Benchmark -- Level: " << level << "\n";
        std::cout << "Model: " << model_name << ", Threads: " << N_THREADS << "\n";
        std::cout << "======================================================================\n";

        double total_time = 0.0;
        for (std::size_t i = 0; i < benchmark_positions().size(); ++i) {
            const auto& pos = benchmark_positions()[i];
            const auto t0 = std::chrono::steady_clock::now();
            const CubeBenchmarkResult result = evaluate_position(ctx, pos);
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(t1 - t0).count();
            total_time += elapsed;

            std::cout << "\n  Position " << (i + 1) << ": " << pos.label << "\n";
            std::cout << "    Action: "
                      << optimal_action(result.should_double, result.should_take, result.is_beaver)
                      << "\n";

            std::cout << std::showpos;
            std::cout << "    ND=" << std::fixed << std::setprecision(4) << result.equity_nd
                      << "  DT=" << result.equity_dt;
            if (is_rollout_level) {
                std::vector<std::string> parts;
                if (result.equity_nd_se.has_value()) {
                    std::ostringstream ss;
                    ss << "ND SE=" << std::fixed << std::setprecision(4) << *result.equity_nd_se;
                    parts.push_back(ss.str());
                }
                if (result.equity_dt_se.has_value()) {
                    std::ostringstream ss;
                    ss << "DT SE=" << std::fixed << std::setprecision(4) << *result.equity_dt_se;
                    parts.push_back(ss.str());
                }
                if (result.cubeless_se.has_value()) {
                    std::ostringstream ss;
                    ss << "CL SE=" << std::fixed << std::setprecision(4) << *result.cubeless_se;
                    parts.push_back(ss.str());
                }
                if (!parts.empty()) {
                    std::cout << "  (";
                    for (std::size_t j = 0; j < parts.size(); ++j) {
                        if (j > 0) {
                            std::cout << ", ";
                        }
                        std::cout << parts[j];
                    }
                    std::cout << ")";
                }
            }
            std::cout << std::noshowpos;
            std::cout << "\n";

            print_probs(result.probs);
            if (is_rollout_level && result.prob_std_errors.has_value()) {
                print_prob_std_errors(*result.prob_std_errors);
            }
            std::cout << "    Time: " << std::fixed << std::setprecision(3) << elapsed << "s\n";
        }

        std::cout << "\n======================================================================\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(3) << total_time << "s\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
