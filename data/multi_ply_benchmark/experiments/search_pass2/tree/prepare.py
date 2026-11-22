# SPDX-License-Identifier: MPL-2.0
"""Generate isolated deterministic-tree ablations from the published control."""
import argparse
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = "33c4aa1f6c28d12b846e66e745eef6c2692c5052"
CPP = "cpp/src/cube_eval.cpp"


def change(s, old, new, n=1):
    if s.count(old) != n:
        raise ValueError(f"Unexpected number of occurrences of {old!r}: {s.count(old)}")
    return s.replace(old, new)


def share4(s):
    s = change(s, "// Local to one pair of 3-ply searches. There are only two move-generation\n// levels: the root and one child for each of its 21 rolls. Parallel workers\n// each own a distinct roll slot; the first traversal joins before any read.",
               "// Local to a pair of live and dead cube searches. The root is path zero;\n// roll r at path p leads to 21*p+r+1. Each path has a single writer and\n// the live traversal joins before the dead traversal reads.")
    s = change(s, "static constexpr int ROOT = 21;", "static constexpr int ROOT = 0;")
    s = change(s, "    std::array<Node, ROOT + 1> nodes;", """    explicit CandidateBatchCache(int plies) : nodes(plies == 4 ? 1 + 21 + 21 * 21 : 1 + 21) {}
    std::vector<Node> nodes;""")
    s = change(s, "child_parallel, r);", "child_parallel, 21 * roll_path + r + 1);", 2)
    s = change(s, "n_plies_ != 3 || full_depth_opponent_", "(n_plies_ != 3 && n_plies_ != 4) || full_depth_opponent_")
    return change(s, "std::make_unique<CandidateBatchCache>()", "std::make_unique<CandidateBatchCache>(n_plies_)")


def terminals(s):
    s = change(s, "    auto eval_roll = [&](int r, float* rollCf,", """    // One die can bear off at most one checker. The opponent cannot
    // finish during the mover's turn; the parent is already nonterminal.
    const int remaining_checkers = player_checkers_on_board(board);
    auto eval_roll = [&](int r, float* rollCf,""")
    s = change(s, "        const auto& roll = ALL_ROLLS_21[r];", """        const auto& roll = ALL_ROLLS_21[r];
        const bool can_finish = remaining_checkers <= (roll.d1 == roll.d2 ? 4 : 2);""")
    s = change(s, "GameResult gr = check_game_over(candidates[c]);", "GameResult gr = can_finish ? check_game_over(candidates[c]) : GameResult::NOT_OVER;")
    s = change(s, "chosen_terminal = (check_game_over(chosen) != GameResult::NOT_OVER);", "chosen_terminal = can_finish && check_game_over(chosen) != GameResult::NOT_OVER;", 2)
    start = "                for (int c = 0; c < n_eval; ++c) {\n                    GameResult gr = check_game_over((*eval_candidates)[c]);"
    s = change(s, start, """                if (!can_finish) {
                    live_idx.resize(n_eval);
                    std::iota(live_idx.begin(), live_idx.end(), 0);
                } else for (int c = 0; c < n_eval; ++c) {
                    GameResult gr = check_game_over((*eval_candidates)[c]);""")
    return change(s, "#include <limits>", "#include <limits>\n#include <numeric>")


def dynamic(s):
    return change(s, """        multipy_parallel_for(21, std::min(ctx.n_threads, 21), [&](int r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        });""", """        std::atomic<int> next_roll{0};
        multipy_parallel_run(std::min(ctx.n_threads, 21), [&] {
            for (int r = next_roll.fetch_add(1, std::memory_order_relaxed); r < 21;
                 r = next_roll.fetch_add(1, std::memory_order_relaxed)) {
                eval_roll(r, roll_cf[r], roll_probs[r]);
            }
        });""")


def winner(s):
    s = share4(s)
    s = change(s, """        std::array<Probs, MAX_CANDIDATES> probs;
        int count = 0;""", """        Board chosen;
        Probs post_probs;
        bool terminal;
        bool ready = false;""")
    s = change(s, "explicit CandidateBatchCache(int plies) : nodes(plies == 4 ? 1 + 21 + 21 * 21 : 1 + 21) {}", "explicit CandidateBatchCache(int plies) : node_count(plies == 4 ? 1 + 21 + 21 * 21 : 1 + 21) {}\n    const int node_count;")
    s = change(s, "        Node& node = nodes[path];", """        // A warm transposition-table hit at the live root needs no storage.
        if (!reading && path == ROOT) nodes.resize(node_count);
        if (nodes.empty()) return nullptr;
        Node& node = nodes[path];""")
    begin = s.index("        if (chosen_terminal) {")
    end = s.index("    };", begin)
    continuation = s[begin:end]
    continuation = change(continuation, """            if (n_eval == 1) {
                // Forced move: no pick batch ran — evaluate the leaf NN now.
                bool lrace = is_race(opp_pre_roll);
                chosen_post_probs =
                    pinned_post_probs(*ctx.strategy, chosen, lrace, ctx.pinned_nn);
            }
            leaf_from_post_probs(opp_pre_roll, aci, expanded_cci, false,
                                 chosen_post_probs, arCfLocal, &probsTmp);""", """            const auto post = chosen_post_probs ? *chosen_post_probs : pinned_post_probs(
                *ctx.strategy, chosen, is_race(opp_pre_roll), ctx.pinned_nn);
            leaf_from_post_probs(opp_pre_roll, aci, expanded_cci, false, post, arCfLocal, &probsTmp);""")
    s = s[:begin] + "        continue_chosen(chosen, chosen_terminal, n_eval > 1 ? &chosen_post_probs : nullptr);\n" + s[end:]
    s = change(s, "        candidates.clear();", """        auto continue_chosen = [&](const Board& chosen, bool chosen_terminal,
                                   const CandidateBatchCache::Probs* chosen_post_probs) {
""" + "".join("    " + line if line.strip() else line for line in continuation.splitlines(keepends=True)) + """        };

        if constexpr (ShareBatches) {
            if (ctx.batches->reading && batch_node && batch_node->rolls[r].ready) {
                const auto& cached = batch_node->rolls[r];
                continue_chosen(cached.chosen, cached.terminal, &cached.post_probs);
                return;
            }
        }

        candidates.clear();""")
    begin = s.index("            if constexpr (ShareBatches) {\n                // Equal pre-roll")
    end = s.index("            // Cubeful pick", begin)
    s = s[:begin] + """            evaluate_batch();
            if constexpr (ShareBatches) {
                if (!ctx.batches->reading && batch_node && eval_candidates == &candidates &&
                    n_eval <= CandidateBatchCache::MAX_CANDIDATES) {
                    // The second traversal ranks this identical ordered batch under
                    // a centered 1/1 unlimited cube. Save its exact tied winner.
                    std::array<float, CandidateBatchCache::MAX_CANDIDATES> equity;
                    for (int i = 0; i < n_eval; ++i) equity[i] = cubeless_equity(cand_probs[i]);
                    const int best = pick_best_with_tiebreak(equity.data(), cand_probs.data(), n_eval);
                    auto& saved = batch_node->rolls[r];
                    saved.chosen = candidates[best];
                    saved.post_probs = cand_probs[best];
                    saved.terminal = check_game_over(saved.chosen) != GameResult::NOT_OVER;
                    saved.ready = true;
                }
            }

""" + s[end:]
    return s


def dead(s):
    s = change(s, "template<bool ShareBatches>\nvoid cubeful_eval_recursive(", "template<bool ShareBatches, bool ProbsOnly = false>\nvoid cubeful_eval_recursive(")
    s = change(s, "template<bool ShareBatches>\nstatic void cubeful_equity_nply_multi_impl(", "template<bool ShareBatches, bool ProbsOnly = false>\nstatic void cubeful_equity_nply_multi_impl(")
    s = change(s, "cubeful_eval_recursive<ShareBatches>(opp_", "cubeful_eval_recursive<ShareBatches, ProbsOnly>(opp_", 2)
    s = change(s, "cubeful_eval_recursive<ShareBatches>(board, aciCubePos, n_cubes", "cubeful_eval_recursive<ShareBatches, ProbsOnly>(board, aciCubePos, n_cubes")
    s = change(s, "cubeful_equity_nply_multi_impl<ShareBatches>(\n        flip(board), &dead", "cubeful_equity_nply_multi_impl<ShareBatches, true>(\n        flip(board), &dead")
    s = change(s, """    CubeInfo aci[EVAL_MAX_CCI * 2];
    make_cube_pos_local(aciCubePos, cci, fTop, aci, true);
    const int expanded_cci = 2 * cci;""", """    CubeInfo aci[ProbsOnly ? 1 : EVAL_MAX_CCI * 2];
    if constexpr (ProbsOnly) {
        // This specialization is entered only with a centered 1/1 unlimited
        // cube. Its sole valid state is invariant under a perspective flip.
        aci[0] = aciCubePos[0];
    } else {
        make_cube_pos_local(aciCubePos, cci, fTop, aci, true);
    }
    const int expanded_cci = ProbsOnly ? 1 : 2 * cci;""")
    start = s.index("    auto eval_roll = [&](int r, float* rollCf,")
    s = s[:start] + """    auto finish_leaf = [&](const Board& opponent, const std::array<float, NUM_OUTPUTS>& post,
                           float* values, std::array<float, NUM_OUTPUTS>* probs) {
        if constexpr (ProbsOnly) {
            *probs = invert_probs(post);
            clamp_probs_to_board(*probs, opponent);
        } else {
            leaf_from_post_probs(opponent, aci, expanded_cci, false, post, values, probs);
        }
    };
""" + s[start:]
    s = change(s, "leaf_from_post_probs(opp_board, aci, expanded_cci, false, post,\n                                     arCfLocal, &probsTmp);", "finish_leaf(opp_board, post, arCfLocal, &probsTmp);")
    s = change(s, "leaf_from_post_probs(opp_pre_roll, aci, expanded_cci, false,\n                                 chosen_post_probs, arCfLocal, &probsTmp);", "finish_leaf(opp_pre_roll, chosen_post_probs, arCfLocal, &probsTmp);")
    s = change(s, "for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];",
               "if constexpr (!ProbsOnly)\n                for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];", 2)
    s = change(s, "                const Board& cand = (*eval_candidates)[c];\n                bool crace", """                if constexpr (ProbsOnly) {
                    cf_vals[c] = cubeless_equity(cand_probs[c]);
                    continue;
                }
                const Board& cand = (*eval_candidates)[c];
                bool crace""")
    s = change(s, "            accumulate_terminal(tp, aci,", "            if constexpr (!ProbsOnly) accumulate_terminal(tp, aci,")
    s = change(s, "        for (int i = 0; i < expanded_cci; i++) arCf[i] += roll_cf[r][i];", "        if constexpr (!ProbsOnly)\n            for (int i = 0; i < expanded_cci; i++) arCf[i] += roll_cf[r][i];")
    s = change(s, "    for (int i = 0; i < expanded_cci; i++) {\n        if (is_money)", "    if constexpr (!ProbsOnly) for (int i = 0; i < expanded_cci; i++) {\n        if (is_money)")
    s = change(s, "    for (int i = 0; i < expanded_cci; i++) {\n        if (aci[i].cube_value > 0)",
               "    if constexpr (!ProbsOnly) for (int i = 0; i < expanded_cci; i++) {\n        if (aci[i].cube_value > 0)")
    return change(s, "    get_ecf3_local(arCubeful, cci, arCf, aci);\n    if (all_dead)", "    if constexpr (!ProbsOnly) get_ecf3_local(arCubeful, cci, arCf, aci);\n    if (all_dead)")


def counters(s):
    # Private, count-only build. The public timers use an uninstrumented DLL.
    s = change(s, "namespace bgbot {", """#ifdef _WIN32
#define TREE_EXPORT __declspec(dllexport)
#else
#define TREE_EXPORT __attribute__((visibility("default")))
#endif
static std::atomic<long long> counters[9]{};
static void count(int i, long long n = 1) { counters[i].fetch_add(n, std::memory_order_relaxed); }
extern "C" TREE_EXPORT void take_tree_counts(long long* values) {
    for (int i = 0; i < 9; ++i) values[i] = counters[i].exchange(0, std::memory_order_relaxed);
}
namespace bgbot {""")
    s = change(s, "        const int n_cand = static_cast<int>(candidates.size());", """        const int n_cand = static_cast<int>(candidates.size());
        count(roll.d1 == roll.d2 ? 0 : 1, n_cand);
        if (!can_finish && prefilter && n_cand > pf_threshold) count(4, n_cand);""")
    s = change(s, "            cand_probs.resize(n_eval);", """            cand_probs.resize(n_eval);
            count(2, n_eval);
            if constexpr (ProbsOnly) count(7, n_eval);""")
    s = change(s, """                if (!can_finish) {
                    live_idx.resize(n_eval);""", """                if (!can_finish) {
                    count(4, n_eval);
                    live_idx.resize(n_eval);""")
    s = change(s, """                    std::copy_n(saved->probs.begin(), n_eval, cand_probs.begin());""",
               """                    count(3, n_eval);
                    count(5);
                    std::copy_n(saved->probs.begin(), n_eval, cand_probs.begin());""")
    s = change(s, "        bool chosen_terminal = false;", "        bool chosen_terminal = false;\n        if (!can_finish) count(4);")
    s = change(s, "        if constexpr (ProbsOnly) {\n            *probs = invert_probs", "        if constexpr (ProbsOnly) {\n            count(6);\n            *probs = invert_probs")
    return change(s, "        if (reading) {\n            if (node.model", "        if (reading) {\n            count(8);\n            if (node.model")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("arms", nargs="*", choices=("share4", "winner", "terminals", "dead", "dynamic", "control"))
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    src = args.source.resolve()
    if not src.is_relative_to(ROOT / "build") or not (src / CPP).is_file():
        raise ValueError("Use an existing private source export under build/")
    s = subprocess.check_output(["git", "show", f"{BASE}:{CPP}"], cwd=ROOT).decode("utf-8")
    for name in args.arms:
        if name != "control":
            s = {"share4": share4, "winner": winner, "terminals": terminals, "dead": dead, "dynamic": dynamic}[name](s)
    if args.profile:
        s = counters(s)
    (src / CPP).write_text(s, encoding="utf-8")


if __name__ == "__main__":
    main()
