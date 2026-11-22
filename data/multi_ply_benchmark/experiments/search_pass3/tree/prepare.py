# SPDX-License-Identifier: MPL-2.0
"""Recreate a private CPU search ablation from its recorded control."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
BASE = "82e63972379b77ab52ebafc716620e813ff7baa1"
CPP = "cpp/src/cube_eval.cpp"

CACHE = """
using GroupProbs = std::array<float, NUM_OUTPUTS>;

struct GroupBase {
    Board board{};
    uint64_t model = 0;
    int slot = -1;
    uint64_t serial = 0;
    std::vector<float> hidden, inputs;
    GroupProbs probs{};
};

struct GroupEndpoint {
    Board board{};
    uint64_t base_serial = 0;
    GroupProbs probs{};
};

struct GroupStore {
    std::array<GroupBase, 256> bases;
    std::vector<GroupEndpoint> endpoints;
    uint64_t next_serial = 0;
};

thread_local GroupStore group_store;
"""

GROUP = """
    thread_local std::vector<int> routes;
    routes.resize(n);
    for (int j = 0; j < n; ++j) {
        const Board& board = cands[idxs[j]];
        routes[j] = (pinned >= 0 && !is_race(board)) ? pinned : s.nn_index_for(board);
        if (routes[j] >= BLENDED_SENTINEL_BASE) {
            out[idxs[j]] = s.probs_with_nn(board, routes[j]);
            routes[j] = -1;
        }
    }
    const uint64_t model = s.eval_identity();
    for (int j = 0; j < n; ++j) {
        const int slot = routes[j];
        if (slot < 0) continue;
        const Board& base = BASE_BOARD;
        const uint64_t base_key = fp_mix(hash_board64(base), slot);
        auto& saved = group_store.bases[(base_key ^ (base_key >> 33)) & 255];
        const bool race_net = slot == 0;
        const NeuralNetwork& nn = *s.nn(slot);
        const bool board_delta = !std::is_same_v<S, GamePlanPairStrategy> ||
            nn.n_inputs() == (race_net ? TESAURO_INPUTS : EXTENDED_CONTACT_INPUTS);
        if (saved.slot != slot || saved.board != base || saved.model != model) {
            ++base_misses;
            saved.hidden.resize(nn.n_hidden());
            saved.inputs.resize(nn.n_inputs());
            const auto compute = [&](const auto& inp) {
                return nn.forward_save_base(inp.data(), saved.hidden.data(), saved.inputs.data());
            };
            saved.probs = race_net ? compute(compute_tesauro_inputs(base))
                                   : compute(compute_extended_contact_inputs(base));
            saved.board = base;
            saved.model = model;
            saved.slot = slot;
            saved.serial = ++group_store.next_serial;
        }
        ++base_requests;
        START_INDEX
        for (int k = FIRST_DELTA; k < n; ++k) {
            if (routes[k] != slot) continue;
            routes[k] = -1;
            const Board& cand = cands[idxs[k]];
            ++delta_requests;
            ENDPOINT_LOOKUP
            if (board_delta) {
                out[idxs[k]] = nn.forward_board_from_base(
                    cand, base, saved.hidden.data(), saved.inputs.data());
            } else {
                const auto compute = [&](const auto& inp) {
                    return nn.forward_from_base(inp.data(), saved.hidden.data(), saved.inputs.data());
                };
                out[idxs[k]] = race_net ? compute(compute_tesauro_inputs(cand))
                                        : compute(compute_extended_contact_inputs(cand));
            }
            ENDPOINT_INSERT
        }
    }
"""

LOOKUP = """
            if (group_store.endpoints.empty()) group_store.endpoints.resize(8192);
            const uint64_t hash = fp_mix(hash_board64(cand), saved.serial);
            auto& endpoint = group_store.endpoints[(hash ^ (hash >> 33)) & 8191];
            if (endpoint.base_serial == saved.serial && endpoint.board == cand) {
                out[idxs[k]] = endpoint.probs;
                ++delta_hits;
                continue;
            }
"""
INSERT = """
            endpoint.board = cand;
            endpoint.base_serial = saved.serial;
            endpoint.probs = out[idxs[k]];
"""


def prepare(arm, profile):
    s = subprocess.check_output(["git", "show", f"{BASE}:{CPP}"], cwd=ROOT).decode()
    if arm == "control":
        return s
    if arm == "node":
        return scoped_node(s)
    if arm == "partial":
        return partial_batches(scoped_node(s))
    anchor = arm.startswith("anchor")
    memo = arm.endswith("memo")
    i = s.index("template <class S>\nvoid eval_groups_pair")
    s = s[:i] + CACHE + s[i:]
    s = s.replace("const int* idxs, int n,\n", "const int* idxs, int n, const Board& parent,\n", 2)
    s = s.replace("cands, idxs, n, out, pinned", "cands, idxs, n, parent, out, pinned")
    s = s.replace("static_cast<int>(live_idx.size()),\n", "static_cast<int>(live_idx.size()), board,\n")
    i = s.index("    thread_local std::vector<int> nn_idx_of;")
    j = s.index("\n}\n\nvoid eval_each_impl", i)
    group = GROUP.replace("BASE_BOARD", "parent" if anchor else "cands[idxs[j]]")
    group = group.replace("START_INDEX", "" if anchor else "routes[j] = -1;\n        out[idxs[j]] = saved.probs;")
    group = group.replace("FIRST_DELTA", "j" if anchor else "j + 1")
    group = group.replace("ENDPOINT_LOOKUP", LOOKUP if memo else "").replace("ENDPOINT_INSERT", INSERT if memo else "")
    counts = ("base_requests", "base_misses", "delta_requests", "delta_hits")
    if profile:
        group = "    uint64_t " + ", ".join(x + " = 0" for x in counts) + ";\n" + group
        for k, name in enumerate(counts):
            group += f"    group_counts[{k}].fetch_add({name}, std::memory_order_relaxed);\n"
    else:
        for c in counts:
            group = group.replace(f"++{c};", "")
    s = s[:i] + group + s[j:]
    if profile:
        s = s.replace("thread_local GroupStore group_store;", """thread_local GroupStore group_store;
std::atomic<uint64_t> group_counts[9]{};
extern "C" __declspec(dllexport) void take_tree_counts(long long* out) {
    for (int i = 0; i < 9; ++i) out[i] = group_counts[i].exchange(0);
}""")
    return s


def scoped_node(s):
    start = s.index("// Grouped delta evaluation for pair-NN strategies.")
    end = s.index("template <class S>", start)
    s = s[:start] + """// A node may distribute its 21 rolls across workers. Each worker saves a
// dense neural base for the pre-roll board the first time it uses a network
// at this node. Its next roll on the same network applies only input deltas.
// An outer base survives recursive children because their depths differ.
struct DiceNode {
    const Board& board;
    int depth;
    uint64_t id;
};
std::atomic<uint64_t> next_dice_node{1};

struct DiceNetworkBase {
    std::vector<float> hidden, inputs;
    uint64_t node = 0;
};

""" + s[end:]
    s = s.replace("const int* idxs, int n,\n", "const int* idxs, int n, const DiceNode& node,\n", 2)
    s = s.replace("cands, idxs, n, out, pinned", "cands, idxs, n, node, out, pinned")
    s = s.replace("static_cast<int>(live_idx.size()),\n", "static_cast<int>(live_idx.size()), dice_node,\n")
    s = s.replace("    thread_local std::vector<float> saved_base, saved_inputs;", """    thread_local DiceNetworkBase scratch;
    thread_local std::vector<std::vector<DiceNetworkBase>> by_depth;
    if (node.id && static_cast<int>(by_depth.size()) <= node.depth)
        by_depth.resize(node.depth + 1);""")
    start = s.index("        if (static_cast<int>(saved_base.size())")
    stop = s.index("        const bool is_pr", start)
    s = s[:start] + """        DiceNetworkBase* slot = &scratch;
        if (node.id) {
            auto& nets = by_depth[node.depth];
            if (static_cast<int>(nets.size()) <= nidx) nets.resize(nidx + 1);
            slot = &nets[nidx];
        }
        auto& saved_base = slot->hidden;
        auto& saved_inputs = slot->inputs;
""" + s[stop:]
    start = s.index("        if (is_pr) {")
    stop = s.index("            if (done[k]", start)
    s = s[:start] + """        const Board& base = node.id ? node.board : cands[idxs[j]];
        if (!node.id || slot->node != node.id) {
            saved_base.resize(nh);
            saved_inputs.resize(ni);
            const auto save = [&](const auto& inputs) {
                return nn.forward_save_base(inputs.data(), saved_base.data(), saved_inputs.data());
            };
            const auto probs = is_pr ? save(compute_tesauro_inputs(base))
                                     : save(compute_extended_contact_inputs(base));
            if (!node.id) out[idxs[j]] = probs;
            slot->node = node.id;
        }
        if (!node.id) done[j] = 1;
        for (int k = node.id ? j : j + 1; k < n; ++k) {
""" + s[stop:]
    s = s.replace("cands[idxs[k]], cands[idxs[j]], saved_base.data()", "cands[idxs[k]], base, saved_base.data()")
    s = s.replace("    auto eval_roll = [&](int r, float* rollCf,", """    // Rollout-internal trees retain their original sibling delta base. A
    // unique node ID prevents any reuse across unrelated searches or mutable
    // evaluators, even when a later call uses the same board.
    const DiceNode dice_node{board, plies, ctx.deep_prefilter ? 0 :
        next_dice_node.fetch_add(1, std::memory_order_relaxed)};
    auto eval_roll = [&](int r, float* rollCf,""")
    return s


def partial_batches(s):
    # Both traversals use one neural delta base per pre-roll board. A raw
    # vector for a filtered survivor is thus identical to its unfiltered one.
    s = s.replace("        int count = 0;\n    };\n    struct Node", """        std::array<int, MAX_CANDIDATES> original;
        int count = 0;
        int total = 0;
        bool subset = false;
    };
    struct Node""")
    s = s.replace("    explicit CandidateBatchCache(int plies)",
        "    const bool share_survivors;\n    explicit CandidateBatchCache(int plies, bool share_survivors)")
    s = s.replace("        : node_count(plies == 4", "        : share_survivors(share_survivors), node_count(plies == 4")
    s = s.replace("    CandidateBatchCache batches(n_plies_);", """    const Strategy* model = base_.get();
    while (const auto* wrapped = dynamic_cast<const BearoffStrategy*>(model)) model = &wrapped->base();
    const bool share_survivors = dynamic_cast<const BackgameAwarePairStrategy*>(model) ||
                                 dynamic_cast<const GamePlanPairStrategy*>(model);
    CandidateBatchCache batches(n_plies_, share_survivors);""")
    s = s.replace("        if (prefilter && n_cand > pf_threshold) {\n            thread_local std::vector<std::pair<double, int>> filter_scores;",
                  "        thread_local std::vector<std::pair<double, int>> filter_scores;\n        if (prefilter && n_cand > pf_threshold) {")
    s = s.replace("            auto evaluate_batch = [&] {", """            using SavedRoll = CandidateBatchCache::Roll;
            auto evaluate_batch = [&](const SavedRoll* partial = nullptr) {
                int reused = 0;""")
    s = s.replace("                    GameResult gr = check_game_over((*eval_candidates)[c]);", """                    if constexpr (ShareBatches) {
                        if (partial && reused < partial->count && partial->original[reused] == c) {
                            cand_probs[c] = partial->probs[reused++];
                            continue;
                        }
                    }
                    GameResult gr = check_game_over((*eval_candidates)[c]);""")
    start = s.index("                // Equal pre-roll board and dice")
    stop = s.index("            } else {\n                evaluate_batch();", start)
    s = s[:start] + """                auto* saved = batch_node ? &batch_node->rolls[r] : nullptr;
                const bool full = eval_candidates == &candidates;
                const bool hit = saved && ctx.batches->reading && full && saved->total == n_cand;
                if (hit && !saved->subset && saved->count == n_eval) {
                    std::copy_n(saved->probs.begin(), n_eval, cand_probs.begin());
                } else {
                    evaluate_batch(hit && saved->subset ? saved : nullptr);
                    // A first-sibling delta or a custom evaluator may depend on
                    // the ordered group. Only a built-in pair with a node base
                    // shares surviving indices between differently ordered lists.
                    if (saved && !ctx.batches->reading && n_eval <= CandidateBatchCache::MAX_CANDIDATES &&
                        (full || (dice_node.id && ctx.batches->share_survivors))) {
                        saved->count = n_eval;
                        saved->total = n_cand;
                        saved->subset = !full;
                        if (full) {
                            std::copy_n(cand_probs.begin(), n_eval, saved->probs.begin());
                        } else {
                            // Record survivors in the generator's order, independently of
                            // PubEval rank. The reader scans them alongside its full list.
                            std::array<int, CandidateBatchCache::MAX_CANDIDATES> ranks;
                            for (int k = 0; k < n_eval; ++k) ranks[k] = k;
                            std::sort(ranks.begin(), ranks.begin() + n_eval,
                                [&](int a, int b) { return filter_scores[a].second < filter_scores[b].second; });
                            for (int k = 0; k < n_eval; ++k) {
                                saved->original[k] = filter_scores[ranks[k]].second;
                                saved->probs[k] = cand_probs[ranks[k]];
                            }
                        }
                    }
                }
""" + s[stop:]
    return s


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("source", type=Path)
    cli.add_argument("arm", choices=("control", "sibling", "sibling_memo", "anchor", "anchor_memo", "node", "partial"))
    cli.add_argument("--profile", action="store_true")
    args = cli.parse_args()
    path = (args.source / CPP).resolve()
    if not path.is_relative_to(ROOT / "build"):
        raise ValueError("Experiment edits must remain inside build/")
    path.write_text(prepare(args.arm, args.profile), encoding="utf8", newline="\n")
