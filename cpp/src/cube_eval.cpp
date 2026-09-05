// SPDX-License-Identifier: MPL-2.0
// Copyright (C) 2026 Mark Higgins
//
// N-ply cubeful evaluation engine: the evaluate-all-and-decide recursion
// behind cube_decision_nply(_multi), cubeful_equity_nply(_multi),
// cubeful_probs_nply and cubeful_probs_and_equity_nply (declared in cube.h;
// the single-cube wrappers live in cube.cpp and compose the multi entries
// defined here).
//
// Key mechanics (see MULTI-PLY.md for the full specification):
//   * Interior move-picks evaluate all candidates of a roll through a batched
//     delta-evaluation kernel (encode base candidate once, sparse hidden-layer
//     updates per sibling) with per-candidate game-plan NN selection, then
//     pick by 1-ply cubeful equity against the primary cube state.
//   * At plies==2 internal nodes the leaf value of the picked move reuses the
//     pick batch's NN output (leaf = invert(NN(chosen))) � no leaf NN evals.
//   * The cubeless probs of the tree are accumulated through the same
//     traversal and can be returned via probs_out � one walk serves both
//     cubeful equities and cubeless probabilities.
//   * Memoization is a per-thread cache keyed by (board, plies, cci, fTop,
//     cube-state fingerprint); entries stay valid across calls and threads
//     never invalidate each other. clear_cubeful_eval_cache() resets it.
//   * The 21 rolls at the top tree levels run in parallel when the caller
//     passes n_threads > 1 (standalone analytics); rollout-internal calls run
//     serial because trial dispatch already saturates the cores.

#include "bgbot/cube.h"
#include "bgbot/neural_net.h"
#include "bgbot/bearoff.h"
#include "bgbot/board.h"
#include "bgbot/moves.h"
#include "bgbot/encoding.h"
#include "bgbot/match_equity.h"
#include "bgbot/pubeval.h"
#include "bgbot/multipy.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace bgbot {

namespace {

constexpr int EVAL_MAX_CCI = 64;   // matches MAX_CCI in cube.cpp

// ======================== Cube-state helpers ========================

float resolve_cube_x_local(
    const std::array<float, NUM_OUTPUTS>& probs,
    const CubeInfo& cube, const Board& board, bool race)
{
    if (cube.cube_x_override >= 0.0f) return cube.cube_x_override;
    auto [pp, op] = pip_counts(board);
    return cube_efficiency(probs, race, pp, op);
}

void make_cube_pos_local(
    const CubeInfo input[], int cci, bool fTop,
    CubeInfo output[], bool fInvert)
{
    for (int ici = 0, i = 0; ici < cci; ici++) {
        if (input[ici].cube_value > 0) {
            output[i] = fInvert ? flip_cube_perspective(input[ici]) : input[ici];
        } else {
            output[i].cube_value = -1;
        }
        i++;

        if (!fTop && input[ici].cube_value > 0 && can_double(input[ici])) {
            CubeInfo dt = input[ici];
            dt.cube_value = 2 * input[ici].cube_value;
            dt.owner = CubeOwner::OPPONENT;
            output[i] = fInvert ? flip_cube_perspective(dt) : dt;
        } else {
            output[i].cube_value = -1;
        }
        i++;
    }
}

void get_ecf3_local(
    float arCubeful[], int cci,
    const float arCf[], const CubeInfo aci[])
{
    for (int ici = 0, i = 0; ici < cci; ici++, i += 2) {
        if (aci[i + 1].cube_value > 0) {
            float rND = arCf[i];
            float rDT;
            bool is_money = aci[i].is_money();

            if (is_money) {
                rDT = 2.0f * arCf[i + 1];
            } else {
                rDT = arCf[i + 1];
            }

            if (is_money && aci[i].beaver && rDT < 0.0f) {
                rDT = 2.0f * rDT;
            }

            float rDP;
            if (is_money) {
                rDP = 1.0f;
            } else {
                rDP = dp_mwc(aci[i].match.away1, aci[i].match.away2,
                             aci[i].cube_value, aci[i].match.is_crawford);
            }

            if (rDT >= rND && rDP >= rND) {
                arCubeful[ici] = (rDT >= rDP) ? rDP : rDT;
            } else {
                arCubeful[ici] = rND;
            }
        } else {
            arCubeful[ici] = arCf[i];
        }
    }
}

// ======================== Per-thread cache ========================
//
// Open-addressing, epoch-free. The key incorporates a fingerprint of the
// cube-state array, so entries remain valid across calls and across cube
// contexts — no global invalidation needed (unlike the old cubeful cache,
// whose process-global epoch bump per top-level call destroyed every other
// thread's in-flight memoization under parallel trial dispatch).

uint64_t fp_mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 12) + (h >> 4);
    return h;
}

uint64_t cube_fp_one(uint64_t h, const CubeInfo& c) {
    h = fp_mix(h, static_cast<uint32_t>(c.cube_value));
    h = fp_mix(h, static_cast<int>(c.owner) + 3);
    h = fp_mix(h, static_cast<uint32_t>(c.match.away1) * 131u
                  + static_cast<uint32_t>(c.match.away2));
    h = fp_mix(h, c.match.is_crawford ? 7 : 11);
    h = fp_mix(h, c.jacoby ? 13 : 17);
    h = fp_mix(h, c.beaver ? 19 : 23);
    h = fp_mix(h, static_cast<uint32_t>(c.max_cube_value) + 29);
    h = fp_mix(h, static_cast<uint32_t>(
                      static_cast<int32_t>(c.cube_x_override * 4096.0f)) + 31);
    return h;
}

uint64_t hash_board64(const Board& b) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 26; ++i) {
        h ^= static_cast<uint32_t>(b[i] + 32);
        h *= 0x100000001b3ULL;
    }
    return h;
}

struct CubefulEvalCacheEntry {
    uint64_t key = 0;            // 0 = empty
    uint64_t cube_fp = 0;
    Board board{};
    int16_t plies = 0;
    int16_t cci = 0;
    uint8_t ftop = 0;
    float vals[16];
    float probs[NUM_OUTPUTS];
};

struct CubefulEvalCache {
    static constexpr size_t SIZE = 1 << 15;   // 32768 entries (~7 MB)
    static constexpr size_t MASK = SIZE - 1;
    static constexpr int MAX_PROBE = 4;
    std::vector<CubefulEvalCacheEntry> entries;

    CubefulEvalCache() : entries(SIZE) {}

    void clear() {
        for (auto& e : entries) e.key = 0;
    }

    bool get(uint64_t key, uint64_t fp, const Board& b, int plies, int cci,
             bool ftop, float* vals, std::array<float, NUM_OUTPUTS>* probs) const
    {
        size_t idx = static_cast<size_t>(key) & MASK;
        for (int p = 0; p < MAX_PROBE; ++p, idx = (idx + 1) & MASK) {
            const auto& e = entries[idx];
            if (e.key != key) continue;
            if (e.cube_fp != fp || e.plies != plies || e.cci != cci ||
                e.ftop != (ftop ? 1 : 0) || !(e.board == b)) continue;
            for (int i = 0; i < cci; ++i) vals[i] = e.vals[i];
            if (probs) {
                for (int k = 0; k < NUM_OUTPUTS; ++k) (*probs)[k] = e.probs[k];
            }
            return true;
        }
        return false;
    }

    void put(uint64_t key, uint64_t fp, const Board& b, int plies, int cci,
             bool ftop, const float* vals,
             const std::array<float, NUM_OUTPUTS>& probs)
    {
        size_t idx = static_cast<size_t>(key) & MASK;
        size_t target = idx;
        for (int p = 0; p < MAX_PROBE; ++p, idx = (idx + 1) & MASK) {
            const auto& e = entries[idx];
            if (e.key == 0 || e.key == key) { target = idx; break; }
        }
        auto& e = entries[target];
        e.key = key;
        e.cube_fp = fp;
        e.board = b;
        e.plies = static_cast<int16_t>(plies);
        e.cci = static_cast<int16_t>(cci);
        e.ftop = ftop ? 1 : 0;
        for (int i = 0; i < cci; ++i) e.vals[i] = vals[i];
        for (int k = 0; k < NUM_OUTPUTS; ++k) e.probs[k] = probs[k];
    }
};

thread_local CubefulEvalCache g_cubeful_eval_cache;

uint64_t make_cache_key(const Board& b, uint64_t fp, int plies, int cci, bool ftop) {
    uint64_t h = hash_board64(b);
    h = fp_mix(h, fp);
    h = fp_mix(h, static_cast<uint64_t>(plies) * 0x517cc1b727220a95ULL);
    h = fp_mix(h, static_cast<uint64_t>(cci) * 0x6c62272e07bb0142ULL);
    h = fp_mix(h, ftop ? 0x2545F4914F6CDD1DULL : 0x9E6C63D0876A9ULL);
    if (h == 0) h = 1;
    return h;
}

// Shared PubEval instance for the aggressive deep-node pre-filter.
const PubEval& static_pubeval() {
    static const PubEval pe(PubEval::WeightSource::TESAURO);
    return pe;
}

// ======================== Batched candidate evaluation ========================

// Grouped delta evaluation for pair-NN strategies. Candidates are classified
// per-candidate (matching the interior-pick convention) and grouped by NN
// index; within a group the first candidate is the delta base.
template <class S>
void eval_groups_pair(const S& s, const std::vector<Board>& cands,
                      const int* idxs, int n,
                      std::array<float, NUM_OUTPUTS>* out, int pinned = -1)
{
    thread_local std::vector<int> nn_idx_of;
    thread_local std::vector<char> done;
    thread_local std::vector<float> saved_base, saved_inputs;

    nn_idx_of.resize(n);
    done.assign(n, 0);
    for (int j = 0; j < n; ++j) {
        nn_idx_of[j] = (pinned >= 0 && !is_race(cands[idxs[j]]))
            ? pinned : s.nn_index_for(cands[idxs[j]]);
        if (nn_idx_of[j] >= BLENDED_SENTINEL_BASE) {
            // Blended backgame sentinel (21-NN hybrid): no shared delta-eval
            // base, evaluate directly via the strategy's blend-aware probs.
            out[idxs[j]] = s.probs_with_nn(cands[idxs[j]], nn_idx_of[j]);
            done[j] = 1;
        }
    }

    for (int j = 0; j < n; ++j) {
        if (done[j]) continue;
        const int nidx = nn_idx_of[j];
        const NeuralNetwork& nn = *s.nn(nidx);
        const int ni = nn.n_inputs();
        const int nh = nn.n_hidden();
        if (static_cast<int>(saved_base.size()) < nh) saved_base.resize(nh);
        if (static_cast<int>(saved_inputs.size()) < ni) saved_inputs.resize(ni);
        const bool is_pr = (nidx == 0);

        if (is_pr) {
            auto inp = compute_tesauro_inputs(cands[idxs[j]]);
            out[idxs[j]] = nn.forward_save_base(
                inp.data(), saved_base.data(), saved_inputs.data());
        } else {
            auto inp = compute_extended_contact_inputs(cands[idxs[j]]);
            out[idxs[j]] = nn.forward_save_base(
                inp.data(), saved_base.data(), saved_inputs.data());
        }
        done[j] = 1;

        for (int k = j + 1; k < n; ++k) {
            if (done[k] || nn_idx_of[k] != nidx) continue;
            if (is_pr) {
                auto inp = compute_tesauro_inputs(cands[idxs[k]]);
                out[idxs[k]] = nn.forward_from_base(
                    inp.data(), saved_base.data(), saved_inputs.data());
            } else {
                auto inp = compute_extended_contact_inputs(cands[idxs[k]]);
                out[idxs[k]] = nn.forward_from_base(
                    inp.data(), saved_base.data(), saved_inputs.data());
            }
            done[k] = 1;
        }
    }
}

void eval_each_impl(const Strategy& strat, const std::vector<Board>& cands,
                    const int* idxs, int n,
                    std::array<float, NUM_OUTPUTS>* out, int pinned = -1)
{
    if (n <= 0) return;

    if (const auto* bs = dynamic_cast<const BearoffStrategy*>(&strat)) {
        const BearoffDB& db = bs->db();
        thread_local std::vector<int> rest;
        rest.clear();
        rest.reserve(n);
        for (int j = 0; j < n; ++j) {
            const Board& c = cands[idxs[j]];
            if (db.is_bearoff(c)) {
                out[idxs[j]] = db.lookup_probs(c, /*post_move=*/true);
            } else {
                rest.push_back(idxs[j]);
            }
        }
        if (!rest.empty()) {
            // Note: rest is thread_local and the recursive call below does not
            // re-enter this BearoffStrategy branch (no double wrapping), so the
            // buffer is not clobbered.
            eval_each_impl(bs->base(), cands, rest.data(),
                           static_cast<int>(rest.size()), out, pinned);
        }
        return;
    }

    if (const auto* bg = dynamic_cast<const BackgameAwarePairStrategy*>(&strat)) {
        eval_groups_pair(*bg, cands, idxs, n, out, pinned);
        return;
    }
    if (const auto* gp = dynamic_cast<const GamePlanPairStrategy*>(&strat)) {
        eval_groups_pair(*gp, cands, idxs, n, out, pinned);
        return;
    }

    // Fallback: per-candidate loop with the interior-pick convention.
    for (int j = 0; j < n; ++j) {
        const Board& c = cands[idxs[j]];
        out[idxs[j]] = strat.evaluate_probs(c, is_race(c));
    }
}

// ======================== Core recursion ========================

struct EvalCtx {
    const Strategy* strategy;
    const Strategy* move_filter;
    int top_plies;       // plies at the entry node — enables the aggressive
                         // PubEval pre-filter at strictly deeper nodes only
    int n_threads = 1;   // top-level roll parallelism (1 = serial; rollout
                         // trial-internal calls are always serial)
    bool deep_prefilter = false;  // aggressive PubEval prune at deep nodes
                                  // (rollout-internal evaluations only)
    int pinned_nn = -1;   // root routing: the net every node uses (-1 = per node)
};

// Post-move probs of `board` under root routing: the pinned net when one is
// set and the board still has contact, else the strategy's own routing.
// Exact bearoff positions keep their DB values.
std::array<float, NUM_OUTPUTS> pinned_post_probs(
    const Strategy& strategy, const Board& board, bool race, int pinned)
{
    if (pinned >= 0 && !race) {
        // Unwrap EVERY bearoff layer: the analyzer wraps its strategy in a
        // BearoffStrategy and the bindings wrap that again, and a single
        // unwrap left the pair strategy unreachable — every dance / forced-
        // move leaf then fell back to the board's own routing.
        const Strategy* s = &strategy;
        while (const auto* bs = dynamic_cast<const BearoffStrategy*>(s)) {
            if (bs->db().is_bearoff(board)) return strategy.evaluate_probs(board, race);
            s = &bs->base();
        }
        if (const auto* bg = dynamic_cast<const BackgameAwarePairStrategy*>(s))
            return bg->probs_with_nn(board, pinned);
    }
    return strategy.evaluate_probs(board, race);
}

// Compute leaf values from precomputed post-move probs of the previous
// mover's chosen board. `board` is the leaf pre-roll position (leaf mover's
// POV); post_probs == strategy.evaluate_probs(flip(board), ...), i.e. the NN
// output for the previous mover's post-move board � the same NN evaluation
// the parent's pick batch already performed, so plies==2 subtrees evaluate
// their leaves for free.
void leaf_from_post_probs(
    const Board& board,
    const CubeInfo aci_in[], int cci, bool fTop,
    const std::array<float, NUM_OUTPUTS>& post_probs,
    float arCubeful[],
    std::array<float, NUM_OUTPUTS>* arProbsOut)
{
    bool race = is_race(board);
    auto pre_roll_probs = invert_probs(post_probs);
    clamp_probs_to_board(pre_roll_probs, board);
    float default_x = resolve_cube_x_local(
        pre_roll_probs,
        (cci > 0 && aci_in[0].cube_value > 0) ? aci_in[0] : CubeInfo{},
        board, race);

    CubeInfo aci[EVAL_MAX_CCI * 2];
    make_cube_pos_local(aci_in, cci, fTop, aci, false);

    float arCf[EVAL_MAX_CCI * 2];
    for (int i = 0; i < 2 * cci; i++) {
        if (aci[i].cube_value <= 0) {
            arCf[i] = 0.0f;
            continue;
        }
        if (cube_is_dead(aci[i])) {
            arCf[i] = cubeless_value(pre_roll_probs, aci[i]);
            continue;
        }
        float x = (aci[i].cube_x_override >= 0.0f)
                   ? aci[i].cube_x_override : default_x;
        if (aci[i].is_money()) {
            arCf[i] = cl2cf_money(pre_roll_probs, aci[i].owner, x,
                                   aci[i].jacoby_active());
        } else {
            arCf[i] = cl2cf_match(pre_roll_probs, aci[i], x);
        }
    }

    get_ecf3_local(arCubeful, cci, arCf, aci);
    if (arProbsOut) *arProbsOut = pre_roll_probs;
}

// Terminal accumulation for a chosen move that ends the game. `tp` is the
// terminal probs in opp-of-mover POV (already inverted). Writes weighted
// values into arCfLocal[0..expanded_cci-1].
void accumulate_terminal(
    const std::array<float, NUM_OUTPUTS>& tp,
    const CubeInfo aci[], int expanded_cci, int weight,
    float arCfLocal[])
{
    for (int i = 0; i < expanded_cci; i++) {
        if (aci[i].cube_value <= 0) {
            arCfLocal[i] = 0.0f;
            continue;
        }
        if (cube_is_dead(aci[i])) {
            arCfLocal[i] = weight * cubeless_value(tp, aci[i]);
            continue;
        }
        if (aci[i].is_money()) {
            if (aci[i].jacoby_active()) {
                arCfLocal[i] = weight * (2.0f * tp[0] - 1.0f);
            } else {
                arCfLocal[i] = weight * cubeless_equity(tp);
            }
        } else {
            arCfLocal[i] = weight * cubeless_mwc(tp,
                aci[i].match.away1, aci[i].match.away2,
                aci[i].cube_value, aci[i].match.is_crawford);
        }
    }
}

struct DiceRoll21 { int d1, d2, weight; };
constexpr std::array<DiceRoll21, 21> ALL_ROLLS_21 = {{
    {1,1,1}, {2,2,1}, {3,3,1}, {4,4,1}, {5,5,1}, {6,6,1},
    {1,2,2}, {1,3,2}, {1,4,2}, {1,5,2}, {1,6,2},
    {2,3,2}, {2,4,2}, {2,5,2}, {2,6,2},
    {3,4,2}, {3,5,2}, {3,6,2},
    {4,5,2}, {4,6,2},
    {5,6,2}
}};

void cubeful_eval_recursive(
    const Board& board,
    const CubeInfo aciCubePos[],
    int cci,
    const EvalCtx& ctx,
    int plies,
    bool fTop,
    float arCubeful[],
    std::array<float, NUM_OUTPUTS>* arProbsOut,
    bool allow_parallel = false)
{
    bool is_money = (cci > 0 && aciCubePos[0].cube_value > 0)
                    ? aciCubePos[0].is_money() : true;

    // Cache lookup — internal nodes only (entries always carry probs).
    // Leaf/terminal nodes are cheap in this engine (leaf NN evals are reused
    // from the parent's pick batch), so caching them would cost more in key
    // hashing than it saves.
    uint64_t fp = 0, key = 0;
    const bool can_cache = (plies >= 2 && cci > 0 && cci <= 16);
    if (can_cache) {
        // Seed with the evaluator identity: the per-thread cache (and the
        // SharedPosCache keys derived from it) is shared across all engine
        // callers on a thread, so entries from different models must never
        // collide. The identity is a process-unique monotonic id (never an
        // address — freed strategies' addresses get reused); equivalent
        // wrappers (BearoffStrategy over the same NN) resolve to the same
        // identity and share entries. The deep-prefilter context is folded
        // in too — the same node evaluated with and without the deep prune
        // has different values, so the two contexts must never serve each
        // other's entries.
        fp = 0xcbf29ce484222325ULL
             ^ (ctx.strategy->eval_identity() * 0x9e3779b97f4a7c15ULL)
             ^ (ctx.deep_prefilter ? 0x6a09e667f3bcc909ULL : 0)
             // Root routing changes every value below the root, so a node
             // pinned to one net must never serve an unpinned (or
             // differently pinned) evaluation of the same board.
             ^ (ctx.pinned_nn >= 0
                    ? (static_cast<uint64_t>(ctx.pinned_nn) + 1) * 0xbb67ae8584caa73bULL
                    : 0);
        for (int i = 0; i < cci; ++i) fp = cube_fp_one(fp, aciCubePos[i]);
        key = make_cache_key(board, fp, plies, cci, fTop);
        if (g_cubeful_eval_cache.get(key, fp, board, plies, cci, fTop,
                             arCubeful, arProbsOut)) {
            return;
        }
    }

    // Terminal check
    GameResult result = check_game_over(board);
    if (result != GameResult::NOT_OVER) {
        auto t_probs = terminal_probs(result);
        for (int ici = 0; ici < cci; ici++) {
            if (aciCubePos[ici].cube_value <= 0) {
                arCubeful[ici] = 0.0f;
                continue;
            }
            if (cube_is_dead(aciCubePos[ici])) {
                arCubeful[ici] = cubeless_value(t_probs, aciCubePos[ici]);
                continue;
            }
            if (aciCubePos[ici].is_money()) {
                if (aciCubePos[ici].jacoby_active()) {
                    arCubeful[ici] = 2.0f * t_probs[0] - 1.0f;
                } else {
                    arCubeful[ici] = cubeless_equity(t_probs);
                }
            } else {
                arCubeful[ici] = cubeless_mwc(t_probs,
                    aciCubePos[ici].match.away1, aciCubePos[ici].match.away2,
                    aciCubePos[ici].cube_value, aciCubePos[ici].match.is_crawford);
            }
        }
        if (arProbsOut) *arProbsOut = t_probs;
        if (can_cache) {
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, t_probs);
        }
        return;
    }

    // Leaf node (plies <= 1): single NN evaluation + Janowski.
    if (plies <= 1) {
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = pinned_post_probs(*ctx.strategy, flipped, race, ctx.pinned_nn);
        std::array<float, NUM_OUTPUTS> pre_out{};
        leaf_from_post_probs(board, aciCubePos, cci, fTop, post_probs,
                             arCubeful, &pre_out);
        if (arProbsOut) *arProbsOut = pre_out;
        if (can_cache) {
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, pre_out);
        }
        return;
    }

    // Dead-cube (cubeless) node sharing across rollout trial threads.
    // When every valid cube state at this node is dead, the node's equities
    // are fully determined by its cubeless probs (arCubeful[i] ==
    // cubeless_equity(probs) for valid slots, 0 for invalidated slots), so
    // the node is shared through the rollout's cross-thread SharedPosCache —
    // the same structure the cubeless N-ply recursion uses. The key is
    // salted so tree-produced entries can never be confused with entries
    // produced by MultiPlyStrategy::evaluate_probs_nply (same cache,
    // different evaluator => not bit-identical values).
    bool all_dead = can_cache && !fTop;
    if (all_dead) {
        for (int i = 0; i < cci; ++i) {
            if (aciCubePos[i].cube_value <= 0) continue;   // invalidated slot
            if (!cube_is_dead(aciCubePos[i])) { all_dead = false; break; }
        }
    }
    SharedPosCache* shared_cache =
        all_dead ? MultiPlyStrategy::get_shared_cache() : nullptr;
    SharedPosCache::Entry* shared_reservation = nullptr;
    if (shared_cache) {
        const uint64_t shared_key = key ^ 0x9e3779b97f4a7c15ULL;
        auto res = shared_cache->lookup_or_reserve(shared_key, 0, board);
        if (res.probs) {
            const auto p = *res.probs;
            for (int i = 0; i < cci; ++i) {
                arCubeful[i] = (aciCubePos[i].cube_value <= 0)
                    ? 0.0f : cubeless_value(p, aciCubePos[i]);
            }
            if (arProbsOut) *arProbsOut = p;
            g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop,
                                     arCubeful, p);
            return;
        }
        shared_reservation = res.reservation;
    }

    // Internal node: expand cube states, iterate 21 rolls.
    CubeInfo aci[EVAL_MAX_CCI * 2];
    make_cube_pos_local(aciCubePos, cci, fTop, aci, true);
    const int expanded_cci = 2 * cci;

    float arCf[EVAL_MAX_CCI * 2] = {};
    std::array<float, NUM_OUTPUTS> probsAccum = {};

    static constexpr int MOVE_FILTER_THRESHOLD = 16;
    static constexpr int MOVE_FILTER_KEEP = 15;
    // Deep-node pre-filter (PubEval): nodes strictly below the entry node's
    // ply are averaged over many rolls/trials, so pruning the PubEval-worst
    // candidates costs little accuracy while removing most of the per-node
    // eval work on big candidate sets (doubles rolls generate 30-90 moves).
    // keep=14 prunes only the deep tail, which is almost never the NN-best
    // candidate; tighter settings produce a measurable systematic negative
    // equity bias (weaker simulated play shrinks equities toward zero).
    // Overridable via BGBOT_CUBEFUL_DEEP_THRESHOLD / _KEEP for tuning.
    static const int DEEP_FILTER_THRESHOLD = []() {
        const char* v = std::getenv("BGBOT_CUBEFUL_DEEP_THRESHOLD");
        return v ? std::atoi(v) : 16;
    }();
    static const int DEEP_FILTER_KEEP = []() {
        const char* v = std::getenv("BGBOT_CUBEFUL_DEEP_KEEP");
        return v ? std::atoi(v) : 14;
    }();

    // Race nodes skip the aggressive filter: their candidate evaluations hit
    // the bearoff DB / cheap race NN, so PubEval scoring would cost more than
    // it saves.
    const bool deep_node = ctx.deep_prefilter && (plies < ctx.top_plies)
                           && !is_race(board);
    const Strategy* prefilter = ctx.move_filter;
    int pf_threshold = MOVE_FILTER_THRESHOLD;
    int pf_keep = MOVE_FILTER_KEEP;
    if (deep_node) {
        if (!prefilter) prefilter = &static_pubeval();
        pf_threshold = DEEP_FILTER_THRESHOLD;
        pf_keep = DEEP_FILTER_KEEP;
    }

    // Per-roll evaluation. Writes the WEIGHTED contribution of roll `r` into
    // rollCf[0..expanded_cci-1] and rollProbs (so serial and parallel
    // execution accumulate the identical floating-point sequence below).
    //
    // Child-level parallelism only pays when there are more threads than the
    // 21 top-level roll tasks; below that it just stacks BLOCKING
    // parallel_for waits inside pool workers (a nested caller sleeps until
    // its queued chunks run), which can exhaust the shared pool and
    // deadlock. Values are thread-count independent (fixed accumulation
    // order), so this is purely a scheduling decision.
    const bool child_parallel = allow_parallel && (plies - 1 > 2)
                                && ctx.n_threads > 21;
    auto eval_roll = [&](int r, float* rollCf,
                         std::array<float, NUM_OUTPUTS>& rollProbs) {
        thread_local std::vector<Board> candidates;
        thread_local std::vector<Board> filtered;
        thread_local std::vector<std::array<float, NUM_OUTPUTS>> cand_probs;
        thread_local std::vector<int> live_idx;

        const auto& roll = ALL_ROLLS_21[r];
        const float w = static_cast<float>(roll.weight);
        float arCfLocal[EVAL_MAX_CCI * 2];

        candidates.clear();
        if (candidates.capacity() < 32) candidates.reserve(32);
        possible_boards_unsorted(board, roll.d1, roll.d2, candidates);
        const int n_cand = static_cast<int>(candidates.size());

        if (n_cand == 0) {
            // Standing pat: flip board, recurse / leaf.
            Board opp_board = flip(board);
            std::array<float, NUM_OUTPUTS> probsTmp{};
            if (plies - 1 <= 1) {
                // Leaf on opp_board. Terminal impossible (board not terminal,
                // no move played). NN eval of flip(opp_board) == board.
                bool lrace = is_race(opp_board);
                auto post = pinned_post_probs(*ctx.strategy, flip(opp_board), lrace, ctx.pinned_nn);
                leaf_from_post_probs(opp_board, aci, expanded_cci, false, post,
                                     arCfLocal, &probsTmp);
            } else {
                cubeful_eval_recursive(opp_board, aci, expanded_cci, ctx, plies - 1,
                               false, arCfLocal, &probsTmp, child_parallel);
            }
            for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];
            for (int k = 0; k < NUM_OUTPUTS; k++)
                rollProbs[k] = w * probsTmp[k];
            return;
        }

        // Determine eval candidate set (cheap pre-filter for big sets).
        const std::vector<Board>* eval_candidates = &candidates;
        if (prefilter && n_cand > pf_threshold) {
            thread_local std::vector<std::pair<double, int>> filter_scores;
            filter_scores.clear();
            filter_scores.reserve(n_cand);
            bool pre_move_race = is_race(board);
            for (int c = 0; c < n_cand; c++) {
                GameResult gr = check_game_over(candidates[c]);
                double eq = (gr != GameResult::NOT_OVER)
                    ? 1e30
                    : prefilter->evaluate(candidates[c], pre_move_race);
                filter_scores.push_back({-eq, c});
            }
            int keep = std::min(pf_keep, n_cand);
            std::partial_sort(filter_scores.begin(), filter_scores.begin() + keep,
                              filter_scores.end());
            filtered.clear();
            filtered.reserve(keep);
            for (int k = 0; k < keep; k++) {
                filtered.push_back(candidates[filter_scores[k].second]);
            }
            eval_candidates = &filtered;
        }

        const int n_eval = static_cast<int>(eval_candidates->size());
        Board chosen;
        std::array<float, NUM_OUTPUTS> chosen_post_probs{};
        bool chosen_terminal = false;

        if (n_eval == 1) {
            chosen = (*eval_candidates)[0];
            chosen_terminal = (check_game_over(chosen) != GameResult::NOT_OVER);
            // chosen_post_probs unused on this path unless plies-1 <= 1 and
            // non-terminal — compute it lazily below.
        } else {
            // Batched per-candidate probs (terminals get terminal_probs;
            // others per-candidate NN classification — pick convention).
            cand_probs.resize(n_eval);
            live_idx.clear();
            live_idx.reserve(n_eval);
            for (int c = 0; c < n_eval; ++c) {
                GameResult gr = check_game_over((*eval_candidates)[c]);
                if (gr != GameResult::NOT_OVER) {
                    cand_probs[c] = terminal_probs(gr);
                } else {
                    live_idx.push_back(c);
                }
            }
            eval_each_impl(*ctx.strategy, *eval_candidates,
                           live_idx.data(), static_cast<int>(live_idx.size()),
                           cand_probs.data(), ctx.pinned_nn);

            // Cubeful pick vs the primary cube state: every interior pick
            // is match/cube-aware without forking the recursion per cube.
            // Ties on the cubeful value are broken by money cubeless equity
            // (pick_best_with_tiebreak in cube.h): at scores where a gammon
            // and a backgammon lose the match alike, EVERY move can score
            // identically, and an arbitrary pick here leaves checkers in the
            // opponent's home board and invents backgammon losses at the leaf.
            thread_local std::vector<float> cf_vals;
            cf_vals.resize(n_eval);
            for (int c = 0; c < n_eval; ++c) {
                const Board& cand = (*eval_candidates)[c];
                bool crace = is_race(cand);
                auto [pp, op] = pip_counts(cand);
                float cube_x = cube_efficiency(cand_probs[c], crace, pp, op);
                cf_vals[c] = cl2cf(cand_probs[c], aciCubePos[0], cube_x);
            }
            const int local_best =
                pick_best_with_tiebreak(cf_vals.data(), cand_probs.data(), n_eval);
            chosen = (*eval_candidates)[local_best];
            chosen_post_probs = cand_probs[local_best];
            chosen_terminal = (check_game_over(chosen) != GameResult::NOT_OVER);
        }

        if (chosen_terminal) {
            auto tp = invert_probs(terminal_probs(check_game_over(chosen)));
            accumulate_terminal(tp, aci, expanded_cci, roll.weight, rollCf);
            for (int k = 0; k < NUM_OUTPUTS; k++)
                rollProbs[k] = w * tp[k];
            return;
        }

        Board opp_pre_roll = flip(chosen);
        std::array<float, NUM_OUTPUTS> probsTmp{};
        if (plies - 1 <= 1) {
            if (n_eval == 1) {
                // Forced move: no pick batch ran — evaluate the leaf NN now.
                bool lrace = is_race(opp_pre_roll);
                chosen_post_probs =
                    pinned_post_probs(*ctx.strategy, chosen, lrace, ctx.pinned_nn);
            }
            leaf_from_post_probs(opp_pre_roll, aci, expanded_cci, false,
                                 chosen_post_probs, arCfLocal, &probsTmp);
        } else {
            cubeful_eval_recursive(opp_pre_roll, aci, expanded_cci, ctx, plies - 1,
                           false, arCfLocal, &probsTmp, child_parallel);
        }
        for (int i = 0; i < expanded_cci; i++) rollCf[i] = w * arCfLocal[i];
        for (int k = 0; k < NUM_OUTPUTS; k++)
            rollProbs[k] = w * probsTmp[k];
    };

    // Execute the 21 rolls (serial inside rollout trials; parallel at the
    // top tree levels of standalone N-ply analytics calls: enabled when
    // the entry passed n_threads > 1 and n_plies > 2, children only while
    // n_plies > 2, children only while plies-1 > 2). Accumulation happens in
    // fixed roll order either way, so thread count never changes the result.
    std::array<std::array<float, NUM_OUTPUTS>, 21> roll_probs{};
    float roll_cf[21][EVAL_MAX_CCI * 2];

    if (allow_parallel && ctx.n_threads > 1) {
        multipy_parallel_for(21, std::min(ctx.n_threads, 21), [&](int r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        });
    } else {
        for (int r = 0; r < 21; ++r) {
            eval_roll(r, roll_cf[r], roll_probs[r]);
        }
    }

    for (int r = 0; r < 21; ++r) {
        for (int i = 0; i < expanded_cci; i++) arCf[i] += roll_cf[r][i];
        for (int k = 0; k < NUM_OUTPUTS; k++) probsAccum[k] += roll_probs[r][k];
    }

    // Average over 36 and flip perspective back to current player.
    for (int i = 0; i < expanded_cci; i++) {
        if (is_money) {
            arCf[i] = -arCf[i] / 36.0f;
        } else {
            arCf[i] = 1.0f - arCf[i] / 36.0f;
        }
    }

    // Probs are accumulated unconditionally (cheap — leaves produce them as a
    // byproduct) so cache entries always carry valid probs regardless of
    // whether THIS caller requested them.
    std::array<float, NUM_OUTPUTS> probs_final{};
    for (int k = 0; k < NUM_OUTPUTS; k++)
        probsAccum[k] /= 36.0f;
    probs_final = invert_probs(probsAccum);
    if (arProbsOut) *arProbsOut = probs_final;

    // Un-invert cube states back to current player's perspective.
    for (int i = 0; i < expanded_cci; i++) {
        if (aci[i].cube_value > 0) {
            aci[i] = flip_cube_perspective(aci[i]);
        }
    }

    get_ecf3_local(arCubeful, cci, arCf, aci);
    if (all_dead) {
        // Derive dead-node equities from the node probs so the value is
        // bit-identical whether computed locally or served from the shared
        // cache (the two accumulation orders differ in the last ulp).
        for (int i = 0; i < cci; ++i) {
            arCubeful[i] = (aciCubePos[i].cube_value <= 0)
                ? 0.0f : cubeless_value(probs_final, aciCubePos[i]);
        }
    }
    if (can_cache) {
        g_cubeful_eval_cache.put(key, fp, board, plies, cci, fTop, arCubeful, probs_final);
    }
    if (shared_reservation) {
        shared_cache->publish(shared_reservation, probs_final);
    }
}

// 1-ply pre-roll cubeful evaluation: one NN evaluation shared across all
// cube states, then Janowski per state.
void eval_pre_roll_1ply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    float* out_raw,                  // money equity / match MWC per cube
    std::array<float, NUM_OUTPUTS>* probs_out,
    int pinned = -1)
{
    Board flipped = flip(board);
    bool race = is_race(board);

    GameResult result = check_game_over(flipped);
    std::array<float, NUM_OUTPUTS> pre_roll_probs;
    bool terminal = (result != GameResult::NOT_OVER);
    if (terminal) {
        pre_roll_probs = invert_probs(terminal_probs(result));
    } else {
        auto post_probs = pinned_post_probs(strategy, flipped, race, pinned);
        pre_roll_probs = invert_probs(post_probs);
        clamp_probs_to_board(pre_roll_probs, board);
    }

    for (int i = 0; i < n_cubes; ++i) {
        const CubeInfo& cube = cubes[i];
        if (terminal) {
            if (cube.is_money()) {
                out_raw[i] = cube.jacoby_active()
                    ? (2.0f * pre_roll_probs[0] - 1.0f)
                    : cubeless_equity(pre_roll_probs);
            } else {
                out_raw[i] = cubeless_mwc(pre_roll_probs,
                    cube.match.away1, cube.match.away2,
                    cube.cube_value, cube.match.is_crawford);
            }
            continue;
        }
        float x = resolve_cube_x_local(pre_roll_probs, cube, board, race);
        if (cube.is_money()) {
            out_raw[i] = cl2cf_money(pre_roll_probs, cube.owner, x,
                                      cube.jacoby_active());
        } else {
            out_raw[i] = cl2cf_match(pre_roll_probs, cube, x);
        }
    }
    if (probs_out) *probs_out = pre_roll_probs;
}

} // namespace

// ======================== Public API ========================

void clear_cubeful_eval_cache() {
    g_cubeful_eval_cache.clear();
}

uint64_t cube_state_fingerprint(const CubeInfo* cubes, int n) {
    uint64_t fp = 0xcbf29ce484222325ULL;
    for (int i = 0; i < n; ++i) fp = cube_fp_one(fp, cubes[i]);
    return fp;
}

void cubeful_equity_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    float* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool fTop,
    std::array<float, NUM_OUTPUTS>* probs_out,
    bool deep_prefilter,
    const Board* root_board)
{
    (void)filter;  // candidate selection inside the tree is single-pick
    if (n_cubes <= 0) return;
    if (n_cubes > EVAL_MAX_CCI) n_cubes = EVAL_MAX_CCI;
    const int pinned = root_board ? strategy.root_pin_for(*root_board) : -1;

    if (n_plies <= 1) {
        float raw[EVAL_MAX_CCI];
        eval_pre_roll_1ply_multi(board, cubes, n_cubes, strategy, raw, probs_out, pinned);
        for (int i = 0; i < n_cubes; ++i) {
            if (cubes[i].is_money()) {
                out[i] = raw[i];
            } else {
                out[i] = mwc2eq(raw[i], cubes[i].match.away1, cubes[i].match.away2,
                                cubes[i].cube_value, cubes[i].match.is_crawford);
            }
        }
        return;
    }

    CubeInfo aciCubePos[EVAL_MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) aciCubePos[i] = cubes[i];

    EvalCtx ctx{&strategy, move_filter, n_plies, n_threads, deep_prefilter};
    ctx.pinned_nn = pinned;
    const bool allow_parallel = (n_threads > 1 && n_plies > 2);
    float arCubeful[EVAL_MAX_CCI];
    cubeful_eval_recursive(board, aciCubePos, n_cubes, ctx, n_plies, fTop,
                   arCubeful, probs_out, allow_parallel);

    for (int i = 0; i < n_cubes; ++i) {
        if (cubes[i].is_money()) {
            out[i] = arCubeful[i];
        } else {
            out[i] = mwc2eq(arCubeful[i], cubes[i].match.away1, cubes[i].match.away2,
                            cubes[i].cube_value, cubes[i].match.is_crawford);
        }
    }
}

void cube_decision_nply_multi(
    const Board& board,
    const CubeInfo* cubes,
    int n_cubes,
    const Strategy& strategy,
    int n_plies,
    CubeDecision* out,
    const MoveFilter& filter,
    int n_threads,
    const Strategy* move_filter,
    bool deep_prefilter)
{
    (void)filter;  // candidate selection inside the tree is single-pick
    if (n_cubes <= 0) return;
    if (n_cubes > EVAL_MAX_CCI / 2) n_cubes = EVAL_MAX_CCI / 2;
    const int pinned = strategy.root_pin_for(board);  // a cube decision's root is the position itself

    if (n_plies <= 1) {
        // 1-ply path: pre-roll probs ONCE (shared), then Janowski per cube.
        Board flipped = flip(board);
        bool race = is_race(board);
        auto post_probs = pinned_post_probs(strategy, flipped, race, pinned);
        auto pre_roll_probs = invert_probs(post_probs);
        clamp_probs_to_board(pre_roll_probs, board);
        for (int i = 0; i < n_cubes; ++i) {
            float x = resolve_cube_x_local(pre_roll_probs, cubes[i], board, race);
            out[i] = cube_decision_1ply(pre_roll_probs, cubes[i], x);
        }
        return;
    }

    const int n_states = 2 * n_cubes;
    CubeInfo aciCubePos[EVAL_MAX_CCI];
    for (int i = 0; i < n_cubes; ++i) {
        aciCubePos[2 * i]     = cubes[i];
        aciCubePos[2 * i + 1] = cubes[i];
        aciCubePos[2 * i + 1].cube_value = 2 * cubes[i].cube_value;
        aciCubePos[2 * i + 1].owner      = CubeOwner::OPPONENT;
    }

    EvalCtx ctx{&strategy, move_filter, n_plies, n_threads, deep_prefilter};
    ctx.pinned_nn = pinned;
    const bool allow_parallel = (n_threads > 1 && n_plies > 2);
    float arCubeful[EVAL_MAX_CCI];
    cubeful_eval_recursive(board, aciCubePos, n_states, ctx, n_plies, /*fTop=*/true,
                   arCubeful, nullptr, allow_parallel);

    // Unpack per-branch decisions — identical to cube_decision_nply_multi.
    for (int i = 0; i < n_cubes; ++i) {
        const CubeInfo& cube = cubes[i];
        float nd_raw = arCubeful[2 * i];
        float dt_raw = arCubeful[2 * i + 1];
        bool is_money = cube.is_money();

        CubeDecision result = {};

        if (is_money) {
            result.equity_nd = nd_raw;
            float actual_dt = 2.0f * dt_raw;
            result.equity_dp = 1.0f;

            if (cube.beaver && actual_dt < 0.0f) {
                result.equity_dt = 2.0f * actual_dt;
                result.is_beaver = true;
            } else {
                result.equity_dt = actual_dt;
            }
        } else {
            int away1 = cube.match.away1;
            int away2 = cube.match.away2;
            int cv = cube.cube_value;
            bool craw = cube.match.is_crawford;
            float dp_m = dp_mwc(away1, away2, cv, craw);
            result.equity_nd = mwc2eq(nd_raw, away1, away2, cv, craw);
            result.equity_dt = mwc2eq(dt_raw, away1, away2, cv, craw);
            result.equity_dp = mwc2eq(dp_m,   away1, away2, cv, craw);
        }

        bool player_can_double = can_double(cube);
        bool auto_double = (!is_money && !cube.match.is_crawford &&
                            player_can_double &&
                            cube.match.away1 > 1 && cube.match.away2 == 1);

        if (!player_can_double) {
            result.should_double = false;
            result.should_take = true;
            result.optimal_equity = result.equity_nd;
        } else if (is_money) {
            float best_double = std::min(result.equity_dt, result.equity_dp);
            result.should_double = (best_double > result.equity_nd);
            result.should_take = (result.equity_dt <= result.equity_dp);
            result.optimal_equity = result.should_double
                ? std::min(result.equity_dt, result.equity_dp)
                : result.equity_nd;
        } else if (auto_double) {
            float dt_m = dt_raw;
            float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                    cube.cube_value, cube.match.is_crawford);
            result.should_double = true;
            result.should_take = (dt_m <= dp_m_val);
            result.optimal_equity = std::min(result.equity_dt, result.equity_dp);
        } else {
            float nd_m = nd_raw;
            float dt_m = dt_raw;
            float dp_m_val = dp_mwc(cube.match.away1, cube.match.away2,
                                    cube.cube_value, cube.match.is_crawford);
            float best_double = std::min(dt_m, dp_m_val);
            result.should_double = (best_double > nd_m);
            result.should_take = (dt_m <= dp_m_val);
            result.optimal_equity = result.should_double
                ? std::min(result.equity_dt, result.equity_dp)
                : result.equity_nd;
        }

        out[i] = result;
    }
}

} // namespace bgbot
