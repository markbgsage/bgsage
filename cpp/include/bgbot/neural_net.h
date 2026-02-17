#pragma once

#include "strategy.h"
#include "encoding.h"
#include "board.h"
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <array>
#include <mutex>

namespace bgbot {

constexpr int NN_INPUTS  = TESAURO_INPUTS;  // 196 (legacy, for backward compat)
constexpr int NN_OUTPUTS = NUM_OUTPUTS;     // 5: P(win), P(gw), P(bw), P(gl), P(bl)

class NeuralNetwork {
public:
    // Construct with random weights drawn from normal(0, eps).
    explicit NeuralNetwork(int n_hidden = 120,
                           int n_inputs = NN_INPUTS,
                           float eps = 0.1f,
                           uint32_t seed = 42);

    // ----- Inference (const, thread-safe) -----

    // Forward pass: inputs -> 5 sigmoid outputs.
    // `inputs` must point to at least n_inputs_ floats.
    std::array<float, NN_OUTPUTS> forward(const float* inputs) const;

    // Overload for std::array<float, 196> backward compat
    std::array<float, NN_OUTPUTS> forward(
        const std::array<float, NN_INPUTS>& inputs) const {
        return forward(inputs.data());
    }

    // Batch forward pass: evaluate multiple input vectors at once.
    // Each input is n_inputs_ floats, stored contiguously (stride = n_inputs_).
    void forward_batch(const float* inputs_array,
                       std::array<float, NN_OUTPUTS>* outputs_array,
                       int count) const;

    // Incremental (delta) evaluation: saves pre-sigmoid hidden sums and input
    // vector for the first candidate, then for subsequent candidates only applies
    // the input delta. This is much faster when most inputs don't change between
    // candidates (typical in backgammon: a move changes ~4-8 of 196/244 inputs).
    //
    // forward_save_base: full forward pass, but also saves pre-sigmoid hidden sums
    // and input vector into caller-provided buffers.
    //   saved_base: must be at least n_hidden_ floats (pre-sigmoid hidden sums)
    //   saved_inputs: must be at least n_inputs_ floats (input vector copy)
    std::array<float, NN_OUTPUTS> forward_save_base(
        const float* inputs, float* saved_base, float* saved_inputs) const;

    // forward_from_base: incremental forward using saved base. Computes delta
    // between new inputs and saved_inputs, applies delta to saved_base (copied
    // internally), applies sigmoid, then output layer.
    //   inputs: new input vector (n_inputs_ floats)
    //   saved_base: pre-sigmoid hidden sums from forward_save_base
    //   saved_inputs: input vector from forward_save_base
    // Both saved_base and saved_inputs are NOT modified.
    std::array<float, NN_OUTPUTS> forward_from_base(
        const float* inputs, const float* saved_base, const float* saved_inputs) const;

    // Ensure transposed hidden weights are initialized for incremental inference.
    void ensure_transposed_weights() const {
        std::call_once(transposed_weights_init_, [this] {
            build_transposed_weights();
        });
    }

    // Legacy overload for std::array<float, 196>
    void forward_batch(
        const std::array<float, NN_INPUTS>* inputs_array,
        std::array<float, NN_OUTPUTS>* outputs_array,
        int count) const {
        forward_batch(reinterpret_cast<const float*>(inputs_array), outputs_array, count);
    }

    // Compute equity from 5 output probabilities.
    // equity = 2*P(win) - 1 + P(gw) - P(gl) + P(bw) - P(bl)
    static double compute_equity(const std::array<float, NN_OUTPUTS>& outputs);

    // ----- Training (mutating, NOT thread-safe) -----

    // Forward pass that caches all intermediate values for td_update().
    // `inputs` must point to at least n_inputs_ floats.
    std::array<float, NN_OUTPUTS> forward_with_gradients(const float* inputs);

    // Overload for backward compat
    std::array<float, NN_OUTPUTS> forward_with_gradients(
        const std::array<float, NN_INPUTS>& inputs) {
        return forward_with_gradients(inputs.data());
    }

    // Apply TD weight update using cached gradients from forward_with_gradients().
    void td_update(const std::array<float, NN_OUTPUTS>& targets, float alpha);

    // ----- Persistence -----

    // Save weights in binary format.
    // Format: [int32 n_hidden] [int32 n_inputs] [int32 n_outputs]
    //         [n_hidden*(n_inputs+1) floats: hidden weights row-major]
    //         [n_outputs*(n_hidden+1) floats: output weights row-major]
    bool save_weights(const std::string& filepath) const;

    // Load weights from binary file. n_inputs in file must match n_inputs_.
    bool load_weights(const std::string& filepath);

    int n_hidden() const { return n_hidden_; }
    int n_inputs() const { return n_inputs_; }

    // Direct weight access (for weight transfer between networks)
    const std::vector<float>& hidden_weights() const { return hidden_weights_; }
    const std::vector<float>& output_weights() const { return output_weights_; }
    std::vector<float>& hidden_weights() { return hidden_weights_; }
    std::vector<float>& output_weights() { return output_weights_; }

private:
    int n_hidden_;
    int n_inputs_;

    // Weight matrices stored as flat vectors (row-major).
    // hidden_weights_: shape [n_hidden, n_inputs_+1]  (last column = bias)
    // output_weights_: shape [NN_OUTPUTS, n_hidden+1]  (last column = bias)
    std::vector<float> hidden_weights_;
    std::vector<float> output_weights_;

    // Transposed hidden weights for fast column access in forward_from_base.
    // hidden_weights_T_: shape [n_inputs_, n_hidden]  (no bias column)
    // Column i of hidden_weights_ (stride n_inputs_+1) becomes row i of this.
    // Built lazily on first forward_from_base call, or after load_weights.
    mutable std::vector<float> hidden_weights_T_;
    mutable std::once_flag transposed_weights_init_;
    void build_transposed_weights() const;

    // ----- Cached state for training (set by forward_with_gradients) -----
    std::vector<float> cached_inputs_;            // size n_inputs_
    std::vector<float> cached_hiddens_;           // size n_hidden+1 (last = 1.0 bias)
    std::array<float, NN_OUTPUTS> cached_outputs_;
    std::array<float, NN_OUTPUTS> cached_prods_;  // outputs * (1 - outputs)
    std::vector<float> cached_hprods_;            // size n_hidden: hiddens * (1-hiddens)

    // Index helpers for flat weight arrays
    float& hw(int h, int i)       { return hidden_weights_[h * (n_inputs_ + 1) + i]; }
    float  hw(int h, int i) const { return hidden_weights_[h * (n_inputs_ + 1) + i]; }
    float& ow(int o, int h)       { return output_weights_[o * (n_hidden_ + 1) + h]; }
    float  ow(int o, int h) const { return output_weights_[o * (n_hidden_ + 1) + h]; }
};


// Strategy subclass that uses a NeuralNetwork for position evaluation.
class NNStrategy : public Strategy {
public:
    using Strategy::best_move_index;  // inherit Board overload

    // Construct from a shared NeuralNetwork.
    explicit NNStrategy(std::shared_ptr<NeuralNetwork> nn);

    // Construct by loading weights from a file.
    explicit NNStrategy(const std::string& weights_path, int n_hidden = 120,
                        int n_inputs = NN_INPUTS);

    // Evaluate a post-move board. Returns equity.
    double evaluate(const Board& board, bool pre_move_is_race) const override;

    // Returns the 5 raw NN output probabilities.
    std::array<float, NN_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;

    // Batched best_move_index
    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;

    NeuralNetwork& network() { return *nn_; }
    const NeuralNetwork& network() const { return *nn_; }
    std::shared_ptr<NeuralNetwork> network_ptr() { return nn_; }

private:
    std::shared_ptr<NeuralNetwork> nn_;
};


// Position type classification
enum class PosType { CONTACT, CRASHED, RACE };

inline PosType classify_position(const Board& board) {
    if (is_race(board)) return PosType::RACE;
    if (is_crashed(board)) return PosType::CRASHED;
    return PosType::CONTACT;
}


// Strategy that uses three separate neural networks for contact, crashed, and race.
class MultiNNStrategy : public Strategy {
public:
    using Strategy::best_move_index;  // inherit Board overload

    // Construct with three networks. crashed_nn can be null (falls back to contact_nn).
    MultiNNStrategy(std::shared_ptr<NeuralNetwork> contact_nn,
                    std::shared_ptr<NeuralNetwork> crashed_nn,
                    std::shared_ptr<NeuralNetwork> race_nn);

    // Construct from weight files (contact + race only, crashed = contact)
    MultiNNStrategy(const std::string& contact_weights,
                    const std::string& race_weights,
                    int n_hidden = 120);

    // Full constructor with all three weight files (same hidden size)
    MultiNNStrategy(const std::string& contact_weights,
                    const std::string& crashed_weights,
                    const std::string& race_weights,
                    int n_hidden = 120);

    // Full constructor with all three weight files and separate hidden sizes
    MultiNNStrategy(const std::string& contact_weights,
                    const std::string& crashed_weights,
                    const std::string& race_weights,
                    int n_hidden_contact,
                    int n_hidden_crashed,
                    int n_hidden_race);

    double evaluate(const Board& board, bool pre_move_is_race) const override;

    std::array<float, NN_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;

    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;

    std::shared_ptr<NeuralNetwork> contact_nn() { return contact_nn_; }
    std::shared_ptr<NeuralNetwork> crashed_nn() { return crashed_nn_; }
    std::shared_ptr<NeuralNetwork> race_nn() { return race_nn_; }

private:
    std::shared_ptr<NeuralNetwork> contact_nn_;
    std::shared_ptr<NeuralNetwork> crashed_nn_;  // may be null (use contact_nn)
    std::shared_ptr<NeuralNetwork> race_nn_;

    double evaluate_with_nn(const Board& board, PosType ptype) const;
    std::array<float, NN_OUTPUTS> probs_with_nn(const Board& board, PosType ptype) const;
    const NeuralNetwork& select_nn(PosType ptype) const;
};


// Strategy that uses five separate neural networks based on game plan classification:
// PureRace (80h, 196 inputs), Racing/Attacking/Priming/Anchoring (120h, 214 inputs).
class GamePlanStrategy : public Strategy {
public:
    GamePlanStrategy(std::shared_ptr<NeuralNetwork> purerace_nn,
                     std::shared_ptr<NeuralNetwork> racing_nn,
                     std::shared_ptr<NeuralNetwork> attacking_nn,
                     std::shared_ptr<NeuralNetwork> priming_nn,
                     std::shared_ptr<NeuralNetwork> anchoring_nn);

    GamePlanStrategy(const std::string& purerace_weights,
                     const std::string& racing_weights,
                     const std::string& attacking_weights,
                     const std::string& priming_weights,
                     const std::string& anchoring_weights,
                     int n_hidden_purerace = 80,
                     int n_hidden_racing = 120,
                     int n_hidden_attacking = 120,
                     int n_hidden_priming = 120,
                     int n_hidden_anchoring = 120);

    double evaluate(const Board& board, bool pre_move_is_race) const override;

    std::array<float, NN_OUTPUTS> evaluate_probs(
        const Board& board, bool pre_move_is_race) const override;
    std::array<float, NN_OUTPUTS> evaluate_probs(
        const Board& board, const Board& pre_move_board) const override;

    int best_move_index(const std::vector<Board>& candidates,
                        bool pre_move_is_race) const override;

    // Override: classify pre-move board's game plan and use that NN for all candidates
    int best_move_index(const std::vector<Board>& candidates,
                        const Board& pre_move_board) const override;

    std::shared_ptr<NeuralNetwork> purerace_nn() { return purerace_nn_; }
    std::shared_ptr<NeuralNetwork> racing_nn() { return racing_nn_; }
    std::shared_ptr<NeuralNetwork> attacking_nn() { return attacking_nn_; }
    std::shared_ptr<NeuralNetwork> priming_nn() { return priming_nn_; }
    std::shared_ptr<NeuralNetwork> anchoring_nn() { return anchoring_nn_; }

    // Batch evaluation: classify pre_move_board once, evaluate all candidates
    // with the appropriate NN. Fills equities[0..n-1] with equity values.
    // Returns the index of the best (highest equity) candidate.
    // This is faster than calling evaluate_probs() per candidate because it
    // avoids repeated game plan classification and virtual dispatch.
    // Handles terminal positions (check_game_over) for each candidate.
    int evaluate_candidates_equity(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities) const;

    // Batch encoding + forward pass: classify once, encode all non-terminal
    // candidates into a contiguous buffer, call forward_batch, compute equities.
    // Returns index of best candidate. Faster than evaluate_candidates_equity
    // when there are many candidates (amortizes NN call overhead).
    int batch_evaluate_candidates_equity(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities) const;

    // Same as batch_evaluate_candidates_equity, but also stores the full
    // 5-output probabilities for each candidate in `probs_out`.
    // Avoids re-evaluating the best candidate when probabilities are needed.
    int batch_evaluate_candidates_equity_probs(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities,
        std::array<float, NUM_OUTPUTS>* probs_out) const;

    // Same as batch_evaluate_candidates_equity, but returns only the
    // best candidate probabilities in `best_probs_out`.
    // Avoids filling probabilities for non-best candidates when only one
    // probability vector is needed.
    // If `equities` is nullptr, the equity outputs are not stored.
    int batch_evaluate_candidates_best_prob(
        const std::vector<Board>& candidates,
        const Board& pre_move_board,
        double* equities,
        std::array<float, NUM_OUTPUTS>* best_probs_out) const;



private:
    std::shared_ptr<NeuralNetwork> purerace_nn_;
    std::shared_ptr<NeuralNetwork> racing_nn_;
    std::shared_ptr<NeuralNetwork> attacking_nn_;
    std::shared_ptr<NeuralNetwork> priming_nn_;
    std::shared_ptr<NeuralNetwork> anchoring_nn_;

    double evaluate_with_nn(const Board& board, GamePlan gp) const;
    std::array<float, NN_OUTPUTS> probs_with_nn(const Board& board, GamePlan gp) const;
    const NeuralNetwork& select_nn(GamePlan gp) const;
};

} // namespace bgbot
