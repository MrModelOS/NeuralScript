#pragma once
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/eval.hpp"
#include "ns/optim/muon.hpp"
#include <map>
#include <string>
#include <vector>

namespace ns {

// Numeric training driver. Executes the *unfused* MLIR forward graph of a
// network with a reverse-mode numeric tape, accumulates gradients into the
// trainable weight buffers (the TENSOR_ALLOC'd matrices of the network), and
// hands them — per parameter — directly to the optimizer layer
// (MuonOptimizer::step_param, which applies Muon for 2D matrices and AdamW
// otherwise). This is the numeric bridge between the forward graph and the
// optimizer library.
//
// Forward numeric semantics deliberately mirror ModuleEvaluator (the trusted
// CPU reference): matmul (row-major [M,K]x[K,N]), elementwise binops with
// modulo-broadcast rhs, activations relu/leaky_relu/sigmoid/tanh/swish/gelu/
// silu/identity, layernorm, softmax, dropout (identity at inference).
class NumericTrainer {
public:
    struct Param {
        std::string id;              // MLIR value id (e.g. "fc1_w")
        int64_t rows = 0, cols = 0;
        std::vector<double> data;    // current weights, row-major rows*cols
        std::vector<double> grad;    // accumulated batch gradient
    };

    NumericTrainer(const MLIRFunction& fn, const MLIRModule& mod);

    // Bind an external tensor value (network input) by id.
    void bind_input(const std::string& id, std::vector<double> data,
                    std::vector<int64_t> shape);

    const std::vector<Param>& params() const { return params_; }
    std::vector<Param>& params() { return params_; }
    Param* param(const std::string& id);
    const TensorBuffer* value(const std::string& id) const;
    const std::vector<std::string>& errors() const { return errors_; }

    // Execute the forward graph, recording the tape. Returns the value of the
    // function return id (logits, [B, C]).
    const TensorBuffer* forward();

    // Seed the reverse pass: expects the gradient of the loss wrt the return
    // value (e.g. (softmax(logits) - labels) / B) and accumulates dL/dw into
    // every registered weight via the numeric tape.
    void backward(const std::vector<double>& grad_return, int64_t B, int64_t C);

    // Zero accumulated gradients (call after each optimizer step).
    void zero_grad();

    // Apply the registered weights through `opt` (per-parameter bridging into
    // the Muon/AdamW optimizer). Params are auto-registered into `opt` on the
    // first call; step_index = 0-based step counter.
    void apply_step(MuonOptimizer& opt, size_t step_index);

private:
    struct Rec {
        MLIROp op = MLIROp::IDENTITY;
        std::string result_id;
        std::vector<std::string> operands;
        std::string attribute;
        TensorBuffer out;                       // forward output (saved)
        std::vector<TensorBuffer> ins;          // forward operand values
        bool is_weight_producer = false;
    };

    const MLIRFunction& fn_;
    const MLIRModule& mod_;
    std::map<std::string, TensorBuffer> values_;
    std::map<std::string, TensorBuffer> inputs_;  // bound inputs (survive forwards)
    std::map<std::string, TensorBuffer> grads_;
    std::vector<Rec> tape_;
    std::vector<Param> params_;
    const TensorBuffer* last_out_ = nullptr;
    std::vector<std::string> errors_;
    bool params_registered_ = false;

    void register_params();
    void record(const MLIRInstr& instr);
    bool eval_matmul(Rec& r);
    bool eval_activation(Rec& r);
    bool eval_binop(Rec& r);
    bool eval_layernorm(Rec& r);
    bool eval_softmax(Rec& r);

    void back_matmul(Rec& r);
    void back_activation(Rec& r);
    void back_binop(Rec& r);
    void back_layernorm(Rec& r);
    void back_softmax(Rec& r);

    void accumulate(const std::string& id, const TensorBuffer& g);
};

} // namespace ns