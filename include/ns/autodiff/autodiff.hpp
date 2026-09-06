#pragma once
#include "ns/parser/ast.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ns {

// Reverse-mode automatic differentiation over a forward computation graph.
//
// The forward graph is built from statements inside a `grad { ... }` block:
// each assignment is turned into a node connecting outputs to their inputs.
// A reverse pass then computes, for every differentiable variable `v`, the
// adjoint `dL/dv` and accumulates it into a per-variable gradient accumulator.
//
// Supported differentiable ops (Phase 2):
//   - matmul (A @ B)
//   - elementwise +,-,*,/ with (possibly) scalar broadcast
//   - ReLU / Swish / Tanh / Sigmoid / GELU activations
//   - cross_entropy (softmax + CE) -> produces dLogits
//   - Dense layer (matmul + bias), when expanded inline
//
// The optimizer consumes the resulting gradient accumulators.

// Kinds of variable a gradient can be requested for / accumulated on.
enum class GradTarget {
    TensorLeaf,   // a plain tensor variable (e.g. weights)
    ScalarLeaf,
    Intermediate   // not a leaf; gradient computed but not accumulated
};

// One node in the forward graph.
struct ADNode {
    enum Op {
        MATMUL, ADD, SUB, MUL, DIV,
        RELU, SWISH, TANH, SIGMOID, GELU,
        CROSS_ENTROPY,   // softmax + cross entropy, terminal (produces scalar loss)
        VAR_IN,          // leaf: an input variable
        CONST_SCALAR     // a scalar constant (e.g. weight * 1)
    };

    Op op;
    std::string name;            // resulting value name
    TensorType type;             // shape of the value this node produces

    // Inputs (edges): operand value names.
    std::vector<std::string> inputs;
    // For binary elementwise: whether each operand is a scalar (broadcast).
    std::vector<bool> input_scalar;
    // Auxiliary: activation name for generic activation, scalar value.
    std::string attr;
    double scalar_value = 0.0;

    // Filled in by the reverse pass: adjoint dL/d(output of this node).
    bool is_loss = false;
};

struct ReqGrad {
    std::string name;
    GradTarget target;
};

class Autodiff {
public:
    // Build the forward graph from the statements of a grad block.
    // `weights` = set of variable names that require gradients (accumulated).
    void build_grad_block(Stmt* grad_body, const std::unordered_set<std::string>& weights);

    // Run the reverse pass, producing gradient expressions for all variables.
    void backward();

    // The reverse pass result: for each requested variable, a list of
    // gradient contribution expressions (each a small RPN/closure string).
    struct GradResult {
        std::string var;
        GradTarget target;
        std::vector<std::string> contributions; // human-readable derivation
    };

    const std::vector<GradResult>& results() const { return results_; }
    const std::vector<ADNode>& forward_nodes() const { return nodes_; }
    bool has_errors() const { return !errors_.empty(); }
    const std::vector<std::string>& errors() const { return errors_; }

    // Reset weight set and state.
    void set_weights(std::unordered_set<std::string> w) { weights_ = std::move(w); }

private:
    std::vector<ADNode> nodes_;
    std::unordered_map<std::string, size_t> name_to_node_; // value name -> node idx
    std::unordered_map<std::string, size_t> grad_of_;      // value name -> adjoint node idx
    std::unordered_set<std::string> weights_;
    std::vector<GradResult> results_;
    std::vector<std::string> errors_;

    // DAG assembly helpers
    void compute_statement(Stmt* stmt);
    void compute_expr(Expr* expr, std::string& out_name, TensorType& out_type);
    void add_node(const ADNode& n);
    void emit_adjoin(size_t node_idx, const std::string& seed_adj_name);
};

// Entry point used by the compiler driver: differentiate the grad block in a
// function and attach gradient info to the network/function.
class GradCompiler {
public:
    // Compile a grad block found in a function; returns reverse-pass results.
    std::vector<Autodiff::GradResult> compile_grad_block(
        Stmt* fn_body, const std::unordered_set<std::string>& weight_params);

    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::vector<std::string> errors_;
};

} // namespace ns
