#pragma once
#include "ns/parser/ast.hpp"
#include <map>
#include <string>
#include <vector>

namespace ns {

// A simplified MLIR-style IR that captures the high-level tensor graph.
// This represents the "ns.tensor" / "ns.layer" dialect before lowering.

enum class MLIROp {
    // Tensor ops
    TENSOR_ALLOC,      // allocate a tensor
    TENSOR_FREE,       // deallocate a tensor
    MATMUL,            // matrix multiply (GEMM)
    ELEMENTWISE_BINOP, // elementwise binary op (+,-,*,/)
    ACTIVATION,        // activation function
    RELU, ACTIVATION_OP = RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    SWISH,
    GELU,
    SILU,
    IDENTITY,
    SOFTMAX,
    DROPOUT,
    LAYERNORM,
    CROSS_ENTROPY,     // loss function
    CONCAT,
    RESHAPE,
    TRANSPOSE,
    CONSTANT,

    // Control flow
    FN_CALL,           // function call
    FORWARD,           // forward pass
    GRAD,              // gradient computation block
    OPT_STEP,          // optimizer step

    // Layer declarations
    LAYER_DENSE,
    LAYER_DROPOUT,
    LAYER_ATTENTION,
    LAYER_EMBEDDING,
    LAYER_LAYERNORM,

    // Storage
    ALLOC_BUFFER,      // allocate GPU buffer
    FREE_BUFFER,       // free GPU buffer

    // Fused kernel (introduced by the fusion pass)
    FUSED             // fused GEMM + (activation | layernorm | elementwise tail)
};

// A fused group produced by the fusion pass: GEMM (A@B->C) followed by an
// elementwise epilogue (activation / layernorm / binop chain) that has no
// consumers other than the fusion boundary.
struct FusedopGroup {
    std::string result_id;            // final fused result value name
    std::string c;                    // intermediate GEMM result (usually internal)
    std::vector<std::string> ops;     // epilogue op names: "relu","gelu","layernorm","+",...
    std::vector<std::string> epilogue_operands; // extra operands for binops
    TensorType result_type;
    bool has_bias = false;
    std::string bias_operand;
};

struct MLIRValue {
    std::string id;
    TensorType type;
    bool is_temporary = false;

    MLIRValue() = default;
    MLIRValue(std::string i, TensorType t) : id(std::move(i)), type(std::move(t)) {}
};

struct MLIRInstr {
    MLIROp op;
    std::string result_id;
    std::vector<std::string> operands;
    TensorType result_type;

    // Auxiliary data
    std::string comment;
    std::string attribute; // e.g., activation type, loss type
    int64_t int_attr = 0;
    double float_attr = 0.0;

    MLIRInstr() : op(MLIROp::CONSTANT), int_attr(0), float_attr(0.0) {}
    MLIRInstr(MLIROp o, std::string res) : op(o), result_id(std::move(res)),
        int_attr(0), float_attr(0.0) {}
};

struct MLIRFunction {
    std::string name;
    std::vector<MLIRInstr> instructions;
    std::string return_id;
};

struct MLIRModule {
    std::vector<MLIRFunction> functions;
    std::vector<std::string> global_buffers;
    // Fusion pass output: groups keyed by the fused result value id.
    std::vector<FusedopGroup> fused_groups;
};

class MLIRCompiler {
public:
    // Convert NS AST to MLIR module
    MLIRModule compile(Program& program);

    // Pretty-print MLIR text representation
    std::string dump(MLIRModule& module);

private:
    int temp_counter_ = 0;
    std::string new_temp(const std::string& prefix = "t");

    // Top-level `type X = <const|alias>` aliases, for resolving layer dims.
    std::map<std::string, int64_t> aliases_;
    // Layer name -> weight buffer id (network layer params map onto *_w ids).
    std::map<std::string, std::string> layer_weight_id_;
    // Layer metadata for forward lowering (activation, dropout rate).
    struct LayerMeta {
        std::string type;          // "Dense", "Dropout", ...
        std::string activation;    // "ReLU"/"GELU"/... (empty if none)
        double dropout_rate = 0.0;
    };
    std::map<std::string, LayerMeta> layer_meta_;
    // Resolve a dimension expression value (int literal or alias) to a const,
    // returning false if it is dynamic/symbolic.
    bool resolve_dim_int(const Expr* expr, int64_t& out);

    void collect_aliases(Program& program);
    void compile_fn(Stmt* stmt, MLIRModule& module);
    void compile_network(Stmt* stmt, MLIRModule& module);
    void compile_stmt(Stmt* stmt, MLIRFunction& fn);
    void compile_expr(Expr* expr, MLIRFunction& fn, MLIRValue& out);

    // Higher-level helpers
    void lower_matmul(Expr* expr, MLIRFunction& fn, MLIRValue& out);
    void lower_pipeline(Expr* expr, MLIRFunction& fn, MLIRValue& out);
    void lower_pipeline_apply(Expr* expr, MLIRFunction& fn, MLIRValue& in, MLIRValue& out);
    void apply_pipeline_stage(const std::string& wname, MLIRValue& in, MLIRFunction& fn, MLIRValue& out);
    void lower_activation(const std::string& act, MLIRValue& in, MLIRFunction& fn, MLIRValue& out);
};

} // namespace ns
