#pragma once
#include "ns/parser/ast.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

namespace ns {

// Result of shape checking / inference
struct ShapeError {
    std::string message;
    uint32_t line;
    uint32_t column;
};

// Layer kinds with dimension inference rules
enum class LayerKind {
    Unknown,
    Dense,
    Linear,
    Dropout,
    LayerNorm,
    Attention,
    Embedding,
    MoE,
    Activation
};

// Inference rule: maps a layer's declared config params to input->output dims.
// Each layer carries: keyword params (in, out, activation, etc.) resolved to
// DimExpr values so we can compute output shapes symbolically.
struct LayerRule {
    LayerKind kind = LayerKind::Unknown;
    // Config values captured from the layer declaration.
    DimExpr in;        // Dense/Linear input features
    DimExpr out;       // Dense/Linear output features
    DimExpr emb_vocab; // Embedding vocab
    DimExpr emb_dim;   // Embedding dim
    DimExpr num_heads; // Attention
    DimExpr num_experts; // MoE capacity (compile-time slot count)
    DimExpr ffn_dim;     // MoE expert FFN hidden width (default 4*emb_dim)
    DimExpr initial_experts; // MoE experts active at init (default = num_experts)
    std::string activation; // activation name
    double rate = 0.0;  // Dropout rate

    // Compute output dims given input dims.
    // Returns false on mismatch (records error through checker via caller).
    std::vector<DimExpr> apply(const std::vector<DimExpr>& in_dims,
                               bool& ok) const;
};

class ShapeChecker {
public:
    // Run full shape check on a program. Returns true if valid.
    bool check(Program& program);

    const std::vector<ShapeError>& errors() const { return errors_; }
    TypePtr get_expr_type(Expr* expr) const;

private:
    std::vector<ShapeError> errors_;
    // Symbol table for type aliases: name -> DimExpr (alias constants)
    std::unordered_map<std::string, DimExpr> type_aliases_;
    // Symbol table for variables: name -> Type
    std::unordered_map<std::string, TypePtr> variables_;
    // Function signatures
    struct FuncSig {
        std::vector<TypePtr> param_types;
        TypePtr return_type;
    };
    std::unordered_map<std::string, FuncSig> functions_;

    // Symbolic dimension constraint store (union-find).
    // Maps a symbolic name to its resolved binding (constant or another symbol).
    struct SymbolBinding {
        bool bound_to_const = false;
        int64_t const_value = 0;
        std::string binds_to;      // another symbolic name, if not const
        bool visited = false;
    };
    std::unordered_map<std::string, SymbolBinding> symbols_;

    // Layer name -> inference rule (declared inside networks)
    std::unordered_map<std::string, LayerRule> layer_rules_;

    void report(uint32_t line, uint32_t column, const std::string& msg);

    // Statement analysis
    void check_stmt(Stmt* stmt);
    void check_var_decl(Stmt* stmt);
    void check_fn_decl(Stmt* stmt);
    void check_network_decl(Stmt* stmt);
    void check_grad_block(Stmt* stmt);
    void check_block(Stmt* stmt);
    void check_return(Stmt* stmt);
    void check_expr_stmt(Stmt* stmt);

    // Expression analysis (infers types, sets expr->inferred_type)
    void check_expr(Expr* expr);

    // Tensor shape inference rules
    bool matmul_compatible(const TensorType& lhs, const TensorType& rhs,
                           uint32_t line, uint32_t col);
    TensorType infer_matmul(const TensorType& lhs, const TensorType& rhs);

    // Dimension unification with symbolic inference
    bool unify_dim(const DimExpr& a, const DimExpr& b, uint32_t line, uint32_t col);

    // Symbolic constraint resolution
    void record_symbol_const(const std::string& name, int64_t value);
    void merge_symbols(const std::string& a, const std::string& b);
    // Resolve a dim expr to a fully-bound form (follow symbol bindings).
    DimExpr resolve_dim(const DimExpr& d) const;
    TensorType resolve_tensor(const TensorType& t) const;
    // Expand type-alias symbolic dims to their bound values.
    void bind_alias_dim(DimExpr& dim);

    // Layer rule helpers
    LayerRule parse_layer_rule(Stmt* layer);
    void apply_layer(const LayerRule& rule, const TensorType& in,
                     TensorType& out, uint32_t line, uint32_t col);

    // Lookup helper
    TypePtr lookup_var(const std::string& name);
};

} // namespace ns
