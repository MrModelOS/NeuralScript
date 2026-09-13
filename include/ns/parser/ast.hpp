#pragma once
#include "ns/lexer/token.hpp"
#include <memory>
#include <string>
#include <vector>
#include <variant>

namespace ns {

// Forward declarations
struct Expr;
struct Stmt;
struct TypeNode;

using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;
using TypePtr = std::unique_ptr<TypeNode>;

// ---- Type System Nodes ----

enum class Dtype {
    Float16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    FP8, FP4, Bool
};

struct DimExpr {
    enum Kind { CONST, SYMBOLIC, DYNAMIC };
    Kind kind;
    int64_t const_value;
    std::string symbolic_name;

    DimExpr() : kind(DYNAMIC), const_value(0) {}
    DimExpr(int64_t v) : kind(CONST), const_value(v) {}
    DimExpr(std::string name, bool is_dynamic = false)
        : kind(is_dynamic ? DYNAMIC : SYMBOLIC), const_value(0), symbolic_name(std::move(name)) {}

    static DimExpr dynamic() { return DimExpr(); }
    static DimExpr constant(int64_t v) { return DimExpr(v); }
    static DimExpr symbolic(const std::string& name) { return DimExpr(name); }

    bool is_dynamic() const { return kind == DYNAMIC; }
    bool is_const() const { return kind == CONST; }
    bool is_symbolic() const { return kind == SYMBOLIC; }
};

struct TensorType {
    std::vector<DimExpr> dims;
    Dtype dtype;
    bool is_mutable;

    TensorType() : dtype(Dtype::Float32), is_mutable(false) {}
    TensorType(std::vector<DimExpr> d, Dtype dt)
        : dims(std::move(d)), dtype(dt), is_mutable(false) {}

    bool is_dynamic() const {
        for (auto& d : dims) if (d.is_dynamic()) return true;
        return false;
    }
};

struct TypeNode {
    enum Kind { TENSOR, SCALAR, FUNCTION, UNKNOWN };
    Kind kind;
    TensorType tensor_type;
    Dtype scalar_type;

    TypeNode() : kind(UNKNOWN), scalar_type(Dtype::Float32) {}
    TypeNode(TensorType tt) : kind(TENSOR), tensor_type(std::move(tt)), scalar_type(Dtype::Float32) {}
    TypeNode(Dtype dt) : kind(SCALAR), scalar_type(dt) {}

    bool is_tensor() const { return kind == TENSOR; }
    bool is_scalar() const { return kind == SCALAR; }
    bool is_dynamic() const { return kind == TENSOR && tensor_type.is_dynamic(); }
};

// ---- Expression Nodes ----

struct Expr {
    enum Kind {
        LITERAL_INT, LITERAL_FLOAT, LITERAL_STRING, LITERAL_BOOL,
        IDENTIFIER, BINARY_OP, UNARY_OP, MATMUL_OP,
        PIPELINE_OP, FUNCTION_CALL, INDEX_OP,
        TYPE_CAST, TENSOR_LITERAL
    };

    Kind kind;
    Token token; // source location + value

    // Binary/unary
    ExprPtr left;
    ExprPtr right;
    ExprPtr operand;

    // Function call
    std::string callee;
    std::vector<ExprPtr> args;

    // Index
    std::vector<ExprPtr> indices;

    // Type inference result
    TypePtr inferred_type;

    Expr(Kind k, Token tok) : kind(k), token(std::move(tok)) {}
};

// ---- Statement Nodes ----

struct Stmt {
    enum Kind {
        VAR_DECL, VAR_ASSIGN, RETURN_STMT, EXPR_STMT,
        IF_STMT, WHILE_STMT, BLOCK, FN_DECL,
        NETWORK_DECL, LAYER_DECL, FORWARD_DECL,
        GRAD_BLOCK, TYPE_DECL, IMPORT_STMT, TRAIN_DECL
    };

    Kind kind;
    Token token;

    // Variable declaration
    std::string var_name;
    TypePtr var_type;
    ExprPtr init_expr;

    // Function declaration
    std::string fn_name;
    struct Param {
        std::string name;
        TypePtr type;
        bool is_mut;
        bool is_ref;
    };
    std::vector<Param> params;
    TypePtr return_type;
    StmtPtr body;

    // Network declaration
    std::string network_name;
    std::vector<StmtPtr> layers;
    std::vector<StmtPtr> methods;

    // Layer declaration
    std::string layer_name;
    std::string layer_type;
    struct LayerParam {
        std::string name;
        ExprPtr value;
    };
    std::vector<LayerParam> layer_params;

    // If/While
    ExprPtr condition;
    StmtPtr else_branch;

    // Block
    std::vector<StmtPtr> statements;

    // Grad block
    StmtPtr grad_body;

    // Type alias
    std::string alias_name;
    TypePtr alias_type;
    ExprPtr alias_expr;

    // Expression statement
    ExprPtr expr;

    Stmt(Kind k, Token tok) : kind(k), token(std::move(tok)) {}
};

// ---- Program ----

struct Program {
    std::vector<StmtPtr> top_level;

    void add(StmtPtr stmt) {
        top_level.push_back(std::move(stmt));
    }
};

// ---- Utility ----

std::string dtype_to_string(Dtype dt);
Dtype token_to_dtype(TokenType tt);
std::string dim_expr_to_string(const DimExpr& dim);
std::string tensor_type_to_string(const TensorType& tt);

} // namespace ns
