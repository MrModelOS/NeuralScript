#include "ns/typechecker/shape_checker.hpp"
#include <sstream>
#include <iostream>
#include <cctype>   // std::isdigit
#include <unordered_set>
#include <algorithm> // std::reverse

namespace ns {

bool ShapeChecker::check(Program& program) {
    errors_.clear();
    // First pass: collect type aliases
    for (auto& stmt : program.top_level) {
        if (stmt->kind == Stmt::TYPE_DECL) {
            if (stmt->alias_expr->kind == Expr::LITERAL_INT) {
                type_aliases_[stmt->alias_name] =
                    DimExpr::constant(std::stoll(stmt->alias_expr->token.value));
            } else if (stmt->alias_expr->kind == Expr::IDENTIFIER) {
                // Could be symbolic name (e.g., Dynamic) or reference to another alias
                type_aliases_[stmt->alias_name] = DimExpr::symbolic(stmt->alias_expr->token.value);
            }
        }
    }

    // Second pass: analyze all statements
    for (auto& stmt : program.top_level) {
        check_stmt(stmt.get());
    }
    return errors_.empty();
}

void ShapeChecker::report(uint32_t line, uint32_t column, const std::string& msg) {
    errors_.push_back({msg, line, column});
}

TypePtr ShapeChecker::get_expr_type(Expr* expr) const {
    if (expr && expr->inferred_type) {
        // Return a copy
        return std::make_unique<TypeNode>(*expr->inferred_type);
    }
    return std::make_unique<TypeNode>(Dtype::Float32);
}

void ShapeChecker::check_stmt(Stmt* stmt) {
    if (!stmt) return;
    switch (stmt->kind) {
        case Stmt::VAR_DECL: check_var_decl(stmt); break;
        case Stmt::FN_DECL: check_fn_decl(stmt); break;
        case Stmt::NETWORK_DECL: check_network_decl(stmt); break;
        case Stmt::GRAD_BLOCK: check_grad_block(stmt); break;
        case Stmt::BLOCK: check_block(stmt); break;
        case Stmt::RETURN_STMT: check_return(stmt); break;
        case Stmt::EXPR_STMT: check_expr_stmt(stmt); break;
        case Stmt::IF_STMT:
            if (stmt->condition) check_expr(stmt->condition.get());
            check_stmt(stmt->body.get());
            check_stmt(stmt->else_branch.get());
            break;
        case Stmt::WHILE_STMT:
            if (stmt->condition) check_expr(stmt->condition.get());
            check_stmt(stmt->body.get());
            break;
        case Stmt::TYPE_DECL: break; // handled in first pass
        case Stmt::LAYER_DECL: {
            // Register layer rule so pipeline chains can resolve and apply it.
            layer_rules_[stmt->layer_name] = parse_layer_rule(stmt);
            break;
        }
        case Stmt::FORWARD_DECL:
            check_stmt(stmt->body.get());
            break;
        case Stmt::TRAIN_DECL:
            check_stmt(stmt->body.get());
            break;
        case Stmt::VAR_ASSIGN:
            if (stmt->init_expr) check_expr(stmt->init_expr.get());
            break;
        case Stmt::IMPORT_STMT:
            break;
    }
}

void ShapeChecker::check_var_decl(Stmt* stmt) {
    if (stmt->var_type) {
        // Expand symbolic dims from type aliases
        if (stmt->var_type->is_tensor()) {
            for (auto& dim : stmt->var_type->tensor_type.dims) {
                if (dim.kind == DimExpr::SYMBOLIC) {
                    auto it = type_aliases_.find(dim.symbolic_name);
                    if (it != type_aliases_.end()) {
                        // For dynamic and symbolic aliases, keep symbolic
                        // For constants, substitute
                        if (it->second.is_const()) {
                            dim = it->second;
                        }
                    }
                    // Unknown symbolic stays symbolic (dependent type)
                }
            }
        }
        variables_[stmt->var_name] = std::make_unique<TypeNode>(stmt->var_type->tensor_type);
    }

    if (stmt->init_expr) {
        check_expr(stmt->init_expr.get());
        if (stmt->init_expr->inferred_type && stmt->var_type) {
            // Type-check init expr against declared type
            if (stmt->var_type->is_tensor() && stmt->init_expr->inferred_type->is_tensor()) {
                auto& declared = stmt->var_type->tensor_type;
                auto& actual = stmt->init_expr->inferred_type->tensor_type;
                if (declared.dims.size() != actual.dims.size()) {
                    report(stmt->token.line, stmt->token.column,
                           "Dimension rank mismatch: declared " + tensor_type_to_string(declared) +
                           " but expression produces rank-" + std::to_string(actual.dims.size()));
                } else {
                    for (size_t i = 0; i < declared.dims.size(); i++) {
                        DimExpr decl_dim = declared.dims[i];
                        unify_dim(decl_dim, actual.dims[i], stmt->token.line, stmt->token.column);
                        // Update variable's type with unified dim
                        stmt->var_type->tensor_type.dims[i] = decl_dim;
                    }
                }
            }
        }
        if (stmt->var_type) {
            variables_[stmt->var_name] = std::make_unique<TypeNode>(stmt->var_type->tensor_type);
        } else if (stmt->init_expr && stmt->init_expr->inferred_type) {
            // Untyped declaration: bind to the inferred type of the init expr.
            variables_[stmt->var_name] =
                std::make_unique<TypeNode>(*stmt->init_expr->inferred_type);
        }
    }
}

void ShapeChecker::check_fn_decl(Stmt* stmt) {
    // Register function signature
    FuncSig sig;
    for (auto& p : stmt->params) {
        if (p.type) {
            sig.param_types.push_back(std::make_unique<TypeNode>(*p.type));
        }
    }
    if (stmt->return_type) {
        sig.return_type = std::make_unique<TypeNode>(*stmt->return_type);
    }
    functions_[stmt->fn_name] = std::move(sig);

    // Push scope
    std::unordered_map<std::string, TypePtr> saved = std::move(variables_);
    variables_.clear();

    for (auto& p : stmt->params) {
        if (p.type) {
            variables_[p.name] = std::make_unique<TypeNode>(*p.type);
        }
    }

    if (stmt->body) {
        check_block(stmt->body.get());
    }

    // Pop scope
    variables_ = std::move(saved);
}

void ShapeChecker::check_network_decl(Stmt* stmt) {
    // Save global variable scope
    std::unordered_map<std::string, TypePtr> saved = std::move(variables_);
    variables_.clear();

    // inputs/outputs are vars
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::VAR_DECL) {
            // Register input/output type
            if (method->var_type) {
                auto tt = method->var_type->tensor_type;
                for (auto& dim : tt.dims) {
                    bind_alias_dim(dim);
                    dim = resolve_dim(dim);
                }
                variables_[method->var_name] = std::make_unique<TypeNode>(tt);
                // refresh source var_type so declared_input/output capture expanded dims
                method->var_type->tensor_type = tt;
            }
        }
    }

    // Check layers (also register inference rules for pipeline propagation)
    for (auto& layer : stmt->layers) {
        layer_rules_[layer->layer_name] = parse_layer_rule(layer.get());
        check_stmt(layer.get());
        // Register the trainable weight matrix as a tensor variable, so train()
        // methods can reference layers (e.g. `batch_x @ fc1`) directly and the
        // weight folds into the network's weight set (no w1/w2 duplication).
        if ((layer->layer_type == "Dense" || layer->layer_type == "Linear") &&
            variables_.count(layer->layer_name) == 0) {
            auto riter = layer_rules_.find(layer->layer_name);
            if (riter != layer_rules_.end() && riter->second.in.kind != DimExpr::DYNAMIC &&
                riter->second.out.kind != DimExpr::DYNAMIC) {
                DimExpr din = riter->second.in;
                DimExpr dout = riter->second.out;
                bind_alias_dim(din);
                bind_alias_dim(dout);
                if (din.is_const() && dout.is_const()) {
                    TensorType wt;
                    wt.dtype = Dtype::Float32;
                    wt.dims = {din, dout};
                    variables_[layer->layer_name] = std::make_unique<TypeNode>(wt);
                }
            }
        }
    }

    // Capture declared network input type (used to type forward params).
    TypePtr declared_input;
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::VAR_DECL && method->var_name == "input"
            && method->var_type) {
            declared_input = std::make_unique<TypeNode>(method->var_type->tensor_type);
        }
    }

    // Check forward
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::FORWARD_DECL) {
            // Register forward params as tensor variables (typed by the input map)
            for (auto& p : method->params) {
                if (!p.type) {
                    // If no explicit type, default to this network's input type
                    if (declared_input) {
                        variables_[p.name] =
                            std::make_unique<TypeNode>(*declared_input);
                    } else {
                        auto it = variables_.find(p.name);
                        if (it != variables_.end()) {
                            variables_[p.name] =
                                std::make_unique<TypeNode>(*it->second);
                        } else {
                            variables_[p.name] =
                                std::make_unique<TypeNode>(Dtype::Float32);
                        }
                    }
                } else {
                    variables_[p.name] = std::make_unique<TypeNode>(*p.type);
                }
            }
        }
    }

    // Register train() params in scope (typed by their declared Tensor types).
    for (auto& method : stmt->methods) {
        if (method->kind != Stmt::TRAIN_DECL) continue;
        for (auto& p : method->params) {
            if (p.type) {
                variables_[p.name] = std::make_unique<TypeNode>(*p.type);
            } else {
                variables_[p.name] = std::make_unique<TypeNode>(Dtype::Float32);
            }
        }
    }

    // Capture declared network output type (for forward verification).
    TypePtr declared_output;
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::VAR_DECL && method->var_name == "output"
            && method->var_type) {
            declared_output = std::make_unique<TypeNode>(method->var_type->tensor_type);
        }
    }

    for (auto& method : stmt->methods) {
        check_stmt(method.get());
    }

    // Verify the forward() return shape matches the declared output.
    if (declared_output && declared_output->is_tensor()) {
        for (auto& method : stmt->methods) {
            if (method->kind != Stmt::FORWARD_DECL || !method->body) continue;
            auto& rets = method->body->statements;
            for (auto& s : rets) {
                if (s->kind != Stmt::RETURN_STMT || !s->init_expr) continue;
                if (s->init_expr->inferred_type &&
                    s->init_expr->inferred_type->is_tensor()) {
                    auto& produced = s->init_expr->inferred_type->tensor_type;
                    auto& expected = declared_output->tensor_type;
                    if (produced.dims.size() != expected.dims.size()) {
                        report(stmt->token.line, stmt->token.column,
                               "Network forward() produces rank-" +
                               std::to_string(produced.dims.size()) +
                               " but declared output has rank-" +
                               std::to_string(expected.dims.size()));
                    } else {
                        for (size_t i = 0; i < expected.dims.size(); i++) {
                            unify_dim(expected.dims[i], produced.dims[i],
                                      stmt->token.line, stmt->token.column);
                        }
                    }
                }
            }
        }
    }

    variables_ = std::move(saved);
}

void ShapeChecker::check_grad_block(Stmt* stmt) {
    if (stmt->grad_body) {
        check_block(stmt->grad_body.get());
    }
}

void ShapeChecker::check_block(Stmt* stmt) {
    for (auto& s : stmt->statements) {
        check_stmt(s.get());
    }
}

void ShapeChecker::check_return(Stmt* stmt) {
    if (stmt->init_expr) {
        check_expr(stmt->init_expr.get());
    }
}

void ShapeChecker::check_expr_stmt(Stmt* stmt) {
    if (stmt->expr) {
        check_expr(stmt->expr.get());
    }
}

void ShapeChecker::check_expr(Expr* expr) {
    if (!expr) return;

    switch (expr->kind) {
        case Expr::LITERAL_INT:
            expr->inferred_type = std::make_unique<TypeNode>(Dtype::Int64);
            break;
        case Expr::LITERAL_FLOAT:
            expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float64);
            break;
        case Expr::LITERAL_BOOL:
            expr->inferred_type = std::make_unique<TypeNode>(Dtype::Bool);
            break;
        case Expr::LITERAL_STRING:
            expr->inferred_type = std::make_unique<TypeNode>(Dtype::Int64); // placeholder
            break;
        case Expr::IDENTIFIER: {
            auto vt = lookup_var(expr->token.value);
            if (vt) {
                expr->inferred_type = std::make_unique<TypeNode>(*vt);
            } else if (layer_rules_.count(expr->token.value)) {
                // Layer reference; treated as an opaque transform object.
                expr->inferred_type = nullptr;
            } else if (expr->token.value == "Dynamic") {
                expr->inferred_type = nullptr;
            } else {
                report(expr->token.line, expr->token.column,
                       "Undefined variable '" + expr->token.value + "'");
            }
            break;
        }
        case Expr::MATMUL_OP: {
            check_expr(expr->left.get());
            check_expr(expr->right.get());
            if (expr->left->inferred_type && expr->right->inferred_type &&
                expr->left->inferred_type->is_tensor() &&
                expr->right->inferred_type->is_tensor()) {
                auto& lhs = expr->left->inferred_type->tensor_type;
                auto& rhs = expr->right->inferred_type->tensor_type;
                if (matmul_compatible(lhs, rhs, expr->token.line, expr->token.column)) {
                    expr->inferred_type = std::make_unique<TypeNode>(infer_matmul(lhs, rhs));
                }
            } else {
                report(expr->token.line, expr->token.column,
                       "Matmul @ requires tensor operands");
            }
            break;
        }
        case Expr::BINARY_OP: {
            check_expr(expr->left.get());
            check_expr(expr->right.get());
            auto t = expr->token.type;
            if (t == TokenType::OP_ASSIGN) {
                expr->inferred_type = expr->left->inferred_type
                    ? std::make_unique<TypeNode>(*expr->left->inferred_type)
                    : nullptr;
            } else if (expr->left->inferred_type && expr->right->inferred_type) {
                // Elementwise array op: same shapes
                if (expr->left->inferred_type->is_tensor() &&
                    expr->right->inferred_type->is_tensor()) {
                    auto& lhs = expr->left->inferred_type->tensor_type;
                    auto& rhs = expr->right->inferred_type->tensor_type;
                    // For elementwise, dims must unify
                    bool ok = true;
                    if (lhs.dims.size() == rhs.dims.size()) {
                        for (size_t i = 0; i < lhs.dims.size(); i++) {
                            if (!unify_dim(const_cast<DimExpr&>(lhs.dims[i]), rhs.dims[i],
                                           expr->token.line, expr->token.column)) {
                                ok = false;
                            }
                        }
                    } else {
                        ok = false;
                        report(expr->token.line, expr->token.column,
                               "Elementwise op shape rank mismatch");
                    }
                    if (ok) {
                        expr->inferred_type = std::make_unique<TypeNode>(const_cast<TensorType&>(lhs));
                    }
                } else if (expr->left->inferred_type->is_scalar()) {
                    expr->inferred_type = std::make_unique<TypeNode>(
                        const_cast<TypeNode&>(*expr->left->inferred_type));
                } else {
                    expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                }
            }
            break;
        }
        case Expr::UNARY_OP:
            check_expr(expr->operand.get());
            if (expr->operand->inferred_type) {
                expr->inferred_type = std::make_unique<TypeNode>(*expr->operand->inferred_type);
            }
            break;
        case Expr::PIPELINE_OP: {
            // x -> fc1 -> drop -> fc2 : thread a tensor through layers.
            // AST is left-associative: ((x -> fc1) -> drop) -> fc2.
            // Walk the left-leaning tree to collect [input, ...stages].
            std::vector<const Expr*> input_args;
            std::vector<const Expr*> stages;
            const Expr* cur = expr;
            while (cur && cur->kind == Expr::PIPELINE_OP) {
                stages.push_back(cur->right.get());
                cur = cur->left.get();
            }
            if (cur) {
                input_args.push_back(cur);
                std::reverse(stages.begin(), stages.end());
            } else {
                report(expr->token.line, expr->token.column,
                       "Invalid pipeline with no input tensor");
                break;
            }

            // Resolve the input tensor type.
            TensorType current_type;
            bool have_type = false;
            check_expr(const_cast<Expr*>(cur));
            if (cur->inferred_type && cur->inferred_type->is_tensor()) {
                current_type = cur->inferred_type->tensor_type;
                have_type = true;
            }

            // Apply each stage in order.
            for (const Expr* stage : stages) {
                if (!have_type) break;
                std::string name = stage->token.value;
                auto it = layer_rules_.find(name);
                if (it != layer_rules_.end()) {
                    TensorType next;
                    apply_layer(it->second, current_type, next,
                                expr->token.line, expr->token.column);
                    current_type = next;
                } else {
                    // Unknown stage (could be inline activation/transform);
                    // treat as shape-preserving passthrough for now.
                    (void)name;
                }
            }

            if (have_type) {
                expr->inferred_type =
                    std::make_unique<TypeNode>(resolve_tensor(current_type));
            }
            break;
        }
        case Expr::FUNCTION_CALL: {
            // Self-contained argument check
            for (auto& arg : expr->args) {
                check_expr(arg.get());
            }
            std::string func_name;
            if (expr->operand->kind == Expr::IDENTIFIER) {
                func_name = expr->operand->token.value;
            }
            // Special-case well-known functions
            if (func_name == "cross_entropy") {
                // loss is scalar
                expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                if (expr->args.size() >= 2) {
                    auto& a0 = expr->args[0];
                    auto& a1 = expr->args[1];
                    if (a0->inferred_type && a1->inferred_type &&
                        a0->inferred_type->is_tensor() && a1->inferred_type->is_tensor()) {
                        auto& preds = a0->inferred_type->tensor_type;
                        auto& labels = a1->inferred_type->tensor_type;
                        // preds: [B, C], labels: [B, C]
                        if (preds.dims.size() != 2 || labels.dims.size() != 2) {
                            report(expr->token.line, expr->token.column,
                                   "cross_entropy expects 2D tensors [B, C]");
                        } else {
                            unify_dim(const_cast<DimExpr&>(preds.dims[0]),
                                      labels.dims[0], expr->token.line, expr->token.column);
                            unify_dim(const_cast<DimExpr&>(preds.dims[1]),
                                      labels.dims[1], expr->token.line, expr->token.column);
                        }
                    }
                }
            } else if (func_name == "Dense" || func_name == "Linear" ||
                       func_name == "Dropout" || func_name == "LayerNorm" ||
                       func_name == "Attention" || func_name == "Embedding") {
                // Layer constructors produce a layer object; infer minimal
                expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
            } else if (func_name == "relu" || func_name == "leaky_relu" ||
                       func_name == "sigmoid" || func_name == "tanh" ||
                       func_name == "silu" || func_name == "swish" ||
                       func_name == "gelu" || func_name == "softmax" ||
                       func_name == "identity" || func_name == "dropout") {
                // Elementwise activation: shape-preserving passthrough.
                if (!expr->args.empty() && expr->args[0]->inferred_type) {
                    expr->inferred_type = std::make_unique<TypeNode>(*expr->args[0]->inferred_type);
                } else {
                    expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                }
            } else if (func_name == "forward" || func_name == "step") {
                // Network method call; result is a tensor (passthrough).
                if (!expr->args.empty() && expr->args[0]->inferred_type) {
                    expr->inferred_type = std::make_unique<TypeNode>(*expr->args[0]->inferred_type);
                } else {
                    expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                }
            } else {
                auto it = functions_.find(func_name);
                if (it != functions_.end()) {
                    if (it->second.return_type) {
                        expr->inferred_type = std::make_unique<TypeNode>(*it->second.return_type);
                    } else {
                        expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                    }
                } else {
                    report(expr->token.line, expr->token.column,
                           "Unknown function '" + func_name + "'");
                    expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                }
            }
            break;
        }
        case Expr::INDEX_OP: {
            check_expr(expr->operand.get());
            for (auto& idx : expr->indices) {
                check_expr(idx.get());
            }
            if (expr->operand->inferred_type && expr->operand->inferred_type->is_tensor()) {
                if (!expr->indices.empty()) {
                    // Reduce dim by index count -> scalar or smaller tensor
                    expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
                }
            }
            break;
        }
        case Expr::TYPE_CAST:
            check_expr(expr->operand.get());
            if (expr->operand->inferred_type) {
                expr->inferred_type = std::make_unique<TypeNode>(*expr->operand->inferred_type);
            }
            break;
        case Expr::TENSOR_LITERAL:
            expr->inferred_type = std::make_unique<TypeNode>(Dtype::Float32);
            break;
    }
}

bool ShapeChecker::matmul_compatible(const TensorType& lhs, const TensorType& rhs,
                                     uint32_t line, uint32_t col) {
    if (lhs.dims.size() != 2 || rhs.dims.size() != 2) {
        report(line, col, "Matmul @ requires 2D tensors, got " +
               std::to_string(lhs.dims.size()) + "D and " + std::to_string(rhs.dims.size()) + "D");
        return false;
    }
    // lhs: [A, B], rhs: [C, D] -> requires B == C
    const DimExpr& inner_lhs = lhs.dims[1];
    const DimExpr& inner_rhs = rhs.dims[0];

    if (!unify_dim(const_cast<DimExpr&>(inner_lhs), inner_rhs, line, col)) {
        return false;
    }
    // Report informative message if mismatch
    if (inner_lhs.is_const() && inner_rhs.is_const() && inner_lhs.const_value != inner_rhs.const_value) {
        report(line, col, "Shape mismatch in @: inner dims " +
               std::to_string(inner_lhs.const_value) + " vs " + std::to_string(inner_rhs.const_value));
        return false;
    }
    return true;
}

TensorType ShapeChecker::infer_matmul(const TensorType& lhs, const TensorType& rhs) {
    TensorType result;
    result.dims.push_back(lhs.dims[0]); // row dim from lhs
    result.dims.push_back(rhs.dims[1]); // col dim from rhs
    result.dtype = lhs.dtype;
    return result;
}

bool ShapeChecker::unify_dim(const DimExpr& a_, const DimExpr& b_, uint32_t line, uint32_t col) {
    DimExpr a = a_;
    DimExpr b = b_;

    // Dynamic unifies with anything -> no new constraint, but return success
    // (a symbolic bound to Dynamic is treated as permanently-unbound).
    if (a.is_dynamic() || b.is_dynamic()) return true;

    if (a.is_const() && b.is_const()) {
        if (a.const_value == b.const_value) return true;
        report(line, col, "Dimension mismatch: constant " + std::to_string(a.const_value) +
               " vs " + std::to_string(b.const_value));
        return false;
    }

    // At least one side is symbolic. Route through the global constraint store.
    if (a.is_symbolic() && b.is_symbolic()) {
        if (a.symbolic_name == b.symbolic_name) return true;
        merge_symbols(a.symbolic_name, b.symbolic_name);
        return true;
    }
    if (a.is_symbolic() && b.is_const()) {
        record_symbol_const(a.symbolic_name, b.const_value);
        return true;
    }
    if (a.is_const() && b.is_symbolic()) {
        record_symbol_const(b.symbolic_name, a.const_value);
        return true;
    }
    // Fallthrough
    return true;
}

void ShapeChecker::record_symbol_const(const std::string& name, int64_t value) {
    auto& sb = symbols_[name];
    if (sb.bound_to_const) {
        if (sb.const_value != value) {
            // Conflict: two different constants for same symbol. Could warn,
            // but within a single program this indicates genuinely incompatible
            // shapes. We keep the first binding to avoid cascading noise.
            return;
        }
        return;
    }
    sb.bound_to_const = true;
    sb.const_value = value;
    sb.binds_to.clear();
}

void ShapeChecker::merge_symbols(const std::string& a, const std::string& b) {
    if (a == b) return;
    auto& sa = symbols_[a];
    auto& sb = symbols_[b];
    // If one side is already bound to a constant, propagate it.
    if (sa.bound_to_const) { record_symbol_const(b, sa.const_value); return; }
    if (sb.bound_to_const) { record_symbol_const(a, sb.const_value); return; }
    // Otherwise chain them.
    if (sa.binds_to.empty()) sa.binds_to = b;
    else merge_symbols(sa.binds_to, b);
}

DimExpr ShapeChecker::resolve_dim(const DimExpr& d) const {
    if (!d.is_symbolic()) return d;
    // Follow bindings
    std::string cur = d.symbolic_name;
    std::unordered_set<std::string> seen;
    while (true) {
        if (seen.count(cur)) break;
        seen.insert(cur);
        auto it = symbols_.find(cur);
        if (it == symbols_.end()) break;
        if (it->second.bound_to_const) {
            return DimExpr::constant(it->second.const_value);
        }
        if (it->second.binds_to.empty()) break;
        cur = it->second.binds_to;
    }
    return DimExpr::symbolic(cur);
}

TensorType ShapeChecker::resolve_tensor(const TensorType& t) const {
    TensorType r = t;
    for (auto& dim : r.dims) {
        dim = resolve_dim(dim);
    }
    return r;
}

std::vector<DimExpr> LayerRule::apply(const std::vector<DimExpr>& in_dims, bool& ok) const {
    ok = false;
    std::vector<DimExpr> base;
    switch (kind) {
        case LayerKind::Dense:
        case LayerKind::Linear: {
            // [..., in_dim] -> [..., out_dim]
            ok = true;
            if (in_dims.empty()) { ok = false; return base; }
            base = in_dims;
            base.back() = out;
            return base;
        }
        case LayerKind::Dropout:
        case LayerKind::LayerNorm:
        case LayerKind::Activation:
            // shape-preserving
            ok = true;
            return in_dims;
        case LayerKind::Embedding: {
            // [B, seq] -> [B, seq, emb_dim]
            ok = true;
            if (in_dims.empty()) { ok = false; return base; }
            base = in_dims;
            base.push_back(emb_dim);
            return base;
        }
        case LayerKind::Attention: {
            // [B, seq, dim] keeps shape (self-attention over an emb dim)
            ok = true;
            return in_dims;
        }
        default:
            ok = false;
            return base;
    }
}

void ShapeChecker::bind_alias_dim(DimExpr& dim) {
    if (dim.kind == DimExpr::SYMBOLIC) {
        auto it = type_aliases_.find(dim.symbolic_name);
        if (it != type_aliases_.end() && it->second.is_const()) {
            dim = it->second;
        } // symbolic alias keeps symbolic (dependent type)
    }
}

LayerRule ShapeChecker::parse_layer_rule(Stmt* layer) {
    LayerRule rule;
    std::string lt = layer->layer_type;
    if (lt == "Dense" || lt == "Linear") rule.kind = LayerKind::Dense;
    else if (lt == "Dropout") rule.kind = LayerKind::Dropout;
    else if (lt == "LayerNorm") rule.kind = LayerKind::LayerNorm;
    else if (lt == "Attention" || lt == "MultiHeadAttention") rule.kind = LayerKind::Attention;
    else if (lt == "Embedding") rule.kind = LayerKind::Embedding;
    else rule.kind = LayerKind::Unknown;

    for (auto& p : layer->layer_params) {
        std::string v;
        if (p.value) {
            if (p.value->kind == Expr::LITERAL_INT) v = p.value->token.value;
            else if (p.value->kind == Expr::IDENTIFIER) v = p.value->token.value;
            else v = p.value->token.value;
        }
        if (p.name == "in" || p.name == "in_features") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.in = DimExpr::constant(std::stoll(v));
            else rule.in = DimExpr::symbolic(v);
        } else if (p.name == "out" || p.name == "out_features") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.out = DimExpr::constant(std::stoll(v));
            else rule.out = DimExpr::symbolic(v);
        } else if (p.name == "activation" || p.name == "act") {
            rule.activation = v;
        } else if (p.name == "rate" || p.name == "p") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.rate = std::stod(v);
        } else if (p.name == "vocab_size") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.emb_vocab = DimExpr::constant(std::stoll(v));
            else rule.emb_vocab = DimExpr::symbolic(v);
        } else if (p.name == "d_model" || p.name == "n_embd" || p.name == "dim") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.emb_dim = DimExpr::constant(std::stoll(v));
            else rule.emb_dim = DimExpr::symbolic(v);
        } else if (p.name == "heads" || p.name == "num_heads") {
            if (!v.empty() && std::isdigit((unsigned char)v[0])) rule.num_heads = DimExpr::constant(std::stoll(v));
            else rule.num_heads = DimExpr::symbolic(v);
        }
    }
    return rule;
}

void ShapeChecker::apply_layer(const LayerRule& rule, const TensorType& in,
                               TensorType& out, uint32_t line, uint32_t col) {
    bool ok = false;
    auto out_dims = rule.apply(in.dims, ok);
    if (!ok) {
        report(line, col, "Layer cannot be applied to input of rank " +
               std::to_string(in.dims.size()));
        out = in;
        return;
    }
    out.dims = out_dims;
    out.dtype = in.dtype;
    // Resolve any symbolic choices introduced by the rule
    for (auto& d : out.dims) {
        d = resolve_dim(d);
        bind_alias_dim(d);
        d = resolve_dim(d);
    }
}

TypePtr ShapeChecker::lookup_var(const std::string& name) {
    auto it = variables_.find(name);
    if (it != variables_.end() && it->second) {
        return std::make_unique<TypeNode>(*it->second);
    }
    return nullptr;
}

} // namespace ns
