#include "ns/mlir/mlir_compiler.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace ns {

std::string MLIRCompiler::new_temp(const std::string& prefix) {
    return prefix + std::to_string(temp_counter_++);
}

MLIRModule MLIRCompiler::compile(Program& program) {
    collect_aliases(program);
    MLIRModule module;
    MLIRFunction main_fn;
    main_fn.name = "main";
    bool main_has_content = false;

    for (auto& stmt : program.top_level) {
        switch (stmt->kind) {
            case Stmt::FN_DECL: compile_fn(stmt.get(), module); break;
            case Stmt::NETWORK_DECL: compile_network(stmt.get(), module); break;
            case Stmt::VAR_DECL:
            case Stmt::VAR_ASSIGN:
            case Stmt::EXPR_STMT:
            case Stmt::RETURN_STMT:
                // Top-level compute goes into an implicit @main function.
                compile_stmt(stmt.get(), main_fn);
                main_has_content = true;
                break;
            default: break; // top-level type decls etc. skipped in IR
        }
    }

    if (main_has_content) {
        module.functions.push_back(std::move(main_fn));
    }
    return module;
}

void MLIRCompiler::collect_aliases(Program& program) {
    for (auto& stmt : program.top_level) {
        if (stmt->kind == Stmt::TYPE_DECL && stmt->alias_expr) {
            Expr* e = stmt->alias_expr.get();
            if (e->kind == Expr::LITERAL_INT) {
                try { aliases_[stmt->alias_name] = std::stoll(e->token.value); }
                catch (...) { }
            } else if (e->kind == Expr::IDENTIFIER) {
                int64_t v;
                if (resolve_dim_int(e, v)) aliases_[stmt->alias_name] = v;
            }
        }
    }
}

bool MLIRCompiler::resolve_dim_int(const Expr* expr, int64_t& out) {
    if (!expr) return false;
    if (expr->kind == Expr::LITERAL_INT) {
        try { out = std::stoll(expr->token.value); return true; }
        catch (...) { return false; }
    }
    if (expr->kind == Expr::IDENTIFIER) {
        auto it = aliases_.find(expr->token.value);
        if (it != aliases_.end()) { out = it->second; return true; }
    }
    return false;
}

void MLIRCompiler::compile_fn(Stmt* stmt, MLIRModule& module) {
    MLIRFunction fn;
    fn.name = stmt->fn_name;
    compile_stmt(stmt->body.get(), fn);
    module.functions.push_back(std::move(fn));
}

void MLIRCompiler::compile_network(Stmt* stmt, MLIRModule& module) {
    MLIRFunction fn;
    fn.name = stmt->network_name;

    // Emit weight buffer allocs for trainable layers BEFORE the forward body.
    // Dense: weight [in, out]. Layer dims may be constant literals or aliases
    // (resolved via aliases_); dynamic dims fall back to Tensor dims only.
    for (auto& layer : stmt->layers) {
        if (!layer) continue;
        LayerMeta meta;
        meta.type = layer->layer_type;
        for (auto& p : layer->layer_params) {
            if (p.name == "activation" && p.value) {
                std::string av = p.value->token.value;
                if (!av.empty() && av != "Identity") meta.activation = av;
            } else if (p.name == "rate" && p.value) {
                try { meta.dropout_rate = std::stod(p.value->token.value); }
                catch (...) {}
            }
        }
        layer_meta_[layer->layer_name] = meta;

        bool trainable = layer->layer_type == "Dense" || layer->layer_type == "Linear";
        if (!trainable) continue;

        int64_t in_d = -1, out_d = -1;
        for (auto& p : layer->layer_params) {
            if (p.name == "in" && p.value) { int64_t v; if (resolve_dim_int(p.value.get(), v)) in_d = v; }
            else if (p.name == "out" && p.value) { int64_t v; if (resolve_dim_int(p.value.get(), v)) out_d = v; }
        }
        if (in_d <= 0 || out_d <= 0) continue; // dynamic weights: not materialized here

        auto instr = MLIRInstr(MLIROp::TENSOR_ALLOC, layer->layer_name + "_w");
        instr.result_type = TensorType({DimExpr::constant(in_d), DimExpr::constant(out_d)},
                                       Dtype::Float32);
        instr.comment = "allocate weight [" + std::to_string(in_d) + ", " +
                        std::to_string(out_d) + "]";
        layer_weight_id_[layer->layer_name] = layer->layer_name + "_w";
        fn.instructions.push_back(instr);
    }

    // Note: because the layer's weight shape may differ from the operand that
    // flows through the pipeline (a dynamic leading dim), we additionally mark
    // the layer operand weight shape on the matmul below. The alloc above gives
    // the codegen a concrete [in, out] to size the weight buffer.

    // Compile forward
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::FORWARD_DECL) {
            compile_stmt(method->body.get(), fn);
        }
    }

    module.functions.push_back(std::move(fn));
}

void MLIRCompiler::compile_stmt(Stmt* stmt, MLIRFunction& fn) {
    if (!stmt) return;
    switch (stmt->kind) {
        case Stmt::EXPR_STMT: {
            MLIRValue out;
            compile_expr(stmt->expr.get(), fn, out);
            break;
        }
        case Stmt::RETURN_STMT: {
            if (stmt->init_expr) {
                MLIRValue out;
                compile_expr(stmt->init_expr.get(), fn, out);
                fn.return_id = out.id;
            }
            break;
        }
        case Stmt::VAR_ASSIGN: {
            if (stmt->init_expr) {
                MLIRValue out;
                compile_expr(stmt->init_expr.get(), fn, out);
            }
            break;
        }
        case Stmt::BLOCK: {
            for (auto& s : stmt->statements) {
                compile_stmt(s.get(), fn);
            }
            break;
        }
        case Stmt::GRAD_BLOCK: {
            // Emit GRAD marker
            auto instr = MLIRInstr(MLIROp::GRAD, "");
            instr.comment = "gradient block";
            fn.instructions.push_back(instr);
            if (stmt->grad_body) {
                compile_stmt(stmt->grad_body.get(), fn);
            }
            break;
        }
        case Stmt::FORWARD_DECL: {
            auto instr = MLIRInstr(MLIROp::FORWARD, "");
            instr.comment = "forward() entry";
            fn.instructions.push_back(instr);
            compile_stmt(stmt->body.get(), fn);
            break;
        }
        case Stmt::LAYER_DECL: {
            MLIROp op = MLIROp::LAYER_DENSE;
            std::string type = stmt->layer_type;
            if (type == "Dense" || type == "Linear") op = MLIROp::LAYER_DENSE;
            else if (type == "Dropout") op = MLIROp::LAYER_DROPOUT;
            else if (type == "Attention") op = MLIROp::LAYER_ATTENTION;
            else if (type == "Embedding") op = MLIROp::LAYER_EMBEDDING;
            else if (type == "LayerNorm") op = MLIROp::LAYER_LAYERNORM;

            auto instr = MLIRInstr(op, stmt->layer_name);
            instr.comment = "layer " + type;
            // Extract params into attr string, e.g. "in:128,out:512"
            std::string params;
            for (size_t i = 0; i < stmt->layer_params.size(); i++) {
                auto& p = stmt->layer_params[i];
                if (i > 0) params += ",";
                params += p.name;
                if (p.value) params += ":" + p.value->token.value;
            }
            instr.attribute = params;
            fn.instructions.push_back(instr);
            break;
        }
        case Stmt::VAR_DECL: {
            if (stmt->var_type && stmt->var_type->is_tensor()) {
                auto instr = MLIRInstr(MLIROp::TENSOR_ALLOC, stmt->var_name);
                instr.result_type = stmt->var_type->tensor_type;
                instr.comment = "allocate " + tensor_type_to_string(stmt->var_type->tensor_type);
                fn.instructions.push_back(instr);
            }
            if (stmt->init_expr) {
                MLIRValue out;
                compile_expr(stmt->init_expr.get(), fn, out);
            }
            break;
        }
        default:
            break;
    }
}

void MLIRCompiler::compile_expr(Expr* expr, MLIRFunction& fn, MLIRValue& out) {
    if (!expr) return;

    switch (expr->kind) {
        case Expr::LITERAL_INT:
        case Expr::LITERAL_FLOAT:
        case Expr::LITERAL_BOOL: {
            auto instr = MLIRInstr(MLIROp::CONSTANT, new_temp("c"));
            instr.attribute = expr->token.value;
            instr.comment = "constant " + expr->token.value;
            fn.instructions.push_back(instr);
            out.id = instr.result_id;
            break;
        }
        case Expr::IDENTIFIER: {
            out.id = expr->token.value;
            if (expr->inferred_type && expr->inferred_type->is_tensor()) {
                out.type = expr->inferred_type->tensor_type;
            }
            break;
        }
        case Expr::MATMUL_OP: {
            lower_matmul(expr, fn, out);
            break;
        }
        case Expr::PIPELINE_OP: {
            lower_pipeline(expr, fn, out);
            break;
        }
        case Expr::BINARY_OP: {
            MLIRValue lhs, rhs;
            compile_expr(expr->left.get(), fn, lhs);
            compile_expr(expr->right.get(), fn, rhs);

            MLIROp op;
            switch (expr->token.type) {
                case TokenType::OP_PLUS: op = MLIROp::ELEMENTWISE_BINOP; break;
                default: op = MLIROp::ELEMENTWISE_BINOP; break;
            }
            auto instr = MLIRInstr(op, new_temp("e"));
            instr.operands = {lhs.id, rhs.id};
            instr.attribute = expr->token.value; // the operator symbol
            fn.instructions.push_back(instr);
            out.id = instr.result_id;
            break;
        }
        case Expr::FUNCTION_CALL: {
            std::string func;
            if (expr->operand) func = expr->operand->token.value;

            MLIROp op;
            if (func == "cross_entropy") op = MLIROp::CROSS_ENTROPY;
            else if (func == "Dense" || func == "Linear") op = MLIROp::LAYER_DENSE;
            else if (func == "Dropout") op = MLIROp::LAYER_DROPOUT;
            else if (func == "LayerNorm") op = MLIROp::LAYER_LAYERNORM;
            else op = MLIROp::FN_CALL;

            auto instr = MLIRInstr(op, new_temp("f"));
            auto& self = expr->operand;
            if (self && self->kind == Expr::FUNCTION_CALL) {
                // skip
            }
            for (auto& arg : expr->args) {
                MLIRValue a;
                compile_expr(arg.get(), fn, a);
                instr.operands.push_back(a.id);
            }
            instr.comment = func + "()";
            fn.instructions.push_back(instr);
            out.id = instr.result_id;
            break;
        }
        case Expr::UNARY_OP: {
            compile_expr(expr->operand.get(), fn, out);
            break;
        }
        case Expr::INDEX_OP: {
            MLIRValue base;
            compile_expr(expr->operand.get(), fn, base);
            out = base;
            break;
        }
        default:
            out.id = "";
            break;
    }
}

void MLIRCompiler::lower_matmul(Expr* expr, MLIRFunction& fn, MLIRValue& out) {
    MLIRValue lhs, rhs;
    compile_expr(expr->left.get(), fn, lhs);
    compile_expr(expr->right.get(), fn, rhs);

    auto instr = MLIRInstr(MLIROp::MATMUL, new_temp("mm"));
    instr.operands = {lhs.id, rhs.id};
    if (expr->inferred_type && expr->inferred_type->is_tensor()) {
        instr.result_type = expr->inferred_type->tensor_type;
        out.type = expr->inferred_type->tensor_type;
    }
    instr.comment = lhs.id + " @ " + rhs.id;
    fn.instructions.push_back(instr);
    out.id = instr.result_id;
}

void MLIRCompiler::lower_pipeline(Expr* expr, MLIRFunction& fn, MLIRValue& out) {
    MLIRValue input;
    compile_expr(expr->left.get(), fn, input);
    lower_pipeline_apply(expr->right.get(), fn, input, out);
}

void MLIRCompiler::lower_activation(const std::string& act, MLIRValue& in,
                                    MLIRFunction& fn, MLIRValue& out) {
    MLIROp op = MLIROp::IDENTITY;
    if (act == "ReLU") op = MLIROp::RELU;
    else if (act == "Sigmoid") op = MLIROp::SIGMOID;
    else if (act == "Tanh") op = MLIROp::TANH;
    else if (act == "GELU") op = MLIROp::GELU;
    else if (act == "Swish" || act == "SiLU") op = MLIROp::SWISH;
    else if (act == "LeakyReLU") op = MLIROp::LEAKY_RELU;
    else if (act == "Softmax") op = MLIROp::SOFTMAX;

    auto instr = MLIRInstr(op, new_temp("act"));
    instr.operands = {in.id};
    instr.comment = act + "(" + in.id + ")";
    fn.instructions.push_back(instr);
    out.id = instr.result_id;
}

void MLIRCompiler::apply_pipeline_stage(const std::string& wname, MLIRValue& in,
                                        MLIRFunction& fn, MLIRValue& out) {
    auto miter = layer_meta_.find(wname);
    if (miter != layer_meta_.end() && miter->second.type == "Dropout") {
        auto instr = MLIRInstr(MLIROp::DROPOUT, new_temp("fc"));
        instr.operands = {in.id};
        instr.comment = "dropout(" + in.id + ")";
        fn.instructions.push_back(instr);
        out.id = instr.result_id;
        return;
    }

    auto wit = layer_weight_id_.find(wname);
    std::string wid = wit != layer_weight_id_.end() ? wit->second : wname;
    auto instr = MLIRInstr(MLIROp::MATMUL, new_temp("fc"));
    instr.operands = {in.id, wid};
    instr.comment = in.id + " -> " + wname;
    fn.instructions.push_back(instr);

    MLIRValue mm = in;
    mm.id = instr.result_id;
    std::string act;
    if (miter != layer_meta_.end()) act = miter->second.activation;
    if (!act.empty()) {
        lower_activation(act, mm, fn, out);
    } else {
        out.id = instr.result_id;
    }
}

void MLIRCompiler::lower_pipeline_apply(Expr* expr, MLIRFunction& fn,
                                        MLIRValue& in, MLIRValue& out) {
    if (expr->kind == Expr::PIPELINE_OP) {
        MLIRValue stage_out;
        lower_pipeline_apply(expr->left.get(), fn, in, stage_out);
        if (expr->right->kind == Expr::PIPELINE_OP) {
            lower_pipeline_apply(expr->right.get(), fn, stage_out, out);
        } else {
            apply_pipeline_stage(expr->right->token.value, stage_out, fn, out);
        }
    } else {
        apply_pipeline_stage(expr->token.value, in, fn, out);
    }
}

std::string MLIRCompiler::dump(MLIRModule& module) {
    std::ostringstream oss;
    oss << "// NeuralScript MLIR\n";
    for (auto& fn : module.functions) {
        oss << "ns.func @" << fn.name << " {\n";
        for (auto& instr : fn.instructions) {
            oss << "  ";
            switch (instr.op) {
                case MLIROp::TENSOR_ALLOC: oss << "ns.tensor.alloc"; break;
                case MLIROp::TENSOR_FREE: oss << "ns.tensor.free"; break;
                case MLIROp::MATMUL: oss << "ns.matmul"; break;
                case MLIROp::ELEMENTWISE_BINOP: oss << "ns.subview_binop"; break;
                case MLIROp::RELU: oss << "ns.activation.relu"; break;
                case MLIROp::LEAKY_RELU: oss << "ns.activation.leaky_relu"; break;
                case MLIROp::SIGMOID: oss << "ns.activation.sigmoid"; break;
                case MLIROp::TANH: oss << "ns.activation.tanh"; break;
                case MLIROp::SWISH: oss << "ns.activation.swish"; break;
                case MLIROp::GELU: oss << "ns.activation.gelu"; break;
                case MLIROp::SILU: oss << "ns.activation.silu"; break;
                case MLIROp::IDENTITY: oss << "ns.activation.identity"; break;
                case MLIROp::SOFTMAX: oss << "ns.activation.softmax"; break;
                case MLIROp::DROPOUT: oss << "ns.dropout"; break;
                case MLIROp::LAYERNORM: oss << "ns.layernorm"; break;
                case MLIROp::CROSS_ENTROPY: oss << "ns.cross_entropy"; break;
                case MLIROp::GRAD: oss << "ns.grad"; break;
                case MLIROp::FORWARD: oss << "ns.forward"; break;
                case MLIROp::LAYER_DENSE: oss << "ns.layer.dense"; break;
                case MLIROp::LAYER_DROPOUT: oss << "ns.layer.dropout"; break;
                case MLIROp::LAYER_ATTENTION: oss << "ns.layer.attention"; break;
                case MLIROp::LAYER_EMBEDDING: oss << "ns.layer.embedding"; break;
                case MLIROp::LAYER_LAYERNORM: oss << "ns.layer.layernorm"; break;
                case MLIROp::CONSTANT: oss << "ns.constant"; break;
                case MLIROp::FN_CALL: oss << "ns.fn.call"; break;
                case MLIROp::FUSED: oss << "ns.fused"; break;
                default: oss << "ns.op"; break;
            }
            if (!instr.result_id.empty()) {
                oss << " %" << instr.result_id;
            }
            for (auto& op : instr.operands) {
                oss << " %" << op;
            }
            if (!instr.result_type.dims.empty()) {
                oss << " : " << tensor_type_to_string(instr.result_type);
            }
            oss << " // " << instr.comment << "\n";
        }
        if (!fn.return_id.empty()) {
            oss << "  ns.return %" << fn.return_id << "\n";
        }
        oss << "}\n";
    }
    return oss.str();
}

} // namespace ns
