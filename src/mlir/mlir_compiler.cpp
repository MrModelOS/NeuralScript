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

std::string MLIRCompiler::activation_name(MLIROp op) {
    switch (op) {
        case MLIROp::RELU: return "relu";
        case MLIROp::LEAKY_RELU: return "leaky_relu";
        case MLIROp::SIGMOID: return "sigmoid";
        case MLIROp::TANH: return "tanh";
        case MLIROp::SWISH: return "swish";
        case MLIROp::GELU: return "gelu";
        case MLIROp::SILU: return "silu";
        case MLIROp::IDENTITY: return "identity";
        case MLIROp::SOFTMAX: return "softmax";
        case MLIROp::LAYERNORM: return "layernorm";
        default: return "identity";
    }
}

void MLIRCompiler::compile_fn(Stmt* stmt, MLIRModule& module) {
    reset_locals();
    MLIRFunction fn;
    fn.name = stmt->fn_name;
    compile_stmt(stmt->body.get(), fn);
    module.functions.push_back(std::move(fn));
}

void MLIRCompiler::compile_network(Stmt* stmt, MLIRModule& module) {
    reset_locals();
    MLIRFunction fn;
    fn.name = stmt->network_name;

    // Weight buffer allocs for trainable layers, shared by forward() AND the
    // train() method (item: single weight set, no w1/w2 duplication).
    std::vector<MLIRInstr> weight_allocs;
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
        weight_allocs.push_back(instr);
    }

    // Note: because the layer's weight shape may differ from the operand that
    // flows through the pipeline (a dynamic leading dim), we additionally mark
    // the layer operand weight shape on the matmul below. The alloc above gives
    // the codegen a concrete [in, out] to size the weight buffer.

    // Forward pass(es) -> inference function named after the network.
    for (auto& w : weight_allocs) fn.instructions.push_back(w);
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::FORWARD_DECL) {
            compile_stmt(method->body.get(), fn);
        }
    }
    module.functions.push_back(std::move(fn));
    module.functions.back().is_train = false;

    // train() method(s) -> a separate function sharing the SAME weight allocs.
    // Reverse-mode ops produced by the grad block bind their gradients to the
    // network weight tensors directly (no separate w1/w2 in the source).
    MLIRFunction tfn;
    tfn.name = stmt->network_name + "_train";
    tfn.is_train = true;
    bool have_train = false;
    for (auto& w : weight_allocs) tfn.instructions.push_back(w);
    for (auto& method : stmt->methods) {
        if (method->kind == Stmt::TRAIN_DECL) {
            compile_stmt(method->body.get(), tfn);
            have_train = true;
        }
    }
    if (have_train) {
        module.functions.push_back(std::move(tfn));
    }
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
                if (!stmt->var_name.empty() && !out.id.empty())
                    local_var_ids_[stmt->var_name] = out.id;
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
            // AOT forward+backward: compile the grad body forward, then lower
            // the reverse-mode tape into backward MLIR ops plus an OPT_STEP.
            auto marker = MLIRInstr(MLIROp::GRAD, "");
            marker.comment = "gradient block (forward)";
            fn.instructions.push_back(marker);

            size_t fwd_start = fn.instructions.size();
            if (stmt->grad_body) {
                compile_stmt(stmt->grad_body.get(), fn);
            }
            size_t fwd_end = fn.instructions.size();

            // Reverse-mode lowering over the range [fwd_start, fwd_end).
            std::map<std::string, std::string> g; // result_id -> grad id
            for (size_t idx = fwd_end; idx > fwd_start; --idx) {
                // Copy by value: backward instrs are appended to the same
                // vector, which would invalidate a reference on realloc.
                const MLIRInstr i = fn.instructions[idx - 1];
                switch (i.op) {
                    case MLIROp::CROSS_ENTROPY: {
                        // Seed: dL/d(preds) = (softmax(preds) - labels) / B.
                        auto lg = MLIRInstr(MLIROp::LOSS_GRAD, new_temp("g"));
                        lg.operands = i.operands; // preds, labels
                        lg.attribute = "cross_entropy";
                        lg.comment = "d(cross_entropy)/d(preds)";
                        fn.instructions.push_back(lg);
                        g[i.result_id] = lg.result_id;
                        if (!i.operands.empty()) g[i.operands[0]] = lg.result_id;
                        break;
                    }
                    case MLIROp::MATMUL: {
                        auto cit = g.find(i.result_id);
                        if (cit == g.end() || i.operands.size() < 2) break;
                        // dA = dC @ B^T
                        auto ia = MLIRInstr(MLIROp::MATMUL_GRAD_A, new_temp("g"));
                        ia.operands = {cit->second, i.operands[1]};
                        ia.comment = "d" + i.operands[0] + " = d" + i.result_id + " @ " + i.operands[1] + "^T";
                        fn.instructions.push_back(ia);
                        // dB = A^T @ dC  (operands: A, dC, Bweight)
                        auto iw = MLIRInstr(MLIROp::MATMUL_GRAD_W, new_temp("g"));
                        iw.operands = {i.operands[0], cit->second, i.operands[1]};
                        iw.comment = "d" + i.operands[1] + " = " + i.operands[0] + "^T @ d" + i.result_id;
                        fn.instructions.push_back(iw);
                        g[i.operands[0]] = ia.result_id;
                        g[i.operands[1]] = iw.result_id;
                        break;
                    }
                    case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
                    case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
                    case MLIROp::SILU: case MLIROp::IDENTITY:
                    case MLIROp::SOFTMAX: case MLIROp::LAYERNORM: {
                        auto cit = g.find(i.result_id);
                        if (cit == g.end() || i.operands.empty()) break;
                        auto ig = MLIRInstr(MLIROp::ACTIVATION_GRAD, new_temp("g"));
                        ig.operands = {cit->second, i.operands[0]};
                        ig.attribute = activation_name(i.op);
                        ig.comment = "d" + i.operands[0] + " = d" + i.result_id + " * act'(in)";
                        fn.instructions.push_back(ig);
                        g[i.operands[0]] = ig.result_id;
                        break;
                    }
                    case MLIROp::DROPOUT: {
                        // Inverted dropout: forward saves a scaled binary mask
                        // m[i] = (keep ? 1/(1-p) : 0); backward is dOut * m.
                        if (i.operands.empty()) break;
                        auto cit = g.find(i.result_id);
                        if (cit == g.end()) break;
                        auto ig = MLIRInstr(MLIROp::ACTIVATION_GRAD, new_temp("g"));
                        ig.operands = {cit->second, i.result_id};
                        ig.attribute = "dropout";
                        ig.comment = "d" + i.operands[0] + " = d" + i.result_id +
                                     " * dropout_mask";
                        fn.instructions.push_back(ig);
                        g[i.operands[0]] = ig.result_id;
                        break;
                    }
                    default:
                        break;
                }
            }

            // Optimizer step: pair each weight alloc in fn with its gradient.
            std::vector<std::string> opt_ops;
            for (auto& instr : fn.instructions) {
                if (instr.op != MLIROp::TENSOR_ALLOC) continue;
                auto git = g.find(instr.result_id);
                if (git != g.end()) {
                    opt_ops.push_back(instr.result_id);
                    opt_ops.push_back(git->second);
                }
            }
            if (!opt_ops.empty()) {
                auto opt = MLIRInstr(MLIROp::OPT_STEP, "");
                opt.operands = std::move(opt_ops);
                opt.attribute = "muon";
                opt.comment = "optimizer step (Muon / AdamW)";
                fn.instructions.push_back(opt);
            }
            break;
        }
        case Stmt::TRAIN_DECL: {
            if (stmt->body) {
                compile_stmt(stmt->body.get(), fn);
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
                if (!stmt->var_name.empty() && !out.id.empty())
                    local_var_ids_[stmt->var_name] = out.id;
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
            // Local var binding first (mirrors source-level shadowing), then
            // network layer names resolve to their weight tensors so train()
            // can write `batch_x @ fc1` directly.
            auto lit = local_var_ids_.find(out.id);
            if (lit != local_var_ids_.end()) {
                out.id = lit->second;
            } else {
                auto wit = layer_weight_id_.find(out.id);
                if (wit != layer_weight_id_.end()) out.id = wit->second;
            }
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
            else if (func == "relu") op = MLIROp::RELU;
            else if (func == "leaky_relu") op = MLIROp::LEAKY_RELU;
            else if (func == "sigmoid") op = MLIROp::SIGMOID;
            else if (func == "tanh") op = MLIROp::TANH;
            else if (func == "silu" || func == "swish") op = MLIROp::SWISH;
            else if (func == "gelu") op = MLIROp::GELU;
            else if (func == "softmax") op = MLIROp::SOFTMAX;
            else if (func == "identity") op = MLIROp::IDENTITY;
            else if (func == "dropout") op = MLIROp::DROPOUT;
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
            if (op == MLIROp::DROPOUT && expr->args.size() >= 2) {
                Expr* rate_arg = expr->args[1].get();
                if (rate_arg->kind == Expr::LITERAL_INT ||
                    rate_arg->kind == Expr::LITERAL_FLOAT) {
                    try { instr.float_attr = std::stod(rate_arg->token.value); }
                    catch (...) { }
                }
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
    if (miter != layer_meta_.end()) {
        if (miter->second.type != "Dense" && miter->second.type != "Linear" &&
            miter->second.type != "Dropout") {
            throw std::runtime_error(
                "Unsupported layer type '" + miter->second.type +
                "' in AOT pipeline lowering (layer '" + wname +
                "'). Only Dense/Linear and Dropout layers can be lowered; " +
                "Attention/Embedding/LayerNorm kernels are not yet implemented.");
        }
        if (miter->second.type == "Dropout") {
            auto instr = MLIRInstr(MLIROp::DROPOUT, new_temp("fc"));
            instr.operands = {in.id};
            instr.float_attr = miter->second.dropout_rate;
            instr.comment = "dropout(" + in.id + ", p=" +
                            std::to_string(miter->second.dropout_rate) + ")";
            fn.instructions.push_back(instr);
            out.id = instr.result_id;
            return;
        }
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
                case MLIROp::MATMUL_GRAD_A: oss << "ns.grad.matmul.a"; break;
                case MLIROp::MATMUL_GRAD_W: oss << "ns.grad.matmul.w"; break;
                case MLIROp::ACTIVATION_GRAD: oss << "ns.grad.activation"; break;
                case MLIROp::LOSS_GRAD: oss << "ns.grad.loss"; break;
                case MLIROp::GRAD: oss << "ns.grad"; break;
                case MLIROp::FORWARD: oss << "ns.forward"; break;
                case MLIROp::LAYER_DENSE: oss << "ns.layer.dense"; break;
                case MLIROp::LAYER_DROPOUT: oss << "ns.layer.dropout"; break;
                case MLIROp::LAYER_ATTENTION: oss << "ns.layer.attention"; break;
                case MLIROp::LAYER_EMBEDDING: oss << "ns.layer.embedding"; break;
                case MLIROp::LAYER_LAYERNORM: oss << "ns.layer.layernorm"; break;
                case MLIROp::CONSTANT: oss << "ns.constant"; break;
                case MLIROp::FN_CALL: oss << "ns.fn.call"; break;
                case MLIROp::OPT_STEP: oss << "ns.opt.step"; break;
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
