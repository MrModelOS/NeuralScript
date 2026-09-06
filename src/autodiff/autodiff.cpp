#include "ns/autodiff/autodiff.hpp"
#include <sstream>
#include <algorithm>

namespace ns {

// Combine a list of gradient contributions: "(c1 + c2 + ...)".
static std::string join(const std::string& open,
                        const std::vector<std::string>& items,
                        const std::string& close);

void Autodiff::add_node(const ADNode& n) {
    name_to_node_[n.name] = nodes_.size();
    nodes_.push_back(n);
}

void Autodiff::build_grad_block(Stmt* grad_body, const std::unordered_set<std::string>& weights) {
    weights_ = weights;
    nodes_.clear();
    name_to_node_.clear();
    grad_of_.clear();
    results_.clear();
    errors_.clear();

    if (!grad_body) return;

    // The grad block is typically a BLOCK of VAR_DECL / EXPR statements.
    std::vector<Stmt*> stmts;
    if (grad_body->kind == Stmt::BLOCK) {
        for (auto& s : grad_body->statements) stmts.push_back(s.get());
    } else {
        stmts.push_back(grad_body);
    }

    for (Stmt* s : stmts) {
        compute_statement(s);
    }
}

void Autodiff::compute_statement(Stmt* stmt) {
    if (!stmt) return;
    switch (stmt->kind) {
        case Stmt::VAR_DECL: {
            if (!stmt->init_expr) return;
            std::string out = stmt->var_name;
            TensorType t;
            compute_expr(stmt->init_expr.get(), out, t);
            break;
        }
        case Stmt::VAR_ASSIGN: {
            if (!stmt->init_expr) return;
            std::string out = stmt->var_name;
            TensorType t;
            compute_expr(stmt->init_expr.get(), out, t);
            break;
        }
        case Stmt::EXPR_STMT: {
            if (!stmt->expr) return;
            std::string out;
            TensorType t;
            compute_expr(stmt->expr.get(), out, t);
            break;
        }
        case Stmt::BLOCK: {
            for (auto& s : stmt->statements) compute_statement(s.get());
            break;
        }
        default:
            break;
    }
}

void Autodiff::compute_expr(Expr* expr, std::string& out_name, TensorType& out_type) {
    if (!expr) return;

    switch (expr->kind) {
        case Expr::MATMUL_OP: {
            std::string lhs, rhs;
            TensorType lt, rt;
            compute_expr(expr->left.get(), lhs, lt);
            compute_expr(expr->right.get(), rhs, rt);

            TensorType result;
            if (expr->inferred_type && expr->inferred_type->is_tensor()) {
                result = expr->inferred_type->tensor_type;
            } else {
                result.dims.push_back(DimExpr::symbolic("B"));
            }

            std::string out = (out_name.empty() || out_name == expr->token.value)
                ? "mm_" + std::to_string(nodes_.size()) : out_name;
            ADNode n;
            n.op = ADNode::MATMUL;
            n.name = out;
            n.type = result;
            n.inputs = {lhs, rhs};
            if (!out_name.empty() && out_name != expr->token.value) out = out_name;
            n.name = out;
            add_node(n);
            out_name = out;
            out_type = result;
            break;
        }
        case Expr::BINARY_OP: {
            std::string lhs, rhs;
            TensorType lt, rt;
            compute_expr(expr->left.get(), lhs, lt);
            compute_expr(expr->right.get(), rhs, rt);

            ADNode::Op op;
            bool ls = lt.dims.empty(), rs = rt.dims.empty();
            switch (expr->token.type) {
                case TokenType::OP_PLUS: op = ADNode::ADD; break;
                case TokenType::OP_MINUS: op = ADNode::SUB; break;
                case TokenType::OP_STAR: op = ADNode::MUL; break;
                case TokenType::OP_SLASH: op = ADNode::DIV; break;
                default: op = ADNode::ADD; break;
            }

            std::string out = (out_name.empty() || out_name == expr->token.value)
                ? "e_" + std::to_string(nodes_.size()) : out_name;
            ADNode n;
            n.op = op;
            n.name = out;
            n.type = ls ? rt : lt;
            if (lt.dims.empty()) n.type = rt; else if (rt.dims.empty()) n.type = lt;
            else n.type = lt;
            n.inputs = {lhs, rhs};
            n.input_scalar = {ls, rs};
            add_node(n);
            out_name = out;
            out_type = n.type;
            break;
        }
        case Expr::IDENTIFIER: {
            // A leaf variable (input tensor or weight).
            out_name = expr->token.value;
            if (expr->inferred_type && expr->inferred_type->is_tensor()) {
                out_type = expr->inferred_type->tensor_type;
            }
            // Register as a VAR_IN node so reverse pass can seed if it's the loss.
            if (name_to_node_.count(out_name) == 0) {
                ADNode n;
                n.op = ADNode::VAR_IN;
                n.name = out_name;
                n.type = out_type;
                add_node(n);
            }
            break;
        }
        case Expr::FUNCTION_CALL: {
            std::string fn;
            if (expr->operand) fn = expr->operand->token.value;

            // Gather arg names/types.
            std::vector<std::string> arg_names;
            std::vector<TensorType> arg_types;
            for (auto& a : expr->args) {
                std::string an;
                TensorType at;
                compute_expr(a.get(), an, at);
                arg_names.push_back(an);
                arg_types.push_back(at);
            }

            if (fn == "cross_entropy") {
                // Terminal loss: [B,C] preds, [B,C] labels -> scalar.
                std::string out = (out_name.empty()) ? "loss" : out_name;
                ADNode n;
                n.op = ADNode::CROSS_ENTROPY;
                n.name = out;
                n.type = TensorType(); // scalar
                if (arg_names.size() >= 1) n.inputs = {arg_names[0]};
                if (arg_names.size() >= 2) n.inputs.push_back(arg_names[1]);
                n.is_loss = true;
                add_node(n);
                out_name = out;
                out_type = TensorType();
            } else {
                // Unknown call: pass-through (no gradient contribution).
                std::string out = (out_name.empty()) ? "t_" + std::to_string(nodes_.size()) : out_name;
                ADNode n;
                n.op = ADNode::VAR_IN; // passthrough placeholder
                n.name = out;
                for (auto& an : arg_names) n.inputs.push_back(an);
                add_node(n);
                out_name = out;
            }
            break;
        }
        case Expr::UNARY_OP: {
            std::string in;
            TensorType it;
            compute_expr(expr->operand.get(), in, it);

            // Detect activation by token value.
            ADNode::Op op = ADNode::RELU;
            switch (expr->operand->token.type) {
                case TokenType::ACT_RELU: op = ADNode::RELU; break;
                case TokenType::ACT_TANH: op = ADNode::TANH; break;
                case TokenType::ACT_SIGMOID: op = ADNode::SIGMOID; break;
                case TokenType::ACT_SWISH: case TokenType::ACT_SILU: op = ADNode::SWISH; break;
                case TokenType::ACT_GELU: op = ADNode::GELU; break;
                default: op = ADNode::RELU; break;
            }
            std::string out = (out_name.empty()) ? "act_" + std::to_string(nodes_.size()) : out_name;
            ADNode n;
            n.op = op;
            n.name = out;
            n.type = it;
            n.inputs = {in};
            add_node(n);
            out_name = out;
            out_type = it;
            break;
        }
        default:
            out_name = (out_name.empty()) ? "t_" + std::to_string(nodes_.size()) : out_name;
            break;
    }
}

void Autodiff::emit_adjoin(size_t node_idx, const std::string& seed_adj_name) {
    // Placeholder: full reverse pass implemented in backward().
    (void)node_idx; (void)seed_adj_name;
}

void Autodiff::backward() {
    results_.clear();
    grad_of_.clear();

    std::vector<std::string> order;
    order.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); i++) order.push_back(nodes_[i].name);

    // Seed adjoints: every node gets an empty gradient set; the loss seeds dL/dL = 1.
    std::unordered_map<std::string, std::vector<std::string>> adj; // name -> contributions
    for (size_t i = 0; i < nodes_.size(); i++) adj[nodes_[i].name].clear();

    // Locate the loss node.
    size_t loss_idx = nodes_.size();
    for (size_t i = 0; i < nodes_.size(); i++) {
        if (nodes_[i].op == ADNode::CROSS_ENTROPY) { loss_idx = i; break; }
    }
    if (loss_idx == nodes_.size()) {
        // No explicit loss; nothing to differentiate past a scalar-less graph.
        return;
    }
    adj[nodes_[loss_idx].name].push_back("1");

    // Reverse topological order (nodes_ already in forward order).
    for (int i = (int)loss_idx; i >= 0; i--) {
        const ADNode& n = nodes_[i];
        auto it = adj.find(n.name);
        if (it == adj.end() || it->second.empty()) continue;
        const std::vector<std::string>& dL = it->second;

        switch (n.op) {
            case ADNode::MATMUL: {
                // C = A @ B ; dA = dC @ B^T ; dB = A^T @ dC
                const std::string& A = n.inputs[0];
                const std::string& B = n.inputs[1];
                std::string aT = "T(" + A + ")";
                std::string bT = "T(" + B + ")";
                // dA = dC @ B^T
                std::string dA = "d(" + A + ") += " + join("(", dL, ")") + " @ " + bT;
                // dB = A^T @ dC
                std::string dB = "d(" + B + ") += " + aT + " @ " + join("(", dL, ")");
                adj[A].push_back(dA);
                adj[B].push_back(dB);
                break;
            }
            case ADNode::ADD: {
                // dL/dA = dL, dL/dB = dL (scalar broadcast: reduce-sum, omitted here)
                adj[n.inputs[0]].push_back(join("(", dL, ")"));
                adj[n.inputs[1]].push_back(join("(", dL, ")"));
                break;
            }
            case ADNode::SUB: {
                adj[n.inputs[0]].push_back(join("(", dL, ")"));
                adj[n.inputs[1]].push_back("-" + join("(", dL, ")"));
                break;
            }
            case ADNode::MUL: {
                // dA += dBval * dC ;  dB += dAval * dC
                adj[n.inputs[0]].push_back(join("(", dL, ")") + " * " + n.inputs[1]);
                adj[n.inputs[1]].push_back(join("(", dL, ")") + " * " + n.inputs[0]);
                break;
            }
            case ADNode::DIV: {
                // C = A/B ; dA = dC/B ; dB = -dC*A/B^2
                std::string dA = join("(", dL, ")") + " / " + n.inputs[1];
                std::string dB = "-" + join("(", dL, ")") + " * " + n.inputs[0] +
                                 " / (" + n.inputs[1] + "^2)";
                adj[n.inputs[0]].push_back(dA);
                adj[n.inputs[1]].push_back(dB);
                break;
            }
            case ADNode::RELU: {
                // dA = dC * (A > 0)
                std::string dA = join("(", dL, ")") + " * (" + n.inputs[0] + " > 0)";
                adj[n.inputs[0]].push_back(dA);
                break;
            }
            case ADNode::SWISH: {
                // swish(a)=a*sigmoid(a); dA = dC * (s + a*s*(1-s))
                std::string a = n.inputs[0];
                std::string s = "sig(" + a + ")";
                std::string dA = join("(", dL, ")") + " * (" + s + " + " + a + "*" +
                                 s + "*(1-" + s + "))";
                adj[n.inputs[0]].push_back(dA);
                break;
            }
            case ADNode::TANH: {
                std::string dA = join("(", dL, ")") + " * (1 - tanh^2(" + n.inputs[0] + "))";
                adj[n.inputs[0]].push_back(dA);
                break;
            }
            case ADNode::SIGMOID: {
                std::string s = "sig(" + n.inputs[0] + ")";
                std::string dA = join("(", dL, ")") + " * " + s + "*(1-" + s + ")";
                adj[n.inputs[0]].push_back(dA);
                break;
            }
            case ADNode::GELU: {
                std::string x = n.inputs[0];
                std::string dA = join("(", dL, ")") + " * (0.5*(1+erf(" + x +
                                  "/sqrt(2))) + " + x + "*exp(-" + x + "^2/2)/sqrt(2*pi))";
                adj[n.inputs[0]].push_back(dA);
                break;
            }
            case ADNode::CROSS_ENTROPY: {
                // CE(pred,target): softmax grad. dpred = softmax(pred) - target (per class),
                // then / B (for mean reduction). Here we emit symbolic form.
                const std::string& pred = n.inputs[0];
                const std::string& tgt = n.inputs[1];
                std::string sm = "softmax(" + pred + ")";
                std::string dpred = "(" + sm + " - " + tgt + ")";
                adj[pred].push_back(dpred);
                break;
            }
            default:
                break;
        }
    }

    // Emit results for requested weights and all intermediate leaves.
    for (const std::string& name : order) {
        auto it = adj.find(name);
        if (it == adj.end() || it->second.empty()) continue;
        if (it->second.size() == 1 && it->second[0] == "1") continue; // skip the loss itself
        GradResult r;
        r.var = name;
        r.target = (weights_.count(name) ? GradTarget::TensorLeaf : GradTarget::Intermediate);
        r.contributions = it->second;
        results_.push_back(std::move(r));
    }
}

static std::string join(const std::string& open, const std::vector<std::string>& items, const std::string& close) {
    if (items.empty()) return open + close;
    std::string s;
    for (size_t i = 0; i < items.size(); i++) {
        if (i) s += " + ";
        s += items[i];
    }
    return open + s + close;
}

std::vector<Autodiff::GradResult> GradCompiler::compile_grad_block(
    Stmt* fn_body, const std::unordered_set<std::string>& weight_params) {
    errors_.clear();
    Autodiff ad;
    if (!fn_body) return {};
    // Find the grad block inside fn_body.
    Stmt* grad = nullptr;
    std::vector<Stmt*> stack{fn_body};
    while (!stack.empty()) {
        Stmt* s = stack.back(); stack.pop_back();
        if (s->kind == Stmt::GRAD_BLOCK) { grad = s; break; }
        if (s->kind == Stmt::BLOCK) {
            for (auto& st : s->statements) { if (st) stack.push_back(st.get()); }
        }
    }
    if (!grad) return {};
    ad.build_grad_block(grad->grad_body.get(), weight_params);
    ad.backward();
    for (auto& e : ad.errors()) errors_.push_back(e);
    return ad.results();
}

} // namespace ns
