#include "ns/mlir/eval.hpp"
#include <cmath>
#include <algorithm>

namespace ns {

int64_t TensorBuffer::numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return shape.empty() ? 1 : n;
}

const TensorBuffer* ModuleEvaluator::getv(const std::string& id) const {
    auto it = values_.find(id);
    return it == values_.end() ? nullptr : &it->second;
}

void ModuleEvaluator::bind(const std::string& id, std::vector<double> data,
                           std::vector<int64_t> shape) {
    TensorBuffer tb;
    tb.data = std::move(data);
    tb.shape = std::move(shape);
    values_[id] = std::move(tb);
}

bool ModuleEvaluator::run(const MLIRFunction& fn, const MLIRModule& mod,
                          std::string want, TensorBuffer& out) {
    errors_.clear();
    for (const auto& instr : fn.instructions) {
        if (!eval_instr(instr, mod)) return false;
    }
    if (want.empty()) want = fn.return_id;
    const TensorBuffer* v = getv(want);
    if (!v) { errors_.push_back("no value for " + want); return false; }
    out = *v;
    return true;
}

void ModuleEvaluator::matmul(const TensorBuffer& a, const TensorBuffer& b, TensorBuffer& c) {
    // a: [M, K], b: [K, N] -> c: [M, N]
    int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
    c.shape = {M, N};
    c.data.assign((size_t)(M * N), 0.0);
    for (int64_t i = 0; i < M; i++)
        for (int64_t k = 0; k < K; k++) {
            double av = a.data[(size_t)(i * K + k)];
            if (av == 0.0) continue;
            for (int64_t j = 0; j < N; j++)
                c.data[(size_t)(i * N + j)] += av * b.data[(size_t)(k * N + j)];
        }
}

void ModuleEvaluator::activation(const std::string& act, TensorBuffer& m) {
    for (auto& v : m.data) {
        if (act == "relu") v = v > 0 ? v : 0.0;
        else if (act == "leaky_relu") v = v > 0 ? v : 0.01 * v;
        else if (act == "sigmoid") v = 1.0 / (1.0 + std::exp(-v));
        else if (act == "tanh") v = std::tanh(v);
        else if (act == "swish") v = v / (1.0 + std::exp(-v));
        else if (act == "gelu") v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
        else if (act == "silu") v = v / (1.0 + std::exp(-v));
        else if (act == "identity") { /* no-op */ }
    }
}

void ModuleEvaluator::elementwise(const TensorBuffer& a, const TensorBuffer* b,
                                  const std::string& op, TensorBuffer& out) {
    out.shape = a.shape;
    out.data.resize(a.data.size());
    for (size_t i = 0; i < a.data.size(); i++) {
        double rhs = b ? b->data[i % b->data.size()] : 0.0;
        if (op == "+") out.data[i] = a.data[i] + rhs;
        else if (op == "-") out.data[i] = a.data[i] - rhs;
        else if (op == "*") out.data[i] = a.data[i] * rhs;
        else if (op == "/") out.data[i] = a.data[i] / rhs;
    }
}

void ModuleEvaluator::layernorm(TensorBuffer& m) {
    // Normalize along the last (contiguous) dimension per row.
    int64_t last = m.shape[m.shape.size() - 1];
    int64_t rows = m.numel() / last;
    for (int64_t r = 0; r < rows; r++) {
        double mean = 0.0, var = 0.0;
        for (int64_t j = 0; j < last; j++) mean += m.data[(size_t)(r * last + j)];
        mean /= last;
        for (int64_t j = 0; j < last; j++) {
            double d = m.data[(size_t)(r * last + j)] - mean;
            var += d * d;
        }
        var /= last;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int64_t j = 0; j < last; j++)
            m.data[(size_t)(r * last + j)] = (m.data[(size_t)(r * last + j)] - mean) * inv;
    }
}

void ModuleEvaluator::softmax(TensorBuffer& m) {
    int64_t last = m.shape[m.shape.size() - 1];
    int64_t rows = m.numel() / last;
    for (int64_t r = 0; r < rows; r++) {
        double mx = m.data[(size_t)(r * last)];
        for (int64_t j = 1; j < last; j++) mx = std::max(mx, m.data[(size_t)(r * last + j)]);
        double s = 0.0;
        for (int64_t j = 0; j < last; j++) s += std::exp(m.data[(size_t)(r * last + j)] - mx);
        for (int64_t j = 0; j < last; j++)
            m.data[(size_t)(r * last + j)] = std::exp(m.data[(size_t)(r * last + j)] - mx) / s;
    }
}

bool ModuleEvaluator::eval_instr(const MLIRInstr& instr, const MLIRModule& mod) {
    switch (instr.op) {
        case MLIROp::MATMUL: {
            const TensorBuffer* a = getv(instr.operands[0]);
            const TensorBuffer* b = getv(instr.operands[1]);
            if (!a || !b) { errors_.push_back("matmul missing operand"); return false; }
            TensorBuffer c;
            matmul(*a, *b, c);
            values_[instr.result_id] = std::move(c);
            return true;
        }
        case MLIROp::DROPOUT: {
            const TensorBuffer* a = getv(instr.operands[0]);
            if (!a) return false;
            values_[instr.result_id] = *a;    // inference: identity
            return true;
        }
        case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
        case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
        case MLIROp::SILU: case MLIROp::IDENTITY: {
            const TensorBuffer* a = getv(instr.operands[0]);
            if (!a) return false;
            std::string act;
            switch (instr.op) {
                case MLIROp::RELU: act = "relu"; break;
                case MLIROp::LEAKY_RELU: act = "leaky_relu"; break;
                case MLIROp::SIGMOID: act = "sigmoid"; break;
                case MLIROp::TANH: act = "tanh"; break;
                case MLIROp::SWISH: act = "swish"; break;
                case MLIROp::GELU: act = "gelu"; break;
                case MLIROp::SILU: act = "silu"; break;
                default: act = "identity"; break;
            }
            TensorBuffer out = *a;
            activation(act, out);
            values_[instr.result_id] = std::move(out);
            return true;
        }
        case MLIROp::ELEMENTWISE_BINOP: {
            const TensorBuffer* a = getv(instr.operands[0]);
            if (!a) return false;
            const TensorBuffer* b = instr.operands.size() > 1 ? getv(instr.operands[1]) : nullptr;
            TensorBuffer out;
            elementwise(*a, b, instr.attribute, out);
            values_[instr.result_id] = std::move(out);
            return true;
        }
        case MLIROp::LAYERNORM: {
            const TensorBuffer* a = getv(instr.operands[0]);
            if (!a) return false;
            TensorBuffer out = *a;
            layernorm(out);
            values_[instr.result_id] = std::move(out);
            return true;
        }
        case MLIROp::SOFTMAX: {
            const TensorBuffer* a = getv(instr.operands[0]);
            if (!a) return false;
            TensorBuffer out = *a;
            softmax(out);
            values_[instr.result_id] = std::move(out);
            return true;
        }
        case MLIROp::FUSED:
            return eval_fused(instr, mod);
        default:
            return true; // non-tensor markers (GRAD, FORWARD, layer decls) skipped
    }
}

bool ModuleEvaluator::eval_fused(const MLIRInstr& instr, const MLIRModule& mod) {
    // Reconstruct the fused computation: GEMM then each epilogue op.
    const TensorBuffer* a = getv(instr.operands[0]);
    const TensorBuffer* b = getv(instr.operands[1]);
    if (!a || !b) return false;

    TensorBuffer c;
    matmul(*a, *b, c);

    // Find the matching group descriptor by result id.
    const FusedopGroup* group = nullptr;
    for (const auto& g : mod.fused_groups) {
        if (g.result_id == instr.result_id) { group = &g; break; }
    }
    if (!group) { errors_.push_back("no fusion group for " + instr.result_id); return false; }

    size_t extra = 0;
    for (const auto& opname : group->ops) {
        if (opname == "+" || opname == "-" || opname == "*" || opname == "/") {
            const TensorBuffer* rhs = nullptr;
            if (extra < group->epilogue_operands.size())
                rhs = getv(group->epilogue_operands[extra]);
            TensorBuffer out;
            elementwise(c, rhs, opname, out);
            c = out;
            extra++;
        } else if (opname == "layernorm") {
            layernorm(c);
        } else if (opname == "softmax") {
            softmax(c);
        } else {
            activation(opname, c);
        }
    }
    values_[instr.result_id] = std::move(c);
    return true;
}

} // namespace ns
