#include "ns/training/trainer.hpp"
#include <cmath>
#include <unordered_set>

namespace ns {

NumericTrainer::NumericTrainer(const MLIRFunction& fn, const MLIRModule& mod)
    : fn_(fn), mod_(mod) {
    register_params();
}

void NumericTrainer::register_params() {
    for (const auto& instr : fn_.instructions) {
        if (instr.op == MLIROp::TENSOR_ALLOC) {
            Param p;
            p.id = instr.result_id;
            const auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) p.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) p.cols = d[1].const_value;
            p.data.assign((size_t)(p.rows * p.cols), 0.0);
            p.grad.assign(p.data.size(), 0.0);
            params_.push_back(std::move(p));
        }
    }
}

void NumericTrainer::bind_input(const std::string& id, std::vector<double> data,
                                std::vector<int64_t> shape) {
    TensorBuffer tb;
    tb.data = std::move(data);
    tb.shape = std::move(shape);
    inputs_[id] = tb;
    values_[id] = std::move(tb);
}

NumericTrainer::Param* NumericTrainer::param(const std::string& id) {
    for (auto& p : params_) if (p.id == id) return &p;
    return nullptr;
}

const TensorBuffer* NumericTrainer::value(const std::string& id) const {
    auto it = values_.find(id);
    return it == values_.end() ? nullptr : &it->second;
}

const TensorBuffer* NumericTrainer::forward() {
    tape_.clear();
    values_ = inputs_;   // restore bound inputs; clear leftover values
    for (const auto& instr : fn_.instructions) {
        if (instr.op == MLIROp::TENSOR_ALLOC) {
            // Materialize the (possibly overridden) weight into a buffer.
            Param* p = param(instr.result_id);
            if (p) {
                TensorBuffer tb;
                tb.data = p->data;
                tb.shape = {p->rows, p->cols};
                values_[p->id] = std::move(tb);
            }
            continue;
        }
        record(instr);
    }
    last_out_ = value(fn_.return_id);
    return last_out_;
}

void NumericTrainer::record(const MLIRInstr& instr) {
    Rec r;
    r.op = instr.op;
    r.result_id = instr.result_id;
    r.operands = instr.operands;
    r.attribute = instr.attribute;

    for (const auto& opnd : instr.operands) {
        auto it = values_.find(opnd);
        if (it != values_.end() && r.ins.size() < 2)
            r.ins.push_back(it->second);
    }
    // Every (first up-to-2) operand must resolve, in declared order; a
    // silently mismatched pairing would corrupt the forward/backward pass.
    if (r.ins.size() != std::min<size_t>(instr.operands.size(), 2)) {
        errors_.push_back("forward: missing operand for " + instr.result_id);
        return;
    }

    bool ok = false;
    switch (instr.op) {
        case MLIROp::MATMUL: ok = eval_matmul(r); break;
        case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
        case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
        case MLIROp::SILU: case MLIROp::IDENTITY: ok = eval_activation(r); break;
        case MLIROp::DROPOUT: {
            if (r.ins.size() >= 1) { r.out = r.ins[0]; ok = true; }
            break;
        }
        case MLIROp::ELEMENTWISE_BINOP: ok = eval_binop(r); break;
        case MLIROp::LAYERNORM: ok = eval_layernorm(r); break;
        case MLIROp::SOFTMAX: ok = eval_softmax(r); break;
        default:
            // GRAD / FORWARD / layer markers: no tensor result.
            return;
    }
    if (!ok) { errors_.push_back("forward failed at " + r.result_id); return; }
    values_[r.result_id] = r.out;
    tape_.push_back(std::move(r));
}

static void matmul_local(const TensorBuffer& a, const TensorBuffer& b, TensorBuffer& c) {
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

bool NumericTrainer::eval_matmul(Rec& r) {
    if (r.ins.size() < 2) return false;
    matmul_local(r.ins[0], r.ins[1], r.out);
    return true;
}

bool NumericTrainer::eval_activation(Rec& r) {
    if (r.ins.size() < 1) return false;
    TensorBuffer out = r.ins[0];
    std::string act;
    switch (r.op) {
        case MLIROp::RELU: act = "relu"; break;
        case MLIROp::LEAKY_RELU: act = "leaky_relu"; break;
        case MLIROp::SIGMOID: act = "sigmoid"; break;
        case MLIROp::TANH: act = "tanh"; break;
        case MLIROp::SWISH: act = "swish"; break;
        case MLIROp::GELU: act = "gelu"; break;
        case MLIROp::SILU: act = "silu"; break;
        default: act = "identity"; break;
    }
    for (auto& v : out.data) {
        if (act == "relu") v = v > 0 ? v : 0.0;
        else if (act == "leaky_relu") v = v > 0 ? v : 0.01 * v;
        else if (act == "sigmoid") v = 1.0 / (1.0 + std::exp(-v));
        else if (act == "tanh") v = std::tanh(v);
        else if (act == "swish") v = v / (1.0 + std::exp(-v));
        else if (act == "gelu") v = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
        else if (act == "silu") v = v / (1.0 + std::exp(-v));
    }
    r.out = std::move(out);
    return true;
}

bool NumericTrainer::eval_binop(Rec& r) {
    if (r.ins.size() < 1) return false;
    const TensorBuffer& a = r.ins[0];
    const TensorBuffer* b = r.ins.size() > 1 ? &r.ins[1] : nullptr;
    TensorBuffer out;
    out.shape = a.shape;
    out.data.resize(a.data.size());
    const std::string& op = r.attribute;
    for (size_t i = 0; i < a.data.size(); i++) {
        double bv = (b && !b->data.empty()) ? b->data[i % b->data.size()] : 0.0;
        double av = a.data[i];
        if (op == "+") out.data[i] = av + bv;
        else if (op == "-") out.data[i] = av - bv;
        else if (op == "*") out.data[i] = av * bv;
        else if (op == "/") out.data[i] = bv == 0.0 ? 0.0 : av / bv;
        else out.data[i] = av;
    }
    r.out = std::move(out);
    return true;
}

bool NumericTrainer::eval_layernorm(Rec& r) {
    if (r.ins.size() < 1) return false;
    const TensorBuffer& in = r.ins[0];
    TensorBuffer out = in;
    int64_t last = in.shape.empty() ? 1 : in.shape.back();
    int64_t rows = out.data.size() / last;
    for (int64_t row = 0; row < rows; row++) {
        double mean = 0.0, var = 0.0;
        for (int64_t j = 0; j < last; j++) mean += out.data[(size_t)(row * last + j)];
        mean /= last;
        for (int64_t j = 0; j < last; j++) {
            double d = out.data[(size_t)(row * last + j)] - mean;
            var += d * d;
        }
        var /= last;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int64_t j = 0; j < last; j++)
            out.data[(size_t)(row * last + j)] =
                (out.data[(size_t)(row * last + j)] - mean) * inv;
    }
    r.out = std::move(out);
    return true;
}

bool NumericTrainer::eval_softmax(Rec& r) {
    if (r.ins.size() < 1) return false;
    const TensorBuffer& in = r.ins[0];
    TensorBuffer out = in;
    int64_t last = in.shape.empty() ? 1 : in.shape.back();
    int64_t rows = out.data.size() / last;
    for (int64_t row = 0; row < rows; row++) {
        double mx = out.data[(size_t)(row * last)];
        for (int64_t j = 1; j < last; j++) mx = std::max(mx, out.data[(size_t)(row * last + j)]);
        double s = 0.0;
        for (int64_t j = 0; j < last; j++) {
            out.data[(size_t)(row * last + j)] =
                std::exp(out.data[(size_t)(row * last + j)] - mx);
            s += out.data[(size_t)(row * last + j)];
        }
        for (int64_t j = 0; j < last; j++)
            out.data[(size_t)(row * last + j)] /= s;
    }
    r.out = std::move(out);
    return true;
}

void NumericTrainer::backward(const std::vector<double>& grad_return,
                              int64_t B, int64_t C) {
    grads_.clear();
    {
        TensorBuffer g;
        g.shape = {B, C};
        g.data = grad_return;
        grads_[fn_.return_id] = std::move(g);
    }
    for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
        Rec& r = *it;
        switch (r.op) {
            case MLIROp::MATMUL: back_matmul(r); break;
            case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
            case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
            case MLIROp::SILU: case MLIROp::IDENTITY: back_activation(r); break;
            case MLIROp::DROPOUT: {
                auto git = grads_.find(r.result_id);
                if (git != grads_.end()) accumulate(r.operands[0], git->second);
                break;
            }
            case MLIROp::ELEMENTWISE_BINOP: back_binop(r); break;
            case MLIROp::LAYERNORM: back_layernorm(r); break;
            case MLIROp::SOFTMAX: back_softmax(r); break;
            default: break;
        }
    }

    // Fold accumulated grads into params.
    for (auto& p : params_) {
        auto it = grads_.find(p.id);
        if (it != grads_.end() && it->second.data.size() == p.grad.size()) {
            for (size_t i = 0; i < p.grad.size(); i++)
                p.grad[i] += it->second.data[i];
        }
    }
    grads_.clear();
}

void NumericTrainer::back_matmul(Rec& r) {
    auto git = grads_.find(r.result_id);
    if (git == grads_.end() || r.ins.size() < 2) return;
    const TensorBuffer& gC = git->second;   // [M, N]
    const TensorBuffer& A = r.ins[0];       // [M, K]
    const TensorBuffer& B = r.ins[1];       // [K, N]

    {   // dA = gC @ B^T
        TensorBuffer ga;
        ga.shape = {gC.shape[0], B.shape[0]};
        ga.data.assign((size_t)(ga.shape[0] * ga.shape[1]), 0.0);
        for (int64_t i = 0; i < gC.shape[0]; i++)
            for (int64_t j = 0; j < B.shape[0]; j++) {
                double s = 0.0;
                for (int64_t k = 0; k < B.shape[1]; k++)
                    s += gC.data[(size_t)(i * gC.shape[1] + k)] * B.data[(size_t)(j * B.shape[1] + k)];
                ga.data[(size_t)(i * ga.shape[1] + j)] = s;
            }
        accumulate(r.operands[0], ga);
    }
    {   // dB = A^T @ gC   (sum over M = rows of A)
        TensorBuffer gb;
        gb.shape = {B.shape[0], gC.shape[1]};
        gb.data.assign((size_t)(gb.shape[0] * gb.shape[1]), 0.0);
        for (int64_t i = 0; i < B.shape[0]; i++)
            for (int64_t j = 0; j < gC.shape[1]; j++) {
                double s = 0.0;
                for (int64_t p = 0; p < A.shape[0]; p++)
                    s += A.data[(size_t)(p * A.shape[1] + i)] *
                         gC.data[(size_t)(p * gC.shape[1] + j)];
                gb.data[(size_t)(i * gb.shape[1] + j)] = s;
            }
        accumulate(r.operands[1], gb);
    }
}

void NumericTrainer::back_activation(Rec& r) {
    auto git = grads_.find(r.result_id);
    if (git == grads_.end() || r.ins.size() < 1) return;
    const TensorBuffer& g = git->second;
    const TensorBuffer& in = r.ins[0];
    TensorBuffer ga = g;
    for (size_t i = 0; i < ga.data.size() && i < in.data.size(); i++) {
        double x = in.data[i];
        double d = 0.0;
        switch (r.op) {
            case MLIROp::RELU: d = x > 0 ? 1.0 : 0.0; break;
            case MLIROp::LEAKY_RELU: d = x > 0 ? 1.0 : 0.01; break;
            case MLIROp::SIGMOID: { double s = 1.0 / (1.0 + std::exp(-x)); d = s * (1 - s); break; }
            case MLIROp::TANH: { double t = std::tanh(x); d = 1 - t * t; break; }
            case MLIROp::SWISH: case MLIROp::SILU: {
                double s = 1.0 / (1.0 + std::exp(-x));
                d = s + x * s * (1 - s);
                break;
            }
            case MLIROp::GELU: {
                d = 0.5 * (1 + std::erf(x / std::sqrt(2.0))) +
                    x * std::exp(-x * x / 2) / std::sqrt(2 * 3.141592653589793);
                break;
            }
            default: d = 1.0; break;
        }
        ga.data[i] *= d;
    }
    accumulate(r.operands[0], ga);
}

void NumericTrainer::back_binop(Rec& r) {
    auto git = grads_.find(r.result_id);
    if (git == grads_.end() || r.ins.size() < 1) return;
    const TensorBuffer& g = git->second;
    const TensorBuffer& a = r.ins[0];
    accumulate(r.operands[0], g);
    if (r.operands.size() > 1 && r.ins.size() > 1) {
        const TensorBuffer& b = r.ins[1];
        const std::string& op = r.attribute;
        TensorBuffer gb;
        gb.shape = b.shape;
        gb.data.assign(b.data.size(), 0.0);
        for (size_t i = 0; i < g.data.size(); i++) {
            if (b.data.empty()) continue;
            size_t k = i % b.data.size();
            double d = 0.0;
            if (op == "+") d = 1.0;
            else if (op == "-") d = -1.0;
            else if (op == "*") d = a.data[i];
            else if (op == "/") d = b.data[k] == 0.0 ? 0.0 : -a.data[i] / (b.data[k] * b.data[k]);
            else d = 0.0;
            gb.data[k] += g.data[i] * d;
        }
        accumulate(r.operands[1], gb);
    }
}

void NumericTrainer::back_layernorm(Rec& r) {
    auto git = grads_.find(r.result_id);
    if (git == grads_.end() || r.ins.size() < 1) return;
    const TensorBuffer& g = git->second;
    const TensorBuffer& in = r.ins[0];
    TensorBuffer ga = r.out; // use saved output values
    int64_t last = r.out.shape.empty() ? 1 : (int64_t)r.out.shape.back();
    int64_t rows = (int64_t)(g.data.size() / last);
    // Recompute mean/var from saved input.
    for (int64_t row = 0; row < rows; row++) {
        double mean = 0.0, var = 0.0;
        for (int64_t j = 0; j < last; j++) mean += in.data[(size_t)(row * last + j)];
        mean /= last;
        for (int64_t j = 0; j < last; j++) {
            double d = in.data[(size_t)(row * last + j)] - mean;
            var += d * d;
        }
        var /= last;
        double inv = 1.0 / std::sqrt(var + 1e-5);
        double wsum = 0.0;
        for (int64_t j = 0; j < last; j++)
            wsum += g.data[(size_t)(row * last + j)] * (in.data[(size_t)(row * last + j)] - mean);
        wsum /= last;
        for (int64_t j = 0; j < last; j++) {
            double gd = g.data[(size_t)(row * last + j)];
            double xc = in.data[(size_t)(row * last + j)] - mean;
            ga.data[(size_t)(row * last + j)] = (gd - wsum - xc * wsum / (var + 1e-5)) * inv;
        }
    }
    accumulate(r.operands[0], ga);
}

void NumericTrainer::back_softmax(Rec& r) {
    auto git = grads_.find(r.result_id);
    if (git == grads_.end() || r.ins.size() < 1) return;
    const TensorBuffer& g = git->second;
    const TensorBuffer& sm = r.out;
    TensorBuffer ga = g;
    int64_t last = sm.shape.empty() ? 1 : (int64_t)sm.shape.back();
    int64_t rows = (int64_t)(g.data.size() / last);
    for (int64_t row = 0; row < rows; row++) {
        double dot = 0.0;
        for (int64_t j = 0; j < last; j++)
            dot += g.data[(size_t)(row * last + j)] * sm.data[(size_t)(row * last + j)];
        for (int64_t j = 0; j < last; j++)
            ga.data[(size_t)(row * last + j)] =
                sm.data[(size_t)(row * last + j)] *
                (g.data[(size_t)(row * last + j)] - dot);
    }
    accumulate(r.operands[0], ga);
}

void NumericTrainer::accumulate(const std::string& id, const TensorBuffer& g) {
    // Propagate gradients for ALL ids through the tape; params are folded into
    // the optimizer-facing accumulators at the end of backward().
    auto& grad = grads_[id];
    if (grad.data.empty()) { grad = g; return; }
    size_t n = std::min(grad.data.size(), g.data.size());
    for (size_t i = 0; i < n; i++) grad.data[i] += g.data[i];
}

void NumericTrainer::zero_grad() {
    for (auto& p : params_) std::fill(p.grad.begin(), p.grad.end(), 0.0);
}

void NumericTrainer::apply_step(MuonOptimizer& opt, size_t step_index) {
    if (!params_registered_) {
        for (auto& p : params_) {
            opt.add_param(p.data.data(), p.rows, p.cols);
        }
        params_registered_ = true;
    }
    opt.set_step(step_index);
    for (size_t i = 0; i < params_.size(); i++) {
        opt.step_param(i, params_[i].grad.data());
    }
}

} // namespace ns