// Full-loop training test (Phase 5 final):
//   forward (trusted CPU eval) -> numerical backprop via NumericTrainer ->
//   gradient accumulation -> Muon::step_param (matrices) / AdamW (vectors).
// Confirms the classic neural-training convergence: Loss -> 0 on XOR with a
// real MLP (Dense + GELU), and cross-checks gradients against central finite
// differences and the forward against ModuleEvaluator.
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/eval.hpp"
#include "ns/training/trainer.hpp"
#include "ns/optim/muon.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using namespace ns;

static double softmax_ce_loss(const std::vector<double>& logits, size_t B, size_t C,
                              const std::vector<double>& onehot) {
    double sum = 0.0;
    for (size_t b = 0; b < B; b++) {
        double mx = logits[b*C];
        for (size_t j = 1; j < C; j++) mx = std::max(mx, logits[b*C + j]);
        double s = 0.0;
        for (size_t j = 0; j < C; j++) s += std::exp(logits[b*C + j] - mx);
        double ce = 0.0;
        for (size_t j = 0; j < C; j++)
            ce += onehot[b*C + j] * (mx - logits[b*C + j] + std::log(s));
        sum += ce;
    }
    return sum / (double)B;
}

static std::vector<double> ce_grad(const std::vector<double>& logits, size_t B, size_t C,
                                   const std::vector<double>& onehot) {
    std::vector<double> g(B * C);
    for (size_t b = 0; b < B; b++) {
        double mx = logits[b*C];
        for (size_t j = 1; j < C; j++) mx = std::max(mx, logits[b*C + j]);
        double s = 0.0;
        for (size_t j = 0; j < C; j++) s += std::exp(logits[b*C + j] - mx);
        for (size_t j = 0; j < C; j++)
            g[b*C + j] = (std::exp(logits[b*C + j] - mx) / s - onehot[b*C + j]) / (double)B;
    }
    return g;
}

int main() {
    // XOR-ish dataset: 4 points, 2 classes (0,0)->0 ; (0,1),(1,0)->1 ; (1,1)->0.
    const size_t F = 4, C = 2, NB = 4;
    std::vector<double> X = {
        0.0, 0.0, 1.0, 1.0,     // sample 0
        0.0, 1.0, 0.0, 1.0,     // sample 1
        1.0, 0.0, 0.0, 1.0,     // sample 2
        1.0, 1.0, 1.0, 1.0,     // sample 3
    };
    std::vector<size_t> labels = {0, 1, 1, 0};
    std::vector<double> Y(NB * C, 0.0);
    for (size_t b = 0; b < NB; b++) Y[b*C + labels[b]] = 1.0;

    // Compile an MLP network through the real pipeline.
    const char* SRC = R"(
type Batch = Dynamic
network Net {
    input:  Tensor[Batch, 4] float32
    output: Tensor[Batch, 2] float32
    layer fc1 = Dense(in: 4, out: 16, activation: GELU)
    layer fc2 = Dense(in: 16, out: 2, activation: Identity)
    forward(x) { return x -> fc1 -> fc2 }
}
)";
    Lexer lexer(SRC);
    auto toks = lexer.tokenize();
    Parser parser(toks);
    Program prog = parser.parse_program();
    ShapeChecker checker;
    checker.check(prog);
    MLIRCompiler mlir;
    auto module = mlir.compile(prog);

    const MLIRFunction* net = nullptr;
    for (auto& f : module.functions) if (f.name == "Net") { net = &f; break; }
    assert(net != nullptr);

    NumericTrainer tr(*net, module);
    tr.bind_input("x", X, {NB, F});

    // Random init the two weights.
    std::mt19937 rng(3);
    std::normal_distribution<double> nd(0.0, 0.35);
    for (auto& p : tr.params()) {
        p.data.assign(p.data.size(), 0.0);
        for (auto& v : p.data) v = nd(rng);
    }

    // --- Forward cross-check vs the trusted ModuleEvaluator. ---
    const TensorBuffer* logits = tr.forward();
    assert(logits != nullptr);
    {
        ModuleEvaluator ev;
        ev.bind("x", X, {NB, F});
        for (auto& p : tr.params()) ev.bind(p.id, p.data, {p.rows, p.cols});
        TensorBuffer ref;
        assert(ev.run(*net, module, "", ref));
        assert(ref.shape == logits->shape);
        double maxe = 0.0;
        for (size_t i = 0; i < logits->data.size(); i++)
            maxe = std::max(maxe, std::fabs(logits->data[i] - ref.data[i]));
        std::printf("forward-vs-eval max_err=%.3e\n", maxe);
        assert(maxe < 1e-12);
    }

    // --- Gradient cross-check vs central finite differences. ---
    {
        NumericTrainer t2(*net, module);
        t2.bind_input("x", X, {NB, F});
        rng.seed(4);
        for (auto& p : t2.params()) {
            p.data.assign(p.data.size(), 0.0);
            for (auto& v : p.data) v = nd(rng);
        }
        auto loss_of = [&](NumericTrainer& t) {
            const TensorBuffer* l = t.forward();
            return softmax_ce_loss(l->data, NB, C, Y);
        };
        auto grad_of = [&](NumericTrainer& t, std::vector<double>& g) {
            const TensorBuffer* l = t.forward();
            g = ce_grad(l->data, NB, C, Y);
            t.zero_grad();
            t.backward(g, NB, C);
        };

        double eps = 1e-6;
        for (size_t pi = 0; pi < t2.params().size(); pi++) {
            auto& p = t2.params()[pi];
            size_t idx = std::min<size_t>(3, p.data.size() - 1);
            double orig = p.data[idx];
            p.data[idx] = orig + eps; double Lp = loss_of(t2);
            p.data[idx] = orig - eps; double Lm = loss_of(t2);
            p.data[idx] = orig;
            double fd = (Lp - Lm) / (2 * eps);

            std::vector<double> g;
            grad_of(t2, g);
            double an = t2.params()[pi].grad[idx];
            std::cerr << "analytic[" << pi << "] idx=" << idx
                      << ": " << an << "  fd: " << fd
                      << "  (diff " << std::fabs(an - fd) << ")\n";
            assert(std::fabs(an - fd) < 1e-3);
        }
    }

    // --- Full training loop: numeric grads -> Muon (matrix weights). ---
    {
        NumericTrainer t3(*net, module);
        t3.bind_input("x", X, {NB, F});
        rng.seed(3);
        for (auto& p : t3.params()) {
            p.data.assign(p.data.size(), 0.0);
            for (auto& v : p.data) v = nd(rng);
        }
        // Also wire a vector (non-matrix) param through AdamW to cover the
        // AdamW bridge: a tiny "bias" with random gradient each step.
        NumericTrainer::Param bias;
        bias.id = "bias_test";
        bias.rows = 8; bias.cols = 1;
        bias.data.assign(8, 0.1);
        bias.grad.assign(8, 0.0);
        t3.params().push_back(std::move(bias));

        MuonParams muon_p;
        muon_p.weight_decay = 0.0;
        muon_p.lr = 0.05;
        MuonOptimizer muon(muon_p);

        double l0 = -1.0;
        for (size_t step = 0; step < 1500; step++) {
            const TensorBuffer* lg = t3.forward();
            std::vector<double> lo = lg->data;
            double loss = softmax_ce_loss(lo, NB, C, Y);
            if (step == 0) l0 = loss;
            t3.zero_grad();
            t3.backward(ce_grad(lo, NB, C, Y), NB, C);

            // random gradient for the vector param (excersies AdamW path)
            auto& vp = t3.params().back();
            for (auto& g : vp.grad) g = nd(rng) * 0.01;

            t3.apply_step(muon, step);
            if (step % 500 == 0)
                std::printf("step %4zu  loss=%.6e\n", step, loss);
        }
        const TensorBuffer* lg = t3.forward();
        std::vector<double> lo = lg->data;
        double lf = softmax_ce_loss(lo, NB, C, Y);
        std::printf("training: l0=%.6e  lf=%.6e\n", l0, lf);
        assert(lf < 1e-6);
        assert(l0 > 0.3); // sanity: it really was unsolved initially

        // AdamW vector param actually moved.
        auto& vp0 = t3.params().back();
        double moved = 0.0;
        for (size_t i = 0; i < vp0.data.size(); i++) moved += std::fabs(vp0.data[i] - 0.1);
        std::printf("adamw-bias moved=%.3e\n", moved);
        assert(moved > 1e-6);
    }

    std::cout << "PASS: full training loop (numeric grads -> Muon/AdamW) converges Loss->0\n";
    return 0;
}