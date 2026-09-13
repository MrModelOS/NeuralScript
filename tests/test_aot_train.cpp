// AOT training test (v1.1): a network with a train() method lowers through the
// compiler's reverse-mode pass, and the generated self-contained runtime driver
// exposes the C-ABI training entry points ns_runtime_train_step /
// ns_objective_loss. The host application (pure C-ABI) trains a tiny XOR MLP,
// then verifies:
//   - the cross-entropy loss decreases over training,
//   - ns_objective_loss agrees with a host-side CE computed from ns_eval_infer
//     outputs (numerical consistency of the forward-only objective),
//   - ns_objective_loss does NOT mutate weights (mode 0 writeback guard),
//   - after training the network classifies all four XOR patterns correctly,
//   - weights actually moved (Muon/AdamW optimizer took a step).
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using namespace ns;

#ifndef NS_REPO_INCLUDE_DIR
#define NS_REPO_INCLUDE_DIR ".."
#endif

static const char* XOR = R"(
type Bs = Dynamic
network XOR {
    input:  Tensor[Bs, 2] float32
    output: Tensor[Bs, 2] float32
    layer fc1 = Dense(in: 2, out: 16, activation: ReLU)
    layer fc2 = Dense(in: 16, out: 2, activation: Identity)
    forward(x) {
        return x -> fc1 -> fc2
    }
    train(x: Tensor[Bs, 2], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h     = relu(x @ fc1)
            var preds = h @ fc2
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

int main() {
    // Generate the self-contained runtime driver (forward + training core).
    Lexer lexer(XOR);
    auto toks = lexer.tokenize();
    Parser parser(toks);
    Program prog = parser.parse_program();
    ShapeChecker checker;
    checker.check(prog);

    MLIRCompiler mlir;
    auto module = mlir.compile(prog);
    FusionPass fuse;
    fuse.run(module);

    CodegenOptions opts;
    opts.backend = TargetBackend::CPU_CXX;
    opts.emit_runtime_driver = true;
    CodeGenerator cg;
    std::string code = cg.generate(module, opts);

    const std::string driver = "/tmp/ns_aot_driver.cpp";
    const std::string host = "/tmp/ns_aot_host.cpp";
    const std::string bin = "/tmp/ns_aot_host";
    {
        std::ofstream of(driver);
        of << code;
    }
    std::ofstream hf(host);
    hf << R"ns(
#include "ns/runtime/ns_runtime.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fail(const char* msg) { std::printf("FAIL: %s\n", msg); return 1; }

int main() {
    // Bipolar XOR dataset (bit 0 -> -1, bit 1 -> +1); class = parity.
    // Bipolar inputs are required here: with {0,1} encoding the (0,0) pattern
    // maps to a zero pre-activation, killing the ReLU layer (linear layers in
    // NeuralScript have no bias term yet), and the pattern becomes unlearnable.
    const float xs[4][2] = { {-1.f, -1.f}, {-1.f, 1.f}, {1.f, -1.f}, {1.f, 1.f} };
    const float ys[4][2] = { {1.f, 0.f}, {0.f, 1.f}, {0.f, 1.f}, {1.f, 0.f} };
    const size_t xn = 4 * 2;

    // Weight layout: fc1_w [2,16], fc2_w [16,2].
    const size_t nw = 2 * 16 + 16 * 2;
    std::vector<float> w(nw);
    unsigned s = 12345u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }

    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");

    // Static weight count must be queryable BEFORE a model exists, and the
    // layout must describe a dense, non-overlapping blob of exactly nw floats.
    if (ns_weight_count_static() != nw) return fail("weight_count_static");
    const ns_weight_layout* lay = ns_model_layout(m);
    if (!lay || lay->num_weights != 2) return fail("layout count");
    size_t expected_off = 0, lay_total = 0;
    for (size_t i = 0; i < lay->num_weights; i++) {
        const ns_weight_desc& d = lay->desc[i];
        if (d.offset != expected_off) return fail("layout offset");
        lay_total += d.count;
        expected_off += d.count;
    }
    if (lay_total != nw) return fail("layout total");

    std::vector<float> w0(nw);
    if (ns_model_get_weights(m, w0.data(), nw) != 0) return fail("get_weights(0)");
    float loss0 = -1.f;
    if (ns_objective_loss(m, xs[0], ys[0], xn, &loss0) != 0) return fail("objective(0)");

    // Loss should be ~ln(2) with random init; guard sanity.
    if (!(loss0 > 0.5f && loss0 < 1.0f)) {
        std::printf("loss0 = %.4f (expected ~ln2)\n", loss0);
        return fail("loss0 sanity");
    }

    float lossT = loss0;
    const int epochs = 400;
    for (int e = 0; e < epochs; e++)
        if (ns_runtime_train_step(m, xs[0], ys[0], xn, &lossT, 0.3f) != 0)
            return fail("train_step");

    if (!(lossT < loss0 * 0.9f)) {
        std::printf("loss0=%.4f lossT=%.4f\n", loss0, lossT);
        return fail("loss did not decrease");
    }

    // Objective must agree with host-side CE over inference outputs.
    std::vector<float> out(4 * 2);
    if (ns_eval_infer(m, xs[0], out.data(), xn) != 0) return fail("eval");
    float ce_ref = 0.f;
    for (int s2 = 0; s2 < 4; s2++) {
        const float* r = &out[s2 * 2];
        float mx = std::max(r[0], r[1]);
        float p0 = std::exp(r[0] - mx), p1 = std::exp(r[1] - mx);
        float sum = p0 + p1;
        float lab = ys[s2][0] > 0.5f ? p0 : p1;
        ce_ref -= std::log(lab / sum);
    }
    ce_ref /= 4.f;
    float obj = -1.f;
    if (ns_objective_loss(m, xs[0], ys[0], xn, &obj) != 0) return fail("objective(final)");
    if (std::fabs(obj - ce_ref) > 1e-3) {
        std::printf("obj=%.5f ce_ref=%.5f\n", obj, ce_ref);
        return fail("objective inconsistency");
    }

    // ns_objective_loss (mode 0) must NOT touch weights.
    std::vector<float> wb(nw);
    if (ns_model_get_weights(m, wb.data(), nw) != 0) return fail("get_weights(final)");
    if (ns_objective_loss(m, xs[0], ys[0], xn, &obj) != 0) return fail("objective(again)");
    std::vector<float> wc(nw);
    if (ns_model_get_weights(m, wc.data(), nw) != 0) return fail("get_weights(re)");
    for (size_t i = 0; i < nw; i++)
        if (wb[i] != wc[i]) return fail("objective mutated weights");

    // Optimizer must have moved the weights.
    bool moved = false;
    for (size_t i = 0; i < nw; i++)
        if (wc[i] != w0[i]) { moved = true; break; }
    if (!moved) return fail("weights unchanged after training");

    // Trained model must classify all four XOR patterns.
    int correct = 0;
    float o[2];
    for (int s2 = 0; s2 < 4; s2++) {
        if (ns_eval_infer(m, xs[s2], o, 2) != 0) return fail("eval single");
        int pred = o[0] > o[1] ? 0 : 1;
        int want = ys[s2][0] > 0.5f ? 0 : 1;
        if (pred == want) correct++;
    }
    if (correct != 4) {
        std::printf("correct=%d/4 lossT=%.4f\n", correct, lossT);
        return fail("classification not converged");
    }

    std::printf("loss0=%.4f lossT=%.4f correct=4/4\n", loss0, lossT);
    ns_free(m);
    return 0;
}
)ns";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " + NS_REPO_INCLUDE_DIR + " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: AOT train host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: AOT train host run failed\n";
        return 1;
    }

    std::cout << "PASS: AOT training core\n";
    return 0;
}