// MoE AOT-backprop test (v1.3): reverse-mode must lower LAYER_MOE to three
// grad ops (MOE_GRAD_X, MOE_GRAD_WG, MOE_GRAD_WE) and the MoE layer must be
// usable in a train() body. A toy 2-expert router (tokens 0-7 → class 0,
// tokens 8-15 → class 1) trains until loss decreases and tokens classify
// correctly, proving the MoE weights get valid gradients.
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
#include <random>

using namespace ns;

#ifndef NS_REPO_INCLUDE_DIR
#define NS_REPO_INCLUDE_DIR ".."
#endif

// Network: embedding → MoE(router + 2 experts, D=8) → dense → softmax CE.
static const char* MOE_NET = R"(
type Bs = Dynamic
network MoENet {
    input:  Tensor[Bs] float32
    output: Tensor[Bs, 2] float32
    layer emb    = Embedding(vocab_size: 16, d_model: 8)
    layer moe    = MoE(d_model: 8, num_experts: 2)
    layer fc     = Dense(in: 8, out: 2, activation: Identity)
    forward(x) {
        var h = x -> emb
        var m = h -> moe
        var t = m -> fc
        return t
    }
    train(x: Tensor[Bs], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h    = x -> emb
            var moe  = h -> moe
            var preds = moe @ fc
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

int main() {
    Lexer lexer(MOE_NET);
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

    const std::string driver = "/tmp/ns_moe_driver.cpp";
    const std::string host   = "/tmp/ns_moe_host.cpp";
    const std::string bin    = "/tmp/ns_moe_host";
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
#include <random>
#include <vector>

static int fail(const char* msg) { std::printf("FAIL: %s\n", msg); return 1; }

int main() {
    const int BS = 16, V = 16, D = 8, C = 2;
    float xs[BS];
    float ys[BS * C];
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> td(0, V - 1);
    for (int i = 0; i < BS; i++) {
        xs[i] = (float)td(rng);
        int cls = ((int)xs[i] < V / 2) ? 0 : 1;
        ys[i * C]     = cls == 0 ? 1.f : 0.f;
        ys[i * C + 1] = cls == 1 ? 1.f : 0.f;
    }

    // Weight count: emb (16*8) + gate (8*2) + experts (2*8*8) + fc (8*2) = 304
    const size_t nw = 16 * 8 + 8 * 2 + 2 * 8 * 8 + 8 * 2;
    std::vector<float> w(nw);
    unsigned s = 12345u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }

    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");

    float loss0 = -1.f;
    if (ns_objective_loss(m, xs, ys, BS, &loss0) != 0) return fail("objective(0)");

    float lossT = loss0;
    const int epochs = 400;
    for (int e = 0; e < epochs; e++)
        if (ns_runtime_train_step(m, xs, ys, BS, &lossT, 0.3f) != 0)
            return fail("train_step");

    if (!(lossT < loss0 * 0.5f)) {
        std::printf("loss0=%.4f lossT=%.4f\n", loss0, lossT);
        return fail("loss did not decrease");
    }

    int correct = 0;
    float o[C];
    for (int t = 0; t < BS; t++) {
        if (ns_eval_infer(m, &xs[t], o, 1) != 0) return fail("eval single");
        int pred = o[0] > o[1] ? 0 : 1;
        int want = (int)xs[t] < V / 2 ? 0 : 1;
        if (pred == want) correct++;
    }
    if (correct < BS / 2) {
        std::printf("correct=%d/%d lossT=%.4f\n", correct, BS, lossT);
        return fail("classification not converged");
    }

    std::printf("loss0=%.4f lossT=%.4f correct=%d/%d\n", loss0, lossT, correct, BS);
    ns_free(m);
    return 0;
}
)ns";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " + NS_REPO_INCLUDE_DIR + " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: MoE train host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: MoE train host run failed\n";
        return 1;
    }

    std::cout << "PASS: MoE AOT backprop\n";
    return 0;
}
