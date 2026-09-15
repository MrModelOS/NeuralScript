// LayerNorm AOT-backprop test (v1.3): a network that pipes through a LayerNorm
// layer in the train() body, so the reverse-mode pass must lower LAYERNORM to
// LAYERNORM_GRAD. The generated self-contained runtime driver is compiled with
// the host compiler and the XOR task must still converge (loss decreases and
// all four patterns classify correctly), proving LayerNorm contributes a valid
// gradient rather than the identity fallback.
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

static const char* XOR_LN = R"(
type Bs = Dynamic
network XORLN {
    input:  Tensor[Bs, 2] float32
    output: Tensor[Bs, 2] float32
    layer fc1 = Dense(in: 2, out: 16, activation: ReLU)
    layer ln  = LayerNorm()
    layer fc2 = Dense(in: 16, out: 2, activation: Identity)
    forward(x) {
        return x -> fc1 -> ln -> fc2
    }
    train(x: Tensor[Bs, 2], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h     = relu(x @ fc1)
            var n     = h -> ln
            var preds = n @ fc2
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

int main() {
    Lexer lexer(XOR_LN);
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

    const std::string driver = "/tmp/ns_ln_driver.cpp";
    const std::string host = "/tmp/ns_ln_host.cpp";
    const std::string bin = "/tmp/ns_ln_host";
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
#include <vector>

static int fail(const char* msg) { std::printf("FAIL: %s\n", msg); return 1; }

int main() {
    const float xs[4][2] = { {-1.f, -1.f}, {-1.f, 1.f}, {1.f, -1.f}, {1.f, 1.f} };
    const float ys[4][2] = { {1.f, 0.f}, {0.f, 1.f}, {0.f, 1.f}, {1.f, 0.f} };
    const size_t xn = 4 * 2;

    const size_t nw = 2 * 16 + 16 * 2;
    std::vector<float> w(nw);
    unsigned s = 12345u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }

    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");

    float loss0 = -1.f;
    if (ns_objective_loss(m, xs[0], ys[0], xn, &loss0) != 0) return fail("objective(0)");
    if (!(loss0 > 0.3f && loss0 < 1.5f)) {
        std::printf("loss0 = %.4f\n", loss0);
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
        std::cerr << "FAIL: LayerNorm train host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: LayerNorm train host run failed\n";
        return 1;
    }

    std::cout << "PASS: LayerNorm AOT backprop\n";
    return 0;
}