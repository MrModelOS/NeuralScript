// Embedding AOT-backprop test (v1.3): the reverse-mode pass must lower
// LAYER_EMBEDDING to EMBEDDING_GRAD_W (scatter-add of dOut rows into dW by the
// token index), and the embedding layer must be usable in a train() body. The
// generated self-contained runtime driver trains a token->class network on a
// deterministic toy vocabulary (tokens 0-3 -> class 0, tokens 4-7 -> class 1)
// and verifies loss decreases and all tokens classify correctly.
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

static const char* EMB_NET = R"(
type Bs = Dynamic
network EmbNet {
    input:  Tensor[Bs] float32
    output: Tensor[Bs, 2] float32
    layer emb = Embedding(vocab_size: 8, d_model: 16)
    layer fc  = Dense(in: 16, out: 2, activation: Identity)
    forward(x) {
        var h = x -> emb
        return h -> fc
    }
    train(x: Tensor[Bs], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h      = x -> emb
            var preds  = h @ fc
            var loss   = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

int main() {
    Lexer lexer(EMB_NET);
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

    const std::string driver = "/tmp/ns_emb_driver.cpp";
    const std::string host = "/tmp/ns_emb_host.cpp";
    const std::string bin = "/tmp/ns_emb_host";
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
    // One-hot class labels: token t is class 0 if t < 4, else class 1.
    const float xs[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    const float ys[8][2] = {{1.f,0.f},{1.f,0.f},{1.f,0.f},{1.f,0.f},
                            {0.f,1.f},{0.f,1.f},{0.f,1.f},{0.f,1.f}};

    const size_t nw = 8 * 16 + 16 * 2;
    std::vector<float> w(nw);
    unsigned s = 12345u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }

    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");

    float loss0 = -1.f;
    if (ns_objective_loss(m, xs, ys[0], 8, &loss0) != 0) return fail("objective(0)");

    float lossT = loss0;
    const int epochs = 400;
    for (int e = 0; e < epochs; e++)
        if (ns_runtime_train_step(m, xs, ys[0], 8, &lossT, 0.3f) != 0)
            return fail("train_step");

    if (!(lossT < loss0 * 0.5f)) {
        std::printf("loss0=%.4f lossT=%.4f\n", loss0, lossT);
        return fail("loss did not decrease");
    }

    int correct = 0;
    float o[2];
    for (int t = 0; t < 8; t++) {
        if (ns_eval_infer(m, &xs[t], o, 1) != 0) return fail("eval single");
        int pred = o[0] > o[1] ? 0 : 1;
        int want = t < 4 ? 0 : 1;
        if (pred == want) correct++;
    }
    if (correct != 8) {
        std::printf("correct=%d/8 lossT=%.4f\n", correct, lossT);
        return fail("classification not converged");
    }

    std::printf("loss0=%.4f lossT=%.4f correct=8/8\n", loss0, lossT);
    ns_free(m);
    return 0;
}
)ns";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " + NS_REPO_INCLUDE_DIR + " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: Embedding train host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: Embedding train host run failed\n";
        return 1;
    }

    std::cout << "PASS: Embedding AOT backprop\n";
    return 0;
}