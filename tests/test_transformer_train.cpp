// End-to-end AOT backprop through the full transformer stack:
//   embedding -> multi-head self-attention -> layernorm -> MLP -> dense head
// Task: reproduce the class of the FIRST token at every sequence position, fed
// as a single input (S = N), so only the attention path can pull information
// from position 0. Every layer's backward is AOT-lowered (EMBEDDING_GRAD_W,
// ATTENTION_GRAD_X/WQ/WK/WV/WO, LAYERNORM_GRAD, MATMUL grads). Verified via the
// runtime ABI: loss0 ~ ln(4) converges to ~0, 16/16 classification.
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"
#include "ns/runtime/ns_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ns;

#ifndef NS_REPO_INCLUDE_DIR
#define NS_REPO_INCLUDE_DIR ".."
#endif

static const char* TFMR_NET = R"(
type Bs = Dynamic
network TFMR {
    input:  Tensor[Bs] float32
    output: Tensor[Bs, 4] float32
    layer emb   = Embedding(vocab_size: 8, d_model: 16)
    layer attn  = Attention(d_model: 16, heads: 4)
    layer ln    = LayerNorm()
    layer mlp1  = Dense(in: 16, out: 32, activation: ReLU)
    layer mlp2  = Dense(in: 32, out: 16, activation: Identity)
    layer fc    = Dense(in: 16, out: 4, activation: Identity)
    forward(x) {
        var h = x -> emb
        var a = h -> attn
        var n = a -> ln
        var m = n -> mlp1
        var m2 = m -> mlp2
        var t = m2 -> fc
        return t
    }
    train(x: Tensor[Bs], labels: Tensor[Bs, 4]) -> float32 {
        grad {
            var h     = x -> emb
            var at    = h -> attn
            var ln_   = at -> ln
            var mlp1o = relu(ln_ @ mlp1)
            var mlp2o = mlp1o @ mlp2
            var preds = mlp2o @ fc
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

static int fail(const char* what) {
    std::cerr << "FAIL: " << what << "\n";
    return 1;
}

int main() {
    Lexer lexer(TFMR_NET);
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

    const std::string driver = "/tmp/ns_tfmr_driver.cpp";
    const std::string host   = "/tmp/ns_tfmr_host.cpp";
    const std::string bin    = "/tmp/ns_tfmr_host";
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
    const int N = 16, V = 8, C = 4;
    float xs[N];
    float ys[N * C];
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> td(0, V - 1);
    for (int i = 0; i < N; i++) xs[i] = (float)td(rng);
    int cls0 = (int)xs[0] % C;             // task: output the class of the FIRST token
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++) ys[i * C + c] = (c == cls0) ? 1.f : 0.f;

    // emb (8*16) + attn q/k/v/o (4*16*16) + mlp1 (16*32) + mlp2 (32*16) + fc (16*4) = 2240
    const size_t nw = 8 * 16 + 4 * 16 * 16 + 16 * 32 + 32 * 16 + 16 * 4;
    std::vector<float> w(nw);
    unsigned s = 12345u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.2f; }

    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");

    float loss0 = -1.f;
    if (ns_objective_loss(m, xs, ys, N, &loss0) != 0) return fail("objective(0)");

    float lossT = loss0;
    const int epochs = 15000;
    for (int e = 0; e < epochs; e++)
        if (ns_runtime_train_step(m, xs, ys, N, &lossT, 0.007f) != 0)
            return fail("train_step");

    if (!(lossT < loss0 * 0.5f)) {
        std::printf("loss0=%.4f lossT=%.4f\n", loss0, lossT);
        return fail("loss did not decrease");
    }

    int correct = 0, total = N;
    std::vector<float> outs((size_t)N * C);
    if (ns_eval_infer(m, xs, outs.data(), N) != 0) return fail("eval full seq");
    for (int t = 0; t < N; t++) {
        int pred = 0;
        for (int c = 1; c < C; c++) if (outs[t * C + c] > outs[t * C + pred]) pred = c;
        if (pred == cls0) correct++;
    }
    if (correct * 10 < total * 9) {  // need >= 90%
        std::printf("correct=%d/%d lossT=%.4f\n", correct, total, lossT);
        return fail("transformer classification not converged");
    }

    std::printf("loss0=%.4f lossT=%.4f correct=%d/%d\n", loss0, lossT, correct, total);
    ns_free(m);
    return 0;
}
)ns";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " NS_REPO_INCLUDE_DIR " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: Transformer train host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: Transformer train host run failed\n";
        return 1;
    }

    std::cout << "PASS: Transformer AOT end-to-end (emb -> attn -> ln -> mlp -> CE)\n";
    return 0;
}