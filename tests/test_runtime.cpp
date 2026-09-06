// C-ABI runtime test: generate a self-contained runtime driver (CPU reference
// backend + ns_* host API), compile it alongside a C host application that
// drives the model exclusively through the C-ABI (ns_runtime_init /
// ns_eval_infer / ns_model_* / ns_free), and verify the output against a
// hand-computed reference.
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

static const char* MLP = R"(
type Batch = Dynamic
network MLP {
    input:  Tensor[Batch, 784] float32
    output: Tensor[Batch, 10] float32
    layer fc1 = Dense(in: 784, out: 32, activation: GELU)
    layer dr  = Dropout(rate: 0.5)
    layer fc2 = Dense(in: 32, out: 10, activation: Identity)
    forward(x) {
        return x -> fc1 -> dr -> fc2
    }
}
)";

int main() {
    // Generate the self-contained runtime driver.
    Lexer lexer(MLP);
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

    const std::string driver = "/tmp/ns_driver.cpp";
    const std::string host = "/tmp/ns_host_app.cpp";
    const std::string bin = "/tmp/ns_host_app";
    {
        std::ofstream of(driver);
        of << code;
    }
    std::ofstream hf(host);
    hf << R"(
#include "ns/runtime/ns_runtime.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t batch = 3, feat = 784, H = 32, C = 10;
    std::vector<float> x(batch * feat);
    for (auto& v : x) v = dist(rng);
    std::vector<float> w(H * feat + C * H);
    for (auto& v : w) v = dist(rng);

    ns_model* m = ns_runtime_init(w.data(), w.size());
    if (!m) { std::printf("INIT_FAIL\n"); return 1; }
    if (ns_model_weight_count(m) != w.size()) { std::printf("WCOUNT_FAIL\n"); return 1; }
    const ns_weight_layout* lay = ns_model_layout(m);
    if (lay->num_weights != 2) { std::printf("LAYOUT_FAIL\n"); return 1; }
    if (std::string(lay->desc[0].name) != "fc1_w" || lay->desc[0].count != H * feat) {
        std::printf("DESC0_FAIL\n"); return 1;
    }
    if (std::string(lay->desc[1].name) != "fc2_w" || lay->desc[1].count != C * H) {
        std::printf("DESC1_FAIL\n"); return 1;
    }

    size_t nout = ns_model_output_numel(m, x.size());
    if (nout != batch * C) { std::printf("OUTNUMEL_FAIL %zu\n", nout); return 1; }
    std::vector<float> out(nout, -999.f);
    if (ns_eval_infer(m, x.data(), out.data(), x.size()) != 0) { std::printf("EVAL_FAIL\n"); return 1; }

    // Reference: GELU(x @ w1) @ w2    (dropout = identity at inference).
    auto gelu = [](float u) { return 0.5f * u * (1.f + std::erf(u / 1.41421356f)); };
    std::vector<float> h(batch * H);
    for (size_t b = 0; b < batch; b++)
        for (size_t j = 0; j < H; j++) {
            float acc = 0.f;
            for (size_t k = 0; k < feat; k++) acc += x[b*feat + k] * w[k*H + j];
            h[b*H + j] = gelu(acc);
        }
    std::vector<float> ref(batch * C);
    for (size_t b = 0; b < batch; b++)
        for (size_t j = 0; j < C; j++) {
            float acc = 0.f;
            for (size_t k = 0; k < H; k++) acc += h[b*H + k] * w[H*feat + k*C + j];
            ref[b*C + j] = acc;
        }
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); i++)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    std::printf("%.3e\n", max_err);

    ns_free(m);
    return max_err < 1e-4 ? 0 : 1;
}
)";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " + NS_REPO_INCLUDE_DIR + " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: C-ABI link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: C-ABI host run failed\n";
        return 1;
    }

    std::cout << "PASS: C-ABI runtime driver\n";
    return 0;
}