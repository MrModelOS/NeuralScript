// CPU-backend integration test: compile an NS program through the full
// pipeline, emit C++ with the CPU reference backend, compile it with the host
// compiler, and execute it against a hand-computed reference. Verifies the
// generated runtime matches exact matmul semantics (batch-aware).
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace ns;

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
    // 1. Parse -> check -> MLIR -> fuse -> CPU codegen.
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
    CodeGenerator cg;
    std::string code = cg.generate(module, opts);

    // Structural invariants of the generated code.
    if (code.find("ns_matmul") == std::string::npos) {
        std::cerr << "FAIL: no matmul kernel emitted\n";
        return 1;
    }
    if (code.find("buf_fc1_w") == std::string::npos) {
        std::cerr << "FAIL: weight buffer fc1_w not declared\n";
        return 1;
    }
    if (code.find("ns_act2") == std::string::npos) {
        std::cerr << "FAIL: activation helper missing\n";
        return 1;
    }
    if (code.find("weights +") == std::string::npos) {
        std::cerr << "FAIL: weight loading missing\n";
        return 1;
    }
    if (code.find("(n) / K") == std::string::npos) {
        std::cerr << "FAIL: runtime batch not derived from input size\n";
        return 1;
    }

    // 2. Write harness: generated impl + a main that compares forward()
    //    against a hand-rolled (x @ w1) @ w2 for batch=3.
    std::string src = R"(
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
)";
    src += code;
    src += R"(
int main() {
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t batch = 3, feat = 784, H = 32, C = 10;
    std::vector<float> x(batch * feat);
    for (auto& v : x) v = dist(rng);
    std::vector<float> w(H * feat + C * H);
    for (auto& v : w) v = dist(rng);
    std::vector<float> out(batch * C, -999.f);
    forward(x.data(), w.data(), out.data(), x.size());
    // A @ B uses row-major [M,K]x[K,N]: w is laid out [784,32] then [32,10].
    // fc1 = GELU(x @ w1); dr = identity (inference); fc2 = act1 @ w2.
    auto gelu = [](float u) {
        return 0.5f * u * (1.f + erf(u / 1.41421356f));
    };
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
    return max_err < 1e-4 ? 0 : 1;
}
)";

    std::string path = "/tmp/ns_cpu_integration.cpp";
    std::ofstream of(path);
    of << src;
    of.close();

    // 3. Compile with the host C++ compiler.
    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -o /tmp/ns_cpu_integration " + path;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: generation compile failed\n";
        return 1;
    }

    // 4. Run and check the reported error bound.
    if (std::system("/tmp/ns_cpu_integration") != 0) {
        std::cerr << "FAIL: runtime mismatch vs reference\n";
        return 1;
    }

    std::cout << "PASS: CPU backend integration (batch=3 matmul chain)\n";
    return 0;
}