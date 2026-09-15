// v1.2 Mixture-of-Experts integration test: a MoE layer (router softmax,
// top-1 dispatch, weighted combine) compiled through the full pipeline
// (parse -> typecheck -> MLIR -> CPU codegen), executed against a naive
// reference. Also validates the CUDA lowering & nvcc (skipped when the CUDA
// toolkit is absent, CTest SKIP_RETURN_CODE=77).
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ns;

static const char* MOENET = R"(
type Batch = Dynamic
network MOENet {
    input:  Tensor[Batch, 4] float32
    output: Tensor[Batch, 4] float32
    layer moe = MoE(d_model: 4, num_experts: 3)
    forward(x) {
        return x -> moe
    }
}
)";

int main() {
    Lexer lexer(MOENET);
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

    const char* needles[] = {"static void ns_moe_fwd"};
    for (auto* nd : needles) {
        if (code.find(nd) == std::string::npos) {
            std::cerr << "FAIL: CPU code missing '" << nd << "'\n";
            return 1;
        }
    }

    // CPU harness vs naive reference.
    std::string src = R"(
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
)";
    src += code;
    src += R"(
int main() {
    const int D = 4, E = 3, N = 5, GATE = D * E;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(N * D);
    for (int i = 0; i < N*D; i++) x[i] = dist(rng);
    std::vector<float> w(GATE + E * D * D); // Wg[D,E] @0, We[E,D,D] @GATE
    for (auto& v : w) v = dist(rng);
    std::vector<float> out(N * D, -777.f);
    forward(x.data(), w.data(), out.data(), (size_t)(N * D));

    const float* Wg = w.data();
    const float* We = w.data() + GATE;
    std::vector<float> ref(N * D);
    for (int i = 0; i < N; i++) {
        std::vector<float> lg(E);
        float mx = -1.0e30f;
        for (int e = 0; e < E; e++) {
            float a = 0.f;
            for (int k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
            lg[e] = a; mx = std::max(mx, a);
        }
        float sum = 0.f;
        for (int e = 0; e < E; e++) { lg[e] = std::exp(lg[e]-mx); sum += lg[e]; }
        int best = 0;
        for (int e = 1; e < E; e++) if (lg[e] > lg[best]) best = e;
        float p = (sum > 0.f) ? lg[best]/sum : 0.f;
        for (int j = 0; j < D; j++) {
            float a = 0.f;
            for (int k = 0; k < D; k++) a += x[i*D+k] * We[best*D*D + k*D + j];
            ref[i*D+j] = p * a;
        }
    }
    double max_err = 0.0;
    for (int i = 0; i < N*D; i++)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    std::printf("%.3e\n", max_err);
    if (!(max_err < 1e-4)) return 1;
    return 0;
}
)";

    std::string path = "/tmp/ns_moe_cpu.cpp";
    { std::ofstream of(path); of << src; }
    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    if (system((cc + " -O2 -o /tmp/ns_moe_cpu " + path).c_str()) != 0) {
        std::cerr << "FAIL: generation compile failed\n";
        return 1;
    }
    if (system("/tmp/ns_moe_cpu") != 0) {
        std::cerr << "FAIL: runtime mismatch vs reference\n";
        return 1;
    }

    // CUDA nvcc compile validation.
    {
        std::string nvcc = "nvcc";
        if (system((nvcc + " --version > /dev/null 2>&1").c_str()) != 0) {
            std::cout << "SKIP: no CUDA toolkit (nvcc)\n";
            return 77;
        }
        CodegenOptions cu = opts;
        cu.backend = TargetBackend::CUDA;
        std::string cu_code = cg.generate(module, cu);
        if (cu_code.find("ns_moe_kernel") == std::string::npos) {
            std::cerr << "FAIL: CUDA code missing 'ns_moe_kernel'\n";
            return 1;
        }
        std::string cu_path = "/tmp/ns_moe.cu";
        { std::ofstream of(cu_path); of << cu_code; }
        if (system((nvcc + " -c -O2 -std=c++11 " + cu_path + " -o /tmp/ns_moe.o 2>&1").c_str()) != 0) {
            std::cerr << "FAIL: nvcc rejected emitted CUDA source\n";
            return 1;
        }
    }

    std::cout << "PASS: MoE router/top-1/weighted-combine\n";
    return 0;
}