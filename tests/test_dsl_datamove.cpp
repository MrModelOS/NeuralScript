// v1.2 data-movement DSL test: exercises the function-call syntax
// slice()/index()/scatter()/concat()/transpose()/reshape() through the full
// pipeline (parse -> typecheck -> MLIR -> CPU codegen), compiles the emitted
// C++ and verifies against a naive reference. Also nvcc-validates the CUDA
// lowering (skipped when the toolkit is absent, CTest SKIP_RETURN_CODE=77).
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

static const char* DMML = R"(
type Batch = Dynamic
network DataMove2 {
    input:  Tensor[Batch, 3] float32
    output: Tensor[Batch, 5] float32
    forward(x) {
        var W1: Tensor[4, 3] float32
        var W2: Tensor[3, 2] float32
        var m = x @ W2
        var h = concat(m, x, 1)
        var s = slice(h, 1, 1, 4)
        var g = index(h, 1, 3, 0)
        var i = index(W1, 0, 1, 2)
        var t = transpose(W1)
        var r = reshape(h, 0, 5)
        return scatter(h, 1, g, 0, 1)
    }
}
)";

int main() {
    Lexer lexer(DMML);
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

    const char* needles[] = {"static void ns_index2", "static void ns_indexrows",
                             "static void ns_scatter2", "static void ns_scatterrows",
                             "static void ns_slice2", "static void ns_slicerows"};
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
    const int C = 3, W2r = 3, W2c = 2, N = 4, Hw = 5, W1r = 4;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(N * C);
    for (int i = 0; i < N*C; i++) x[i] = dist(rng);
    std::vector<float> w(18); // W1[4,3]=12 floats @0, W2[3,2]=6 floats @12
    for (auto& v : w) v = dist(rng);
    std::vector<float> out(N * Hw, -777.f);
    forward(x.data(), w.data(), out.data(), (size_t)(N * C));

    const float* W1 = w.data();
    const float* W2 = w.data() + 12;
    std::vector<float> h(N * Hw);
    for (int i = 0; i < N; i++) {
        h[i*Hw+0] = 0.f; h[i*Hw+1] = 0.f;
        for (int k = 0; k < W2r; k++) {
            h[i*Hw+0] += x[i*C+k] * W2[k*W2c+0];
            h[i*Hw+1] += x[i*C+k] * W2[k*W2c+1];
        }
        h[i*Hw+2] = x[i*C+0]; h[i*Hw+3] = x[i*C+1]; h[i*Hw+4] = x[i*C+2];
    }
    std::vector<float> ref = h; // scatter(h,1,g,0,1): copy, then out[:0]=g[:0]=h[:3], out[:1]=g[:1]=h[:0]
    for (int i = 0; i < N; i++) {
        ref[i*Hw+0] = h[i*Hw+3];
        ref[i*Hw+1] = h[i*Hw+0];
    }
    double max_err = 0.0;
    for (int i = 0; i < N*Hw; i++)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    std::printf("%.3e\n", max_err);
    if (!(max_err < 1e-4)) return 1;
    // sanity: ref must not equal the untouched h by noise (verify scatter ran)
    bool changed = false;
    for (int i = 0; i < N; i++) if (ref[i*Hw+0] != h[i*Hw+0] || ref[i*Hw+1] != h[i*Hw+1]) changed = true;
    return changed ? 0 : 1;
}
)";

    std::string path = "/tmp/ns_dsl_datamove_cpu.cpp";
    { std::ofstream of(path); of << src; }
    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    if (system((cc + " -O2 -o /tmp/ns_dsl_datamove_cpu " + path).c_str()) != 0) {
        std::cerr << "FAIL: generation compile failed\n";
        return 1;
    }
    if (system("/tmp/ns_dsl_datamove_cpu") != 0) {
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
        const char* cu_needles[] = {"ns_slice2_kernel", "ns_index_kernel",
                                    "ns_indexrows_kernel", "ns_scatter_kernel",
                                    "ns_scatterrows_kernel"};
        for (auto* nd : cu_needles) {
            if (cu_code.find(nd) == std::string::npos) {
                std::cerr << "FAIL: CUDA code missing '" << nd << "'\n";
                return 1;
            }
        }
        std::string cu_path = "/tmp/ns_dsl_datamove.cu";
        { std::ofstream of(cu_path); of << cu_code; }
        if (system((nvcc + " -c -O2 -std=c++11 " + cu_path + " -o /tmp/ns_dsl_datamove.o 2>&1").c_str()) != 0) {
            std::cerr << "FAIL: nvcc rejected emitted CUDA source\n";
            return 1;
        }
    }

    std::cout << "PASS: DSL slice/index/scatter/concat/transpose/reshape\n";
    return 0;
}