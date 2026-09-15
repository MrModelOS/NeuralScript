// v1.2 data-movement integration test: builds an MLIRModule programmatically
// that exercises TRANSPOSE, CONCAT (axis 1), RESHAPE and LAYERNORM on top of
// an EMBEDDING-fed subgraph, emits CPU C++ and verifies it against a naive
// reference. Also asserts the CUDA lowering splices the new kernels.
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/codegen/codegen.hpp"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ns;

static MLIRModule build_module() {
    MLIRModule mod;
    MLIRFunction fn;
    fn.name = "DataMove";
    fn.is_train = false;

    auto alloc = [](const std::string& id, int64_t r, int64_t c) {
        MLIRInstr in(MLIROp::TENSOR_ALLOC, id);
        in.result_type = TensorType({DimExpr::constant(r), DimExpr::constant(c)},
                                    Dtype::Float32);
        return in;
    };
    fn.instructions.push_back(alloc("emb_w", 4, 3));   // [V=4, D=3]
    fn.instructions.push_back(alloc("d1_w", 3, 2));    // [3, 2]
    fn.instructions.push_back(alloc("t1_w", 2, 3));    // transpose source

    MLIRInstr e(MLIROp::LAYER_EMBEDDING, "e1");
    e.operands = {"emb_w", "input"};
    e.result_type = TensorType({DimExpr::dynamic(), DimExpr::constant(3)}, Dtype::Float32);
    fn.instructions.push_back(e);

    MLIRInstr mm(MLIROp::MATMUL, "m1");
    mm.operands = {"e1", "d1_w"};
    mm.result_type = TensorType({DimExpr::dynamic(), DimExpr::constant(2)}, Dtype::Float32);
    fn.instructions.push_back(mm);

    MLIRInstr tr(MLIROp::TRANSPOSE, "t1");
    tr.operands = {"t1_w"};
    tr.result_type = TensorType({DimExpr::constant(3), DimExpr::constant(2)}, Dtype::Float32);
    fn.instructions.push_back(tr);

    MLIRInstr cc(MLIROp::CONCAT, "c1");
    cc.operands = {"m1", "e1"};
    cc.attribute = "1"; // axis 1 (features)
    cc.result_type = TensorType({DimExpr::dynamic(), DimExpr::constant(5)}, Dtype::Float32);
    fn.instructions.push_back(cc);

    MLIRInstr rs(MLIROp::RESHAPE, "r1");
    rs.operands = {"c1"};
    rs.result_type = cc.result_type;
    fn.instructions.push_back(rs);

    MLIRInstr ln(MLIROp::LAYERNORM, "ln1");
    ln.operands = {"c1"};
    ln.result_type = cc.result_type;
    fn.instructions.push_back(ln);

    fn.return_id = "ln1";
    mod.functions.push_back(std::move(fn));
    return mod;
}

int main() {
    auto mod = build_module();
    CodegenOptions opts;
    opts.backend = TargetBackend::CPU_CXX;
    CodeGenerator cg;
    std::string code = cg.generate(mod, opts);

    const char* needles[] = {"ns_embedding", "ns_transpose2d", "ns_concat2",
                             "ns_layernorm", "buf_ln1"};
    for (auto* nd : needles) {
        if (code.find(nd) == std::string::npos) {
            std::cerr << "FAIL: CPU code missing '" << nd << "'\n";
            return 1;
        }
    }

    CodegenOptions cu = opts;
    cu.backend = TargetBackend::CUDA;
    std::string cuda = cg.generate(mod, cu);
    const char* cu_needles[] = {"ns_transpose2d_kernel", "ns_concat2_kernel",
                                "ns_embedding_kernel", "ns_concat0_kernel"};
    for (auto* nd : cu_needles) {
        if (cuda.find(nd) == std::string::npos) {
            std::cerr << "FAIL: CUDA code missing '" << nd << "'\n";
            return 1;
        }
    }

    // ---- CPU harness vs naive reference. ----
    std::string src = R"(
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
)";
    src += code;
    src += R"(
int main() {
    const int V = 4, D = 3, K = 3, N = 2, S = 2;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(S);
    for (int i = 0; i < S; i++) x[i] = (float)(i % V);
    // alloc order: emb_w[4,3] @0 (12), d1_w[3,2] @12 (6), t1_w[2,3] @18 (6)
    std::vector<float> w(24);
    for (auto& v : w) v = dist(rng);
    std::vector<float> out(S * 5, -777.f);
    forward(x.data(), w.data(), out.data(), x.size());

    const float* emb = w.data();
    const float* d1 = w.data() + 12;
    std::vector<float> E(S*D), M(S*N);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) E[i*D+j] = emb[(int)x[i]*D + j];
    for (int i = 0; i < S; i++)
        for (int j = 0; j < N; j++) {
            float a = 0.f;
            for (int k = 0; k < K; k++) a += E[i*K+k] * d1[k*N+j];
            M[i*N+j] = a;
        }
    std::vector<float> ref(S * 5);
    int W = 5; // concat axis 1: M(2) ++ E(3)
    for (int i = 0; i < S; i++) {
        ref[i*W+0] = M[i*N+0]; ref[i*W+1] = M[i*N+1];
        ref[i*W+2] = E[i*D+0]; ref[i*W+3] = E[i*D+1]; ref[i*W+4] = E[i*D+2];
    }
    for (int i = 0; i < S; i++) { // layernorm, last = 5
        float mean = 0.f, var = 0.f;
        for (int j = 0; j < W; j++) mean += ref[i*W+j];
        mean /= W;
        for (int j = 0; j < W; j++) { float d_ = ref[i*W+j] - mean; var += d_*d_; }
        var /= W;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < W; j++) ref[i*W+j] = (ref[i*W+j] - mean) * inv;
    }
    double max_err = 0.0;
    for (int i = 0; i < S*W; i++)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    std::printf("%.3e\n", max_err);
    return max_err < 1e-4 ? 0 : 1;
}
)";

    std::string path = "/tmp/ns_datamove_cpu.cpp";
    {
        std::ofstream of(path);
        of << src;
    }
    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    if (std::system((cc + " -O2 -o /tmp/ns_datamove_cpu " + path).c_str()) != 0) {
        std::cerr << "FAIL: generation compile failed\n";
        return 1;
    }
    if (std::system("/tmp/ns_datamove_cpu") != 0) {
        std::cerr << "FAIL: runtime mismatch vs reference\n";
        return 1;
    }

    // ---- CUDA nvcc compile validation (device-only, no GPU required). ----
    {
        std::string nvcc = "nvcc";
        if (system((nvcc + " --version > /dev/null 2>&1").c_str()) != 0) {
            std::cout << "SKIP: no CUDA toolkit (nvcc)\n";
            return 77;
        }
        CodegenOptions cu = opts;
        cu.backend = TargetBackend::CUDA;
        std::string cu_code = cg.generate(mod, cu);
        std::string cu_path = "/tmp/ns_datamove.cu";
        { std::ofstream of(cu_path); of << cu_code; }
        if (system((nvcc + " -c -O2 -std=c++11 " + cu_path + " -o /tmp/ns_datamove.o 2>&1").c_str()) != 0) {
            std::cerr << "FAIL: nvcc rejected emitted CUDA source\n";
            return 1;
        }
    }

    std::cout << "PASS: data-movement (embedding+transpose+concat+reshape+layernorm)\n";
    return 0;
}