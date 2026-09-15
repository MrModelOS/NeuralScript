// Transformer-forward integration test: Embedding -> multi-head Attention ->
// LayerNorm through the full pipeline (parse -> typecheck -> MLIR -> codegen),
// emitted as CPU C++, compiled with the host compiler and executed against a
// naive scalar reference. Also asserts the CUDA lowering contains the new
// v1.2 data-movement/attention kernels.
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
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

static const char* TRANSFORMER = R"(
type Batch = Dynamic
type Seq = Dynamic
network Transformer {
    input:  Tensor[Batch, Seq] float32
    output: Tensor[Batch, Seq, 8] float32
    layer emb = Embedding(vocab_size: 16, d_model: 8)
    layer attn = Attention(d_model: 8, heads: 2)
    layer ln  = LayerNorm()
    forward(x) {
        var h = x -> emb
        var a = h -> attn
        return a -> ln
    }
}
)";

int main() {
    Lexer lexer(TRANSFORMER);
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

    // Structural invariants: v1.2 kernels + attention weight buffers present.
    const char* needles[] = {
        "ns_embedding", "ns_attention_fwd", "ns_layernorm",
        "buf_attn_q_w", "buf_attn_k_w", "buf_attn_v_w", "buf_attn_o_w",
        "buf_emb_w",
    };
    for (auto* nd : needles) {
        if (code.find(nd) == std::string::npos) {
            std::cerr << "FAIL: generated CPU code missing '" << nd << "'\n";
            return 1;
        }
    }

    // CUDA lowering structural check (compile-only; device run happens on the
    // GPU runner).
    CodegenOptions cu = opts;
    cu.backend = TargetBackend::CUDA;
    std::string cuda = cg.generate(module, cu);
    const char* cu_needles[] = {
        "ns_attention_core_kernel", "ns_embedding_kernel",
        "ns_transpose2d_kernel", "ns_concat2_kernel",
    };
    for (auto* nd : cu_needles) {
        if (cuda.find(nd) == std::string::npos) {
            std::cerr << "FAIL: generated CUDA code missing '" << nd << "'\n";
            return 1;
        }
    }

    // ---- CPU harness: run forward() vs a naive scalar reference. ----
    std::string src = R"(
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
)";
    src += code;
    src += R"(
namespace {
void gemm(const std::vector<float>& A, const std::vector<float>& B,
          std::vector<float>& C, int M, int K, int N) {
    C.assign((size_t)M * N, 0.f);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float a = 0.f;
            for (int k = 0; k < K; k++) a += A[i*K+k] * B[k*N+j];
            C[i*N+j] = a;
        }
}
void layernorm(std::vector<float>& m, int rows, int last) {
    for (int r = 0; r < rows; r++) {
        float mean = 0.f, var = 0.f;
        for (int j = 0; j < last; j++) mean += m[r*last+j];
        mean /= last;
        for (int j = 0; j < last; j++) { float d = m[r*last+j] - mean; var += d*d; }
        var /= last;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < last; j++) m[r*last+j] = (m[r*last+j] - mean) * inv;
    }
}
}
int main() {
    const int V = 16, D = 8, H = 2, Dk = D / H, S = 4;
    std::mt19937 rng(2025);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(S);
    for (int i = 0; i < S; i++) x[i] = (float)(i % V);
    // weight layout (TENSOR_ALLOC order): emb[V*D], attn_q, attn_k, attn_v, attn_o
    size_t nw = (size_t)V*D + 4*(size_t)D*D;
    std::vector<float> w(nw);
    for (auto& v : w) v = dist(rng);
    std::vector<float> out(S * D, -999.f);
    forward(x.data(), w.data(), out.data(), x.size());

    const float* emb = w.data();
    const float* Wq = w.data() + V*D;
    const float* Wk = w.data() + V*D + D*D;
    const float* Wv = w.data() + V*D + 2*D*D;
    const float* Wo = w.data() + V*D + 3*D*D;
    std::vector<float> E(S*D), Qv(S*D), Kv(S*D), Vv(S*D), sc(S*S);
    std::vector<float> ctx(S*D), preout, ref(S*D);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) E[i*D+j] = emb[(int)x[i]*D + j];
    gemm(E, std::vector<float>(Wq, Wq+D*D), Qv, S, D, D);
    gemm(E, std::vector<float>(Wk, Wk+D*D), Kv, S, D, D);
    gemm(E, std::vector<float>(Wv, Wv+D*D), Vv, S, D, D);
    float scale = 1.0f / sqrtf((float)Dk);
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < S; i++)
            for (int j = 0; j < S; j++) {
                float a = 0.f;
                for (int d = 0; d < Dk; d++)
                    a += Qv[i*D+h*Dk+d] * Kv[j*D+h*Dk+d];
                sc[i*S+j] = a * scale;
            }
        for (int i = 0; i < S; i++) {
            float mx = sc[i*S];
            for (int j = 1; j < S; j++) mx = std::max(mx, sc[i*S+j]);
            float s = 0.f;
            for (int j = 0; j < S; j++) { sc[i*S+j] = expf(sc[i*S+j]-mx); s += sc[i*S+j]; }
            for (int j = 0; j < S; j++) sc[i*S+j] /= s;
        }
        for (int i = 0; i < S; i++)
            for (int d = 0; d < Dk; d++) {
                float a = 0.f;
                for (int j = 0; j < S; j++) a += sc[i*S+j] * Vv[j*D+h*Dk+d];
                ctx[i*D+h*Dk+d] = a;
            }
    }
    gemm(ctx, std::vector<float>(Wo, Wo+D*D), preout, S, D, D);
    ref = preout;
    layernorm(ref, S, D);

    double max_err = 0.0;
    for (int i = 0; i < S*D; i++)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    std::printf("%.3e\n", max_err);
    return max_err < 1e-4 ? 0 : 1;
}
)";

    std::string path = "/tmp/ns_transformer_cpu.cpp";
    {
        std::ofstream of(path);
        of << src;
    }

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -o /tmp/ns_transformer_cpu " + path;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: generation compile failed\n";
        return 1;
    }
    if (std::system("/tmp/ns_transformer_cpu") != 0) {
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
        std::string cu_code = cg.generate(module, cu);
        std::string cu_path = "/tmp/ns_transformer.cu";
        { std::ofstream of(cu_path); of << cu_code; }
        if (system((nvcc + " -c -O2 -std=c++11 " + cu_path + " -o /tmp/ns_transformer.o 2>&1").c_str()) != 0) {
            std::cerr << "FAIL: nvcc rejected emitted CUDA source\n";
            return 1;
        }
    }

    std::cout << "PASS: transformer forward (embedding+attention+layernorm)\n";
    return 0;
}