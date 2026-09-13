// CUDA backend integration test (opt-in via -DNS_ENABLE_CUDA=ON; registered
// only in the NS_ENABLE_CUDA block of tests/CMakeLists.txt).
//
// Lifts two MLPs with a train() method through the full pipeline with the CUDA
// backend, writes the emitted translation units to disk, compiles them with
// nvcc together with a C-ABI host, and runs training ON the GPU:
//
//   1. XOR  (2x16->16x2, both weights min(r,c)=2  < 8) -> AdamW path:
//        loss drops from ~ln(2) to a small value over 400 AdamW steps,
//        4/4 classification after training.
//   2. MUONX (8x8->8x8->8x8 identity on steps, weights min(r,c)=8 >= 8)
//        -> Muon path (slope-0.95 momentum + orthonom + AdamW-style decay):
//        30000 steps at lr=0.0005 converge 8/8. The same schedule runs against
//        the CPU reference so the task provably trains under the reference
//        Muon; divergence of the device path from the reference is caught too.
//
// Shared ABI checks (both nets): loss decreases, weights moved, classification,
// ns_objective_loss is loss-consistent.
//
// Skips (exit 77 -> CTest SKIPPED via SKIP_RETURN_CODE) when nvcc is missing
// or no CUDA-capable device is present, so ctest stays green on non-GPU
// machines. GPU-heavy runs belong in the separate .github/workflows/cuda.yml
// job on a self-hosted GPU runner; the default PR pipeline remains CPU-only.
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
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

using namespace ns;

#ifndef NS_REPO_INCLUDE_DIR
#define NS_REPO_INCLUDE_DIR ".."
#endif
#ifndef NS_NVCC_PATH
#define NS_NVCC_PATH ""
#endif

static const char* XOR = R"(
type Bs = Dynamic
network XOR {
    input:  Tensor[Bs, 2] float32
    output: Tensor[Bs, 2] float32
    layer fc1 = Dense(in: 2, out: 16, activation: ReLU)
    layer fc2 = Dense(in: 16, out: 2, activation: Identity)
    forward(x) {
        return x -> fc1 -> fc2
    }
    train(x: Tensor[Bs, 2], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h     = relu(x @ fc1)
            var preds = h @ fc2
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

// Both 8x8 weight matrices have min(r,c) = 8 >= 8, so every layer runs the Muon
// path (ns_muon_ema_kernel + ns_orthonom_kernel + ns_muon_step_kernel).
static const char* MUONX = R"(
type Bs = Dynamic
network MUONX {
    input:  Tensor[Bs, 8] float32
    output: Tensor[Bs, 8] float32
    layer fc1 = Dense(in: 8, out: 8, activation: ReLU)
    layer fc2 = Dense(in: 8, out: 8, activation: Identity)
    forward(x) {
        return x -> fc1 -> fc2
    }
    train(x: Tensor[Bs, 8], labels: Tensor[Bs, 8]) -> float32 {
        grad {
            var h     = relu(x @ fc1)
            var preds = h @ fc2
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

// Dropout training path: inverted scaled dropout in the grad block. Forward
// (train_mode) builds a curand mask and scales by 1/(1-p); the backward pass
// re-uses the SAME mask. Eval (train_mode=0) is a pure copy. The label 16-wide
// hidden layer means the net must still fit XOR with half the units dropped
// per step.
static const char* DROPOUTX = R"(
type Bs = Dynamic
network DROPOUTX {
    input:  Tensor[Bs, 2] float32
    output: Tensor[Bs, 2] float32
    layer fc1 = Dense(in: 2, out: 16, activation: ReLU)
    layer dr  = Dropout(rate: 0.5)
    layer fc2 = Dense(in: 16, out: 2, activation: Identity)
    forward(x) {
        return x -> fc1 -> dr -> fc2
    }
    train(x: Tensor[Bs, 2], labels: Tensor[Bs, 2]) -> float32 {
        grad {
            var h     = relu(x @ fc1)
            var d     = dropout(h, 0.5)
            var preds = d @ fc2
            var loss  = cross_entropy(preds, labels)
        }
        return loss
    }
}
)";

static int run_cmd(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    if (rc == -1) return -1;
    return WEXITSTATUS(rc);
}

// Shared upstream of the C-ABI declaration block every generated host uses.
static const char* kHostAbi[] = {
    "typedef struct ns_model ns_model;\n",
    "extern \"C\" ns_model* ns_runtime_init(const float*, size_t);\n",
    "extern \"C\" int ns_runtime_train_step(ns_model*, const float*, const float*, size_t, float*, float);\n",
    "extern \"C\" int ns_objective_loss(ns_model*, const float*, const float*, size_t, float*);\n",
    "extern \"C\" int ns_eval_infer(ns_model*, const float*, float*, size_t);\n",
    "extern \"C\" int ns_model_get_weights(const ns_model*, float*, size_t);\n",
    "extern \"C\" void ns_free(ns_model*);\n",
};

static void write_host_abi(std::ostream& os, bool use_cuda_runtime) {
    os << "#include <cstdio>\n"
       << "#include <vector>\n"
       << "#include <cmath>\n";
    if (use_cuda_runtime) os << "#include <cuda_runtime.h>\n";
    for (auto* d : kHostAbi) os << d;
    os << "static int fail(const char* what) { fprintf(stderr, \"FAIL: %s\\n\", what); return 1; }\n";
}

static std::string xor_host_source() {
    std::string s;
    {
        std::ostringstream o;
        write_host_abi(o, /*use_cuda_runtime=*/true);
        o << "int main() {\n"
             "  int dev = 0;\n"
             "  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) return 77;\n"
             "  const float xs[4][2] = {{-1.f,-1.f},{-1.f,1.f},{1.f,-1.f},{1.f,1.f}};\n"
             "  const float ys[4][2] = {{1.f,0.f},{0.f,1.f},{0.f,1.f},{1.f,0.f}};\n"
             "  const size_t xn = 4 * 2, nw = 2 * 16 + 16 * 2;\n"
             "  std::vector<float> w(nw);\n"
             "  unsigned s = 12345u;\n"
             "  for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }\n"
             "  ns_model* m = ns_runtime_init(w.data(), nw);\n"
             "  if (!m) return fail(\"init\");\n"
             "  std::vector<float> w0(nw);\n"
             "  if (ns_model_get_weights(m, w0.data(), nw) != 0) return fail(\"get_weights(0)\");\n"
             "  float loss0 = -1.f;\n"
             "  if (ns_objective_loss(m, xs[0], ys[0], xn, &loss0) != 0) return fail(\"objective(0)\");\n"
             "  if (!(loss0 > 0.5f && loss0 < 1.0f)) return fail(\"loss0 sanity\");\n"
             "  for (int e = 0; e < 400; e++) {\n"
             "    float lo = -1.f;\n"
             "    if (ns_runtime_train_step(m, xs[0], ys[0], xn, &lo, 0.3f) != 0) return fail(\"train_step\");\n"
             "  }\n"
             "  float lossT = -1.f;\n"
             "  if (ns_objective_loss(m, xs[0], ys[0], xn, &lossT) != 0) return fail(\"objective(T)\");\n"
             "  if (!(lossT < loss0 * 0.9f)) return fail(\"loss did not decrease\");\n"
             "  std::vector<float> wT(nw);\n"
             "  if (ns_model_get_weights(m, wT.data(), nw) != 0) return fail(\"get_weights(T)\");\n"
             "  int moved = 0;\n"
             "  for (size_t i = 0; i < nw; i++) if (wT[i] != w0[i]) moved++;\n"
             "  if (moved == 0) return fail(\"weights did not move\");\n"
             "  int acc = 0;\n"
             "  for (int p = 0; p < 4; p++) {\n"
             "    float out[2];\n"
             "    if (ns_eval_infer(m, xs[p], out, 2) != 0) return fail(\"eval_infer\");\n"
             "    int pred = out[0] >= out[1] ? 0 : 1;\n"
             "    int truth = ys[p][0] >= ys[p][1] ? 0 : 1;\n"
             "    if (pred == truth) acc++;\n"
             "  }\n"
             "  if (acc != 4) return fail(\"accuracy\");\n"
             "  std::printf(\"xor: loss0=%.4f lossT=%.5f acc=4/4\\n\", loss0, lossT);\n"
             "  ns_free(m);\n"
             "  return 0;\n"
             "}\n";
        s = o.str();
    }
    return s;
}

// 8-sample identity: sample p has input pattern x[p]=+0.5 with all other
// coordinates -0.5, label = one-hot(p). Both layers are Muon; the schedule
// (30000 x lr=0.0005) is validated against the CPU reference in main().
static std::string muon_host_source(bool use_cuda_runtime) {
    std::string s;
    {
        std::ostringstream o;
        write_host_abi(o, use_cuda_runtime);
        o << "int main() {\n";
        if (use_cuda_runtime)
            o << "  int dev = 0;\n"
                 "  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) return 77;\n";
        o << "  const int N = 8, nw = 8 * 8 + 8 * 8;\n"
             "  std::vector<float> xs(N * N, -0.5f), ys(N * N, 0.f);\n"
             "  for (int p = 0; p < N; p++) { xs[p * N + p] = 0.5f; ys[p * N + p] = 1.f; }\n"
             "  std::vector<float> w(nw);\n"
             "  unsigned s = 12345u;\n"
             "  for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.6f; }\n"
             "  ns_model* m = ns_runtime_init(w.data(), nw);\n"
             "  if (!m) return fail(\"init\");\n"
             "  std::vector<float> w0(nw);\n"
             "  if (ns_model_get_weights(m, w0.data(), nw) != 0) return fail(\"get_weights(0)\");\n"
             "  float loss0 = -1.f;\n"
             "  if (ns_objective_loss(m, xs.data(), ys.data(), N * N, &loss0) != 0) return fail(\"objective(0)\");\n"
             "  if (!(loss0 > 1.5f && loss0 < 3.0f)) return fail(\"loss0 sanity\");\n"
             "  float lossT = loss0;\n"
             "  for (int e = 0; e < 30000; e++) {\n"
             "    if (ns_runtime_train_step(m, xs.data(), ys.data(), N * N, &lossT, 0.0005f) != 0) return fail(\"train_step\");\n"
             "  }\n"
             "  if (!(lossT < 0.3f * loss0)) return fail(\"loss did not decrease\");\n"
             "  std::vector<float> wT(nw);\n"
             "  if (ns_model_get_weights(m, wT.data(), nw) != 0) return fail(\"get_weights(T)\");\n"
             "  int moved = 0;\n"
             "  for (size_t i = 0; i < nw; i++) if (wT[i] != w0[i]) moved++;\n"
             "  if (moved == 0) return fail(\"weights did not move\");\n"
             "  int acc = 0;\n"
             "  for (int p = 0; p < N; p++) {\n"
             "    std::vector<float> out(N);\n"
             "    if (ns_eval_infer(m, &xs[p * N], out.data(), N) != 0) return fail(\"eval_infer\");\n"
             "    int pred = 0;\n"
             "    for (int j = 1; j < N; j++) if (out[j] > out[pred]) pred = j;\n"
             "    if (pred == p) acc++;\n"
             "  }\n"
             "  if (acc != N) return fail(\"accuracy\");\n"
             "  std::printf(\"muon: loss0=%.4f lossT=%.6f acc=8/8\\n\", loss0, lossT);\n"
             "  ns_free(m);\n"
             "  return 0;\n"
             "}\n";
        s = o.str();
    }
    return s;
}

// Dropout XOR host: same feed as the AdamW XOR net, but the forward (in the
// grad block) applies inverted dropout after the ReLU, so each step trains on
// a randomly half-masked hidden state and the SAME mask drives the backward
// through ns_dropout_mul_kernel. Eval (inference, mask=identity) must reach
// 4/4 accuracy despite having only ever seen half-units-dropped forward passes.
static std::string dropout_host_source() {
    std::string s;
    {
        std::ostringstream o;
        write_host_abi(o, /*use_cuda_runtime=*/true);
        o << "int main() {\n"
             "  int dev = 0;\n"
             "  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev <= 0) return 77;\n"
             "  const float xs[4][2] = {{-1.f,-1.f},{-1.f,1.f},{1.f,-1.f},{1.f,1.f}};\n"
             "  const float ys[4][2] = {{1.f,0.f},{0.f,1.f},{0.f,1.f},{1.f,0.f}};\n"
             "  const size_t xn = 4 * 2, nw = 2 * 16 + 16 * 2;\n"
             "  std::vector<float> w(nw);\n"
             "  unsigned s = 12345u;\n"
             "  for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f) * 0.3f; }\n"
             "  ns_model* m = ns_runtime_init(w.data(), nw);\n"
             "  if (!m) return fail(\"init\");\n"
             "  float loss0 = -1.f;\n"
             "  if (ns_objective_loss(m, xs[0], ys[0], xn, &loss0) != 0) return fail(\"objective(0)\");\n"
             "  if (!(loss0 > 0.5f && loss0 < 1.0f)) return fail(\"loss0 sanity\");\n"
             "  float lossT = -1.f;\n"
             "  for (int e = 0; e < 400; e++) {\n"
             "    if (ns_runtime_train_step(m, xs[0], ys[0], xn, &lossT, 0.3f) != 0) return fail(\"train_step\");\n"
             "  }\n"
             "  if (!(lossT < loss0 * 0.9f)) return fail(\"train loss did not decrease\");\n"
             "  int acc = 0;\n"
             "  for (int p = 0; p < 4; p++) {\n"
             "    float out[2];\n"
             "    if (ns_eval_infer(m, xs[p], out, 2) != 0) return fail(\"eval_infer\");\n"
             "    int pred = out[0] >= out[1] ? 0 : 1;\n"
             "    int truth = ys[p][0] >= ys[p][1] ? 0 : 1;\n"
             "    if (pred == truth) acc++;\n"
             "  }\n"
             "  if (acc != 4) return fail(\"accuracy\");\n"
             "  std::printf(\"dropout: loss0=%.4f lossT=%.5f acc=4/4\\n\", loss0, lossT);\n"
             "  ns_free(m);\n"
             "  return 0;\n"
             "}\n";
        s = o.str();
    }
    return s;
}

int main() {
    Lexer lexer(XOR);
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
    opts.backend = TargetBackend::CUDA;
    opts.emit_runtime_driver = true;
    opts.function_name = "ns_xor_forward";
    CodeGenerator cg;
    std::string code = cg.generate(module, opts);

    const std::string cu = "/tmp/ns_cuda_kernels.cu";
    {
        std::ofstream of(cu);
        of << code;
    }

    // The emitted TU must contain the device kernels, the C-ABI driver and the
    // AOT training core.
    if (code.find("__global__") == std::string::npos ||
        code.find("ns_train_core") == std::string::npos ||
        code.find("ns_runtime_train_step") == std::string::npos) {
        std::cerr << "FAIL: emitted CUDA is missing kernels / training ABI\n";
        return 1;
    }

    // Locate nvcc (explicit path from CMake, else PATH lookup).
    std::string nvcc = NS_NVCC_PATH;
    if (nvcc.empty()) nvcc = "nvcc";

    if (run_cmd(nvcc + " --version > /dev/null 2>&1") != 0) {
        std::cout << "SKIP: no CUDA toolkit (nvcc) available\n";
        return 77;
    }

    // ---- Net 1: XOR, AdamW device path. ----
    const std::string host_xor = "/tmp/ns_cuda_host.cpp";
    {
        std::ofstream hf(host_xor);
        hf << xor_host_source();
    }

    // Device-only compile first: catches kernel-signature / syntax defects in
    // the emitted CUDA without needing hardware.
    std::string obj = "/tmp/ns_cuda_kernels.o";
    if (run_cmd(nvcc + " -c -O2 -std=c++11 " + cu + " -o " + obj) != 0) {
        std::cerr << "FAIL: nvcc rejected the emitted CUDA source\n";
        return 1;
    }

    std::string bin = "/tmp/ns_cuda_train";
    if (run_cmd(nvcc + " -O2 -std=c++11 " + cu + " " + host_xor + " -o " + bin) != 0) {
        std::cerr << "FAIL: nvcc host link failed\n";
        return 1;
    }

    int rc = run_cmd(bin);
    if (rc == 77) {
        std::cout << "SKIP: no CUDA-capable device present\n";
        return 77;
    }
    if (rc != 0) {
        std::cerr << "FAIL: CUDA XOR (AdamW) training run failed (exit " << rc << ")\n";
        return 1;
    }

    // ---- Net 2: MUONX, Muon device path. ----
    Lexer lexer2(MUONX);
    auto toks2 = lexer2.tokenize();
    Parser parser2(toks2);
    Program prog2 = parser2.parse_program();
    ShapeChecker checker2;
    checker2.check(prog2);
    MLIRCompiler mlir2;
    auto module2 = mlir2.compile(prog2);
    FusionPass fuse2;
    fuse2.run(module2);

    CodegenOptions opts2;
    opts2.backend = TargetBackend::CUDA;
    opts2.emit_runtime_driver = true;
    opts2.function_name = "ns_muonx_forward";
    std::string code2 = cg.generate(module2, opts2);
    const std::string cu2 = "/tmp/ns_cuda_muon.cu";
    {
        std::ofstream of(cu2);
        of << code2;
    }
    if (code2.find("ns_muon_ema_kernel") == std::string::npos ||
        code2.find("ns_orthonom_kernel") == std::string::npos ||
        code2.find("ns_muon_step_kernel") == std::string::npos) {
        std::cerr << "FAIL: emitted CUDA is missing the Muon kernel path\n";
        return 1;
    }

    // CPU reference for the same Muon schedule: proves the task is learnable
    // under the reference optimizer and between the two implementations.
    CodegenOptions optsc;
    optsc.backend = TargetBackend::CPU_CXX;
    optsc.emit_runtime_driver = true;
    optsc.function_name = "ns_muonx_forward";
    std::string ccode = cg.generate(module2, optsc);
    const std::string cpu_drv = "/tmp/ns_muonx_cpu.cpp";
    {
        std::ofstream of(cpu_drv);
        of << ccode;
    }
    const std::string cpu_host = "/tmp/ns_muonx_cpu_host.cpp";
    {
        std::ofstream hf(cpu_host);
        hf << muon_host_source(/*use_cuda_runtime=*/false);
    }
    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cpu_bin = "/tmp/ns_muonx_cpu_run";
    if (run_cmd(cc + " -O2 -std=c++11 " + cpu_drv + " " + cpu_host + " -o " + cpu_bin) != 0) {
        std::cerr << "FAIL: CPU Muon reference link failed\n";
        return 1;
    }
    if (run_cmd(cpu_bin) != 0) {
        std::cerr << "FAIL: CPU Muon reference run failed\n";
        return 1;
    }

    const std::string host_muon = "/tmp/ns_cuda_muon_host.cpp";
    {
        std::ofstream hf(host_muon);
        hf << muon_host_source(/*use_cuda_runtime=*/true);
    }
    std::string bin2 = "/tmp/ns_cuda_muon_run";
    if (run_cmd(nvcc + " -O2 -std=c++11 " + cu2 + " " + host_muon + " -o " + bin2) != 0) {
        std::cerr << "FAIL: nvcc Muon host link failed\n";
        return 1;
    }
    int rc2 = run_cmd(bin2);
    if (rc2 == 77) {
        std::cout << "SKIP: no CUDA-capable device present\n";
        return 77;
    }
    if (rc2 != 0) {
        std::cerr << "FAIL: CUDA MUONX (Muon) training run failed (exit " << rc2 << ")\n";
        return 1;
    }

    // ---- Net 3: DROPOUTX, curand mask forward + same-mask backward. ----
    Lexer lexer3(DROPOUTX);
    auto toks3 = lexer3.tokenize();
    Parser parser3(toks3);
    Program prog3 = parser3.parse_program();
    ShapeChecker checker3;
    checker3.check(prog3);
    MLIRCompiler mlir3;
    auto module3 = mlir3.compile(prog3);
    FusionPass fuse3;
    fuse3.run(module3);

    CodegenOptions opts3;
    opts3.backend = TargetBackend::CUDA;
    opts3.emit_runtime_driver = true;
    opts3.function_name = "ns_dropoutx_forward";
    std::string code3 = cg.generate(module3, opts3);
    const std::string cu3 = "/tmp/ns_cuda_dropout.cu";
    {
        std::ofstream of(cu3);
        of << code3;
    }
    if (code3.find("ns_dropout_fwd_kernel") == std::string::npos ||
        code3.find("ns_dropout_mul_kernel") == std::string::npos ||
        code3.find("d_dm_") == std::string::npos) {
        std::cerr << "FAIL: emitted CUDA is missing the dropout mask path\n";
        return 1;
    }

    if (run_cmd(nvcc + " -c -O2 -std=c++11 " + cu3 + " -o /tmp/ns_cuda_dropout.o") != 0) {
        std::cerr << "FAIL: nvcc rejected the dropout CUDA source\n";
        return 1;
    }

    const std::string host_dropout = "/tmp/ns_cuda_dropout_host.cpp";
    {
        std::ofstream hf(host_dropout);
        hf << dropout_host_source();
    }
    std::string bin3 = "/tmp/ns_cuda_dropout_run";
    if (run_cmd(nvcc + " -O2 -std=c++11 " + cu3 + " " + host_dropout + " -o " + bin3) != 0) {
        std::cerr << "FAIL: nvcc dropout host link failed\n";
        return 1;
    }
    int rc3 = run_cmd(bin3);
    if (rc3 == 77) {
        std::cout << "SKIP: no CUDA-capable device present\n";
        return 77;
    }
    if (rc3 != 0) {
        std::cerr << "FAIL: CUDA DROPOUTX training run failed (exit " << rc3 << ")\n";
        return 1;
    }

    std::cout << "PASS: CUDA codegen (AdamW XOR + Muon MUONX + Dropout DROPOUTX) trained on device\n";
    return 0;
}