// MoE expert-lifecycle test (v1.3): dynamic birth/merge/kill of experts on a
// compiled CPU runtime driver, plus NSM2 checkpoint round-trips (mask
// persistence) and NSM1 (legacy, all-alive) fallback loading.
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
#include <iostream>
#include <string>
#include <vector>

using namespace ns;

#ifndef NS_REPO_INCLUDE_DIR
#define NS_REPO_INCLUDE_DIR ".."
#endif

// Network: 4-wide MoE with capacity 4, only 1 initial expert live.
static const char* LC_NET = R"(
type Bs = Dynamic
network LCNet {
    input:  Tensor[Bs, 4] float32
    output: Tensor[Bs, 4] float32
    layer moe = MoE(d_model: 4, num_experts: 4, ffn_dim: 8, initial_experts: 1)
    forward(x) {
        return x -> moe
    }
}
)";

int main() {
    Lexer lexer(LC_NET);
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

    const std::string driver = "/tmp/ns_lc_driver.cpp";
    const std::string host   = "/tmp/ns_lc_host.cpp";
    const std::string bin    = "/tmp/ns_lc_host";
    {
        std::ofstream of(driver);
        of << code;
    }
    std::ofstream hf(host);
    hf << R"ns(
#include "ns/runtime/ns_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fail(const char* msg) { std::printf("FAIL: %s\n", msg); return 1; }

int main() {
    const int D = 4, E = 4, H = 8, GATE = D * E;
    const size_t nw = GATE + E * D * H + E * H * D; // gate + e1 + e2 for E=4
    std::vector<float> w(nw);
    unsigned s = 987u;
    for (auto& v : w) { s = s * 1103515245u + 12345u; v = ((float)(s >> 16) / 65535.f - 0.5f); }

    const char* fwd = "/tmp/ns_lc_fwd.nsm2";
    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) return fail("init");
    if (ns_expert_count(m) != 1) return fail("count must be K0=1 after init");

    float x[D] = {0.3f, -0.7f, 0.2f, 1.1f};
    float out0[D], out_ref[D];

    if (ns_expert_birth(m, 2) != 3) return fail("birth(2) -> 3");
    if (ns_expert_merge(m, 0, 1) != 2) return fail("merge(0,1) -> 2");
    if (ns_expert_kill(m, 2) != 1) return fail("kill(2) -> 1");

    if (ns_eval_infer(m, x, out_ref, 1) != 0) return fail("eval (1 expert)");
    if (ns_save_checkpoint(m, fwd) != 0) return fail("save");

    // Mutate: grow another expert + kill, then reload must restore mask+weights.
    if (ns_expert_birth(m, 1) != 2) return fail("birth(1) -> 2");
    if (ns_expert_kill(m, 0) != 1) return fail("kill(0) -> 1");
    if (ns_load_checkpoint(m, fwd) != 0) return fail("load");
    if (ns_expert_count(m) != 1) return fail("count after load");

    if (ns_eval_infer(m, x, out0, 1) != 0) return fail("eval (after load)");
    for (int k = 0; k < D; k++)
        if (out0[k] != out_ref[k]) return fail("output changed across checkpoint round-trip");

    // The written file must be NSM2 and carry the 1-live mask.
    {
        std::FILE* fp = std::fopen(fwd, "rb");
        if (!fp) return fail("open saved ckpt");
        unsigned magic = 0; size_t cnt = 0; unsigned layers = 0; size_t cap = 0;
        if (std::fread(&magic, 4, 1, fp) != 1) return fail("read magic");
        if (std::fread(&cnt, sizeof(cnt), 1, fp) != 1) return fail("read count");
        std::fseek(fp, (long)(cnt * sizeof(float)), SEEK_CUR);
        if (std::fread(&layers, 4, 1, fp) != 1) return fail("read n_layers");
        if (std::fread(&cap, sizeof(cap), 1, fp) != 1) return fail("read cap");
        std::vector<unsigned char> mask(cap, 0);
        if (std::fread(mask.data(), 1, cap, fp) != cap) return fail("read mask");
        std::fclose(fp);
        if (magic != 0x4E534D32u || layers != 1u || cap != (size_t)E) return fail("bad NSM2 header");
        int live = 0;
        for (size_t e = 0; e < cap; e++) live += mask[e];
        if (live != 1) return fail("NSM2 mask does not persist dead/birth state");
    }

    // NSM1 legacy file: weights only -> loads with ALL experts alive.
    {
        ns_model* m1 = ns_runtime_init(w.data(), nw);
        if (!m1) return fail("init 2");
        const char* leg = "/tmp/ns_lc_legacy.nsm1";
        std::FILE* fp = std::fopen(leg, "wb");
        if (!fp) return fail("open legacy for write");
        unsigned magic = 0x4E534D31u;
        if (std::fwrite(&magic, 4, 1, fp) != 1) return fail("w magic");
        if (std::fwrite(&nw, sizeof(nw), 1, fp) != 1) return fail("w count");
        std::vector<float> wbuf(nw, 0.f);
        if (ns_model_get_weights(m, wbuf.data(), nw) != 0) return fail("get weights");
        if (std::fwrite(wbuf.data(), sizeof(float), nw, fp) != nw) return fail("w weights");
        std::fclose(fp);
        if (ns_load_checkpoint(m1, leg) != 0) return fail("load NSM1");
        if (ns_expert_count(m1) != E) return fail("NSM1 fallback must be all-alive");
        if (ns_eval_infer(m1, x, out0, 1) != 0) return fail("eval NSM1");
        ns_free(m1);
    }

    ns_free(m);
    std::printf("lifecycle: birth/merge/kill + NSM2 round-trip + NSM1 fallback OK\n");
    return 0;
}
)ns";
    hf.close();

    std::string cc = std::getenv("CXX") ? std::getenv("CXX") : "g++";
    std::string cmd = cc + " -O2 -std=c++11 -I " + NS_REPO_INCLUDE_DIR + " " +
                      driver + " " + host + " -o " + bin;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "FAIL: lifecycle host link failed\n";
        return 1;
    }
    if (std::system(bin.c_str()) != 0) {
        std::cerr << "FAIL: lifecycle host run failed\n";
        return 1;
    }

    std::cout << "PASS: MoE expert lifecycle (birth/merge/kill, NSM2 round-trip)\n";
    return 0;
}