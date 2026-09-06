// Kernel fusion pass tests (Phase 3).
//
// Builds MLIR modules programmatically, verifies:
//   1. A GEMM + activation chain fuses into a single FUSED op.
//   2. The fused computation is numerically identical to the unfused one.
//   3. A GEMM + layernorm chain fuses and stays numerically identical.
//   4. A GEMM whose result has multiple consumers is NOT fused.
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/mlir/eval.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using namespace ns;

TensorType t2(int64_t m, int64_t n) {
    TensorType t;
    t.dims.push_back(DimExpr::constant(m));
    t.dims.push_back(DimExpr::constant(n));
    t.dtype = Dtype::Float32;
    return t;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

MLIRModule make_mlp_module(bool with_layernorm) {
    MLIRModule mod;
    MLIRFunction fn;
    fn.name = "test";

    // x [3,4] @ w1 [4,5] -> h1 [3,5]
    auto mm = [&](const char* res, const char* a, const char* b, int64_t m, int64_t n) {
        MLIRInstr i(MLIROp::MATMUL, res);
        i.operands = {a, b};
        i.result_type = t2(m, n);
        fn.instructions.push_back(i);
    };
    mm("h1", "x", "w1", 3, 5);
    if (with_layernorm) {
        MLIRInstr ln(MLIROp::LAYERNORM, "r1");
        ln.operands = {"h1"};
        ln.result_type = t2(3, 5);
        fn.instructions.push_back(ln);
    } else {
        MLIRInstr relu(MLIROp::RELU, "r1");
        relu.operands = {"h1"};
        relu.result_type = t2(3, 5);
        fn.instructions.push_back(relu);
    }
    fn.return_id = "r1";
    mod.functions.push_back(std::move(fn));
    return mod;
}

MLIRModule make_multi_consumer_module() {
    MLIRModule mod;
    MLIRFunction fn;
    fn.name = "test";
    MLIRInstr mm(MLIROp::MATMUL, "h1");
    mm.operands = {"x", "w1"};
    mm.result_type = t2(3, 5);
    fn.instructions.push_back(mm);
    // h1 used by TWO consumers -> must NOT fuse with the GEMM.
    MLIRInstr relu1(MLIROp::RELU, "r1");
    relu1.operands = {"h1"};
    relu1.result_type = t2(3, 5);
    fn.instructions.push_back(relu1);
    MLIRInstr relu2(MLIROp::SIGMOID, "r2");
    relu2.operands = {"h1"};
    relu2.result_type = t2(3, 5);
    fn.instructions.push_back(relu2);
    fn.return_id = "r2";
    mod.functions.push_back(std::move(fn));
    return mod;
}

void bind_inputs(ModuleEvaluator& ev) {
    // x [3,4]
    ev.bind("x", {0.5, 0.1, -0.3, 0.9,
                   1.0, -1.5, 0.2, 0.4,
                   -0.7, 0.8, 0.6, -0.2}, {3, 4});
    // w1 [4,5]
    ev.bind("w1", {0.1, 0.2, 0.3, -0.4, 0.5,
                   0.6, -0.2, 0.1, 0.7, 0.3,
                   -0.9, 0.4, 0.2, 0.1, 0.8,
                    0.3, 0.5, -0.6, 0.2, 0.4}, {4, 5});
}

void test_fuse_activation() {
    MLIRModule mod = make_mlp_module(false);

    // Reference: unfused result.
    ModuleEvaluator ev;
    bind_inputs(ev);
    TensorBuffer ref;
    assert(ev.run(mod.functions[0], mod, "r1", ref));
    std::vector<double> ref_data = ref.data;

    // Run fusion.
    FusionPass fp;
    size_t groups = fp.run(mod);
    assert(groups == 1);

    // Module now should have a single FUSED instruction.
    assert(mod.functions[0].instructions.size() == 1);
    assert(mod.functions[0].instructions[0].op == MLIROp::FUSED);
    assert(mod.fused_groups.size() == 1);
    assert(!mod.fused_groups[0].ops.empty());
    assert(mod.fused_groups[0].ops[0] == "relu");

    // Evaluate the fused module.
    ModuleEvaluator evf;
    bind_inputs(evf);
    TensorBuffer fused;
    assert(evf.run(mod.functions[0], mod, "r1", fused));

    double diff = max_abs_diff(ref_data, fused.data);
    std::cout << "fuse+relu: diff=" << diff << "\n";
    assert(diff < 1e-12);
}

void test_fuse_layernorm() {
    MLIRModule mod = make_mlp_module(true);

    ModuleEvaluator ev;
    bind_inputs(ev);
    TensorBuffer ref;
    assert(ev.run(mod.functions[0], mod, "r1", ref));

    FusionPass fp;
    size_t groups = fp.run(mod);
    assert(groups == 1);
    assert(mod.fused_groups[0].ops[0] == "layernorm");

    ModuleEvaluator evf;
    bind_inputs(evf);
    TensorBuffer fused;
    assert(evf.run(mod.functions[0], mod, "r1", fused));

    double diff = max_abs_diff(ref.data, fused.data);
    std::cout << "fuse+layernorm: diff=" << diff << "\n";
    assert(diff < 1e-12);
}

void test_multi_consumer_not_fused() {
    MLIRModule mod = make_multi_consumer_module();
    FusionPass fp;
    size_t groups = fp.run(mod);
    // GEMM result has 2 consumers; must not fuse.
    assert(groups == 0);
    assert(mod.functions[0].instructions.size() == 3);
}

void test_fused_gemm_gemm_chain() {
    // Two-layer MLP: x @ W1 -> relu -> @ W2 (no fusion of second GEMM since
    // first GEMM's relu result r1 is consumed by the 2nd GEMM — the second
    // GEMM's result is fused instead).
    MLIRModule mod;
    MLIRFunction fn;
    fn.name = "mlp";
    auto mm = [&](const char* res, const char* a, const char* b, int64_t m, int64_t n) {
        MLIRInstr i(MLIROp::MATMUL, res);
        i.operands = {a, b};
        i.result_type = t2(m, n);
        fn.instructions.push_back(i);
    };
    mm("h1", "x", "w1", 3, 5);
    MLIRInstr relu(MLIROp::RELU, "a1");
    relu.operands = {"h1"};
    relu.result_type = t2(3, 5);
    fn.instructions.push_back(relu);
    mm("h2", "a1", "w2", 3, 2);
    MLIRInstr gelu(MLIROp::GELU, "out");
    gelu.operands = {"h2"};
    gelu.result_type = t2(3, 2);
    fn.instructions.push_back(gelu);
    fn.return_id = "out";
    mod.functions.push_back(std::move(fn));

    ModuleEvaluator ev;
    ev.bind("x", {0.5, 0.1, -0.3, 0.9, 1.0, -1.5, 0.2, 0.4, -0.7, 0.8, 0.6, -0.2}, {3, 4});
    ev.bind("w1", {0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.2, 0.1, 0.7, 0.3,
                   -0.9, 0.4, 0.2, 0.1, 0.8, 0.3, 0.5, -0.6, 0.2, 0.4}, {4, 5});
    ev.bind("w2", {0.7, -0.3, 0.2, 0.5, -0.4, 0.1, 0.6, 0.9, -0.2, 0.8}, {5, 2});
    TensorBuffer ref;
    assert(ev.run(mod.functions[0], mod, "out", ref));

    FusionPass fp;
    size_t groups = fp.run(mod);
    // Expect exactly 2 fusions (each GEMM+activation).
    assert(groups == 2);

    ModuleEvaluator evf;
    evf.bind("x", {0.5, 0.1, -0.3, 0.9, 1.0, -1.5, 0.2, 0.4, -0.7, 0.8, 0.6, -0.2}, {3, 4});
    evf.bind("w1", {0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.2, 0.1, 0.7, 0.3,
                    -0.9, 0.4, 0.2, 0.1, 0.8, 0.3, 0.5, -0.6, 0.2, 0.4}, {4, 5});
    evf.bind("w2", {0.7, -0.3, 0.2, 0.5, -0.4, 0.1, 0.6, 0.9, -0.2, 0.8}, {5, 2});
    TensorBuffer fused;
    assert(evf.run(mod.functions[0], mod, "out", fused));

    double diff = max_abs_diff(ref.data, fused.data);
    std::cout << "mlp(2 fused): diff=" << diff << "\n";
    assert(diff < 1e-12);
}

} // namespace

int main() {
    test_fuse_activation();
    test_fuse_layernorm();
    test_multi_consumer_not_fused();
    test_fused_gemm_gemm_chain();
    std::cout << "Fusion tests PASSED\n";
    return 0;
}