// Negative-codegen tests (v1.1): the compiler must REFUSE to emit code when a
// non-batch dim is dynamic. Runtime-sized batch (Dynamic) is the only allowed
// dynamic dimension; every other dynamic dim must be a hard compile error so a
// model that cannot be AOT-compiled is caught at build time, never at runtime.
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"

#include <iostream>
#include <string>

using namespace ns;

// fc1's input width is dynamic (not the batch): the weight becomes [Dynamic,16]
// and the matmul's K cannot be resolved statically.
static const char* DYNAMIC_IN_DENSE = R"(
type Bs = Dynamic
network N {
    input:  Tensor[Bs, 4] float32
    output: Tensor[Bs, 4] float32
    layer fc1 = Dense(in: Dynamic, out: 16, activation: ReLU)
    layer fc2 = Dense(in: 16, out: 4, activation: Identity)
    forward(x) {
        return x -> fc1 -> fc2
    }
}
)";

int main() {
    // 1. Dynamic dim in a layer weight -> unifyable here, codegen must throw.
    {
        Lexer lexer(DYNAMIC_IN_DENSE);
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
        try {
            (void)cg.generate(module, opts);
            std::cerr << "FAIL: dynamic-dims network was compiled\n";
            return 1;
        } catch (const std::exception& e) {
            std::string what = e.what();
            if (what.find("cannot infer static shape") == std::string::npos ||
                what.find("dynamic dims are not supported") == std::string::npos) {
                std::cerr << "FAIL: unexpected error: " << what << "\n";
                return 1;
            }
        }
    }

    // 2. The same network must ALSO be rejected by the CUDA backend (the
    //    host-side hard error is codegen-wide, not CPU-only).
    {
        Lexer lexer(DYNAMIC_IN_DENSE);
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
        CodeGenerator cg;
        try {
            (void)cg.generate(module, opts);
            std::cerr << "FAIL: dynamic-dims network was compiled (CUDA)\n";
            return 1;
        } catch (const std::exception& e) {
            std::string what = e.what();
            if (what.find("cannot infer static shape") == std::string::npos ||
                what.find("dynamic dims are not supported") == std::string::npos) {
                std::cerr << "FAIL: unexpected error: " << what << "\n";
                return 1;
            }
        }
    }

    std::cout << "PASS: dynamic-dims hard errors (CPU + CUDA)\n";
    return 0;
}