#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/codegen/codegen.hpp"
#include <cassert>
#include <iostream>

using namespace ns;

static MLIRModule compile(const std::string& src) {
    Lexer lexer(src);
    auto toks = lexer.tokenize();
    Parser parser(toks);
    Program prog = parser.parse_program();
    ShapeChecker checker;
    checker.check(prog);
    MLIRCompiler mlir;
    return mlir.compile(prog);
}

int main() {
    // MLIR generation from a simple program
    {
        std::string src = R"(
var input: Tensor[B, 784] float32
var weights: Tensor[784, 10] float32
var logits = input @ weights
)";
        auto module = compile(src);
        assert(!module.functions.empty());
        bool found_matmul = false;
        for (auto& fn : module.functions) {
            for (auto& instr : fn.instructions) {
                if (instr.op == MLIROp::MATMUL) found_matmul = true;
            }
        }
        assert(found_matmul);
    }

    // Codegen CUDA
    {
        std::string src = R"(
network Net {
    input:  Tensor[B, 784] float32
    output: Tensor[B, 10] float32
    layer fc = Dense(in: 784, out: 10, activation: ReLU)
    forward(x) {
        return x -> fc
    }
}
)";
        auto module = compile(src);
        CodegenOptions opts;
        opts.backend = TargetBackend::CUDA;
        CodeGenerator cg;
        std::string code = cg.generate(module, opts);
        assert(code.find("__global__ void") != std::string::npos);
    }

    // Codegen CPU
    {
        std::string src = R"(
var input: Tensor[B, 784] float32
)";
        auto module = compile(src);
        CodegenOptions opts;
        opts.backend = TargetBackend::CPU_CXX;
        CodeGenerator cg;
        std::string code = cg.generate(module, opts);
        assert(code.find("extern \"C\"") != std::string::npos);
    }

    std::cout << "Codegen + MLIR tests passed.\n";
    return 0;
}
