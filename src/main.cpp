#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include "ns/codegen/codegen.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "NeuralScript compiler (nsc) v1.1.0\n\n"
              << "Usage:\n"
              << "  " << prog << " <file.ns> [--mlir] [--cpp] [--cuda] [--runtime] [--check]\n\n"
              << "Options:\n"
              << "  --mlir     dump MLIR intermediate representation\n"
              << "  --cpp      generate CPU C++ reference backend\n"
              << "  --cuda     generate CUDA backend (default)\n"
              << "  --runtime  append the C-ABI runtime driver (ns_* host API)\n"
              << "  --check    run static shape checking only\n";
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    bool dump_mlir = false;
    bool use_cuda = true;
    bool check_only = false;
    bool emit_runtime = false;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--mlir") dump_mlir = true;
        else if (arg == "--cpp") use_cuda = false;
        else if (arg == "--cuda") use_cuda = true;
        else if (arg == "--check") check_only = true;
        else if (arg == "--runtime") emit_runtime = true;
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    try {
        std::string source = read_file(input_path);

        // Stage 1: Lex
        ns::Lexer lexer(source);
        auto tokens = lexer.tokenize();

        // Stage 2: Parse
        ns::Parser parser(tokens);
        ns::Program program = parser.parse_program();

        // Stage 3: Type + shape checking
        ns::ShapeChecker checker;
        if (!checker.check(program)) {
            std::cerr << "Static shape/type errors:\n";
            for (auto& e : checker.errors()) {
                std::cerr << "  " << e.line << ":" << e.column << "  " << e.message << "\n";
            }
            return 1;
        }

        if (check_only) {
            std::cout << "Shape checking passed.\n";
            return 0;
        }

        // Stage 4: MLIR lowering
        ns::MLIRCompiler mlir;
        auto module = mlir.compile(program);
        if (dump_mlir) {
            std::cout << mlir.dump(module);
            return 0;
        }

        // Stage 4b: kernel fusion pass (matmul + activation/layernorm/bias).
        ns::FusionPass fuse;
        size_t nfused = fuse.run(module);
        std::cerr << "Fusion: " << nfused << " groups fused\n";

        // Stage 5: Codegen
        ns::CodegenOptions opts;
        opts.backend = use_cuda ? ns::TargetBackend::CUDA : ns::TargetBackend::CPU_CXX;
        opts.emit_runtime_driver = emit_runtime;
        ns::CodeGenerator cg;
        std::string code = cg.generate(module, opts);
        std::cout << code;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
