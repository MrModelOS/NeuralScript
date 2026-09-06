#pragma once
#include "ns/mlir/mlir_compiler.hpp"
#include <string>
#include <vector>
#include <set>

namespace ns {

enum class TargetBackend {
    CPU_CXX,     // Generic C++ (scalar reference)
    CPU_SIMD,    // C++ with AVX-512 intrinsics (placeholder)
    CUDA,        // NVIDIA CUDA kernels
    ROCM,        // AMD ROCm kernels (placeholder)
    METAL        // Apple Metal (placeholder)
};

struct CodegenOptions {
    TargetBackend backend = TargetBackend::CUDA;
    bool fuse_kernels = true;
    bool enable_fp16 = false;
    bool emit_runtime_driver = false;  // append self-contained C-ABI layer
    std::string function_name = "forward";
};

// Result of codegen: emits a self-contained source file (e.g. .cu or .cpp)
class CodeGenerator {
public:
    std::string generate(MLIRModule& module, const CodegenOptions& opts);

private:
    std::string gen_cuda(const MLIRModule& module, const CodegenOptions& opts);
    std::string gen_cpu(const MLIRModule& module, const CodegenOptions& opts);

    // Kernel emission helpers
    std::string emit_gemm_kernel(const std::string& a, const std::string& b,
                                 const std::string& c, const TensorType& type);
    std::string emit_activation_kernel(const std::string& act, const std::string& in,
                                       const std::string& out, const TensorType& type);
    std::string emit_elementwise_kernel(const std::string& a, const std::string& b,
                                        const std::string& c, const TensorType& type,
                                        const std::string& op);
    std::string emit_layernorm_kernel(const std::string& in, const std::string& out,
                                      const TensorType& type);
    std::string emit_dropout_kernel(const std::string& in, const std::string& out,
                                    const TensorType& type);
    std::string emit_softmax_kernel(const std::string& in, const std::string& out,
                                    const TensorType& type);

    std::string dtype_c_name(Dtype d) const;
    std::string shape_c_name(const TensorType& t) const;
    size_t numel(const TensorType& t) const;

    // Kernel fusion: collect a run of activations/elementwise into a single kernel
    std::vector<const MLIRInstr*> fuse_run(const MLIRFunction& fn,
                                           size_t start_idx) const;
};

} // namespace ns
