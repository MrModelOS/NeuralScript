#pragma once
#include "ns/mlir/mlir_compiler.hpp"
#include <string>
#include <vector>
#include <set>

namespace ns {

// CUDA target layer (src/codegen/cuda_backend.cpp): device kernel sources and
// launch/runtime snippets owned by the target, spliced by CodeGenerator::gen_cuda.
namespace cuda {
    std::string device_helpers_source();   // launcher macros + scalar act helpers
    std::string forward_kernels_source();  // tiled GEMM, act, copy, fill, binop, ln, softmax
    std::string train_kernels_source();    // grad + AdamW/Muon/orthonom kernels
    std::string runtime_utils_source();    // canonical buffers + ns_cu_reserve
    std::string runtime_utils_u8_source(); // byte allocator (MoE liveness, opt-in)
} // namespace cuda

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

    // Emit the AOT training core (forward + backward + optimizer step) as a
    // C-ABI slave of ns_runtime_train_step / ns_objective_loss.
    std::string emit_train_core(const MLIRFunction& tfn, int64_t in_cols);

    // CUDA twin of emit_train_core: the same reverse-mode step, but every
    // operation becomes a device-kernel launch over persistent device buffers
    // (weights stay resident in video memory; only the batch input and the
    // loss scalar cross PCIe per step).
    std::string emit_train_core_cuda(const MLIRFunction& tfn, int64_t in_cols, int64_t out_cols);

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
