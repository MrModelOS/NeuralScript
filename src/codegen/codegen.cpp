#include "ns/codegen/codegen.hpp"
#include "ns/optim/optim_params.hpp"
#include <sstream>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>

namespace ns {

namespace {
// Exact decimal of a float constant, for splicing into the generated source.
std::string fmt_float(float v) {
    std::ostringstream o;
    o.precision(9);
    o << v;
    return o.str();
}

// Emit the compiled-in LR schedule as a standalone helper for the runtime
// driver. The body is a constant-fold of optim_params.hpp so the generated TU
// stays self-contained (no library linkage).
std::string lr_schedule_source() {
    std::ostringstream o;
    o << "static float ns_lr_schedule(int64_t step) {\n"
      << "    if (" << (int)optim::kLRSchedule << " == 0) return 1.0f;\n"
      << "    if (" << optim::kLrWarmupSteps << " <= 0) return 1.0f;\n"
      << "    if (step < " << optim::kLrWarmupSteps
      << ") return (float)step / (float)" << optim::kLrWarmupSteps << ";\n"
      << "    const int64_t end = " << optim::kLrTotalSteps << " > " << optim::kLrWarmupSteps
      << " ? " << optim::kLrTotalSteps << " : " << (optim::kLrWarmupSteps + 1) << ";\n"
      << "    const int64_t s = step >= end ? end : step;\n"
      << "    const float t = (float)(s - " << optim::kLrWarmupSteps
      << ") / (float)(end - " << optim::kLrWarmupSteps << ");\n"
      << "    return " << fmt_float(optim::kLrMinFactor) << " + 0.5f * (1.0f - "
      << fmt_float(optim::kLrMinFactor) << ") * (1.0f + cosf(3.14159265f * t));\n"
      << "}\n";
    return o.str();
}

// Identify the runtime data input of an inference instruction. GEMMs consume
// it as A; v1.2 layer ops consume it via their tensor operand (embedding: the
// index vector, operand[1]).
std::string input_op_of(const MLIRInstr& instr) {
    if (instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED)
        return instr.operands.size() >= 2 ? instr.operands[0] : "";
    if (instr.op == MLIROp::LAYER_EMBEDDING)
        return instr.operands.size() >= 2 ? instr.operands[1] : "";
    if (instr.op == MLIROp::LAYER_ATTENTION || instr.op == MLIROp::LAYERNORM ||
        instr.op == MLIROp::LAYER_MOE ||
        instr.op == MLIROp::SOFTMAX || instr.op == MLIROp::TRANSPOSE ||
        instr.op == MLIROp::CONCAT || instr.op == MLIROp::RESHAPE ||
        instr.op == MLIROp::SLICE || instr.op == MLIROp::INDEX ||
        instr.op == MLIROp::SCATTER ||
        instr.op == MLIROp::DROPOUT || instr.op == MLIROp::ACTIVATION ||
        instr.op == MLIROp::RELU || instr.op == MLIROp::LEAKY_RELU ||
        instr.op == MLIROp::SIGMOID || instr.op == MLIROp::TANH ||
        instr.op == MLIROp::SWISH || instr.op == MLIROp::GELU ||
        instr.op == MLIROp::SILU || instr.op == MLIROp::IDENTITY)
        return instr.operands.size() >= 1 ? instr.operands[0] : "";
    return "";
}
} // namespace

std::string CodeGenerator::dtype_c_name(Dtype d) const {
    switch (d) {
        case Dtype::Float16: return "half";
        case Dtype::Float32: return "float";
        case Dtype::Float64: return "double";
        case Dtype::Int8: return "int8_t";
        case Dtype::Int16: return "int16_t";
        case Dtype::Int32: return "int32_t";
        case Dtype::Int64: return "int64_t";
        case Dtype::FP8: return "int8_t";     // placeholder fp8
        case Dtype::FP4: return "int8_t";     // placeholder fp4
        case Dtype::Bool: return "bool";
    }
    return "float";
}

size_t CodeGenerator::numel(const TensorType& t) const {
    size_t n = 1;
    for (auto& d : t.dims) {
        if (d.is_const()) n *= (size_t)d.const_value;
        // Dynamic/symbolic treated as 1 for layout purposes
    }
    return n;
}

std::string CodeGenerator::shape_c_name(const TensorType& t) const {
    std::string s;
    for (size_t i = 0; i < t.dims.size(); i++) {
        if (i > 0) s += "][";
        if (t.dims[i].is_const()) s += std::to_string(t.dims[i].const_value);
        else s += "1"; // dynamic dim placeholder
    }
    return s;
}

std::string CodeGenerator::generate(MLIRModule& module, const CodegenOptions& opts) {
    switch (opts.backend) {
        case TargetBackend::CUDA:
            return gen_cuda(module, opts);
        case TargetBackend::CPU_CXX:
        case TargetBackend::CPU_SIMD:
            return gen_cpu(module, opts);
        case TargetBackend::ROCM:
        case TargetBackend::METAL:
        default:
            return gen_cuda(module, opts);
    }
}

std::string CodeGenerator::emit_gemm_kernel(const std::string& a, const std::string& b,
                                            const std::string& c, const TensorType& type) {
    (void)c;  // result buffer named by caller
    // type = result type [M, N]; A is [M, K], B is [K, N]
    size_t M = type.dims.size() > 0 && type.dims[0].is_const() ? type.dims[0].const_value : 1;
    size_t N = type.dims.size() > 1 && type.dims[1].is_const() ? type.dims[1].const_value : 1;
    size_t K = (M > 0) ? (numel(type) / (M > 0 ? M : 1) / (N > 0 ? N : 1)) : 1;
    if (K == 0) K = 1;

    std::string t = dtype_c_name(type.dtype);
    std::ostringstream oss;
    oss << "__global__ void gemm_" << a << "_" << b << "(\n"
        << "    const " << t << "* __restrict__ A,\n"
        << "    const " << t << "* __restrict__ B,\n"
        << "    " << t << "* __restrict__ C,\n"
        << "    int M, int N, int K) {\n"
        << "  const int tidx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        << "  const int tidy = blockIdx.y * blockDim.y + threadIdx.y;\n"
        << "  const int total_x = gridDim.x * blockDim.x;\n"
        << "  const int total_y = gridDim.y * blockDim.y;\n"
        << "  for (int i = tidy; i < M; i += total_y) {\n"
        << "    for (int j = tidx; j < N; j += total_x) {\n"
        << "      float acc = 0.0f;\n"
        << "      for (int k = 0; k < K; ++k) {\n"
        << "        acc += A[i * K + k] * B[k * N + j];\n"
        << "      }\n"
        << "      C[i * N + j] = (" << t << ")acc;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";
    return oss.str();
}

std::string CodeGenerator::emit_activation_kernel(const std::string& act, const std::string& in,
                                                  const std::string& out, const TensorType& type) {
    std::string t = dtype_c_name(type.dtype);
    size_t n = numel(type);
    std::ostringstream oss;
    oss << "__global__ void act_" << in << "_" << out << "(\n"
        << "    const " << t << "* __restrict__ in,\n"
        << "    " << t << "* __restrict__ out) {\n"
        << "  const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    if (act == "ReLU") {
        oss << "  out[idx] = in[idx] > (" << t << ")0 ? in[idx] : (" << t << ")0;\n";
    } else if (act == "Sigmoid") {
        oss << "  out[idx] = (" << t << ")(1.0f / (1.0f + expf(-(float)in[idx])));\n";
    } else if (act == "Tanh") {
        oss << "  out[idx] = (" << t << ")tanhf((float)in[idx]);\n";
    } else if (act == "GELU") {
        oss << "  float x = (float)in[idx];\n"
            << "  out[idx] = (" << t << ")(0.5f * x * (1.0f + erf(x / sqrtf(2.0f))));\n";
    } else if (act == "Swish" || act == "SiLU") {
        oss << "  float x = (float)in[idx];\n"
            << "  out[idx] = (" << t << ")(x / (1.0f + expf(-x)));\n";
    } else if (act == "LeakyReLU") {
        oss << "  out[idx] = in[idx] > 0 ? in[idx] : in[idx] * (" << t << ")0.01f;\n";
    } else if (act == "Identity") {
        oss << "  out[idx] = in[idx];\n";
    }
    oss << "}\n\n";
    (void)n;
    return oss.str();
}

std::string CodeGenerator::emit_elementwise_kernel(const std::string& a, const std::string& b,
                                                   const std::string& c, const TensorType& type,
                                                   const std::string& op) {
    std::string t = dtype_c_name(type.dtype);
    std::ostringstream oss;
    oss << "__global__ void binop_" << a << "_" << b << "_" << c << "(\n"
        << "    const " << t << "* __restrict__ A,\n"
        << "    const " << t << "* __restrict__ B,\n"
        << "    " << t << "* __restrict__ C) {\n"
        << "  const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    if (op == "+") oss << "  C[idx] = A[idx] + B[idx];\n";
    else if (op == "-") oss << "  C[idx] = A[idx] - B[idx];\n";
    else if (op == "*") oss << "  C[idx] = A[idx] * B[idx];\n";
    else if (op == "/") oss << "  C[idx] = A[idx] / B[idx];\n";
    oss << "}\n\n";
    return oss.str();
}

std::string CodeGenerator::emit_layernorm_kernel(const std::string& in, const std::string& out,
                                                 const TensorType& type) {
    std::string t = dtype_c_name(type.dtype);
    std::string shape = shape_c_name(type);
    std::ostringstream oss;
    oss << "__global__ void layernorm_" << in << "_" << out << "(\n"
        << "    const " << t << "* __restrict__ in,\n"
        << "    " << t << "* __restrict__ out) {\n"
        << "  const int row = blockIdx.x;\n"
        << "  int cols = 1; // dynamic placeholder\n"
        << "  float mean = 0.0f, var = 0.0f;\n"
        << "  for (int j = 0; j < cols; ++j) mean += in[row * cols + j];\n"
        << "  mean /= cols;\n"
        << "  for (int j = 0; j < cols; ++j) { float d = in[row * cols + j] - mean; var += d * d; }\n"
        << "  var /= cols;\n"
        << "  float inv = rsqrtf(var + 1e-5f);\n"
        << "  for (int j = 0; j < cols; ++j) out[row * cols + j] = (in[row * cols + j] - mean) * inv;\n"
        << "}\n\n";
    (void)shape;
    return oss.str();
}

std::string CodeGenerator::emit_softmax_kernel(const std::string& in, const std::string& out,
                                               const TensorType& type) {
    std::string t = dtype_c_name(type.dtype);
    std::ostringstream oss;
    oss << "__global__ void softmax_" << in << "_" << out << "(\n"
        << "    const " << t << "* __restrict__ in,\n"
        << "    " << t << "* __restrict__ out) {\n"
        << "  const int row = blockIdx.x;\n"
        << "  int cols = 1; // dynamic placeholder\n"
        << "  float m = -3.4e38f;\n"
        << "  for (int j = 0; j < cols; ++j) m = fmaxf(m, in[row * cols + j]);\n"
        << "  float s = 0.0f;\n"
        << "  for (int j = 0; j < cols; ++j) s += expf(in[row * cols + j] - m);\n"
        << "  for (int j = 0; j < cols; ++j) out[row * cols + j] = expf(in[row * cols + j] - m) / s;\n"
        << "}\n\n";
    return oss.str();
}

namespace {

// Per-id static shape metadata used by the CUDA emitters (mirrors the CPU
// reference emitter, but keeps the forward and train function views separate).
struct CuIdInfo {
    bool is_weight = false;
    bool is_input = false;
    int64_t rows = -1;
    int64_t cols = -1;
    int64_t static_numel = -1;
};

size_t cu_numel(const TensorType& t) {
    size_t n = 1;
    for (auto& d : t.dims)
        if (d.is_const()) n *= (size_t)d.const_value;
    return n;
}

void fill_cu_ids(const MLIRFunction& fn, std::map<std::string, CuIdInfo>& ids) {
    for (auto& instr : fn.instructions) {
        if (instr.op == MLIROp::TENSOR_ALLOC) {
            auto& inf = ids[instr.result_id];
            inf.is_weight = true;
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            if (d.empty() || std::all_of(d.begin(), d.end(),
                                         [](const DimExpr& x){ return x.is_const(); }))
                inf.static_numel = (int64_t)cu_numel(instr.result_type);
        } else if (!instr.result_id.empty()) {
            auto& inf = ids[instr.result_id];
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            bool all_const = d.empty() || std::all_of(d.begin(), d.end(),
                                                      [](const DimExpr& x){ return x.is_const(); });
            if (all_const) inf.static_numel = (int64_t)cu_numel(instr.result_type);
        }
    }
    for (auto& instr : fn.instructions)
        for (auto& op : instr.operands)
            if (!ids.count(op)) ids[op] = CuIdInfo();
}

// Activation-forward code matching gen_cpu's ns_act2.
std::string cu_act_code(MLIROp op) {
    switch (op) {
        case MLIROp::RELU: return "1";
        case MLIROp::LEAKY_RELU: return "2";
        case MLIROp::SIGMOID: return "3";
        case MLIROp::TANH: return "4";
        case MLIROp::SWISH: case MLIROp::SILU: return "5";
        case MLIROp::GELU: return "6";
        default: return "0";
    }
}

std::string cu_act_code_attr(const std::string& a) {
    if (a == "relu") return "1";
    if (a == "leaky_relu") return "2";
    if (a == "sigmoid") return "3";
    if (a == "tanh") return "4";
    if (a == "swish" || a == "silu") return "5";
    if (a == "gelu") return "6";
    return "0";
}

} // namespace

std::string CodeGenerator::gen_cuda(const MLIRModule& module, const CodegenOptions& opts) {
    using namespace std;
    // ---- Select the function to lower (mirrors gen_cpu). ----
    const MLIRFunction* fn = nullptr;
    const MLIRFunction* tfn = nullptr;
    for (auto& f : module.functions) if (f.name == "infer") fn = &f;
    if (!fn) for (auto& f : module.functions) if (f.name != "main" && f.name != "train_step" && !f.is_train) { fn = &f; break; }
    if (!fn) for (auto& f : module.functions) if (f.name != "main" && !f.is_train) { fn = &f; break; }
    if (!fn) for (auto& f : module.functions) { fn = &f; break; }
    for (auto& f : module.functions) if (f.is_train) { tfn = &f; break; }
    if (!fn) {
        ostringstream o;
        o << "// Generated by NeuralScript compiler\n// Target: CUDA\n"
          << "#include <cuda_runtime.h>\n"
          << "extern \"C\" void " << opts.function_name << "(const float*, float*, size_t) {}\n";
        return o.str();
    }

    // ---- Per-id metadata for the inference function. ----
    map<string, CuIdInfo> ids;
    fill_cu_ids(*fn, ids);
    // Identify the runtime data input. GEMMs consume it as A; v1.2 layer ops
    // consume it via their tensor operand (embedding: the index vector).
    string x_id;
    for (auto& instr : fn->instructions) {
        string op0 = input_op_of(instr);
        if (!op0.empty() && !ids[op0].is_weight) { x_id = op0; break; }
    }
    if (!x_id.empty()) ids[x_id].is_input = true;

    vector<string> worder;
    for (auto& instr : fn->instructions)
        if (instr.op == MLIROp::TENSOR_ALLOC) worder.push_back(instr.result_id);
    map<string, size_t> woff;
    size_t weight_total = 0;
    {
        size_t off = 0;
        for (auto& w : worder) {
            woff[w] = off;
            int64_t n = ids[w].static_numel > 0 ? ids[w].static_numel : 1;
            off += (size_t)n;
            weight_total += (size_t)n;
        }
    }

    // Static input/output geometry.
    int64_t in_cols = -1, out_cols = -1;
    for (auto& instr : fn->instructions) {
        bool gemm = instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED;
        if (gemm && instr.operands.size() >= 2) {
            if (in_cols < 0) in_cols = ids[instr.operands[1]].rows;
            out_cols = ids[instr.operands[1]].cols;
        }
        // v1.2: non-GEMM heads (embedding/attention/layernorm/concat/softmax)
        // also fix geometry. Embedding consumes one scalar index per row, so
        // the input's per-row width is 1 (M == number of indices).
        if ((instr.op == MLIROp::LAYER_EMBEDDING || instr.op == MLIROp::LAYER_ATTENTION ||
             instr.op == MLIROp::LAYER_MOE ||
             instr.op == MLIROp::LAYERNORM || instr.op == MLIROp::SOFTMAX ||
             instr.op == MLIROp::CONCAT || instr.op == MLIROp::SLICE ||
             instr.op == MLIROp::INDEX || instr.op == MLIROp::SCATTER ||
             instr.op == MLIROp::TRANSPOSE || instr.op == MLIROp::RESHAPE) && !instr.result_id.empty()) {
            if (instr.op == MLIROp::LAYER_EMBEDDING && in_cols < 0) in_cols = 1;
            int64_t c = ids[instr.result_id].cols;
            if (c > 0) out_cols = c;
        }
    }
    if (in_cols < 0) in_cols = 1;
    if (out_cols < 0) out_cols = 1;
    if (!x_id.empty() && ids[x_id].is_input) ids[x_id].cols = in_cols;

    ostringstream oss;
    oss << "// Generated by NeuralScript compiler (CUDA backend)\n"
        << "// Lowered function: @" << fn->name << "\n"
        << "#include <cuda_runtime.h>\n"
        << "#include <curand_kernel.h>\n"
        << "#include <math.h>\n"
        << "#include <stdio.h>\n"
        << "#include <stdlib.h>\n"
        << "#include <cstring>\n"
        << "#include <algorithm>\n\n";

    // ---- Device scalar helpers + uniform forward kernels (owned by the CUDA
    //      target layer in cuda_backend.cpp). ----
    oss << cuda::device_helpers_source()
        << cuda::forward_kernels_source();

    if (tfn) {
        // ---- Reverse-mode + optimizer kernels (AOT training, owned by the
        //      CUDA target layer). ----
        oss << cuda::train_kernels_source();
    }

    // ---- Runtime utilities: canonical buffers + lazy device allocator. ----
    oss << cuda::runtime_utils_source();

    // ---- Per-model device context (weights + scratch + optimizer state).
    //      Every allocation is owned by the model and released in the context
    //      destructor, so multiple models are independent and ns_free() frees
    //      ALL device memory (no more file-scope statics => no shared-state /
    //      VRAM-leak hazard). ----
    {
        string ty_id;
        const MLIRFunction* cef = tfn ? tfn : fn;
        for (auto& instr : cef->instructions)
            if (instr.op == MLIROp::CROSS_ENTROPY && instr.operands.size() >= 2)
                ty_id = instr.operands[1];
        if (!ty_id.empty()) ids[ty_id].is_input = true;
        set<string> allids;
        auto collect = [&](const MLIRFunction& f) {
            for (auto& instr : f.instructions) {
                if (instr.op == MLIROp::CROSS_ENTROPY) continue;
                for (auto& op : instr.operands) allids.insert(op);
                if (!instr.result_id.empty()) allids.insert(instr.result_id);
            }
        };
        collect(*fn);
        if (tfn) collect(*tfn);
        vector<string> sorted(allids.begin(), allids.end());
        std::sort(sorted.begin(), sorted.end());

        vector<string> frees;  // per-field cudaFree statement for the dtor
        auto add_free = [&](const string& field, const string& cap) {
            frees.push_back("    cudaFree(" + field + "); " + cap + " = 0;\n");
        };

        oss << "// ---- Per-model device context (weights + scratch + optimizer state) ----\n"
            << "typedef struct NSContext {\n"
            << "  float* d_wb = nullptr; size_t d_wb_cap = 0;   // canonical device weights\n"
            << "  float* d_xi = nullptr; size_t d_xi_cap = 0;   // input upload\n"
            << "  float* d_yl = nullptr; size_t d_yl_cap = 0;   // labels upload\n"
            << "  float* d_row = nullptr; size_t d_row_cap = 0; // CE row losses\n"
            << "  float* hrow = nullptr;  size_t hrow_cap = 0;  // CE host reduction\n";
        for (auto& id : sorted) {
            if (ids.count(id) && ids[id].is_weight) continue;
            if (id == x_id) continue;
            if (ids.count(id) && ids[id].is_input) continue;
            oss << "  float* d_" << id << " = nullptr; size_t d_" << id << "_cap = 0;\n";
            add_free("d_" + id, "d_" + id + "_cap");
        }
        // Dropout masks and optimizer moments are only touched by the training
        // core, but they are per-model state, so they live in the context too.
        if (tfn) {
            for (auto& instr : tfn->instructions) {
                if (instr.op == MLIROp::DROPOUT) {
                    oss << "  float* d_dm_" << instr.result_id << " = nullptr; size_t d_dm_"
                        << instr.result_id << "_cap = 0; // dropout mask (scaled)\n";
                    add_free("d_dm_" + instr.result_id, "d_dm_" + instr.result_id + "_cap");
                }
            }
            for (auto& instr : tfn->instructions) {
                if (instr.op != MLIROp::OPT_STEP) continue;
                for (size_t i = 0; i + 1 < instr.operands.size(); i += 2) {
                    string wt = instr.operands[i];
                    oss << "  float* d_am" << wt << " = nullptr; size_t d_am" << wt
                        << "_cap = 0; // AdamW/Muon momentum\n"
                        << "  float* d_av" << wt << " = nullptr; size_t d_av" << wt
                        << "_cap = 0; // AdamW variance\n";
                    add_free("d_am" + wt, "d_am" + wt + "_cap");
                    add_free("d_av" + wt, "d_av" + wt + "_cap");
                }
            }
        }
        oss << "  size_t adam_step = 0;    // AdamW bias-correction step counter\n"
            << "  int64_t lr_step = 0;     // LR schedule step counter (runtime wrapper)\n"
            << "  unsigned drop_seed = 0;  // dropout RNG seed counter\n"
            << "  NSContext() {}\n"
            << "  ~NSContext() {\n"
            << "    cudaFree(d_wb); d_wb_cap = 0;\n"
            << "    cudaFree(d_xi); d_xi_cap = 0;\n"
            << "    cudaFree(d_yl); d_yl_cap = 0;\n"
            << "    cudaFree(d_row); d_row_cap = 0;\n"
            << "    if (hrow) { free((void*)hrow); hrow = nullptr; hrow_cap = 0; }\n";
        for (auto& s : frees) oss << s;
        oss << "  }\n"
            << "  NSContext(const NSContext&) = delete;\n"
            << "  NSContext& operator=(const NSContext&) = delete;\n"
            << "} NSContext;\n\n";
    }

    auto dbuf = [&](const string& id) -> string {
        if (ids[id].is_weight && woff.count(id)) return "(ctx->d_wb + " + to_string(woff[id]) + ")";
        if (ids[id].is_input) return "ctx->d_xi";
        return "ctx->d_" + id;
    };

    // INDEX/SCATTER carry constant int64 index lists; materialize them as
    // device-side constants (referenced by the launched kernels).
    for (auto& instr : fn->instructions) {
        if ((instr.op == MLIROp::INDEX || instr.op == MLIROp::SCATTER) &&
            !instr.ints_attr.empty() && !instr.result_id.empty()) {
            oss << "__device__ const int64_t ns_idx_" << instr.result_id << "["
                << instr.ints_attr.size() << "] = { ";
            for (size_t k = 0; k < instr.ints_attr.size(); k++) {
                oss << instr.ints_attr[k];
                if (k + 1 < instr.ints_attr.size()) oss << ", ";
            }
            oss << " };\n";
        }
    }

    // ---- Forward launcher (runs entirely on device scratch buffers). ----
    oss << "extern \"C\" void " << opts.function_name << "(\n"
        << "    NSContext* ctx, const float* input, float* output, size_t n) {\n"
        << "  if (!ctx || !ctx->d_wb || n == 0) return;\n"
        << "  const int M = (int)(n / " << in_cols << ");\n"
        << "  if (M <= 0) return;\n"
        << "  if (ns_cu_reserve(&ctx->d_xi, &ctx->d_xi_cap, n * sizeof(float), 0)) return;\n"
        << "  cudaMemcpy(ctx->d_xi, input, n * sizeof(float), cudaMemcpyHostToDevice);\n";

    // Feature width of each per-row tensor, propagated forward over the
    // instruction list (ops are in topological order). Ops lowered from bare
    // function calls (dropout, plain activations, ...) carry no result_type in
    // CuIdInfo, so width comes from the producing instruction instead.
    std::map<string, int64_t> wcols;
    {
        map<string, int64_t> cur;
        for (auto& a : fn->instructions) {
            string ida = a.result_id;
            // Seed widths from statically typed operands first; do NOT use
            // operator[] (it would insert 0 entries that break row_dims).
            for (auto& op : a.operands) {
                auto it = ids.find(op);
                if (it != ids.end() && it->second.cols > 0) cur[op] = it->second.cols;
            }
            if ((a.op == MLIROp::MATMUL || a.op == MLIROp::FUSED) && a.operands.size() >= 2) {
                auto it = ids.find(a.operands[1]);
                if (it != ids.end() && it->second.cols > 0) cur[ida] = it->second.cols;
            } else if (!ida.empty() && !a.operands.empty()) {
                auto it = cur.find(a.operands[0]);
                if (it != cur.end() && it->second > 0) cur[ida] = it->second;
            }
        }
        wcols = cur;
    }
    auto row_dims = [&](const string& v) -> int64_t {
        if (!v.empty()) {
            auto wc = wcols.find(v);
            if (wc != wcols.end() && wc->second > 0) return wc->second;
        }
        if (!v.empty() && ids[v].cols > 0) return ids[v].cols;
        for (auto& a : fn->instructions)
            if ((a.op == MLIROp::MATMUL || a.op == MLIROp::FUSED) &&
                a.operands.size() >= 2 && a.operands[0] == v) {
                int64_t r = ids[a.operands[1]].rows;
                if (r > 0) return r;
            }
        return 1;
    };

    string result_id;
    for (auto& instr : fn->instructions) {
        string id = instr.result_id;
        switch (instr.op) {
            case MLIROp::MATMUL: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for matmul '" + id + "' (dynamic dims are not supported here)");
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << Ns << " + 15) / 16), "
                    << dbuf(A)
                    << ", " << dbuf(B) << ", ctx->d_" << id << ", M, " << Ks << ", " << Ns << ");\n";
                break;
            }
            case MLIROp::FUSED: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for fused group '" + id + "' (dynamic dims are not supported here)");
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << Ns << " + 15) / 16), "
                    << dbuf(A)
                    << ", " << dbuf(B) << ", ctx->d_" << id << ", M, " << Ks << ", " << Ns << ");\n";
                size_t group_idx = (size_t)-1;
                for (size_t gi = 0; gi < module.fused_groups.size(); gi++)
                    if (module.fused_groups[gi].result_id == instr.result_id) group_idx = gi;
                if (group_idx != (size_t)-1) {
                    const auto& g = module.fused_groups[group_idx];
                    size_t extra = 0;
                    for (auto& opname : g.ops) {
                        if (opname == "+" || opname == "-" || opname == "*" || opname == "/") {
                            int code = (opname == "-") ? 1 : (opname == "*") ? 2 : (opname == "/") ? 3 : 0;
                            string rhs = extra < g.epilogue_operands.size() ? g.epilogue_operands[extra] : "";
                            int nb = ids.count(rhs) && ids[rhs].static_numel > 0 ? (int)ids[rhs].static_numel : 1;
                            oss << "  NS_LAUNCH1(ns_binop_kernel, M * " << Ns << ", ctx->d_" << id
                                << ", " << dbuf(rhs) << ", ctx->d_" << id << ", M * " << Ns
                                << ", " << nb << ", " << code << ");\n";
                            extra++;
                        } else if (opname == "layernorm") {
                            oss << "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_" << id
                                << ", M * " << Ns << ", " << Ns << ");\n";
                        } else if (opname == "softmax") {
                            oss << "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_" << id
                                << ", M * " << Ns << ", " << Ns << ");\n";
                        } else {
                            oss << "  NS_LAUNCH1(ns_act_kernel, M * " << Ns << ", ctx->d_" << id
                                << ", ctx->d_" << id << ", M * " << Ns << ", "
                                << cu_act_code_attr(opname) << ");\n";
                        }
                    }
                }
                break;
            }
            case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
            case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
            case MLIROp::SILU: case MLIROp::IDENTITY: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_act_kernel, M * " << Ns << ", " << dbuf(in)
                    << ", ctx->d_" << id << ", M * " << Ns << ", " << cu_act_code(instr.op) << ");\n";
                break;
            }
            case MLIROp::DROPOUT: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_copy_kernel, M * " << Ns << ", " << dbuf(in)
                    << ", ctx->d_" << id << ", M * " << Ns << ");\n";
                break;
            }
            case MLIROp::ELEMENTWISE_BINOP: {
                string a = instr.operands.size() > 0 ? instr.operands[0] : "";
                string b = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ns = row_dims(a);
                int code = 0;
                if (instr.attribute == "-") code = 1;
                else if (instr.attribute == "*") code = 2;
                else if (instr.attribute == "/") code = 3;
                int nb = ids.count(b) && ids[b].static_numel > 0 ? (int)ids[b].static_numel : 1;
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_binop_kernel, M * " << Ns << ", " << dbuf(a)
                    << ", " << dbuf(b) << ", ctx->d_" << id << ", M * " << Ns << ", " << nb
                    << ", " << code << ");\n";
                break;
            }
            case MLIROp::LAYERNORM: case MLIROp::SOFTMAX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t last = row_dims(in);
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << last << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_copy_kernel, M * " << last << ", " << dbuf(in)
                    << ", ctx->d_" << id << ", M * " << last << ");\n";
                if (instr.op == MLIROp::LAYERNORM)
                    oss << "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_" << id << ", M * "
                        << last << ", " << last << ");\n";
                else
                    oss << "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_" << id << ", M * "
                        << last << ", " << last << ");\n";
                break;
            }
            case MLIROp::CONSTANT:
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_fill_kernel, 1, ctx->d_" << id << ", 1, "
                    << "(" << instr.attribute << ")f);\n";
                break;

            // ---- v1.2: data-movement ops (CUDA) ----
            case MLIROp::TRANSPOSE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t R = ids[in].rows > 0 ? ids[in].rows : 1;
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)(" << C << " * " << R << ") * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH_BLOCKS(ns_transpose2d_kernel, ((" << C << " + 15) / 16) * ((" << R << " + 15) / 16), "
                    << dbuf(in) << ", ctx->d_" << id << ", " << R << ", " << C << ");\n";
                break;
            }
            case MLIROp::CONCAT: {
                string a = instr.operands.size() > 0 ? instr.operands[0] : "";
                string b = instr.operands.size() > 1 ? instr.operands[1] : "";
                int axis = (!instr.attribute.empty() && instr.attribute == "0") ? 0 : 1;
                int64_t Ca = ids[a].cols > 0 ? ids[a].cols : 1;
                int64_t Cb = ids[b].cols > 0 ? ids[b].cols : 1;
                int64_t Co = Ca + Cb;
                result_id = id;
                if (axis == 1) {
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * " << Co << " * sizeof(float), 0)) return;\n";
                    oss << "  NS_LAUNCH_BLOCKS(ns_concat2_kernel, ((M + 15) / 16) * ((" << Co << " + 15) / 16), "
                        << dbuf(a) << ", " << dbuf(b) << ", ctx->d_" << id << ", M, " << Ca << ", " << Co << ");\n";
                } else {
                    int64_t Ba = ids[a].rows > 0 ? ids[a].rows : 1;
                    int64_t Bb = ids[b].rows > 0 ? ids[b].rows : 1;
                    int64_t D = ids[a].cols > 0 ? ids[a].cols : 1;
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)(" << (Ba + Bb) << " * " << D << ") * sizeof(float), 0)) return;\n";
                    oss << "  NS_LAUNCH_BLOCKS(ns_concat0_kernel, ((" << (Ba + Bb) << " + 15) / 16) * ((" << D << " + 15) / 16), "
                        << dbuf(a) << ", " << dbuf(b) << ", ctx->d_" << id << ", " << Ba << ", " << Bb << ", " << D << ");\n";
                }
                break;
            }
            case MLIROp::RESHAPE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << Ns << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_copy_kernel, M * " << Ns << ", " << dbuf(in)
                    << ", ctx->d_" << id << ", M * " << Ns << ");\n";
                break;
            }
            case MLIROp::SLICE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int axis = 1, s = 0, e = 0;
                {
                    std::istringstream iss(instr.attribute);
                    char c;
                    iss >> axis >> c >> s >> c >> e;
                }
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                int64_t sr = ids[in].rows > 0 ? ids[in].rows : -1;
                string rowsE = sr > 0 ? to_string(sr) : "M";
                result_id = id;
                if (axis == 0) {
                    int64_t OW = (e - s) * C;
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << OW << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_slicerows_kernel, ((" << OW << " + 255) / 256), "
                        << dbuf(in) << ", ctx->d_" << id << ", " << C << ", " << s << ", " << e << ");\n";
                } else {
                    int64_t OW = e - s;
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << rowsE << " * " << OW << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_slice2_kernel, ((" << rowsE << " + 15) / 16) * (("
                        << OW << " + 15) / 16), " << dbuf(in) << ", ctx->d_" << id << ", "
                        << rowsE << ", " << C << ", " << s << ", " << e << ");\n";
                }
                break;
            }
            case MLIROp::INDEX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int axis = (int)instr.int_attr;
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                int64_t L = (int64_t)instr.ints_attr.size();
                int64_t sr = ids[in].rows > 0 ? ids[in].rows : -1;
                string rowsE = sr > 0 ? to_string(sr) : "M";
                result_id = id;
                if (axis == 0) {
                    int64_t OW = L * C;
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << OW << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_indexrows_kernel, ((" << OW << " + 255) / 256), "
                        << dbuf(in) << ", ns_idx_" << id << ", ctx->d_" << id << ", " << L << ", " << C << ");\n";
                } else {
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << rowsE << " * " << L << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_index_kernel, ((" << rowsE << " + 15) / 16) * (("
                        << L << " + 15) / 16), " << dbuf(in) << ", ns_idx_" << id << ", ctx->d_"
                        << id << ", " << rowsE << ", " << C << ", " << L << ");\n";
                }
                break;
            }
            case MLIROp::SCATTER: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                string upd = instr.operands.size() > 1 ? instr.operands[1] : "";
                int axis = (int)instr.int_attr;
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                int64_t L = (int64_t)instr.ints_attr.size();
                int64_t sr = ids[in].rows > 0 ? ids[in].rows : -1;
                string rowsE = sr > 0 ? to_string(sr) : "M";
                result_id = id;
                if (axis == 0) {
                    int64_t OW = (sr > 0 ? sr : 0) * C;
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << OW << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_scatterrows_kernel, ((" << OW << " + 255) / 256), "
                        << dbuf(in) << ", ns_idx_" << id << ", " << dbuf(upd) << ", ctx->d_" << id
                        << ", " << (sr > 0 ? sr : 1) << ", " << C << ", " << L << ");\n";
                } else {
                    oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id
                        << "_cap, (size_t)" << rowsE << " * " << C << " * sizeof(float), 0)) return;\n"
                        << "  NS_LAUNCH_BLOCKS(ns_scatter_kernel, ((" << rowsE << " + 15) / 16) * (("
                        << C << " + 15) / 16), " << dbuf(in) << ", ns_idx_" << id << ", " << dbuf(upd)
                        << ", ctx->d_" << id << ", " << rowsE << ", " << C << ", " << L << ");\n";
                }
                break;
            }
            case MLIROp::LAYER_EMBEDDING: {
                string wt = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << D << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH_BLOCKS(ns_embedding_kernel, ((M + 15) / 16) * ((" << D << " + 15) / 16), "
                    << dbuf(wt) << ", " << dbuf(idx) << ", ctx->d_" << id << ", M, " << V << ", " << D << ");\n";
                break;
            }
            case MLIROp::LAYER_ATTENTION: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wq = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wk = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wv = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wo = instr.operands.size() > 4 ? instr.operands[4] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : (ids[x].cols > 0 ? ids[x].cols : 1);
                int H = instr.attribute.empty() ? 1 : std::stoi(instr.attribute);
                result_id = id;
                // Scratch: Q, K, V and the pre-projection context (M*D floats).
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << D << " * sizeof(float), 0)) return;\n"
                    << "  { float* qs = 0; float* ks = 0; float* vs = 0; float* cs = 0;\n"
                    << "    if (cudaMalloc(&qs, (size_t)M * " << D << " * sizeof(float)) ||\n"
                    << "        cudaMalloc(&ks, (size_t)M * " << D << " * sizeof(float)) ||\n"
                    << "        cudaMalloc(&vs, (size_t)M * " << D << " * sizeof(float)) ||\n"
                    << "        cudaMalloc(&cs, (size_t)M * " << D << " * sizeof(float))) return;\n"
                    << "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << D << " + 15) / 16), "
                    << dbuf(x) << ", " << dbuf(Wq) << ", qs, M, " << D << ", " << D << ");\n"
                    << "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << D << " + 15) / 16), "
                    << dbuf(x) << ", " << dbuf(Wk) << ", ks, M, " << D << ", " << D << ");\n"
                    << "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << D << " + 15) / 16), "
                    << dbuf(x) << ", " << dbuf(Wv) << ", vs, M, " << D << ", " << D << ");\n"
                    << "    ns_attention_core_kernel<<<" << H << ", 256, (size_t)M * M * sizeof(float)>>>"
                    << "(qs, ks, vs, cs, M, " << D << ", " << H << ", 1.0f / sqrtf((float)(" << D << " / " << H << ")));\n"
                    << "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" << D << " + 15) / 16), "
                    << "cs, " << dbuf(Wo) << ", ctx->d_" << id << ", M, " << D << ", " << D << ");\n"
                    << "    cudaFree(qs); cudaFree(ks); cudaFree(vs); cudaFree(cs); }\n";
                break;
            }
            case MLIROp::LAYER_MOE: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wg = instr.operands.size() > 1 ? instr.operands[1] : "";
                string We = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : (ids[x].cols > 0 ? ids[x].cols : 1);
                int E = instr.attribute.empty() ? 4 : std::stoi(instr.attribute);
                result_id = id;
                oss << "  if (ns_cu_reserve(&ctx->d_" << id << ", &ctx->d_" << id << "_cap, (size_t)M * "
                    << D << " * sizeof(float), 0)) return;\n"
                    << "  NS_LAUNCH1(ns_moe_kernel, M, " << dbuf(x) << ", " << dbuf(Wg)
                    << ", " << dbuf(We) << ", ctx->d_" << id << ", M, " << D << ", "
                    << E << ");\n";
                break;
            }
            default:
                break;
        }
    }
    oss << "  cudaDeviceSynchronize();\n"
        << "  if (output) {\n";
    if (result_id.empty()) {
        oss << "    (void)output;\n";
    } else {
        oss << "    cudaMemcpy(output, ctx->d_" << result_id
            << ", (size_t)M * " << out_cols << " * sizeof(float), cudaMemcpyDeviceToHost);\n";
    }
    oss << "  }\n"
        << "}\n\n";

    if (opts.emit_runtime_driver) {
        oss << "// ---- CUDA C-ABI runtime driver ----\n"
            << "typedef struct ns_model { float* h_w; NSContext* ctx; size_t n; } ns_model;\n"
            << "typedef struct ns_weight_desc { const char* name; size_t offset; size_t count; } ns_weight_desc;\n"
            << "typedef struct ns_weight_layout { size_t num_weights; const ns_weight_desc* desc; } ns_weight_layout;\n\n"
            << "namespace { \n";
        if (worder.empty()) {
            oss << "static const ns_weight_desc ns_desc[1] = {};\n";
        } else {
            oss << "static const ns_weight_desc ns_desc[" << worder.size() << "] = {\n";
            for (size_t i = 0; i < worder.size(); i++) {
                int64_t szn = ids[worder[i]].static_numel > 0 ? ids[worder[i]].static_numel : 1;
                oss << "    {\"" << worder[i] << "\", " << woff[worder[i]] << ", " << szn << "}";
                oss << (i + 1 < worder.size() ? ",\n" : "\n");
            }
            oss << "};\n";
        }
        oss << "static const ns_weight_layout ns_layout = { " << worder.size() << ", ns_desc };\n"
            << "static const size_t ns_weight_total = " << weight_total << ";\n"
            << "static const int64_t ns_in_cols = " << in_cols << ", ns_out_cols = " << out_cols << ";\n"
            << "}\n\n"
            << "extern \"C\" ns_model* ns_runtime_init(const float* weights, size_t num_floats) {\n"
            << "    if (num_floats != ns_weight_total) return nullptr;\n"
            << "    float* h_w = new float[ns_weight_total];\n"
            << "    NSContext* ctx = new NSContext();\n"
            << "    if (!h_w || !ctx) { delete[] h_w; delete ctx; return nullptr; }\n"
            << "    std::memcpy(h_w, weights, ns_weight_total * sizeof(float));\n"
            << "    if (ns_cu_reserve(&ctx->d_wb, &ctx->d_wb_cap, ns_weight_total * sizeof(float), 0)) { delete[] h_w; delete ctx; return nullptr; }\n"
            << "    cudaMemcpy(ctx->d_wb, weights, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n"
            << "    ns_model* m = new ns_model{ h_w, ctx, ns_weight_total };\n"
            << "    if (!m) { delete[] h_w; delete ctx; return nullptr; }\n"
            << "    return m;\n"
            << "}\n\n"
            << "extern \"C\" int ns_eval_infer(ns_model* m, const float* input, float* output, size_t input_numel) {\n"
            << "    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n"
            << "    " << opts.function_name << "(m->ctx, input, output, input_numel);\n"
            << "    return 0;\n"
            << "}\n\n"
            << "extern \"C\" size_t ns_model_output_numel(const ns_model* m, size_t input_numel) {\n"
            << "    (void)m; return (size_t)((int64_t)input_numel / ns_in_cols) * (size_t)ns_out_cols;\n"
            << "}\n\n"
            << "extern \"C\" size_t ns_model_weight_count(const ns_model* m) {\n"
            << "    (void)m; return ns_weight_total;\n"
            << "}\n\n"
            << "extern \"C\" size_t ns_weight_count_static(void) {\n"
            << "    return ns_weight_total;\n"
            << "}\n\n"
            << "extern \"C\" int ns_model_get_weights(const ns_model* m, float* out, size_t n) {\n"
            << "    if (!m || !m->ctx || !out || n != ns_weight_total) return -1;\n"
            << "    std::memcpy(out, m->h_w, n * sizeof(float));\n"
            << "    return 0;\n"
            << "}\n\n"
            << "extern \"C\" const ns_weight_layout* ns_model_layout(const ns_model* m) {\n"
            << "    (void)m; return &ns_layout;\n"
            << "}\n\n"
            << "extern \"C\" void ns_free(ns_model* m) {\n"
            << "    if (!m) return; delete m->ctx; delete[] m->h_w; delete m;\n"
            << "}\n\n"
            << "// Persist / restore the weight blob. Format: 4-byte magic \"NSM1\",\n"
            << "// size_t float count, then the raw weights (host-endian).\n"
            << "extern \"C\" int ns_save_checkpoint(const ns_model* m, const char* path) {\n"
            << "    if (!m || !m->h_w || !path) return -1;\n"
            << "    FILE* fp = fopen(path, \"wb\");\n"
            << "    if (!fp) return -1;\n"
            << "    const unsigned magic = 0x4E534D31u; /* \"NSM1\" */\n"
            << "    if (fwrite(&magic, sizeof(magic), 1, fp) != 1 ||\n"
            << "        fwrite(&ns_weight_total, sizeof(ns_weight_total), 1, fp) != 1 ||\n"
            << "        fwrite(m->h_w, sizeof(float), ns_weight_total, fp) != ns_weight_total) {\n"
            << "        fclose(fp); return -1;\n"
            << "    }\n"
            << "    fclose(fp); return 0;\n"
            << "}\n\n"
            << "extern \"C\" int ns_load_checkpoint(ns_model* m, const char* path) {\n"
            << "    if (!m || !m->ctx || !m->h_w || !path) return -1;\n"
            << "    FILE* fp = fopen(path, \"rb\");\n"
            << "    if (!fp) return -1;\n"
            << "    unsigned magic = 0; size_t n = 0;\n"
            << "    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||\n"
            << "        fread(&n, sizeof(n), 1, fp) != 1 ||\n"
            << "        magic != 0x4E534D31u || n != ns_weight_total ||\n"
            << "        fread(m->h_w, sizeof(float), n, fp) != n) {\n"
            << "        fclose(fp); return -1;\n"
            << "    }\n"
            << "    fclose(fp);\n"
            << "    cudaMemcpy(m->ctx->d_wb, m->h_w, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n"
            << "    return 0;\n"
            << "}\n";

        if (tfn) {
            oss << "\n// ---- CUDA training core (forward + backward + optimizer on device) ----\n"
                << emit_train_core_cuda(*tfn, in_cols, out_cols)
                << "\n" << lr_schedule_source()
                << "\nextern \"C\" int ns_runtime_train_step(ns_model* m, const float* input,\n"
                << "                                        const float* labels, size_t input_numel,\n"
                << "                                        float* loss_out, float lr) {\n"
                << "    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n"
                << "    int64_t st = m->ctx->lr_step;\n"
                << "    if (st < 9223372036854775807LL) m->ctx->lr_step = st + 1;\n"
                << "    const float lr_eff = lr * ns_lr_schedule(st);\n"
                << "    ns_train_core(m->ctx, input, input_numel, labels, m->h_w, m->h_w, loss_out, lr_eff, 1);\n"
                << "    return 0;\n"
                << "}\n\n"
                << "extern \"C\" int ns_objective_loss(ns_model* m, const float* input,\n"
                << "                                   const float* labels, size_t input_numel,\n"
                << "                                   float* loss_out) {\n"
                << "    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n"
                << "    ns_train_core(m->ctx, input, input_numel, labels, m->h_w, (float*)0, loss_out, 0.f, 0);\n"
                << "    return 0;\n"
                << "}\n";
        }
    }

    oss << "// All device buffers live in the per-model NSContext; they are released\n"
        << "// when ns_free() destroys the context.\n";
    return oss.str();
}
std::string CodeGenerator::gen_cpu(const MLIRModule& module, const CodegenOptions& opts) {
    // Emit a runnable CPU-reference implementation that lowers a SINGLE
    // function (inference entry) into real C++ kernels. Weight shapes are
    // resolved statically from TENSOR_ALLOC result types; the input's leading
    // dim (batch) is resolved at runtime from the input buffer size. All
    // data buffers are std::vector<float> resized at runtime, so arbitrary
    // batch sizes are supported.
    //
    // Host contract:
    //   forward(input, weights, output, n)
    //     input   - batch-major input, n = numel(input)
    //     weights - concatenation of ALL allocated weight buffers in
    //               discovery order (offsets/sizes documented in a comment)
    //     output  - result buffer, same leading dim as input
    using namespace std;

    // ---- Select the function to lower. ----
    const MLIRFunction* fn = nullptr;
    for (auto& f : module.functions) if (f.name == "infer") fn = &f;
    if (!fn) for (auto& f : module.functions) if (f.name != "main" && f.name != "train_step" && !f.is_train) { fn = &f; break; }
    if (!fn) for (auto& f : module.functions) if (f.name != "main" && !f.is_train) { fn = &f; break; }
    if (!fn) for (auto& f : module.functions) { fn = &f; break; }
    if (!fn) {
        ostringstream o;
        o << "// Generated by NeuralScript compiler (CPU reference)\n"
          << "#include <cstddef>\n"
          << "extern \"C\" void " << opts.function_name << "(const float*, const float*, float*, size_t) {}\n";
        return o.str();
    }

    // ---- Per-id metadata. ----
    struct IdInfo {
        bool is_weight = false;   // TENSOR_ALLOC: concrete weight buffer
        bool is_input = false;    // runtime-sized data input
        int64_t rows = -1;        // static row count, -1 = dynamic
        int64_t cols = -1;        // static col count, -1 = dynamic
        int64_t static_numel = -1;// product of constant dims, -1 if dynamic dim present
    };
    map<string, IdInfo> ids;
    for (auto& instr : fn->instructions) {
        if (instr.op == MLIROp::TENSOR_ALLOC) {
            auto& inf = ids[instr.result_id];
            inf.is_weight = true;
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            if (d.empty() || std::all_of(d.begin(), d.end(), [](const DimExpr& x){ return x.is_const(); }))
                inf.static_numel = numel(instr.result_type);
        } else if (!instr.result_id.empty()) {
            auto& inf = ids[instr.result_id];
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            bool all_const = d.empty() || std::all_of(d.begin(), d.end(), [](const DimExpr& x){ return x.is_const(); });
            if (all_const) inf.static_numel = numel(instr.result_type);
        }
    }
    for (auto& instr : fn->instructions)
        for (auto& op : instr.operands)
            if (!ids.count(op)) ids[op] = IdInfo();
    // Mark input = first instruction that consumes the raw input.
    bool marked = false;
    for (auto& instr : fn->instructions) {
        if (marked) break;
        string op0 = input_op_of(instr);
        if (!op0.empty() && !ids[op0].is_weight) {
            ids[op0].is_input = true;
            marked = true;
        }
    }

    // Propagate the input feature width into metadata so data-move ops that
    // consume the raw input (concat etc.) can derive column counts. Mirrors
    // the C-ABI geometry computed later for the driver.
    {
        int64_t in_w = -1;
        for (auto& instr : fn->instructions) {
            bool gemm = instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED;
            if (gemm && instr.operands.size() >= 2 && in_w < 0)
                in_w = ids[instr.operands[1]].rows;
            if (instr.op == MLIROp::LAYER_EMBEDDING && in_w < 0) in_w = 1;
            if (in_w >= 0) break;
        }
        if (in_w < 0) in_w = 1;
        for (auto& e : ids)
            if (e.second.is_input) e.second.cols = in_w;
    }

    // Weight buffers in ALLOC order.
    vector<string> worder;
    for (auto& instr : fn->instructions)
        if (instr.op == MLIROp::TENSOR_ALLOC)
            worder.push_back(instr.result_id);

    ostringstream oss;
    oss << "// Generated by NeuralScript compiler (CPU reference)\n"
        << "// Lowered function: @" << fn->name << "\n"
        << "#include <cstddef>\n"
        << "#include <cmath>\n"
        << "#include <cstring>\n"
        << "#include <cstdio>\n"
        << "#include <vector>\n"
        << "#include <random>\n"
        << "#include <algorithm>\n\n";

    oss << "static inline float ns_act2(float x, int code) {\n"
        << "  switch (code) {\n"
        << "    case 0: return x;                                   // identity\n"
        << "    case 1: return x > 0 ? x : 0.f;                     // relu\n"
        << "    case 2: return x > 0 ? x : 0.01f * x;               // leaky_relu\n"
        << "    case 3: return 1.f / (1.f + expf(-x));              // sigmoid\n"
        << "    case 4: return tanhf(x);                            // tanh\n"
        << "    case 5: return x / (1.f + expf(-x));                // swish/silu\n"
        << "    case 6: return 0.5f * x * (1.f + erff(x / 1.41421356f)); // gelu\n"
        << "    default: return x;\n"
        << "  }\n"
        << "}\n\n";

    oss << "static void ns_matmul(const float* A, const float* B, float* C,\n"
        << "                     int64_t M, int64_t K, int64_t N) {\n"
        << "  for (int64_t i = 0; i < M; i++)\n"
        << "    for (int64_t j = 0; j < N; j++) {\n"
        << "      float acc = 0.f;\n"
        << "      for (int64_t k = 0; k < K; k++) acc += A[(size_t)i*K + k] * B[(size_t)k*N + j];\n"
        << "      C[(size_t)i*N + j] = acc;\n"
        << "    }\n"
        << "}\n\n";

    oss << "static void ns_binop(const float* A, size_t na, const float* B, size_t nb,\n"
        << "                     float* C, int op) {\n"
        << "  for (size_t i = 0; i < na; i++) {\n"
        << "    float b = nb > 0 ? B[i % nb] : 0.f;\n"
        << "    switch (op) {\n"
        << "      case 0: C[i] = A[i] + b; break;\n"
        << "      case 1: C[i] = A[i] - b; break;\n"
        << "      case 2: C[i] = A[i] * b; break;\n"
        << "      case 3: C[i] = A[i] / b; break;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    oss << "static void ns_layernorm(float* m, size_t numel, int64_t last) {\n"
        << "  int64_t rows = (int64_t)numel / last;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    float mean = 0.f, var = 0.f;\n"
        << "    for (int64_t j = 0; j < last; j++) mean += m[(size_t)r*last + j];\n"
        << "    mean /= last;\n"
        << "    for (int64_t j = 0; j < last; j++) { float d = m[(size_t)r*last + j] - mean; var += d*d; }\n"
        << "    var /= last;\n"
        << "    float inv = 1.f / sqrtf(var + 1e-5f);\n"
        << "    for (int64_t j = 0; j < last; j++) m[(size_t)r*last + j] = (m[(size_t)r*last + j] - mean) * inv;\n"
        << "  }\n"
        << "}\n\n";

    oss << "static void ns_softmax(float* m, size_t numel, int64_t last) {\n"
        << "  int64_t rows = (int64_t)numel / last;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    float mx = m[(size_t)r*last];\n"
        << "    for (int64_t j = 1; j < last; j++) mx = std::max(mx, m[(size_t)r*last + j]);\n"
        << "    float s = 0.f;\n"
        << "    for (int64_t j = 0; j < last; j++) { m[(size_t)r*last + j] = expf(m[(size_t)r*last + j] - mx); s += m[(size_t)r*last + j]; }\n"
        << "    for (int64_t j = 0; j < last; j++) m[(size_t)r*last + j] /= s;\n"
        << "  }\n"
        << "}\n\n";

    // ---- Reverse-mode kernels (AOT training, item v1.1) ----
    oss << "static float ns_act_deriv(float x, int code) {\n"
        << "  switch (code) {\n"
        << "    case 1: return x > 0.f ? 1.f : 0.f;                    // relu\n"
        << "    case 2: return x > 0.f ? 1.f : 0.01f;                  // leaky_relu\n"
        << "    case 3: { float s = 1.f / (1.f + expf(-x)); return s * (1.f - s); } // sigmoid\n"
        << "    case 4: { float t = tanhf(x); return 1.f - t * t; }    // tanh\n"
        << "    case 5: { float s = x / (1.f + expf(-x)); return s + x * (1.f - s); } // swish/silu\n"
        << "    case 6: { float u = x / 1.41421356f; float p = 0.5f * (1.f + erff(u)); return p + x * expf(-u*u) / 2.50662827f; } // gelu\n"
        << "    default: return 1.f;\n"
        << "  }\n"
        << "}\n\n";

    oss << "// dA[i,k] += dC[i,j] * B[k,j]   (forward: C = A @ B)\n"
        << "static void ns_matmul_grad_a(const float* dC, const float* B, float* dA,\n"
        << "                             int64_t M, int64_t K, int64_t N) {\n"
        << "  for (int64_t i = 0; i < M; i++)\n"
        << "    for (int64_t k = 0; k < K; k++) {\n"
        << "      float acc = 0.f;\n"
        << "      for (int64_t j = 0; j < N; j++) acc += dC[(size_t)i*N + j] * B[(size_t)k*N + j];\n"
        << "      dA[(size_t)i*K + k] = acc;\n"
        << "    }\n"
        << "}\n\n";

    oss << "// dB[k,j] += A[i,k] * dC[i,j]   (forward: C = A @ B)\n"
        << "static void ns_matmul_grad_w(const float* A, const float* dC, float* dB,\n"
        << "                             int64_t M, int64_t K, int64_t N) {\n"
        << "  for (int64_t k = 0; k < K; k++)\n"
        << "    for (int64_t j = 0; j < N; j++) {\n"
        << "      float acc = 0.f;\n"
        << "      for (int64_t i = 0; i < M; i++) acc += A[(size_t)i*K + k] * dC[(size_t)i*N + j];\n"
        << "      dB[(size_t)k*N + j] = acc;\n"
        << "    }\n"
        << "}\n\n";

    oss << "// dOut = dIn * act'(actInput)   (elementwise)\n"
        << "static void ns_act_grad(const float* dIn, const float* actInput, float* dOut,\n"
        << "                       size_t n, int code) {\n"
        << "  for (size_t i = 0; i < n; i++) dOut[i] = dIn[i] * ns_act_deriv(actInput[i], code);\n"
        << "}\n\n";

    oss << "// LayerNorm backward: dx = (dout - mean(dout) - (x - mean_x) * mean((x-mean_x)*dout)) * inv\n"
        << "static void ns_layernorm_grad(const float* dout, const float* x, float* dx,\n"
        << "                               size_t numel, int64_t last) {\n"
        << "  int64_t rows = (int64_t)numel / last;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    float mean_x = 0.f, mean_dout = 0.f;\n"
        << "    for (int64_t j = 0; j < last; j++) {\n"
        << "      mean_x   += x[(size_t)r*last + j];\n"
        << "      mean_dout += dout[(size_t)r*last + j];\n"
        << "    }\n"
        << "    mean_x   /= last;\n"
        << "    mean_dout /= last;\n"
        << "    float var_x = 0.f, gamma = 0.f;\n"
        << "    for (int64_t j = 0; j < last; j++) {\n"
        << "      float xj = x[(size_t)r*last + j] - mean_x;\n"
        << "      var_x += xj * xj;\n"
        << "      gamma += xj * dout[(size_t)r*last + j];\n"
        << "    }\n"
        << "    var_x /= last;\n"
        << "    gamma /= last;\n"
        << "    float inv = 1.f / sqrtf(var_x + 1e-5f);\n"
        << "    for (int64_t j = 0; j < last; j++) {\n"
        << "      float xj = x[(size_t)r*last + j] - mean_x;\n"
        << "      dx[(size_t)r*last + j] = (dout[(size_t)r*last + j] - mean_dout - xj * gamma) * inv;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    oss << "// Embedding weight gradient: scatter-add dOut rows into dW[indices[i]].\n"
        << "static void ns_embedding_grad_w(const float* dout, const float* idx, float* dW,\n"
        << "                                size_t M, int64_t emb_dim, int64_t vocab_size) {\n"
        << "  for (int64_t k = 0; k < vocab_size * emb_dim; k++) dW[k] = 0.f;\n"
        << "  for (size_t i = 0; i < M; i++) {\n"
        << "    int64_t row = (int64_t)idx[i];\n"
        << "    if (row < 0 || row >= vocab_size) continue;\n"
        << "    for (int64_t j = 0; j < emb_dim; j++)\n"
        << "      dW[(size_t)row * emb_dim + j] += dout[(size_t)i * emb_dim + j];\n"
        << "  }\n"
        << "}\n\n";

    oss << "// MoE backward: recompute router/expert routing and contribute to dx.\n"
        << "static void ns_moe_grad_x(const float* dout, const float* x,\n"
        << "                         const float* Wg, const float* We, float* dx,\n"
        << "                         int64_t M, int64_t D, int64_t E) {\n"
        << "  std::vector<float> lg((size_t)E), p((size_t)E);\n"
        << "  for (int64_t i = 0; i < M; i++) {\n"
        << "    float mx = -1.0e30f;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];\n"
        << "      lg[(size_t)e] = a;\n"
        << "      if (a > mx) mx = a;\n"
        << "    }\n"
        << "    float sum = 0.f, pb = 0.f; int64_t best = 0;\n"
        << "    for (int64_t e = 0; e < E; e++) { lg[(size_t)e] = expf(lg[(size_t)e] - mx); sum += lg[(size_t)e]; }\n"
        << "    for (int64_t e = 1; e < E; e++) if (lg[(size_t)e] > lg[(size_t)best]) best = e;\n"
        << "    if (sum > 0.f) { for (int64_t e = 0; e < E; e++) p[(size_t)e] = lg[(size_t)e] / sum; pb = p[(size_t)best]; }\n"
        << "    float dpb = 0.f;\n"
        << "    for (int64_t j = 0; j < D; j++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * We[best*D*D + k*D + j];\n"
        << "      dpb += a * dout[i*D+j];\n"
        << "    }\n"
        << "    float sd = dpb * pb;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float dg = p[(size_t)e] * ((e == best ? dpb : 0.f) - sd);\n"
        << "      for (int64_t k = 0; k < D; k++) dx[i*D+k] += dg * Wg[k*E+e];\n"
        << "    }\n"
        << "    for (int64_t k = 0; k < D; k++) {\n"
        << "      float acc = 0.f;\n"
        << "      for (int64_t j = 0; j < D; j++) acc += We[best*D*D + k*D + j] * dout[i*D+j];\n"
        << "      dx[i*D+k] += pb * acc;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    oss << "// MoE router-weight gradient dWg[D,E]: dWg[k,e] += x[i,k] * dlg[i,e].\n"
        << "static void ns_moe_grad_wg(const float* dout, const float* x,\n"
        << "                           const float* Wg, const float* We, float* dWg,\n"
        << "                           int64_t M, int64_t D, int64_t E) {\n"
        << "  for (int64_t k = 0; k < D*E; k++) dWg[k] = 0.f;\n"
        << "  std::vector<float> lg((size_t)E), p((size_t)E);\n"
        << "  for (int64_t i = 0; i < M; i++) {\n"
        << "    float mx = -1.0e30f;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];\n"
        << "      lg[(size_t)e] = a;\n"
        << "      if (a > mx) mx = a;\n"
        << "    }\n"
        << "    float sum = 0.f, pb = 0.f; int64_t best = 0;\n"
        << "    for (int64_t e = 0; e < E; e++) { lg[(size_t)e] = expf(lg[(size_t)e] - mx); sum += lg[(size_t)e]; }\n"
        << "    for (int64_t e = 1; e < E; e++) if (lg[(size_t)e] > lg[(size_t)best]) best = e;\n"
        << "    if (sum > 0.f) { for (int64_t e = 0; e < E; e++) p[(size_t)e] = lg[(size_t)e] / sum; pb = p[(size_t)best]; }\n"
        << "    float dpb = 0.f;\n"
        << "    for (int64_t j = 0; j < D; j++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * We[best*D*D + k*D + j];\n"
        << "      dpb += a * dout[i*D+j];\n"
        << "    }\n"
        << "    float sd = dpb * pb;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float dg = p[(size_t)e] * ((e == best ? dpb : 0.f) - sd);\n"
        << "      for (int64_t k = 0; k < D; k++) dWg[k*E+e] += x[i*D+k] * dg;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    oss << "// MoE expert-weight gradient dWe[E,D,D]: dWe[best,:,:] += x^T @ dz.\n"
        << "static void ns_moe_grad_we(const float* dout, const float* x,\n"
        << "                           const float* Wg, const float* We, float* dWe,\n"
        << "                           int64_t M, int64_t D, int64_t E) {\n"
        << "  for (int64_t k = 0; k < (int64_t)E*D*D; k++) dWe[k] = 0.f;\n"
        << "  std::vector<float> lg((size_t)E), p((size_t)E);\n"
        << "  for (int64_t i = 0; i < M; i++) {\n"
        << "    float mx = -1.0e30f;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];\n"
        << "      lg[(size_t)e] = a;\n"
        << "      if (a > mx) mx = a;\n"
        << "    }\n"
        << "    float sum = 0.f, pb = 0.f; int64_t best = 0;\n"
        << "    for (int64_t e = 0; e < E; e++) { lg[(size_t)e] = expf(lg[(size_t)e] - mx); sum += lg[(size_t)e]; }\n"
        << "    for (int64_t e = 1; e < E; e++) if (lg[(size_t)e] > lg[(size_t)best]) best = e;\n"
        << "    if (sum > 0.f) pb = lg[(size_t)best] / sum;\n"
        << "    for (int64_t k = 0; k < D; k++)\n"
        << "      for (int64_t j = 0; j < D; j++)\n"
        << "        dWe[best*D*D + k*D + j] += x[i*D+k] * dout[i*D+j] * pb;\n"
        << "  }\n"
        << "}\n\n";

    oss << "// dL/d(preds) = (softmax(preds) - labels) / B   (cross-entropy seed)\n"
        << "static void ns_loss_grad(const float* preds, const float* labels, float* d,\n"
        << "                         size_t numel, int64_t C) {\n"
        << "  int64_t rows = (int64_t)numel / C;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    float mx = preds[(size_t)r*C];\n"
        << "    for (int64_t j = 1; j < C; j++) mx = std::max(mx, preds[(size_t)r*C + j]);\n"
        << "    float s = 0.f;\n"
        << "    for (int64_t j = 0; j < C; j++) s += expf(preds[(size_t)r*C + j] - mx);\n"
        << "    for (int64_t j = 0; j < C; j++) {\n"
        << "      float p = expf(preds[(size_t)r*C + j] - mx) / s;\n"
        << "      d[(size_t)r*C + j] = (p - labels[(size_t)r*C + j]) / (float)rows;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    oss << "// Cross-entropy loss over one-hot labels: -sum(y * log(p)).\n"
        << "static float ns_cross_entropy(const float* preds, const float* labels,\n"
        << "                              size_t numel, int64_t C) {\n"
        << "  int64_t rows = (int64_t)numel / C;\n"
        << "  float loss = 0.f;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    float mx = preds[(size_t)r*C];\n"
        << "    for (int64_t j = 1; j < C; j++) mx = std::max(mx, preds[(size_t)r*C + j]);\n"
        << "    float s = 0.f;\n"
        << "    for (int64_t j = 0; j < C; j++) s += expf(preds[(size_t)r*C + j] - mx);\n"
        << "    for (int64_t j = 0; j < C; j++) {\n"
        << "      float p = expf(preds[(size_t)r*C + j] - mx) / s;\n"
        << "      if (labels[(size_t)r*C + j] > 0.f) loss -= logf(p);\n"
        << "    }\n"
        << "  }\n"
        << "  return loss / (float)rows;\n"
        << "}\n\n";

    oss << "// Modified Gram-Schmidt: orthonormalize the columns of X (M>=N),\n"
        << "// or the rows of X (M<N). Matches the CPU-reference Muon semantics.\n"
        << "static void ns_orthonom(float* G, size_t M, size_t N) {\n"
        << "  bool transposed = M < N;\n"
        << "  size_t R = transposed ? N : M;\n"
        << "  size_t C = transposed ? M : N;\n"
        << "  auto at = [&](size_t k, size_t j) -> float& { return transposed ? G[j*R + k] : G[k*N + j]; };\n"
        << "  for (size_t j = 0; j < C; j++) {\n"
        << "    for (size_t i = 0; i < j; i++) {\n"
        << "      float dot = 0.f;\n"
        << "      for (size_t k = 0; k < R; k++) dot += at(k, i) * at(k, j);\n"
        << "      for (size_t k = 0; k < R; k++) at(k, j) -= dot * at(k, i);\n"
        << "    }\n"
        << "    float nrm = 0.f;\n"
        << "    for (size_t k = 0; k < R; k++) nrm += at(k, j) * at(k, j);\n"
        << "    nrm = sqrtf(nrm) + 1e-30f;\n"
        << "    for (size_t k = 0; k < R; k++) at(k, j) /= nrm;\n"
        << "  }\n"
        << "}\n\n";

    // ---- Data-movement and attention kernels (v1.2) ----

    // TRANSPOSE: out[j,i] = in[i,j]   (2D)
    oss << "static void ns_transpose2d(const float* in, float* out,\n"
        << "                           int64_t rows, int64_t cols) {\n"
        << "  for (int64_t i = 0; i < rows; i++)\n"
        << "    for (int64_t j = 0; j < cols; j++)\n"
        << "      out[j*rows + i] = in[i*cols + j];\n"
        << "}\n\n";

    // CONCAT along axis 1 (features): A[rows,Ca], B[rows,Cb] → out[rows,Ca+Cb]
    oss << "static void ns_concat2(const float* a, const float* b, float* out,\n"
        << "                       int64_t rows, int64_t ca, int64_t cb) {\n"
        << "  int64_t co = ca + cb;\n"
        << "  for (int64_t r = 0; r < rows; r++) {\n"
        << "    memcpy(out + r*co, a + r*ca, ca*sizeof(float));\n"
        << "    memcpy(out + r*co + ca, b + r*cb, cb*sizeof(float));\n"
        << "  }\n"
        << "}\n\n";

    // CONCAT along axis 0 (batch): A[Ba,D], B[Bb,D] → out[Ba+Bb,D]
    oss << "static void ns_concat0(const float* a, const float* b, float* out,\n"
        << "                       int64_t ba, int64_t bb, int64_t d) {\n"
        << "  memcpy(out, a, ba*d*sizeof(float));\n"
        << "  memcpy(out + ba*d, b, bb*d*sizeof(float));\n"
        << "}\n\n";

    // SLICE along axis 1 (features): in[M,C], cols [cs,ce) → out[M,ce-cs]
    oss << "static void ns_slice2(const float* in, float* out,\n"
        << "                      int64_t M, int64_t C, int64_t cs, int64_t ce) {\n"
        << "  int64_t ow = ce - cs;\n"
        << "  for (int64_t r = 0; r < M; r++)\n"
        << "    memcpy(out + r*ow, in + r*C + cs, ow*sizeof(float));\n"
        << "}\n\n";

    // SLICE along axis 0 (rows): rows [rs,re) of width C → out[re-rs,C]
    oss << "static void ns_slicerows(const float* in, float* out,\n"
        << "                         int64_t C, int64_t rs, int64_t re) {\n"
        << "  for (int64_t r = rs; r < re; r++)\n"
        << "    memcpy(out + (r-rs)*C, in + r*C, C*sizeof(float));\n"
        << "}\n\n";

    // INDEX along axis 1 (feature gather): out[r,k] = in[r*C + idx[k]]
    oss << "static void ns_index2(const float* in, const int64_t* idx, float* out,\n"
        << "                      int64_t M, int64_t C, int64_t L) {\n"
        << "  for (int64_t r = 0; r < M; r++)\n"
        << "    for (int64_t k = 0; k < L; k++)\n"
        << "      out[r*L + k] = in[r*C + idx[k]];\n"
        << "}\n\n";

    // INDEX along axis 0 (row gather): out[k,:] = in[idx[k],:]
    oss << "static void ns_indexrows(const float* in, const int64_t* idx, float* out,\n"
        << "                         int64_t L, int64_t C) {\n"
        << "  for (int64_t k = 0; k < L; k++)\n"
        << "    memcpy(out + k*C, in + idx[k]*C, C*sizeof(float));\n"
        << "}\n\n";

    // SCATTER along axis 1: out = copy(in); out[r, idx[k]] = upd[r*L + k]
    oss << "static void ns_scatter2(const float* in, const int64_t* idx, const float* upd,\n"
        << "                        float* out, int64_t M, int64_t C, int64_t L) {\n"
        << "  memcpy(out, in, M*C*sizeof(float));\n"
        << "  for (int64_t r = 0; r < M; r++)\n"
        << "    for (int64_t k = 0; k < L; k++)\n"
        << "      out[r*C + idx[k]] = upd[r*L + k];\n"
        << "}\n\n";

    // SCATTER along axis 0: out = copy(in); out[idx[k],:] = upd[k,:]
    oss << "static void ns_scatterrows(const float* in, const int64_t* idx, const float* upd,\n"
        << "                           float* out, int64_t R, int64_t C, int64_t L) {\n"
        << "  memcpy(out, in, R*C*sizeof(float));\n"
        << "  for (int64_t k = 0; k < L; k++)\n"
        << "    memcpy(out + idx[k]*C, upd + k*C, C*sizeof(float));\n"
        << "}\n\n";

    // EMBEDDING: weight[V,D], indices[n] → out[n,D]
    // Indices are float32 values cast to int64 at runtime.
    oss << "static void ns_embedding(const float* w, const float* idx, float* out,\n"
        << "                        int64_t n, int64_t V, int64_t D) {\n"
        << "  for (int64_t i = 0; i < n; i++) {\n"
        << "    int64_t k = (int64_t)idx[i];\n"
        << "    if (k < 0 || k >= V) k = 0;\n"
        << "    memcpy(out + i*D, w + k*D, D*sizeof(float));\n"
        << "  }\n"
        << "}\n\n";

    // MULTI-HEAD SELF-ATTENTION (fused forward).
    // x[BS,D]  Wq[D,D]  Wk[D,D]  Wv[D,D]  Wo[D,D]  →  out[BS,D]
    // tmp must hold at least 3*BS*Dk + S*S floats (scratch for Q/K/V heads and scores).
    // head_dim = D/num_heads.  Sequence length S = BS / B.
    oss << "static void ns_attention_fwd(const float* x,\n"
        << "    const float* Wq, const float* Wk, const float* Wv, const float* Wo,\n"
        << "    float* Q, float* K, float* V, float* sc, float* proj, float* out,\n"
        << "    int64_t BS, int64_t D, int64_t H, int64_t S) {\n"
        << "  int64_t Dk = D / H;\n"
        << "  float scale = 1.0f / sqrtf((float)Dk);\n"
        << "  // Q = x @ Wq,  K = x @ Wk,  V = x @ Wv  (all [BS, D])\n"
        << "  auto gemm = [](const float* A, const float* B, float* C,\n"
        << "                 int64_t M, int64_t K, int64_t N) {\n"
        << "    for (int64_t i = 0; i < M; i++)\n"
        << "      for (int64_t j = 0; j < N; j++) {\n"
        << "        float a = 0.f;\n"
        << "        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*B[k*N+j];\n"
        << "        C[i*N+j] = a;\n"
        << "      }\n"
        << "  };\n"
        << "  gemm(x, Wq, Q, BS, D, D);\n"
        << "  gemm(x, Wk, K, BS, D, D);\n"
        << "  gemm(x, Wv, V, BS, D, D);\n"
        << "  int64_t B = BS / S;\n"
        << "  // Per-head attention, then scatter into out.\n"
        << "  for (int64_t b = 0; b < B; b++) {\n"
        << "    for (int64_t h = 0; h < H; h++) {\n"
        << "      // sc[i,j] = Q[b*S+i, h*Dk..] · K[b*S+j, h*Dk..] * scale\n"
        << "      for (int64_t i = 0; i < S; i++) {\n"
        << "        for (int64_t j = 0; j < S; j++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t d = 0; d < Dk; d++)\n"
        << "            a += Q[((b*S+i)*D)+h*Dk+d] * K[((b*S+j)*D)+h*Dk+d];\n"
        << "          sc[i*S+j] = a * scale;\n"
        << "        }\n"
        << "      }\n"
        << "      // softmax over rows of sc\n"
        << "      for (int64_t i = 0; i < S; i++) {\n"
        << "        float mx = sc[i*S];\n"
        << "        for (int64_t j = 1; j < S; j++) mx = std::max(mx, sc[i*S+j]);\n"
        << "        float s = 0.f;\n"
        << "        for (int64_t j = 0; j < S; j++) { sc[i*S+j] = expf(sc[i*S+j]-mx); s += sc[i*S+j]; }\n"
        << "        for (int64_t j = 0; j < S; j++) sc[i*S+j] /= s;\n"
        << "      }\n"
        << "      // out_head[i,d] = sc[i,:] · V[b*S+:, h*Dk+d]\n"
        << "      for (int64_t i = 0; i < S; i++)\n"
        << "        for (int64_t d = 0; d < Dk; d++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t j = 0; j < S; j++)\n"
        << "            a += sc[i*S+j] * V[((b*S+j)*D)+h*Dk+d];\n"
        << "          out[((b*S+i)*D)+h*Dk+d] = a;\n"
        << "        }\n"
        << "    }\n"
        << "  }\n"
        << "  // output projection: proj = out @ Wo\n"
        << "  gemm(out, Wo, proj, BS, D, D);\n"
        << "  memcpy(out, proj, BS*D*sizeof(float));\n"
        << "}\n\n";

    oss << "// Multi-head attention backward: recompute Q/K/V and the per-head\n"
        << "// softmax, then push the upstream dL/dout through O -> head-concat ->\n"
        << "// softmax -> Q/K/V -> the four DxD projections.  The caller selects a\n"
        << "// SINGLE target output (dX or one of dWq/dWk/dWv/dWo) by passing a\n"
        << "// non-null pointer; the layer input gradient accumulates the gate + \n"
        << "// expert paths in the recomputed stack.\n"
        << "static void ns_attention_bwd(const float* dout, const float* x,\n"
        << "                            const float* Wq, const float* Wk, const float* Wv, const float* Wo,\n"
        << "                            float* dX, float* dWq, float* dWk, float* dWv, float* dWo,\n"
        << "                            int64_t BS, int64_t D, int64_t H, int64_t S) {\n"
        << "  if (D <= 0 || H <= 0 || S <= 0) return;\n"
        << "  int64_t Dk = D / H;\n"
        << "  float scale = 1.0f / sqrtf((float)Dk);\n"
        << "  int64_t B = BS / S;\n"
        << "  if (B <= 0) B = 1;\n"
        << "  auto gemm = [](const float* A, const float* Bw, float* C,\n"
        << "                  int64_t M, int64_t K, int64_t N) {\n"
        << "    for (int64_t i = 0; i < M; i++)\n"
        << "      for (int64_t j = 0; j < N; j++) {\n"
        << "        float a = 0.f;\n"
        << "        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*Bw[k*N+j];\n"
        << "        C[i*N+j] = a;\n"
        << "      }\n"
        << "  };\n"
        << "  auto gtx = [](const float* A, const float* Bw, float* C,\n"
        << "                int64_t M, int64_t K, int64_t N) {  // C = A^T @ Bw\n"
        << "    for (int64_t k = 0; k < K; k++)\n"
        << "      for (int64_t n = 0; n < N; n++) {\n"
        << "        float a = 0.f;\n"
        << "        for (int64_t i = 0; i < M; i++) a += A[i*K+k]*Bw[i*N+n];\n"
        << "        C[k*N+n] = a;\n"
        << "      }\n"
        << "  };\n"
        << "  auto gmt = [](const float* A, const float* Bw, float* C,\n"
        << "                int64_t M, int64_t K, int64_t N) {  // C = A @ Bw^T\n"
        << "    for (int64_t i = 0; i < M; i++)\n"
        << "      for (int64_t j = 0; j < N; j++) {\n"
        << "        float a = 0.f;\n"
        << "        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*Bw[j*K+k];\n"
        << "        C[i*N+j] = a;\n"
        << "      }\n"
        << "  };\n"
        << "  std::vector<float> Qq(BS*D), Kk(BS*D), Vv(BS*D);\n"
        << "  std::vector<float> dQ2(BS*D), dK2(BS*D), dV2(BS*D), dPre(BS*D), oh(BS*D), sc((size_t)S*S);\n"
        << "  gemm(x, Wq, Qq.data(), BS, D, D);\n"
        << "  gemm(x, Wk, Kk.data(), BS, D, D);\n"
        << "  gemm(x, Wv, Vv.data(), BS, D, D);\n"
        << "  for (int64_t i = 0; i < BS; i++)\n"
        << "    for (int64_t k = 0; k < D; k++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t n = 0; n < D; n++) a += dout[i*D+n] * Wo[k*D+n];\n"
        << "      dPre[i*D+k] = a;\n"
        << "    }\n"
        << "  std::fill(dQ2.begin(), dQ2.end(), 0.f);\n"
        << "  std::fill(dK2.begin(), dK2.end(), 0.f);\n"
        << "  std::fill(dV2.begin(), dV2.end(), 0.f);\n"
        << "  std::fill(oh.begin(), oh.end(), 0.f);\n"
        << "  for (int64_t b = 0; b < B; b++) {\n"
        << "    for (int64_t h = 0; h < H; h++) {\n"
        << "      // scores + softmax\n"
        << "      for (int64_t i = 0; i < S; i++) {\n"
        << "        for (int64_t j = 0; j < S; j++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t d = 0; d < Dk; d++)\n"
        << "            a += Qq[((b*S+i)*D)+h*Dk+d] * Kk[((b*S+j)*D)+h*Dk+d];\n"
        << "          sc[i*S+j] = a * scale;\n"
        << "        }\n"
        << "        float mx = sc[i*S];\n"
        << "        for (int64_t j = 1; j < S; j++) mx = std::max(mx, sc[i*S+j]);\n"
        << "        float s = 0.f;\n"
        << "        for (int64_t j = 0; j < S; j++) { sc[i*S+j] = expf(sc[i*S+j]-mx); s += sc[i*S+j]; }\n"
        << "        for (int64_t j = 0; j < S; j++) sc[i*S+j] /= s;\n"
        << "      }\n"
        << "      // out_head (for dWo) and dP\n"
        << "      for (int64_t i = 0; i < S; i++)\n"
        << "        for (int64_t d = 0; d < Dk; d++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t j = 0; j < S; j++)\n"
        << "            a += sc[i*S+j] * Vv[((b*S+j)*D)+h*Dk+d];\n"
        << "          oh[((b*S+i)*D)+h*Dk+d] = a;\n"
        << "        }\n"
        << "      // V grad and P grad (dPre is the head output grad)\n"
        << "      for (int64_t j = 0; j < S; j++)\n"
        << "        for (int64_t d = 0; d < Dk; d++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t i = 0; i < S; i++)\n"
        << "            a += sc[i*S+j] * dPre[((b*S+i)*D)+h*Dk+d];\n"
        << "          dV2[((b*S+j)*D)+h*Dk+d] += a;\n"
        << "        }\n"
        << "      std::vector<float> dp((size_t)S*S), ds((size_t)S*S);\n"
        << "      for (int64_t i = 0; i < S; i++) {\n"
        << "        for (int64_t j = 0; j < S; j++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t d = 0; d < Dk; d++)\n"
        << "            a += Vv[((b*S+j)*D)+h*Dk+d] * dPre[((b*S+i)*D)+h*Dk+d];\n"
        << "          dp[i*S+j] = a;\n"
        << "        }\n"
        << "        float dot = 0.f;\n"
        << "        for (int64_t j = 0; j < S; j++) dot += sc[i*S+j]*dp[i*S+j];\n"
        << "        for (int64_t j = 0; j < S; j++) ds[i*S+j] = sc[i*S+j]*(dp[i*S+j]-dot);\n"
        << "      }\n"
        << "      // Q, K grads\n"
        << "      for (int64_t i = 0; i < S; i++)\n"
        << "        for (int64_t d = 0; d < Dk; d++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t j = 0; j < S; j++) a += ds[i*S+j]*Kk[((b*S+j)*D)+h*Dk+d];\n"
        << "          dQ2[((b*S+i)*D)+h*Dk+d] += a * scale;\n"
        << "        }\n"
        << "      for (int64_t j = 0; j < S; j++)\n"
        << "        for (int64_t d = 0; d < Dk; d++) {\n"
        << "          float a = 0.f;\n"
        << "          for (int64_t i = 0; i < S; i++) a += ds[i*S+j]*Qq[((b*S+i)*D)+h*Dk+d];\n"
        << "          dK2[((b*S+j)*D)+h*Dk+d] += a * scale;\n"
        << "        }\n"
        << "    }\n"
        << "  }\n"
        << "  // projection grads and input grad\n"
        << "  if (dWq) { for (int64_t n = 0; n < D*D; n++) dWq[n] = 0.f; gtx(x, dQ2.data(), dWq, BS, D, D); }\n"
        << "  if (dWk) { for (int64_t n = 0; n < D*D; n++) dWk[n] = 0.f; gtx(x, dK2.data(), dWk, BS, D, D); }\n"
        << "  if (dWv) { for (int64_t n = 0; n < D*D; n++) dWv[n] = 0.f; gtx(x, dV2.data(), dWv, BS, D, D); }\n"
        << "  if (dWo) { for (int64_t n = 0; n < D*D; n++) dWo[n] = 0.f; gtx(oh.data(), dout, dWo, BS, D, D); }\n"
        << "  if (dX) {\n"
        << "    std::vector<float> gq(BS*D), gk(BS*D), gv(BS*D);\n"
        << "    gmt(dQ2.data(), Wq, gq.data(), BS, D, D);  // dQ2 @ Wq^T\n"
        << "    gmt(dK2.data(), Wk, gk.data(), BS, D, D);\n"
        << "    gmt(dV2.data(), Wv, gv.data(), BS, D, D);\n"
        << "    for (int64_t i = 0; i < BS*D; i++) dX[i] = gq[i] + gk[i] + gv[i];\n"
        << "  }\n"
        << "}\n\n";

    oss << "// MIXTURE-OF-EXPERTS (fused router + top-1 dispatch + weighted combine).\n"
        << "// x[M,D]  Wg[D,E]  We[E,D,D]  →  out[M,D]\n"
        << "// Router logits = x @ Wg, softmax over experts, per token the top-1\n"
        << "// expert is selected and its weight matrix is applied to x, scaled by\n"
        << "// the (softmaxed) routing probability.\n"
        << "static void ns_moe_fwd(const float* x, const float* Wg, const float* We,\n"
        << "                      float* out, int64_t M, int64_t D, int64_t E) {\n"
        << "  std::vector<float> lg((size_t)E);\n"
        << "  for (int64_t i = 0; i < M; i++) {\n"
        << "    float mx = -1.0e30f;\n"
        << "    for (int64_t e = 0; e < E; e++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];\n"
        << "      lg[(size_t)e] = a;\n"
        << "      if (a > mx) mx = a;\n"
        << "    }\n"
        << "    float sum = 0.f;\n"
        << "    for (int64_t e = 0; e < E; e++) { lg[(size_t)e] = expf(lg[(size_t)e] - mx); sum += lg[(size_t)e]; }\n"
        << "    int64_t best = 0;\n"
        << "    for (int64_t e = 1; e < E; e++) if (lg[(size_t)e] > lg[(size_t)best]) best = e;\n"
        << "    float p = (sum > 0.f) ? lg[(size_t)best] / sum : 0.f;\n"
        << "    for (int64_t j = 0; j < D; j++) {\n"
        << "      float a = 0.f;\n"
        << "      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * We[best*D*D + k*D + j];\n"
        << "      out[i*D+j] = p * a;\n"
        << "    }\n"
        << "  }\n"
        << "}\n\n";

    // ---- Backward-mode attention seed (dL/dout from upstream) ----
    // Backprop through attention is complex; for v1.2 we only support forward
    // (inference-only) attention in AOT training.  Gradient support is deferred
    // to a future version with automatic-differentiation infrastructure.
    for (auto& [id, inf] : ids) {
        oss << "static std::vector<float> buf_" << id << ";\n";
    }
    oss << "\n";

    // Helper to emit "numel of operand" expression.
    auto numel_expr = [&](const string& id) -> string {
        return ids[id].is_input ? string("n") : ("buf_" + id + ".size()");
    };

    string body;
    for (auto& instr : fn->instructions) {
        string id = instr.result_id;
        switch (instr.op) {
            case MLIROp::MATMUL: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for matmul '" + id + "' (dynamic dims are not supported here)");
                body += "  { int64_t K = " + to_string(Ks) + ", N = " + to_string(Ns) + ";\n";
                body += "    int64_t M = (int64_t)(" + numel_expr(A) + ") / K;\n";
                body += "    buf_" + id + ".resize((size_t)(M * N));\n";
                body += "    ns_matmul(" + (ids[A].is_input ? string("input") : ("buf_" + A + ".data()")) +
                        ", buf_" + B + ".data(), buf_" + id + ".data(), M, K, N); }\n";
                break;
            }
            case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
            case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
            case MLIROp::SILU: case MLIROp::IDENTITY: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int code = 0;
                switch (instr.op) {
                    case MLIROp::RELU: code = 1; break;
                    case MLIROp::LEAKY_RELU: code = 2; break;
                    case MLIROp::SIGMOID: code = 3; break;
                    case MLIROp::TANH: code = 4; break;
                    case MLIROp::SWISH: case MLIROp::SILU: code = 5; break;
                    case MLIROp::GELU: code = 6; break;
                    default: break;
                }
                body += "  { size_t zn = " + numel_expr(in) + ";\n";
                body += "    buf_" + id + ".resize(zn);\n";
                body += "    const float* src = " + (ids[in].is_input ? string("input") : ("buf_" + in + ".data()")) + ";\n";
                body += "    for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = ns_act2(src[i], " + to_string(code) + "); }\n";
                break;
            }
            case MLIROp::DROPOUT: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                body += "  { size_t zn = " + numel_expr(in) + ";\n";
                body += "    buf_" + id + ".resize(zn);\n";
                body += "    const float* src = " + (ids[in].is_input ? string("input") : ("buf_" + in + ".data()")) + ";\n";
                body += "    for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = src[i]; }\n";
                break;
            }
            case MLIROp::ELEMENTWISE_BINOP: {
                string a = instr.operands.size() > 0 ? instr.operands[0] : "";
                string b = instr.operands.size() > 1 ? instr.operands[1] : "";
                int code = 0;
                if (instr.attribute == "-") code = 1;
                else if (instr.attribute == "*") code = 2;
                else if (instr.attribute == "/") code = 3;
                body += "  { size_t na = " + numel_expr(a) + ";\n";
                body += "    buf_" + id + ".resize(na);\n";
                body += "    ns_binop(" + (ids[a].is_input ? string("input") : ("buf_" + a + ".data()")) +
                        ", na, buf_" + b + ".data(), buf_" + b + ".size(), buf_" + id + ".data(), " + to_string(code) + "); }\n";
                break;
            }
            case MLIROp::LAYERNORM: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t last = ids[id].cols > 0 ? ids[id].cols : (ids[in].cols > 0 ? ids[in].cols : 1);
                body += "  { buf_" + id + " = " + (ids[in].is_input ? string("std::vector<float>(input, input + n)") : ("buf_" + in)) + ";\n";
                body += "    ns_layernorm(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(last) + "); }\n";
                break;
            }
            case MLIROp::SOFTMAX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t last = ids[id].cols > 0 ? ids[id].cols : (ids[in].cols > 0 ? ids[in].cols : 1);
                body += "  { buf_" + id + " = " + (ids[in].is_input ? string("std::vector<float>(input, input + n)") : ("buf_" + in)) + ";\n";
                body += "    ns_softmax(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(last) + "); }\n";
                break;
            }
            case MLIROp::FUSED: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for fused group '" + id + "' (dynamic dims are not supported here)");
                body += "  { int64_t K = " + to_string(Ks) + ", N = " + to_string(Ns) + ";\n";
                body += "    int64_t M = (int64_t)(" + numel_expr(A) + ") / K;\n";
                body += "    buf_" + id + ".resize((size_t)(M * N));\n";
                body += "    ns_matmul(" + (ids[A].is_input ? string("input") : ("buf_" + A + ".data()")) +
                        ", buf_" + B + ".data(), buf_" + id + ".data(), M, K, N);\n";
                size_t group_idx = (size_t)-1;
                for (size_t gi = 0; gi < module.fused_groups.size(); gi++)
                    if (module.fused_groups[gi].result_id == instr.result_id) group_idx = gi;
                if (group_idx != (size_t)-1) {
                    const auto& g = module.fused_groups[group_idx];
                    size_t extra = 0;
                    for (auto& opname : g.ops) {
                        if (opname == "+" || opname == "-" || opname == "*" || opname == "/") {
                            string rhs = extra < g.epilogue_operands.size() ? g.epilogue_operands[extra] : "";
                            int code = 0; if (opname == "-") code = 1; else if (opname == "*") code = 2; else if (opname == "/") code = 3;
                            body += "    { size_t zn = buf_" + id + ".size();\n";
                            body += "      ns_binop(buf_" + id + ".data(), zn, buf_" + rhs + ".data(), buf_" + rhs + ".size(), buf_" + id + ".data(), " + to_string(code) + "); }\n";
                            extra++;
                        } else if (opname == "layernorm") {
                            body += "    ns_layernorm(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(Ns) + ");\n";
                        } else if (opname == "softmax") {
                            body += "    ns_softmax(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(Ns) + ");\n";
                        } else {
                            int code = 0;
                            if (opname == "relu") code = 1;
                            else if (opname == "leaky_relu") code = 2;
                            else if (opname == "sigmoid") code = 3;
                            else if (opname == "tanh") code = 4;
                            else if (opname == "swish" || opname == "silu") code = 5;
                            else if (opname == "gelu") code = 6;
                            body += "    { size_t zn = buf_" + id + ".size();\n";
                            body += "      for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = ns_act2(buf_" + id + "[i], " + to_string(code) + "); }\n";
                        }
                    }
                }
                body += "  }\n";
                break;
            }
            case MLIROp::CONSTANT:
                body += "  { buf_" + id + ".assign(1, (float)(" + instr.attribute + ")); }\n";
                break;
            case MLIROp::TENSOR_ALLOC:
                break;

            // ---- v1.2: data-movement ops ----
            case MLIROp::TRANSPOSE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                if (ids[in].rows > 0) {
                    int64_t R = ids[in].rows;
                    body += "  { buf_" + id + ".resize((size_t)(" + to_string(C) + " * " + to_string(R) + "));\n";
                    body += "    ns_transpose2d(" + (ids[in].is_input ? string("input") : ("buf_" + in + ".data()")) +
                            ", buf_" + id + ".data(), " + to_string(R) + ", " + to_string(C) + "); }\n";
                } else {
                    body += "  { int64_t R = (int64_t)(" + numel_expr(in) + ") / " + to_string(C) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)(R * " + to_string(C) + "));\n";
                    body += "    ns_transpose2d(" + (ids[in].is_input ? string("input") : ("buf_" + in + ".data()")) +
                            ", buf_" + id + ".data(), R, " + to_string(C) + "); }\n";
                }
                break;
            }
            case MLIROp::CONCAT: {
                string a = instr.operands.size() > 0 ? instr.operands[0] : "";
                string b = instr.operands.size() > 1 ? instr.operands[1] : "";
                int axis = (!instr.attribute.empty() && instr.attribute == "0") ? 0 : 1;
                auto src_a = [&](const string& s) { return ids[s].is_input ? string("input") : ("buf_" + s + ".data()"); };
                if (axis == 1) {
                    int64_t Ca = ids[a].cols > 0 ? ids[a].cols : 1;
                    int64_t Cb = ids[b].cols > 0 ? ids[b].cols : 1;
                    body += "  { int64_t R = (int64_t)(" + numel_expr(a) + ") / " + to_string(Ca) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)(R * (" + to_string(Ca) + " + " + to_string(Cb) + ")));\n";
                    body += "    ns_concat2(" + src_a(a) + ", " + src_a(b) + ", buf_" + id +
                            ".data(), R, " + to_string(Ca) + ", " + to_string(Cb) + "); }\n";
                } else {
                    int64_t D = ids[a].cols > 0 ? ids[a].cols : 1;
                    body += "  { int64_t Ba = (int64_t)(" + numel_expr(a) + ") / " + to_string(D) + ";\n";
                    body += "    int64_t Bb = (int64_t)(" + numel_expr(b) + ") / " + to_string(D) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)((Ba + Bb) * " + to_string(D) + "));\n";
                    body += "    ns_concat0(" + src_a(a) + ", " + src_a(b) + ", buf_" + id +
                            ".data(), Ba, Bb, " + to_string(D) + "); }\n";
                }
                break;
            }
            case MLIROp::RESHAPE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                body += "  buf_" + id + " = " + (ids[in].is_input ? string("std::vector<float>(input, input+n)") : ("buf_" + in)) + ";\n";
                break;
            }
            case MLIROp::SLICE: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int axis = 1, s = 0, e = 0;
                {
                    std::istringstream iss(instr.attribute);
                    char c;
                    iss >> axis >> c >> s >> c >> e;
                }
                string src = ids[in].is_input ? string("input") : ("buf_" + in + ".data()");
                if (axis == 0) {
                    int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                    body += "  { buf_" + id + ".resize((size_t)(" + to_string((e - s) * C) + "));\n";
                    body += "    ns_slicerows(" + src + ", buf_" + id + ".data(), " +
                            to_string(C) + ", " + to_string(s) + ", " + to_string(e) + "); }\n";
                } else {
                    int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                    body += "  { int64_t M = (int64_t)(" + numel_expr(in) + ") / " + to_string(C) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)(M * " + to_string(e - s) + "));\n";
                    body += "    ns_slice2(" + src + ", buf_" + id + ".data(), M, " + to_string(C) +
                            ", " + to_string(s) + ", " + to_string(e) + "); }\n";
                }
                break;
            }
            case MLIROp::INDEX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int axis = (int)instr.int_attr;
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                int64_t L = (int64_t)instr.ints_attr.size();
                string src = ids[in].is_input ? string("input") : ("buf_" + in + ".data()");
                body += "  static const int64_t idx_" + id + "[" + to_string(L) + "] = {";
                for (size_t k = 0; k < instr.ints_attr.size(); k++) {
                    body += to_string(instr.ints_attr[k]);
                    if (k + 1 < instr.ints_attr.size()) body += ",";
                }
                body += "};\n";
                if (axis == 0) {
                    body += "  buf_" + id + ".resize((size_t)(" + to_string(L * C) + "));\n";
                    body += "  ns_indexrows(" + src + ", idx_" + id + ", buf_" + id + ".data(), " +
                            to_string(L) + ", " + to_string(C) + ");\n";
                } else {
                    body += "  { int64_t M = (int64_t)(" + numel_expr(in) + ") / " + to_string(C) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)(M * " + to_string(L) + "));\n";
                    body += "    ns_index2(" + src + ", idx_" + id + ", buf_" + id + ".data(), M, " +
                            to_string(C) + ", " + to_string(L) + "); }\n";
                }
                break;
            }
            case MLIROp::SCATTER: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                string upd = instr.operands.size() > 1 ? instr.operands[1] : "";
                int axis = (int)instr.int_attr;
                int64_t C = ids[in].cols > 0 ? ids[in].cols : 1;
                int64_t L = (int64_t)instr.ints_attr.size();
                string src = ids[in].is_input ? string("input") : ("buf_" + in + ".data()");
                string upds = ids[upd].is_input ? string("input") : ("buf_" + upd + ".data()");
                body += "  static const int64_t idx_" + id + "[" + to_string(L) + "] = {";
                for (size_t k = 0; k < instr.ints_attr.size(); k++) {
                    body += to_string(instr.ints_attr[k]);
                    if (k + 1 < instr.ints_attr.size()) body += ",";
                }
                body += "};\n";
                if (axis == 0) {
                    int64_t R = ids[in].rows > 0 ? ids[in].rows : 0;
                    body += "  buf_" + id + ".resize((size_t)(" + to_string(R * C) + "));\n";
                    body += "  ns_scatterrows(" + src + ", idx_" + id + ", " + upds +
                            ", buf_" + id + ".data(), " + to_string(R) + ", " + to_string(C) +
                            ", " + to_string(L) + ");\n";
                } else {
                    body += "  { int64_t M = (int64_t)(" + numel_expr(in) + ") / " + to_string(C) + ";\n";
                    body += "    buf_" + id + ".resize((size_t)(M * " + to_string(C) + "));\n";
                    body += "    ns_scatter2(" + src + ", idx_" + id + ", " + upds +
                            ", buf_" + id + ".data(), M, " + to_string(C) + ", " + to_string(L) + "); }\n";
                }
                break;
            }
            case MLIROp::LAYER_EMBEDDING: {
                string wt = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                body += "  { int64_t n_idx = " + numel_expr(idx) + ";\n";
                body += "    buf_" + id + ".resize((size_t)(n_idx * " + to_string(D) + "));\n";
                body += "    ns_embedding(" + (ids[wt].is_input ? string("input") : ("buf_" + wt + ".data()")) +
                        ", " + (ids[idx].is_input ? string("input") : ("buf_" + idx + ".data()")) +
                        ", buf_" + id + ".data(), n_idx, " + to_string(V) + ", " + to_string(D) + "); }\n";
                break;
            }
            case MLIROp::LAYER_ATTENTION: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wq = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wk = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wv = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wo = instr.operands.size() > 4 ? instr.operands[4] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : (ids[x].cols > 0 ? ids[x].cols : 1);
                int H = instr.attribute.empty() ? 1 : std::stoi(instr.attribute);
                body += "  { int64_t BS = " + numel_expr(x) + " / " + to_string(D) + ";\n";
                body += "    int64_t S = BS;\n"; // B=1 assumption for v1.2
                body += "    std::vector<float> q_b(BS*" + to_string(D) + "), k_b(BS*" + to_string(D) + "), v_b(BS*" + to_string(D) + "), proj_b(BS*" + to_string(D) + "), sc_b(S*S);\n";
                body += "    buf_" + id + ".resize((size_t)(BS*" + to_string(D) + "));\n";
                body += "    ns_attention_fwd(" +
                        (ids[x].is_input ? string("input") : ("buf_" + x + ".data()")) + ", " +
                        (ids[Wq].is_input ? string("input") : ("buf_" + Wq + ".data()")) + ", " +
                        (ids[Wk].is_input ? string("input") : ("buf_" + Wk + ".data()")) + ", " +
                        (ids[Wv].is_input ? string("input") : ("buf_" + Wv + ".data()")) + ", " +
                        (ids[Wo].is_input ? string("input") : ("buf_" + Wo + ".data()")) + ",\n";
                body += "        q_b.data(), k_b.data(), v_b.data(), sc_b.data(), proj_b.data(), buf_" + id + ".data(),\n";
                body += "        BS, " + to_string(D) + ", " + to_string(H) + ", S); }\n";
                break;
            }
            case MLIROp::LAYER_MOE: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wg = instr.operands.size() > 1 ? instr.operands[1] : "";
                string We = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : (ids[x].cols > 0 ? ids[x].cols : 1);
                int E = instr.attribute.empty() ? 4 : std::stoi(instr.attribute);
                body += "  { int64_t M = (int64_t)(" + numel_expr(x) + ") / " + to_string(D) + ";\n";
                body += "    buf_" + id + ".resize((size_t)(M * " + to_string(D) + "));\n";
                body += "    ns_moe_fwd(" +
                        (ids[x].is_input ? string("input") : ("buf_" + x + ".data()")) + ", " +
                        (ids[Wg].is_input ? string("input") : ("buf_" + Wg + ".data()")) + ", " +
                        (ids[We].is_input ? string("input") : ("buf_" + We + ".data()")) + ", buf_" +
                        id + ".data(), M, " + to_string(D) + ", " + to_string(E) + "); }\n";
                break;
            }

            default:
                break;
        }
    }

    // Host entry.
    oss << "extern \"C\" void " << opts.function_name << "(\n"
        << "    const float* input, const float* weights, float* output, size_t n) {\n";
    size_t off = 0;
    for (auto& w : worder) {
        int64_t szn = ids[w].static_numel > 0 ? ids[w].static_numel : 1;
        oss << "  // weight " << w << " @ offset " << off << " floats, count " << szn << "\n";
        oss << "  buf_" << w << ".assign(weights + " << off << ", weights + " << off + szn << ");\n";
        off += (size_t)szn;
    }

    oss << body;

    string result_id = fn->return_id;
    if (result_id.empty())
        for (auto& instr : fn->instructions)
            if (instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED || instr.op == MLIROp::SOFTMAX || !instr.result_id.empty())
                result_id = instr.result_id;
    if (result_id.empty()) {
        oss << "  (void)input; (void)output; (void)n;\n";
    } else {
        oss << "  { size_t rn = buf_" << result_id << ".size();\n";
        oss << "    memcpy(output, buf_" << result_id << ".data(), rn * sizeof(float)); }\n";
    }
    oss << "}\n";

    if (opts.emit_runtime_driver) {
        // ---- Self-contained C-ABI runtime driver. ----
        // Layout-compatible with the declarations in ns/runtime/ns_runtime.h.
        size_t total = 0;
        for (auto& w : worder) total += (size_t)(ids[w].static_numel > 0 ? ids[w].static_numel : 1);

        // Static input/output geometry: first GEMM's K (or embedding's 1) and
        // final result columns.
        int64_t in_cols = -1, out_cols = -1;
        for (auto& instr : fn->instructions) {
            if (instr.op != MLIROp::TENSOR_ALLOC && !instr.result_id.empty() && instr.result_id == fn->return_id)
                out_cols = ids[instr.result_id].cols;
        }
        for (auto& instr : fn->instructions) {
            bool gemm = instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED;
            if (gemm && instr.operands.size() >= 2 && in_cols < 0) in_cols = ids[instr.operands[1]].rows;
            if (gemm && instr.operands.size() >= 2) out_cols = ids[instr.operands[1]].cols;
            if (instr.op == MLIROp::LAYER_EMBEDDING && in_cols < 0) in_cols = 1;
            if ((instr.op == MLIROp::LAYER_ATTENTION || instr.op == MLIROp::LAYERNORM ||
                 instr.op == MLIROp::LAYER_MOE ||
                 instr.op == MLIROp::SOFTMAX || instr.op == MLIROp::CONCAT ||
                 instr.op == MLIROp::SLICE || instr.op == MLIROp::INDEX ||
                 instr.op == MLIROp::SCATTER || instr.op == MLIROp::TRANSPOSE ||
                 instr.op == MLIROp::RESHAPE) &&
                !instr.result_id.empty()) {
                int64_t c = ids[instr.result_id].cols;
                if (c > 0) out_cols = c;
            }
        }
        if (in_cols < 0) in_cols = 1;
        if (out_cols < 0) out_cols = 1;

        oss << "\n// ---- C-ABI (host entry points) ----\n";
        oss << "#include <cstring>\n";

        // ---- Per-model CPU context (intermediate buffers, dropout masks,
        //      optimizer moments and the dropout RNG). Everything that used to
        //      live in function-scope statics of ns_train_core now belongs to
        //      the model instance, so two ns_model objects (or re-entrant
        //      calls) in one process can never interfere. Mirrors the CUDA
        //      NSContext design. ----
        const MLIRFunction* tfn_ = nullptr;
        for (auto& f : module.functions) { if (f.is_train) { tfn_ = &f; break; } }

        {
            struct IX { bool is_weight = false; };
            map<string, IX> sc;
            set<string> opt_wts;
            if (tfn_) {
                for (auto& ins : tfn_->instructions) {
                    if (ins.result_id.empty()) continue;
                    auto& e = sc[ins.result_id];
                    e.is_weight = ins.op == MLIROp::TENSOR_ALLOC;
                }
                for (auto& ins : tfn_->instructions)
                    for (auto& op : ins.operands) sc.emplace(std::make_pair(op, IX()));
                // Mirror emit_train_core's runtime I/O detection: the raw data inputs
                // (first GEMM A, or an embedding index stream) and the CE label
                // operand are passed in as raw pointers, never stored in context.
                string xid_, yid_;
                for (auto& ins : tfn_->instructions) {
                    if (ins.op == MLIROp::LAYER_EMBEDDING && ins.operands.size() >= 2 &&
                        !sc[ins.operands[1]].is_weight && xid_.empty())
                        xid_ = ins.operands[1];
                    if (ins.op == MLIROp::MATMUL && ins.operands.size() >= 2 &&
                        !sc[ins.operands[0]].is_weight && xid_.empty())
                        xid_ = ins.operands[0];
                    if (ins.op == MLIROp::CROSS_ENTROPY && ins.operands.size() >= 2)
                        yid_ = ins.operands[1];
                }
                oss << "typedef struct ns_cpu_ctx {\n";
                for (auto& [id, ix] : sc) {
                    if (id == xid_ || id == yid_) continue;
                    oss << "  std::vector<float> buf_" << id << ";\n";
                }
                for (auto& ins : tfn_->instructions)
                    if (ins.op == MLIROp::DROPOUT)
                        oss << "  std::vector<float> buf_dm_" << ins.result_id << ";\n";
                for (auto& ins : tfn_->instructions) {
                    if (ins.op != MLIROp::OPT_STEP) continue;
                    for (size_t i = 0; i + 1 < ins.operands.size(); i += 2) {
                        string wt = ins.operands[i];
                        if (!opt_wts.count(wt)) {
                            oss << "  std::vector<float> st_m_" << wt << ", st_am_" << wt << ", st_av_" << wt << ";\n"
                                << "  size_t st_t_" << wt << " = 0;\n";
                            opt_wts.insert(wt);
                        }
                    }
                }
                oss << "  std::mt19937 rng;\n"
                    << "  int64_t lr_step = 0;   // LR schedule step counter (runtime wrapper)\n"
                    << "  ns_cpu_ctx() : rng(0x9E3779B9u) {}\n"
                    << "} ns_cpu_ctx;\n";
            } else {
                oss << "typedef struct ns_cpu_ctx { std::mt19937 rng; ns_cpu_ctx() : rng(0x9E3779B9u) {} } ns_cpu_ctx;\n";
            }
        }
        oss << "typedef struct ns_model { float* w; size_t n; ns_cpu_ctx* ctx; } ns_model;\n"
            << "typedef struct ns_weight_desc { const char* name; size_t offset; size_t count; } ns_weight_desc;\n"
            << "typedef struct ns_weight_layout { size_t num_weights; const ns_weight_desc* desc; } ns_weight_layout;\n\n";

        oss << "namespace { \n";
        if (worder.empty()) {
            oss << "static const ns_weight_desc ns_desc[1] = {};\n";
        } else {
            oss << "static const ns_weight_desc ns_desc[" << worder.size() << "] = {\n";
            size_t off = 0;
            for (size_t i = 0; i < worder.size(); i++) {
                int64_t szn = ids[worder[i]].static_numel > 0 ? ids[worder[i]].static_numel : 1;
                oss << "    {\"" << worder[i] << "\", " << off << ", " << szn << "}";
                oss << (i + 1 < worder.size() ? ",\n" : "\n");
                off += (size_t)szn;
            }
            oss << "};\n";
        }
        oss << "static const ns_weight_layout ns_layout = { " << worder.size()
            << ", ns_desc };\n";
        oss << "static const size_t ns_weight_total = " << total << ";\n";
        oss << "static const int64_t ns_in_cols = " << in_cols << ", ns_out_cols = " << out_cols << ";\n";
        oss << "}\n\n";

        oss << "extern \"C\" ns_model* ns_runtime_init(const float* weights, size_t num_floats) {\n";
        oss << "    if (num_floats != ns_weight_total) return nullptr;\n";
        oss << "    ns_cpu_ctx* ctx = new ns_cpu_ctx();\n";
        oss << "    ns_model* m = new ns_model{ new float[ns_weight_total], ns_weight_total, ctx };\n";
        oss << "    if (!m->w) { delete ctx; delete m; return nullptr; }\n";
        oss << "    std::memcpy(m->w, weights, ns_weight_total * sizeof(float));\n";
        oss << "    return m;\n";
        oss << "}\n\n";
        oss << "extern \"C\" int ns_eval_infer(ns_model* m, const float* input, float* output, size_t input_numel) {\n";
        oss << "    if (!m || !m->w) return -1;\n";
        oss << "    " << opts.function_name << "(input, m->w, output, input_numel);\n";
        oss << "    return 0;\n";
        oss << "}\n\n";
        oss << "extern \"C\" size_t ns_model_output_numel(const ns_model* m, size_t input_numel) {\n";
        oss << "    (void)m; return (size_t)((int64_t)input_numel / ns_in_cols) * (size_t)ns_out_cols;\n";
        oss << "}\n\n";
        oss << "extern \"C\" size_t ns_model_weight_count(const ns_model* m) {\n";
        oss << "    (void)m; return ns_weight_total;\n";
        oss << "}\n\n";
        oss << "extern \"C\" size_t ns_weight_count_static(void) {\n";
        oss << "    return ns_weight_total;\n";
        oss << "}\n\n";
        oss << "extern \"C\" int ns_model_get_weights(const ns_model* m, float* out, size_t n) {\n";
        oss << "    if (!m || !m->w || !out || n != ns_weight_total) return -1;\n";
        oss << "    std::memcpy(out, m->w, n * sizeof(float));\n";
        oss << "    return 0;\n";
        oss << "}\n\n";
        oss << "extern \"C\" const ns_weight_layout* ns_model_layout(const ns_model* m) {\n";
        oss << "    (void)m; return &ns_layout;\n";
        oss << "}\n\n";
        oss << "extern \"C\" void ns_free(ns_model* m) {\n";
        oss << "    if (!m) return; delete m->ctx; delete[] m->w; delete m;\n";
        oss << "}\n\n";
        oss << "// Persist / restore the weight blob. Format: 4-byte magic \"NSM1\",\n";
        oss << "// size_t float count, then the raw weights (host-endian).\n";
        oss << "extern \"C\" int ns_save_checkpoint(const ns_model* m, const char* path) {\n";
        oss << "    if (!m || !m->w || !path) return -1;\n";
        oss << "    FILE* fp = fopen(path, \"wb\");\n";
        oss << "    if (!fp) return -1;\n";
        oss << "    const unsigned magic = 0x4E534D31u; /* \"NSM1\" */\n";
        oss << "    if (fwrite(&magic, sizeof(magic), 1, fp) != 1 ||\n";
        oss << "        fwrite(&ns_weight_total, sizeof(ns_weight_total), 1, fp) != 1 ||\n";
        oss << "        fwrite(m->w, sizeof(float), ns_weight_total, fp) != ns_weight_total) {\n";
        oss << "        fclose(fp); return -1;\n";
        oss << "    }\n";
        oss << "    fclose(fp); return 0;\n";
        oss << "}\n\n";
        oss << "extern \"C\" int ns_load_checkpoint(ns_model* m, const char* path) {\n";
        oss << "    if (!m || !m->w || !path) return -1;\n";
        oss << "    FILE* fp = fopen(path, \"rb\");\n";
        oss << "    if (!fp) return -1;\n";
        oss << "    unsigned magic = 0; size_t n = 0;\n";
        oss << "    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||\n";
        oss << "        fread(&n, sizeof(n), 1, fp) != 1 ||\n";
        oss << "        magic != 0x4E534D31u || n != ns_weight_total ||\n";
        oss << "        fread(m->w, sizeof(float), n, fp) != n) {\n";
        oss << "        fclose(fp); return -1;\n";
        oss << "    }\n";
        oss << "    fclose(fp); return 0;\n";
        oss << "}\n";

        // ---- AOT training (network train() method present) ----
        for (auto& f : module.functions) {
            if (!f.is_train) continue;
            const MLIRFunction& t = f;
            oss << "\n// ---- Training core (forward + backward + optimizer) ----\n";
            oss << emit_train_core(t, in_cols);
            oss << "\n" << lr_schedule_source();
            oss << "\nextern \"C\" int ns_runtime_train_step(ns_model* m, const float* input,\n"
                << "                                        const float* labels, size_t input_numel,\n"
                << "                                        float* loss_out, float lr) {\n"
                << "    if (!m || !m->w) return -1;\n"
                << "    int64_t st = m->ctx->lr_step;\n"
                << "    if (st < 9223372036854775807LL) m->ctx->lr_step = st + 1;\n"
                << "    const float lr_eff = lr * ns_lr_schedule(st);\n"
                << "    ns_train_core(m->ctx, input, input_numel, labels, m->w, m->w, loss_out, lr_eff, 1);\n"
                << "    return 0;\n"
                << "}\n\n"
                << "extern \"C\" int ns_objective_loss(ns_model* m, const float* input,\n"
                << "                                   const float* labels, size_t input_numel,\n"
                << "                                   float* loss_out) {\n"
                << "    if (!m || !m->w) return -1;\n"
                << "    ns_train_core(m->ctx, input, input_numel, labels, m->w, (float*)0, loss_out, 0.f, 0);\n"
                << "    return 0;\n"
                << "}\n";
            break;
        }
    }

    return oss.str();
}

std::string CodeGenerator::emit_train_core(const MLIRFunction& tfn, int64_t in_cols) {
    using namespace std;
    // Per-id static shape metadata (mirrors gen_cpu).
    struct IdInfo {
        bool is_weight = false;
        int64_t rows = -1, cols = -1;
        int64_t static_numel = -1;
    };
    map<string, IdInfo> ids;
    for (auto& instr : tfn.instructions) {
        if (instr.op == MLIROp::TENSOR_ALLOC) {
            auto& inf = ids[instr.result_id];
            inf.is_weight = true;
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            if (d.empty() || std::all_of(d.begin(), d.end(),
                                         [](const DimExpr& x){ return x.is_const(); }))
                inf.static_numel = numel(instr.result_type);
        } else if (!instr.result_id.empty()) {
            auto& inf = ids[instr.result_id];
            auto& d = instr.result_type.dims;
            if (d.size() > 0 && d[0].is_const()) inf.rows = d[0].const_value;
            if (d.size() > 1 && d[1].is_const()) inf.cols = d[1].const_value;
            bool all_const = d.empty() || std::all_of(d.begin(), d.end(),
                                                      [](const DimExpr& x){ return x.is_const(); });
            if (all_const) inf.static_numel = numel(instr.result_type);
        }
    }
    for (auto& instr : tfn.instructions)
        for (auto& op : instr.operands)
            if (!ids.count(op)) ids[op] = IdInfo();

    // Detect the two dynamic inputs: batch_x (first GEMM A or embedding index
    // stream) and batch_y (CE labels). For embedding-first networks the raw
    // input feeds LAYER_EMBEDDING's index operand, not a GEMM.
    string x_id, y_id;
    for (auto& instr : tfn.instructions) {
        if (instr.op == MLIROp::LAYER_EMBEDDING && instr.operands.size() >= 2 &&
            !ids[instr.operands[1]].is_weight && x_id.empty())
            x_id = instr.operands[1];
        if (instr.op == MLIROp::MATMUL && instr.operands.size() >= 2 &&
            !ids[instr.operands[0]].is_weight && x_id.empty())
            x_id = instr.operands[0];
        if (instr.op == MLIROp::CROSS_ENTROPY && instr.operands.size() >= 2)
            y_id = instr.operands[1];
    }

    // Weight buffers in train-fn ALLOC order (layout-compatible with the
    // network's forward blob for network train methods).
    vector<string> worder_t;
    for (auto& instr : tfn.instructions)
        if (instr.op == MLIROp::TENSOR_ALLOC) worder_t.push_back(instr.result_id);

    auto src_of = [&](const string& id) -> string {
        return id == x_id ? string("x") : ("buf_" + id + ".data()");
    };
    auto numel_of = [&](const string& id) -> string {
        return id == x_id ? string("nx") : ("buf_" + id + ".size()");
    };

    // Column-width propagation (mirrors CUDA wcols): forwards widths from
    // statically typed operands / weights through the instruction graph so
    // layernorm_grad (and future ops needing the column dim) get the right
    // value when result_type is not set on intermediate ops (e.g. RELU).
    map<string, int64_t> wcols;
    {
        for (auto& instr : tfn.instructions) {
            string ida = instr.result_id;
            for (auto& op : instr.operands) {
                auto it = ids.find(op);
                if (it != ids.end() && it->second.cols > 0) wcols[op] = it->second.cols;
            }
            if ((instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED) &&
                instr.operands.size() >= 2) {
                auto it = ids.find(instr.operands[1]);
                if (it != ids.end() && it->second.cols > 0) wcols[ida] = it->second.cols;
            } else if (!ida.empty() && !instr.operands.empty()) {
                auto it = wcols.find(instr.operands[0]);
                if (it != wcols.end() && it->second > 0) wcols[ida] = it->second;
            }
        }
    }
    auto train_row_dims = [&](const string& v) -> int64_t {
        if (!v.empty()) {
            auto wc = wcols.find(v);
            if (wc != wcols.end() && wc->second > 0) return wc->second;
        }
        if (!v.empty()) {
            auto it = ids.find(v);
            if (it != ids.end() && it->second.cols > 0) return it->second.cols;
        }
        return 1;
    };

    ostringstream oss;
    oss << "extern \"C\" void ns_train_core(ns_cpu_ctx* ctx, const float* x, size_t nx, const float* y,\n"
        << "                              const float* w, float* wout, float* loss_out,\n"
        << "                              float lr, int train_mode) {\n";
    oss << "  (void)y;\n";
    for (auto& [id, inf] : ids) {
        if (id == x_id || id == y_id) continue;
        oss << "  std::vector<float>& buf_" << id << " = ctx->buf_" << id << ";\n";
    }
    for (auto& instr : tfn.instructions)
        if (instr.op == MLIROp::DROPOUT)
            oss << "  std::vector<float>& buf_dm_" << instr.result_id
                << " = ctx->buf_dm_" << instr.result_id << ";  // dropout mask (inverted, scaled)\n";

    // Load weights from the host blob (offsets match the forward layout).
    {
        size_t off = 0;
        for (auto& wt : worder_t) {
            int64_t n = ids[wt].static_numel > 0 ? ids[wt].static_numel : 1;
            oss << "  buf_" << wt << ".assign(w + " << off << ", w + " << off + n << ");\n";
            off += (size_t)n;
        }
    }

    string fwd_body, train_body, opt_body;
    for (auto& instr : tfn.instructions) {
        string id = instr.result_id;
        switch (instr.op) {
            case MLIROp::MATMUL: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for matmul '" + id + "' (dynamic dims are not supported here)");
                fwd_body += "  { int64_t K = " + to_string(Ks) + ", N = " + to_string(Ns) + ";\n";
                fwd_body += "    int64_t M = (int64_t)(" + numel_of(A) + ") / K;\n";
                fwd_body += "    buf_" + id + ".resize((size_t)(M * N));\n";
                fwd_body += "    ns_matmul(" + src_of(A) + ", buf_" + B + ".data(), buf_" +
                            id + ".data(), M, K, N); }\n";
                break;
            }
            case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
            case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
            case MLIROp::SILU: case MLIROp::IDENTITY: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int code = 0;
                switch (instr.op) {
                    case MLIROp::RELU: code = 1; break;
                    case MLIROp::LEAKY_RELU: code = 2; break;
                    case MLIROp::SIGMOID: code = 3; break;
                    case MLIROp::TANH: code = 4; break;
                    case MLIROp::SWISH: case MLIROp::SILU: code = 5; break;
                    case MLIROp::GELU: code = 6; break;
                    default: break;
                }
                fwd_body += "  { size_t zn = " + numel_of(in) + ";\n";
                fwd_body += "    buf_" + id + ".resize(zn);\n";
                fwd_body += "    const float* src = " + src_of(in) + ";\n";
                fwd_body += "    for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = ns_act2(src[i], " + to_string(code) + "); }\n";
                break;
            }
            case MLIROp::LAYER_EMBEDDING: {
                string wt = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                fwd_body += "  { int64_t n_idx = " + numel_of(idx) + ";\n";
                fwd_body += "    buf_" + id + ".resize((size_t)(n_idx * " + to_string(D) + "));\n";
                fwd_body += "    ns_embedding(" + src_of(wt) + ", " + src_of(idx) + ", buf_" +
                            id + ".data(), n_idx, " + to_string(V) + ", " + to_string(D) + "); }\n";
                break;
            }
            case MLIROp::LAYER_MOE: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wg = instr.operands.size() > 1 ? instr.operands[1] : "";
                string We = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                fwd_body += "  { int64_t M = (int64_t)(" + numel_of(x) + ") / " + to_string(D) + ";\n";
                fwd_body += "    buf_" + id + ".resize((size_t)(M * " + to_string(D) + "));\n";
                fwd_body += "    ns_moe_fwd(" + src_of(x) + ", " + src_of(Wg) + ", " + src_of(We) +
                            ", buf_" + id + ".data(), M, " + to_string(D) + ", " + to_string(E) + "); }\n";
                break;
            }
            case MLIROp::LAYER_ATTENTION: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wq = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wk = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wv = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wo = instr.operands.size() > 4 ? instr.operands[4] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t H = instr.attribute.empty() ? 1 : std::atol(instr.attribute.c_str());
                fwd_body += "  { int64_t BS = (int64_t)(" + numel_of(x) + ") / " + to_string(D) + ";\n";
                fwd_body += "    int64_t S = BS;\n";
                fwd_body += "    std::vector<float> q_b(BS*" + to_string(D) + "), k_b(BS*" + to_string(D) + "), v_b(BS*" + to_string(D) + "), proj_b(BS*" + to_string(D) + "), sc_b(S*S);\n";
                fwd_body += "    buf_" + id + ".resize((size_t)(BS*" + to_string(D) + "));\n";
                fwd_body += "    ns_attention_fwd(" + src_of(x) + ", " + src_of(Wq) + ", " + src_of(Wk) +
                            ", " + src_of(Wv) + ", " + src_of(Wo) + ",\n";
                fwd_body += "        q_b.data(), k_b.data(), v_b.data(), sc_b.data(), proj_b.data(), buf_" + id + ".data(),\n";
                fwd_body += "        BS, " + to_string(D) + ", " + to_string(H) + ", S); }\n";
                break;
            }
            case MLIROp::LAYERNORM: case MLIROp::SOFTMAX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t last = train_row_dims(in);
                last = last > 0 ? last : 1;
                fwd_body += "  { size_t zn = " + numel_of(in) + ";\n";
                fwd_body += "    buf_" + id + ".resize(zn);\n";
                fwd_body += "    const float* src = " + src_of(in) + ";\n";
                fwd_body += "    for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = src[i];\n";
                if (instr.op == MLIROp::LAYERNORM)
                    fwd_body += "    ns_layernorm(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(last) + "); }\n";
                else
                    fwd_body += "    ns_softmax(buf_" + id + ".data(), buf_" + id + ".size(), " + to_string(last) + "); }\n";
                break;
            }
            case MLIROp::DROPOUT: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                double rate = instr.float_attr;
                if (rate < 0.0) rate = 0.0;
                if (rate >= 1.0) rate = 0.999999;
                string r = std::to_string(rate);
                fwd_body += "  { size_t zn = " + numel_of(in) + ";\n";
                fwd_body += "    buf_" + id + ".resize(zn);\n";
                fwd_body += "    buf_dm_" + id + ".resize(zn);\n";
                fwd_body += "    const float* src = " + src_of(in) + ";\n";
                fwd_body += "    if (train_mode) {\n";
                fwd_body += "      std::bernoulli_distribution keep(1.0 - " + r + ");\n";
                fwd_body += "      const float scale = 1.0f / (float)(1.0 - " + r + ");\n";
                fwd_body += "      for (size_t i = 0; i < zn; i++) { float m = keep(ctx->rng) ? scale : 0.f; buf_dm_" + id + "[i] = m; buf_" + id + "[i] = src[i] * m; }\n";
                fwd_body += "    } else {\n";
                fwd_body += "      for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = src[i];\n";
                fwd_body += "    }\n";
                fwd_body += "  }\n";
                break;
            }
            case MLIROp::CROSS_ENTROPY: {
                string preds = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t C = ids[preds].cols > 0 ? ids[preds].cols : in_cols;
                fwd_body += "  { buf_" + id + ".assign(1, 0.f);\n";
                fwd_body += "    buf_" + id + "[0] = ns_cross_entropy(buf_" + preds + ".data(), y, buf_" +
                            preds + ".size(), " + to_string(C) + "); }\n";
                break;
            }
            case MLIROp::LOSS_GRAD: {
                string preds = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t C = ids[preds].cols > 0 ? ids[preds].cols : in_cols;
                train_body += "  { size_t zn = buf_" + preds + ".size();\n";
                train_body += "    buf_" + id + ".resize(zn);\n";
                train_body += "    ns_loss_grad(buf_" + preds + ".data(), y, buf_" + id +
                              ".data(), zn, " + to_string(C) + "); }\n";
                break;
            }
            case MLIROp::MATMUL_GRAD_A: {
                string dc = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for grad_a '" + id + "' (dynamic dims are not supported here)");
                train_body += "  { int64_t K = " + to_string(Ks) + ", N = " + to_string(Ns) + ";\n";
                train_body += "    int64_t M = (int64_t)(buf_" + dc + ".size()) / N;\n";
                train_body += "    buf_" + id + ".resize((size_t)(M * K));\n";
                train_body += "    ns_matmul_grad_a(buf_" + dc + ".data(), buf_" + B + ".data(), buf_" +
                              id + ".data(), M, K, N); }\n";
                break;
            }
            case MLIROp::MATMUL_GRAD_W: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string dc = instr.operands.size() > 1 ? instr.operands[1] : "";
                string B = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0 || A.empty() || dc.empty())
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for grad_w '" + id + "' (dynamic dims are not supported here)");
                train_body += "  { int64_t K = " + to_string(Ks) + ", N = " + to_string(Ns) + ";\n";
                train_body += "    int64_t M = (int64_t)(buf_" + dc + ".size()) / N;\n";
                train_body += "    buf_" + id + ".resize((size_t)(K * N));\n";
                train_body += "    ns_matmul_grad_w(" + src_of(A) + ", buf_" + dc + ".data(), buf_" +
                              id + ".data(), M, K, N); }\n";
                break;
            }
            case MLIROp::EMBEDDING_GRAD_W: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                string wt = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                train_body += "  { size_t M = buf_" + dout + ".size() / " + to_string(D) + ";\n";
                train_body += "    buf_" + id + ".resize((size_t)(" + to_string(V) + " * " + to_string(D) + "));\n";
                train_body += "    ns_embedding_grad_w(buf_" + dout + ".data(), " + src_of(idx) + ", buf_" +
                              id + ".data(), M, " + to_string(D) + ", " + to_string(V) + "); }\n";
                break;
            }
            case MLIROp::MOE_GRAD_X: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                train_body += "  { int64_t M = (int64_t)(buf_" + dout + ".size()) / " + to_string(D) + ";\n";
                train_body += "    buf_" + id + ".resize((size_t)(M * " + to_string(D) + "));\n";
                train_body += "    for (size_t q = 0; q < buf_" + id + ".size(); q++) buf_" + id + "[q] = 0.f;\n";
                train_body += "    ns_moe_grad_x(buf_" + dout + ".data(), " + src_of(xo) + ", " + src_of(Wg) +
                              ", " + src_of(We) + ", buf_" + id + ".data(), M, " + to_string(D) + ", " + to_string(E) + "); }\n";
                break;
            }
            case MLIROp::MOE_GRAD_WG: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                train_body += "  { int64_t M = (int64_t)(buf_" + dout + ".size()) / " + to_string(D) + ";\n";
                train_body += "    buf_" + id + ".resize((size_t)(" + to_string(D) + " * " + to_string(E) + "));\n";
                train_body += "    ns_moe_grad_wg(buf_" + dout + ".data(), " + src_of(xo) + ", " + src_of(Wg) +
                              ", " + src_of(We) + ", buf_" + id + ".data(), M, " + to_string(D) + ", " + to_string(E) + "); }\n";
                break;
            }
            case MLIROp::MOE_GRAD_WE: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                train_body += "  { int64_t M = (int64_t)(buf_" + dout + ".size()) / " + to_string(D) + ";\n";
                train_body += "    buf_" + id + ".resize((size_t)(" + to_string(E) + " * " + to_string(D) + " * " + to_string(D) + "));\n";
                train_body += "    ns_moe_grad_we(buf_" + dout + ".data(), " + src_of(xo) + ", " + src_of(Wg) +
                              ", " + src_of(We) + ", buf_" + id + ".data(), M, " + to_string(D) + ", " + to_string(E) + "); }\n";
                break;
            }
            case MLIROp::ATTENTION_GRAD_X:
            case MLIROp::ATTENTION_GRAD_WQ:
            case MLIROp::ATTENTION_GRAD_WK:
            case MLIROp::ATTENTION_GRAD_WV:
            case MLIROp::ATTENTION_GRAD_WO: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wq = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wk = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wv = instr.operands.size() > 4 ? instr.operands[4] : "";
                string Wo = instr.operands.size() > 5 ? instr.operands[5] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t H = instr.attribute.empty() ? 1 : std::atol(instr.attribute.c_str());
                int64_t DD = D * D;
                string nele = (instr.op == MLIROp::ATTENTION_GRAD_X)
                                  ? ("(int64_t)(buf_" + dout + ".size())") : to_string(DD);
                string tgt =
                    (instr.op == MLIROp::ATTENTION_GRAD_X) ? ("buf_" + id + ".data()") : ("buf_" + id + ".data()");
                train_body += "  { int64_t BS = (int64_t)(buf_" + dout + ".size()) / " + to_string(D) + ";\n";
                train_body += "    buf_" + id + ".resize((size_t)(" + nele + "));\n";
                train_body += "    ns_attention_bwd(buf_" + dout + ".data(), " + src_of(xo) + ", " + src_of(Wq) +
                              ", " + src_of(Wk) + ", " + src_of(Wv) + ", " + src_of(Wo) + ",\n";
                if (instr.op == MLIROp::ATTENTION_GRAD_X)
                    train_body += "        " + tgt + ", nullptr, nullptr, nullptr, nullptr,\n";
                else if (instr.op == MLIROp::ATTENTION_GRAD_WQ)
                    train_body += "        nullptr, " + tgt + ", nullptr, nullptr, nullptr,\n";
                else if (instr.op == MLIROp::ATTENTION_GRAD_WK)
                    train_body += "        nullptr, nullptr, " + tgt + ", nullptr, nullptr,\n";
                else if (instr.op == MLIROp::ATTENTION_GRAD_WV)
                    train_body += "        nullptr, nullptr, nullptr, " + tgt + ", nullptr,\n";
                else
                    train_body += "        nullptr, nullptr, nullptr, nullptr, " + tgt + ",\n";
                train_body += "        BS, " + to_string(D) + ", " + to_string(H) + ", BS); }\n";
                break;
            }
            case MLIROp::LAYERNORM_GRAD: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string act_in = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t last = train_row_dims(act_in);
                last = last > 0 ? last : 1;
                train_body += "  { size_t zn = buf_" + dout + ".size();\n";
                train_body += "    buf_" + id + ".resize(zn);\n";
                train_body += "    ns_layernorm_grad(buf_" + dout + ".data(), buf_" + act_in +
                              ".data(), buf_" + id + ".data(), zn, " + to_string(last) + "); }\n";
                break;
            }
            case MLIROp::ACTIVATION_GRAD: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string act_in = instr.operands.size() > 1 ? instr.operands[1] : "";
                if (instr.attribute == "dropout") {
                    train_body += "  { size_t zn = buf_" + dout + ".size();\n";
                    train_body += "    buf_" + id + ".resize(zn);\n";
                    train_body += "    const float* dg = buf_" + dout + ".data();\n";
                    train_body += "    const float* dm = buf_dm_" + act_in + ".data();\n";
                    train_body += "    for (size_t i = 0; i < zn; i++) buf_" + id + "[i] = dg[i] * dm[i]; }\n";
                    break;
                }
                int code = 0;
                string a = instr.attribute;
                if (a == "relu") code = 1;
                else if (a == "leaky_relu") code = 2;
                else if (a == "sigmoid") code = 3;
                else if (a == "tanh") code = 4;
                else if (a == "swish" || a == "silu") code = 5;
                else if (a == "gelu") code = 6;
                train_body += "  { size_t zn = buf_" + dout + ".size();\n";
                train_body += "    buf_" + id + ".resize(zn);\n";
                train_body += "    ns_act_grad(buf_" + dout + ".data(), buf_" + act_in +
                              ".data(), buf_" + id + ".data(), zn, " + to_string(code) + "); }\n";
                break;
            }
            case MLIROp::OPT_STEP: {
                for (size_t i = 0; i + 1 < instr.operands.size(); i += 2) {
                    string wt = instr.operands[i], g = instr.operands[i + 1];
                    int64_t r = ids[wt].rows > 0 ? ids[wt].rows : 1;
                    int64_t c = ids[wt].cols > 0 ? ids[wt].cols : 1;
                    int64_t n = r * c;
                    // Muon only for sufficiently wide matrices; small/vector
                    // params fall back to bias-corrected AdamW (robust, and
                    // orthonormalizing a tiny-rank momentum is unstable).
                    bool use_muon = (r > 1 && c > 1 && std::min(r, c) >= 8);
opt_body += "  { size_t n_ = " + to_string(n) + ";\n";
                    if (use_muon) {
                        opt_body += "    std::vector<float>& st_m = ctx->st_m_" + wt + "; std::vector<float>& st_am = ctx->st_am_" + wt + "; std::vector<float>& st_av = ctx->st_av_" + wt + ";\n";
                        opt_body += "    if (st_m.size() != n_) { st_m.assign(n_, 0.f); st_am.assign(n_, 0.f); st_av.assign(n_, 0.f); }\n";
                        opt_body += "    for (size_t i = 0; i < n_; i++) st_m[i] = " + fmt_float(optim::kMuonMomentum) + "f * st_m[i] + buf_" + g + "[i];\n";
                        opt_body += "    ns_orthonom(st_m.data(), (size_t)" + to_string(r) + ", (size_t)" + to_string(c) + ");\n";
                        opt_body += "    float rank = sqrtf((float)std::min((int64_t)" + to_string(r) + ", (int64_t)" + to_string(c) + "));\n";
                        opt_body += "    float le = lr * rank;\n";
                        opt_body += "    for (size_t i = 0; i < n_; i++) buf_" + wt + "[i] -= le * st_m[i] + (" + fmt_float(optim::kMuonDecay) + "f * lr) * buf_" + wt + "[i];\n";
                        opt_body += "    std::fill(st_m.begin(), st_m.end(), 0.f);\n";
                    } else {
                        opt_body += "    std::vector<float>& st_m = ctx->st_m_" + wt + "; std::vector<float>& st_am = ctx->st_am_" + wt + "; std::vector<float>& st_av = ctx->st_av_" + wt + "; size_t& st_t = ctx->st_t_" + wt + ";\n";
                        opt_body += "    size_t t = ++st_t;\n";
                        opt_body += "    if (st_m.size() != n_) { st_m.assign(n_, 0.f); st_am.assign(n_, 0.f); st_av.assign(n_, 0.f); }\n";
                        opt_body += "    float b1t = 1.f - std::pow(" + fmt_float(optim::kAdamWBeta1) + "f, (float)t), b2t = 1.f - std::pow(" + fmt_float(optim::kAdamWBeta2) + "f, (float)t);\n";
                        opt_body += "    for (size_t i = 0; i < n_; i++) {\n";
                        opt_body += "      float gg = buf_" + g + "[i];\n";
                        opt_body += "      st_am[i] = " + fmt_float(optim::kAdamWBeta1) + "f * st_am[i] + " + fmt_float(optim::kOneMinusBeta1) + "f * gg;\n";
                        opt_body += "      st_av[i] = " + fmt_float(optim::kAdamWBeta2) + "f * st_av[i] + " + fmt_float(optim::kOneMinusBeta2) + "f * gg * gg;\n";
                        opt_body += "      float mh = st_am[i] / b1t, vh = st_av[i] / b2t;\n";
                        opt_body += "      buf_" + wt + "[i] -= (lr * mh / (sqrtf(vh) + " + fmt_float(optim::kAdamWEps) + "f)) + (" + fmt_float(optim::kAdamWDecay) + "f * lr) * buf_" + wt + "[i];\n";
                        opt_body += "    }\n";
                    }
                    opt_body += "  }\n";
                }
                break;
            }
            default:
                break;
        }
    }

    oss << fwd_body;
    oss << "  if (train_mode) {\n";
    oss << train_body;
    oss << opt_body;
    oss << "  }\n";

    // Loss output: train() return value is the loss scalar.
    string loss_id = tfn.return_id;
    if (loss_id.empty())
        for (auto& instr : tfn.instructions)
            if (instr.op == MLIROp::CROSS_ENTROPY) loss_id = instr.result_id;
    if (!loss_id.empty() && ids.count(loss_id))
        oss << "  if (loss_out) *loss_out = buf_" << loss_id << "[0];\n";

    // Write back the (possibly updated) weights.
    oss << "  if (train_mode && wout) {\n";
    {
        size_t off = 0;
        for (auto& wt : worder_t) {
            int64_t n = ids[wt].static_numel > 0 ? ids[wt].static_numel : 1;
            oss << "    std::memcpy(wout + " << off << ", buf_" << wt << ".data(), "
                << n << " * sizeof(float));\n";
            off += (size_t)n;
        }
    }
    oss << "  }\n";
oss << "}\n";
    return oss.str();
}

std::string CodeGenerator::emit_train_core_cuda(const MLIRFunction& tfn,
                                                int64_t in_cols, int64_t out_cols) {
    using namespace std;
    map<string, CuIdInfo> ids;
    fill_cu_ids(tfn, ids);

    // Dynamic inputs of the train function: batch_x (first non-weight GEMM A),
    // batch_y (CE labels).
    string x_id, y_id;
    for (auto& instr : tfn.instructions) {
        if (instr.op == MLIROp::LAYER_EMBEDDING && instr.operands.size() >= 2 &&
            !ids[instr.operands[1]].is_weight && x_id.empty())
            x_id = instr.operands[1];
        if (instr.op == MLIROp::MATMUL && instr.operands.size() >= 2 &&
            !ids[instr.operands[0]].is_weight && x_id.empty())
            x_id = instr.operands[0];
        if (instr.op == MLIROp::CROSS_ENTROPY && instr.operands.size() >= 2)
            y_id = instr.operands[1];
    }

    // Weight buffers in train-fn ALLOC order (layout-compatible with the
    // network's forward blob for network train methods).
    vector<string> worder_t;
    for (auto& instr : tfn.instructions)
        if (instr.op == MLIROp::TENSOR_ALLOC) worder_t.push_back(instr.result_id);
    map<string, size_t> woff;
    size_t weight_total = 0;
    {
        size_t off = 0;
        for (auto& wt : worder_t) {
            woff[wt] = off;
            int64_t n = ids[wt].static_numel > 0 ? ids[wt].static_numel : 1;
            off += (size_t)n;
            weight_total += (size_t)n;
        }
    }

    int64_t K1 = ids[x_id].cols > 0 ? ids[x_id].cols : in_cols;
    if (K1 <= 0) K1 = 1;
    int64_t C = out_cols > 0 ? out_cols : 2;

    auto bufv = [&](const string& id) -> string {
        if (id == x_id) return "ctx->d_xi";
        if (id == y_id) return "ctx->d_yl";
        if (ids[id].is_weight && woff.count(id)) return "(ctx->d_wb + " + to_string(woff[id]) + ")";
        return "ctx->d_" + id;
    };

    // Feature width of each per-row tensor, propagated forward over the
    // instruction list (ops are in topological order). Ops lowered from bare
    // function calls (dropout, plain activations, ...) carry no result_type in
    // CuIdInfo, so width comes from the producing instruction instead. Do NOT
    // use operator[] here: it would insert 0 entries that break row_dims.
    std::map<string, int64_t> wcols;
    {
        map<string, int64_t> cur;
        for (auto& a : tfn.instructions) {
            string ida = a.result_id;
            for (auto& op : a.operands) {
                auto it = ids.find(op);
                if (it != ids.end() && it->second.cols > 0) cur[op] = it->second.cols;
            }
            if ((a.op == MLIROp::MATMUL || a.op == MLIROp::FUSED) && a.operands.size() >= 2) {
                auto it = ids.find(a.operands[1]);
                if (it != ids.end() && it->second.cols > 0) cur[ida] = it->second.cols;
            } else if (!ida.empty() && !a.operands.empty()) {
                auto it = cur.find(a.operands[0]);
                if (it != cur.end() && it->second > 0) cur[ida] = it->second;
            }
        }
        // Materials covered here are also actual SSA results.
        wcols = cur;
    }
    auto row_dims = [&](const string& v) -> int64_t {
        if (!v.empty()) {
            auto wc = wcols.find(v);
            if (wc != wcols.end() && wc->second > 0) return wc->second;
        }
        if (!v.empty() && ids[v].cols > 0) return ids[v].cols;
        for (auto& a : tfn.instructions)
            if ((a.op == MLIROp::MATMUL || a.op == MLIROp::FUSED) &&
                a.operands.size() >= 2 && a.operands[0] == v) {
                int64_t r = ids[a.operands[1]].rows;
                if (r > 0) return r;
            }
        return 1;
    };

    ostringstream oss;
    oss << "extern \"C\" void ns_train_core(NSContext* ctx, const float* x, size_t nx, const float* y,\n"
        << "                              const float* w, float* wout, float* loss_out,\n"
        << "                              float lr, int train_mode) {\n"
        << "  (void)w; (void)y;\n"
        << "  if (!ctx || !ctx->d_wb || nx == 0) return;\n"
        << "  const int M = (int)(nx / " << K1 << ");\n"
        << "  if (M <= 0) return;\n"
        << "  const size_t ny = (size_t)M * " << C << ";\n"
        << "  if (ns_cu_reserve(&ctx->d_xi, &ctx->d_xi_cap, nx * sizeof(float), 0)) return;\n"
        << "  if (ns_cu_reserve(&ctx->d_yl, &ctx->d_yl_cap, ny * sizeof(float), 0)) return;\n"
        << "  cudaMemcpy(ctx->d_xi, x, nx * sizeof(float), cudaMemcpyHostToDevice);\n"
        << "  cudaMemcpy(ctx->d_yl, y, ny * sizeof(float), cudaMemcpyHostToDevice);\n";

    // Batch-element-count helper for emission.
    string fwd_text, train_text;
    for (auto& instr : tfn.instructions) {
        string id = instr.result_id;
        switch (instr.op) {
            case MLIROp::MATMUL: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for matmul '" + id + "' (dynamic dims are not supported here)");
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(Ns) + " * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" + to_string(Ns) + " + 15) / 16), "
                    + bufv(A) + ", " + bufv(B) + ", ctx->d_" + id + ", M, "
                    + to_string(Ks) + ", " + to_string(Ns) + ");\n";
                break;
            }
            case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
            case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
            case MLIROp::SILU: case MLIROp::IDENTITY: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                Ns = Ns > 0 ? Ns : 1;
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(Ns) + " * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH1(ns_act_kernel, M * " + to_string(Ns) + ", "
                    + bufv(in) + ", ctx->d_" + id + ", M * " + to_string(Ns) + ", "
                    + cu_act_code(instr.op) + ");\n";
                break;
            }
            case MLIROp::LAYER_EMBEDDING: {
                string wt = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(D) + " * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH1(ns_embedding_kernel, M * " + to_string(D) + ", "
                    + bufv(wt) + ", " + bufv(idx) + ", ctx->d_" + id + ", M, "
                    + to_string(V) + ", " + to_string(D) + ");\n";
                break;
            }
            case MLIROp::LAYER_MOE: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wg = instr.operands.size() > 1 ? instr.operands[1] : "";
                string We = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)(M * "
                    + to_string(D) + ") * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH_BLOCKS(ns_moe_kernel, M, "
                    + bufv(x) + ", " + bufv(Wg) + ", " + bufv(We) + ", ctx->d_" + id + ", "
                    + "M, " + to_string(D) + ", " + to_string(E) + ");\n";
                break;
            }
            case MLIROp::LAYER_ATTENTION: {
                string x = instr.operands.size() > 0 ? instr.operands[0] : "";
                string Wq = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wk = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wv = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wo = instr.operands.size() > 4 ? instr.operands[4] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t H = instr.attribute.empty() ? 1 : std::atol(instr.attribute.c_str());
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(D) + " * sizeof(float), 0)) return;\n";
                fwd_text += "  { float* qs = 0; float* ks = 0; float* vs = 0; float* cs = 0;\n";
                fwd_text += "    if (cudaMalloc(&qs, (size_t)M * " + to_string(D) + " * sizeof(float)) ||\n";
                fwd_text += "        cudaMalloc(&ks, (size_t)M * " + to_string(D) + " * sizeof(float)) ||\n";
                fwd_text += "        cudaMalloc(&vs, (size_t)M * " + to_string(D) + " * sizeof(float)) ||\n";
                fwd_text += "        cudaMalloc(&cs, (size_t)M * " + to_string(D) + " * sizeof(float))) return;\n";
                fwd_text += "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" + to_string(D) + " + 15) / 16), "
                    + bufv(x) + ", " + bufv(Wq) + ", qs, M, " + to_string(D) + ", " + to_string(D) + ");\n";
                fwd_text += "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" + to_string(D) + " + 15) / 16), "
                    + bufv(x) + ", " + bufv(Wk) + ", ks, M, " + to_string(D) + ", " + to_string(D) + ");\n";
                fwd_text += "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" + to_string(D) + " + 15) / 16), "
                    + bufv(x) + ", " + bufv(Wv) + ", vs, M, " + to_string(D) + ", " + to_string(D) + ");\n";
                fwd_text += "    ns_attention_core_kernel<<<" + to_string(H) + ", 256, (size_t)M * M * sizeof(float)>>>"
                    + "(qs, ks, vs, cs, M, " + to_string(D) + ", " + to_string(H)
                    + ", 1.0f / sqrtf((float)(" + to_string(D) + " / " + to_string(H) + ")));\n";
                fwd_text += "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * ((" + to_string(D) + " + 15) / 16), "
                    + "cs, " + bufv(Wo) + ", ctx->d_" + id + ", M, " + to_string(D) + ", " + to_string(D) + ");\n";
                fwd_text += "    cudaFree(qs); cudaFree(ks); cudaFree(vs); cudaFree(cs); }\n";
                break;
            }
            case MLIROp::LAYERNORM: case MLIROp::SOFTMAX: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                Ns = Ns > 0 ? Ns : 1;
                string kn = "M * " + to_string(Ns);
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)(" + kn
                    + ") * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH1(ns_copy_kernel, " + kn + ", "
                    + bufv(in) + ", ctx->d_" + id + ", " + kn + ");\n";
                if (instr.op == MLIROp::LAYERNORM)
                    fwd_text += "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_" + id + ", " + kn + ", " + to_string(Ns) + ");\n";
                else
                    fwd_text += "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_" + id + ", " + kn + ", " + to_string(Ns) + ");\n";
                break;
            }
            case MLIROp::DROPOUT: {
                string in = instr.operands.size() > 0 ? instr.operands[0] : "";
                int64_t Ns = row_dims(in);
                Ns = Ns > 0 ? Ns : 1;
                double rate = instr.float_attr;
                if (rate < 0.0) rate = 0.0;
                if (rate >= 1.0) rate = 0.999999;
                string ne = "M * " + to_string(Ns);
                fwd_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)(" + ne
                    + ") * sizeof(float), 0)) return;\n";
                fwd_text += "  if (train_mode) {\n";
                fwd_text += "    if (ns_cu_reserve(&ctx->d_dm_" + id + ", &ctx->d_dm_" + id + "_cap, (size_t)(" + ne
                    + ") * sizeof(float), 0)) return;\n";
                fwd_text += "    NS_LAUNCH1(ns_dropout_fwd_kernel, " + ne + ", "
                    + bufv(in) + ", ctx->d_" + id + ", ctx->d_dm_" + id + ", " + ne + ", "
                    + to_string(rate) + "f, ctx->drop_seed++);\n";
                fwd_text += "  } else {\n";
                fwd_text += "    NS_LAUNCH1(ns_copy_kernel, " + ne + ", "
                    + bufv(in) + ", ctx->d_" + id + ", " + ne + ");\n";
                fwd_text += "  }\n";
                break;
            }
            case MLIROp::CROSS_ENTROPY: {
                string preds = instr.operands.size() > 0 ? instr.operands[0] : "";
                fwd_text += "  if (ns_cu_reserve(&ctx->d_row, &ctx->d_row_cap, (size_t)M * sizeof(float), 0)) return;\n";
                fwd_text += "  NS_LAUNCH1(ns_ce_row_kernel, M, " + bufv(preds) + ", ctx->d_yl, ctx->d_row, M * "
                    + to_string(C) + ", " + to_string(C) + ");\n";
                fwd_text += "  cudaDeviceSynchronize();\n";
                fwd_text += "  if (!ctx->hrow || ctx->hrow_cap < (size_t)M) { free((void*)ctx->hrow); ctx->hrow = (float*)malloc((size_t)M * sizeof(float)); ctx->hrow_cap = (size_t)M; }\n";
                fwd_text += "  cudaMemcpy(ctx->hrow, ctx->d_row, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);\n";
                fwd_text += "  float loss = 0.f;\n";
                fwd_text += "  for (int r = 0; r < M; r++) loss += ctx->hrow[r];\n";
                fwd_text += "  loss /= (float)M;\n";
                fwd_text += "  if (loss_out) *loss_out = loss;\n";
                break;
            }
            case MLIROp::LOSS_GRAD: {
                string preds = instr.operands.size() > 0 ? instr.operands[0] : "";
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(C) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_loss_grad_kernel, M, " + bufv(preds) + ", ctx->d_yl, ctx->d_"
                    + id + ", M * " + to_string(C) + ", " + to_string(C) + ");\n";
                break;
            }
            case MLIROp::MATMUL_GRAD_A: {
                string dc = instr.operands.size() > 0 ? instr.operands[0] : "";
                string B = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0)
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for grad_a '" + id + "' (dynamic dims are not supported here)");
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(Ks) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_grad_a_kernel, M * " + to_string(Ks) + ", "
                    + bufv(dc) + ", " + bufv(B) + ", ctx->d_" + id + ", M, "
                    + to_string(Ks) + ", " + to_string(Ns) + ");\n";
                break;
            }
            case MLIROp::MATMUL_GRAD_W: {
                string A = instr.operands.size() > 0 ? instr.operands[0] : "";
                string dc = instr.operands.size() > 1 ? instr.operands[1] : "";
                string B = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t Ks = ids[B].rows, Ns = ids[B].cols;
                if (Ks <= 0 || Ns <= 0 || A.empty() || dc.empty())
                    throw std::runtime_error("codegen: cannot infer static shape of operand '" + B +
                        "' for grad_w '" + id + "' (dynamic dims are not supported here)");
                int64_t n = Ks * Ns;
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, "
                    + to_string(n) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_grad_w_kernel, " + to_string(n) + ", "
                    + bufv(A) + ", " + bufv(dc) + ", ctx->d_" + id + ", M, "
                    + to_string(Ks) + ", " + to_string(Ns) + ");\n";
                break;
            }
            case MLIROp::EMBEDDING_GRAD_W: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string idx = instr.operands.size() > 1 ? instr.operands[1] : "";
                string wt = instr.operands.size() > 2 ? instr.operands[2] : "";
                int64_t V = ids[wt].rows > 0 ? ids[wt].rows : 1;
                int64_t D = ids[wt].cols > 0 ? ids[wt].cols : 1;
                int64_t n = V * D;
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, "
                    + to_string(n) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_fill_kernel, " + to_string(n) + ", ctx->d_" + id
                    + ", " + to_string(n) + ", 0.0f);\n";
                train_text += "  NS_LAUNCH1(ns_embedding_grad_w_kernel, M * " + to_string(D) + ", "
                    + bufv(dout) + ", " + bufv(idx) + ", ctx->d_" + id + ", M, "
                    + to_string(D) + ", " + to_string(V) + ");\n";
                break;
            }
            case MLIROp::MOE_GRAD_X: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(D) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH_BLOCKS(ns_moe_grad_x_kernel, M, "
                    + bufv(dout) + ", " + bufv(xo) + ", " + bufv(Wg) + ", " + bufv(We)
                    + ", ctx->d_" + id + ", M, " + to_string(D) + ", " + to_string(E) + ");\n";
                break;
            }
            case MLIROp::MOE_GRAD_WG: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                string nwg = to_string(D) + " * " + to_string(E);
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)("
                    + nwg + ") * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_fill_kernel, " + nwg + ", ctx->d_" + id + ", " + nwg + ", 0.0f);\n";
                train_text += "  NS_LAUNCH_BLOCKS(ns_moe_grad_wg_kernel, M, "
                    + bufv(dout) + ", " + bufv(xo) + ", " + bufv(Wg) + ", " + bufv(We)
                    + ", ctx->d_" + id + ", M, " + to_string(D) + ", " + to_string(E) + ");\n";
                break;
            }
            case MLIROp::MOE_GRAD_WE: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wg = instr.operands.size() > 2 ? instr.operands[2] : "";
                string We = instr.operands.size() > 3 ? instr.operands[3] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t E = instr.attribute.empty() ? 4 : std::atol(instr.attribute.c_str());
                string nwe = to_string(E) + " * " + to_string(D) + " * " + to_string(D);
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)("
                    + nwe + ") * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_fill_kernel, " + nwe + ", ctx->d_" + id + ", " + nwe + ", 0.0f);\n";
                train_text += "  NS_LAUNCH_BLOCKS(ns_moe_grad_we_kernel, M, "
                    + bufv(dout) + ", " + bufv(xo) + ", " + bufv(Wg) + ", " + bufv(We)
                    + ", ctx->d_" + id + ", M, " + to_string(D) + ", " + to_string(E) + ");\n";
                break;
            }
            case MLIROp::ATTENTION_GRAD_X:
            case MLIROp::ATTENTION_GRAD_WQ:
            case MLIROp::ATTENTION_GRAD_WK:
            case MLIROp::ATTENTION_GRAD_WV:
            case MLIROp::ATTENTION_GRAD_WO: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string xo = instr.operands.size() > 1 ? instr.operands[1] : "";
                string Wq = instr.operands.size() > 2 ? instr.operands[2] : "";
                string Wk = instr.operands.size() > 3 ? instr.operands[3] : "";
                string Wv = instr.operands.size() > 4 ? instr.operands[4] : "";
                string Wo = instr.operands.size() > 5 ? instr.operands[5] : "";
                int64_t D = instr.int_attr > 0 ? instr.int_attr : 1;
                int64_t H = instr.attribute.empty() ? 1 : std::atol(instr.attribute.c_str());
                string DD2 = to_string(D) + " * " + to_string(D);
                bool is_x = (instr.op == MLIROp::ATTENTION_GRAD_X);
                int which = is_x ? 0 : (instr.op == MLIROp::ATTENTION_GRAD_WQ ? 1
                    : (instr.op == MLIROp::ATTENTION_GRAD_WK ? 2
                       : (instr.op == MLIROp::ATTENTION_GRAD_WV ? 3 : 4)));
                if (is_x)
                    train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                        + to_string(D) + " * sizeof(float), 0)) return;\n";
                else
                    train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)("
                        + DD2 + ") * sizeof(float), 0)) return;\n";
                if (!is_x)
                    train_text += "  NS_LAUNCH1(ns_fill_kernel, " + DD2 + ", ctx->d_" + id + ", " + DD2 + ", 0.0f);\n";
                train_text += "  { size_t sh = (size_t)M * M * sizeof(float) + 8 * (size_t)M * "
                    + to_string(D / H) + " * sizeof(float);\n";
                train_text += "  ns_attention_grad_kernel<<<" + to_string(H) + ", 256, sh>>>"
                    + "(" + bufv(dout) + ", " + bufv(xo) + ", " + bufv(Wq) + ", " + bufv(Wk)
                    + ", " + bufv(Wv) + ", " + bufv(Wo) + ", "
                    + (is_x ? "ctx->d_" + id : "0") + ", "
                    + (which == 1 ? ("ctx->d_" + id) : "0") + ", "
                    + (which == 2 ? ("ctx->d_" + id) : "0") + ", "
                    + (which == 3 ? ("ctx->d_" + id) : "0") + ", "
                    + (which == 4 ? ("ctx->d_" + id) : "0") + ", M, "
                    + to_string(D) + ", " + to_string(H) + ", M, " + to_string(which) + "); }\n";
                break;
            }
            case MLIROp::LAYERNORM_GRAD: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string act_in = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ns = row_dims(act_in);
                Ns = Ns > 0 ? Ns : 1;
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(Ns) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_layernorm_grad_kernel, M, "
                    + bufv(dout) + ", " + bufv(act_in) + ", ctx->d_" + id + ", M * "
                    + to_string(Ns) + ", " + to_string(Ns) + ");\n";
                break;
            }
            case MLIROp::ACTIVATION_GRAD: {
                string dout = instr.operands.size() > 0 ? instr.operands[0] : "";
                string act_in = instr.operands.size() > 1 ? instr.operands[1] : "";
                int64_t Ns = row_dims(act_in);
                Ns = Ns > 0 ? Ns : 1;
                if (instr.attribute == "dropout") {
                    train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                        + to_string(Ns) + " * sizeof(float), 0)) return;\n";
                    train_text += "  NS_LAUNCH1(ns_dropout_mul_kernel, M * " + to_string(Ns) + ", "
                        + bufv(dout) + ", ctx->d_dm_" + act_in + ", ctx->d_" + id + ", M * "
                        + to_string(Ns) + ");\n";
                    break;
                }
                train_text += "  if (ns_cu_reserve(&ctx->d_" + id + ", &ctx->d_" + id + "_cap, (size_t)M * "
                    + to_string(Ns) + " * sizeof(float), 0)) return;\n";
                train_text += "  NS_LAUNCH1(ns_act_grad_kernel, M * " + to_string(Ns) + ", "
                    + bufv(dout) + ", " + bufv(act_in) + ", ctx->d_" + id + ", M * "
                    + to_string(Ns) + ", " + cu_act_code_attr(instr.attribute) + ");\n";
                break;
            }
            case MLIROp::OPT_STEP: {
                for (size_t i = 0; i + 1 < instr.operands.size(); i += 2) {
                    string wt = instr.operands[i], g = instr.operands[i + 1];
                    int64_t r = ids[wt].rows > 0 ? ids[wt].rows : 1;
                    int64_t c = ids[wt].cols > 0 ? ids[wt].cols : 1;
                    int64_t n = r * c;
                    bool use_muon = (r > 1 && c > 1 && std::min(r, c) >= 8);
                    train_text += "  {\n";
                    if (use_muon) {
                        train_text += "    if (ns_cu_reserve(&ctx->d_am" + wt + ", &ctx->d_am" + wt + "_cap, " + to_string(n)
                            + " * sizeof(float), 1)) return;\n";
                        train_text += "    (void)ctx->adam_step;\n";
                        train_text += "    NS_LAUNCH1(ns_muon_ema_kernel, " + to_string(n) + ", ctx->d_am" + wt
                            + ", " + bufv(g) + ", " + to_string(n) + ", " + fmt_float(optim::kMuonMomentum) + "f);\n";
                        train_text += "    ns_orthonom_kernel<<<1, 1>>>(ctx->d_am" + wt + ", " + to_string(r)
                            + ", " + to_string(c) + ");\n";
                        train_text += "    float le = lr * sqrtf((float)std::min(" + to_string(r) + ", "
                            + to_string(c) + "));\n";
                        train_text += "    NS_LAUNCH1(ns_muon_step_kernel, " + to_string(n) + ", "
                            + bufv(wt) + ", ctx->d_am" + wt + ", " + to_string(n) + ", le, "
                            + fmt_float(optim::kMuonDecay) + "f * lr);\n";
                    } else {
                        train_text += "    if (ns_cu_reserve(&ctx->d_am" + wt + ", &ctx->d_am" + wt + "_cap, " + to_string(n)
                            + " * sizeof(float), 1)) return;\n";
                        train_text += "    if (ns_cu_reserve(&ctx->d_av" + wt + ", &ctx->d_av" + wt + "_cap, " + to_string(n)
                            + " * sizeof(float), 1)) return;\n";
                        train_text += "    size_t t = ++ctx->adam_step;\n";
                        train_text += "    float b1t = 1.f - powf(" + fmt_float(optim::kAdamWBeta1) + "f, (float)t), b2t = 1.f - powf(" + fmt_float(optim::kAdamWBeta2) + "f, (float)t);\n";
                        train_text += "    NS_LAUNCH1(ns_adamw_kernel, " + to_string(n) + ", " + bufv(wt)
                            + ", " + bufv(g) + ", ctx->d_am" + wt + ", ctx->d_av" + wt + ", " + to_string(n)
                            + ", lr, b1t, b2t, " + fmt_float(optim::kAdamWBeta1) + "f, " + fmt_float(optim::kAdamWBeta2)
                            + "f, " + fmt_float(optim::kOneMinusBeta1) + "f, " + fmt_float(optim::kOneMinusBeta2)
                            + "f, " + fmt_float(optim::kAdamWEps) + "f, " + fmt_float(optim::kAdamWDecay) + "f * lr);\n";
                    }
                    train_text += "  }\n";
                }
                break;
            }
            default:
                break;
        }
    }

    oss << fwd_text;
    oss << "  if (train_mode) {\n";
    oss << train_text;
    oss << "  }\n";
    oss << "  cudaDeviceSynchronize();\n";
    oss << "  if (train_mode && wout)\n"
        << "    cudaMemcpy(wout, ctx->d_wb, " << weight_total
        << " * sizeof(float), cudaMemcpyDeviceToHost);\n";
    oss << "}\n";
    return oss.str();
}

} // namespace ns
