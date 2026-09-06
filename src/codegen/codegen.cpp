#include "ns/codegen/codegen.hpp"
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>

namespace ns {

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

std::string CodeGenerator::gen_cuda(const MLIRModule& module, const CodegenOptions& opts) {
    std::ostringstream oss;
    oss << "// Generated by NeuralScript compiler\n"
        << "// Target: CUDA\n"
        << "#include <cuda_runtime.h>\n"
        << "#include <math.h>\n"
        << "#include <stdio.h>\n\n";

    // First pass: collect all kernels referenced by ops
    for (auto& fn : module.functions) {
        for (auto& instr : fn.instructions) {
            switch (instr.op) {
                case MLIROp::MATMUL:
                    if (instr.operands.size() >= 2) {
                        oss << emit_gemm_kernel(instr.operands[0], instr.operands[1],
                                                instr.result_id, instr.result_type);
                    }
                    break;
                case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
                case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
                case MLIROp::SILU: case MLIROp::IDENTITY: {
                    std::string act;
                    switch (instr.op) {
                        case MLIROp::RELU: act = "ReLU"; break;
                        case MLIROp::LEAKY_RELU: act = "LeakyReLU"; break;
                        case MLIROp::SIGMOID: act = "Sigmoid"; break;
                        case MLIROp::TANH: act = "Tanh"; break;
                        case MLIROp::SWISH: act = "Swish"; break;
                        case MLIROp::GELU: act = "GELU"; break;
                        case MLIROp::SILU: act = "SiLU"; break;
                        default: act = "Identity"; break;
                    }
                    if (!instr.operands.empty()) {
                        oss << emit_activation_kernel(act, instr.operands[0],
                                                      instr.result_id, instr.result_type);
                    }
                    break;
                }
                case MLIROp::ELEMENTWISE_BINOP:
                    if (instr.operands.size() >= 2) {
                        oss << emit_elementwise_kernel(instr.operands[0], instr.operands[1],
                                                       instr.result_id, instr.result_type,
                                                       instr.attribute);
                    }
                    break;
                case MLIROp::LAYERNORM:
                    if (!instr.operands.empty()) {
                        oss << emit_layernorm_kernel(instr.operands[0], instr.result_id,
                                                     instr.result_type);
                    }
                    break;
                case MLIROp::SOFTMAX:
                    if (!instr.operands.empty()) {
                        oss << emit_softmax_kernel(instr.operands[0], instr.result_id,
                                                   instr.result_type);
                    }
                    break;
                default: break;
            }
        }
    }

    // Emit host launcher function
    std::string dtype = dtype_c_name(Dtype::Float32);
    oss << "extern \"C\" void " << opts.function_name << "(\n";
    // gather all buffer names referenced
    std::vector<std::string> bufs;
    for (auto& fn : module.functions) {
        for (auto& instr : fn.instructions) {
            for (auto& op : instr.operands) {
                bool found = false;
                for (auto& b : bufs) if (b == op) { found = true; break; }
                if (!found) bufs.push_back(op);
            }
            if (!instr.result_id.empty()) {
                bool found = false;
                for (auto& b : bufs) if (b == instr.result_id) { found = true; break; }
                if (!found) bufs.push_back(instr.result_id);
            }
        }
    }
    oss << "    const " << dtype << "* input,\n    " << dtype << "* output";
    for (size_t i = 0; i < bufs.size(); i++) {
        (void)i;
    }
    oss << ") {\n";

    oss << "  // Kernel launch sequence\n";
    for (auto& fn : module.functions) {
        for (auto& instr : fn.instructions) {
            switch (instr.op) {
                case MLIROp::MATMUL:
                    if (instr.operands.size() >= 2) {
                        oss << "  // launch gemm_" << instr.operands[0] << "_" << instr.operands[1] << "\n";
                    }
                    break;
                case MLIROp::RELU: case MLIROp::LEAKY_RELU: case MLIROp::SIGMOID:
                case MLIROp::TANH: case MLIROp::SWISH: case MLIROp::GELU:
                case MLIROp::SILU: case MLIROp::IDENTITY:
                    oss << "  // launch activation\n";
                    break;
                default: break;
            }
        }
    }
    oss << "  (void)input; (void)output;\n";
    oss << "  printf(\"[NeuralScript] compiled forward() stub ==\\n\");\n";
    oss << "}\n";
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
    if (!fn) for (auto& f : module.functions) if (f.name != "main" && f.name != "train_step") { fn = &f; break; }
    if (!fn) for (auto& f : module.functions) if (f.name != "main") { fn = &f; break; }
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
    // Mark input = first matmul's A operand.
    bool marked = false;
    for (auto& instr : fn->instructions) {
        if (marked) break;
        bool is_gemm = instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED;
        if (is_gemm && instr.operands.size() >= 2 && !ids[instr.operands[0]].is_weight) {
            ids[instr.operands[0]].is_input = true;
            marked = true;
        }
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
        << "#include <vector>\n"
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

    // Buffer declarations.
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
                if (Ks <= 0 || Ns <= 0) { body += "  // matmul " + id + ": B shape unknown, skipped\n"; break; }
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
                if (Ks <= 0 || Ns <= 0) { body += "  // fused " + id + ": B shape unknown, skipped\n"; break; }
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

        // Static input/output geometry: first GEMM's K and final result columns.
        int64_t in_cols = -1, out_cols = -1;
        for (auto& instr : fn->instructions) {
            if (instr.op != MLIROp::TENSOR_ALLOC && !instr.result_id.empty() && instr.result_id == fn->return_id)
                out_cols = ids[instr.result_id].cols;
        }
        for (auto& instr : fn->instructions) {
            bool gemm = instr.op == MLIROp::MATMUL || instr.op == MLIROp::FUSED;
            if (gemm && instr.operands.size() >= 2 && in_cols < 0) in_cols = ids[instr.operands[1]].rows;
            if (gemm && instr.operands.size() >= 2) out_cols = ids[instr.operands[1]].cols;
        }
        if (in_cols < 0) in_cols = 1;
        if (out_cols < 0) out_cols = 1;

        oss << "\n// ---- C-ABI (host entry points) ----\n";
        oss << "#include <cstring>\n";
        oss << "typedef struct ns_model { float* w; size_t n; } ns_model;\n"
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
        oss << "    ns_model* m = new ns_model{ new float[ns_weight_total], ns_weight_total };\n";
        oss << "    if (!m->w) { delete m; return nullptr; }\n";
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
        oss << "extern \"C\" const ns_weight_layout* ns_model_layout(const ns_model* m) {\n";
        oss << "    (void)m; return &ns_layout;\n";
        oss << "}\n\n";
        oss << "extern \"C\" void ns_free(ns_model* m) {\n";
        oss << "    if (!m) return; delete[] m->w; delete m;\n";
        oss << "}\n";
    }

    return oss.str();
}
} // namespace ns
