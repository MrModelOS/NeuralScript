#pragma once
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/fusion.hpp"
#include <map>
#include <string>
#include <vector>

namespace ns {

// A concrete tensor buffer for evaluation.
struct TensorBuffer {
    std::vector<double> data;
    std::vector<int64_t> shape; // dims; empty => scalar
    int64_t numel() const;
};

// Minimal numeric interpreter over the MLIR module. Used by the fusion test to
// prove that the fused module computes the identical result as the unfused one
// (and that both match a hand-checked reference).
class ModuleEvaluator {
public:
    // Bind an external value (input tensor or weight) by id.
    void bind(const std::string& id, std::vector<double> data,
              std::vector<int64_t> shape);

    // Evaluate an instruction list starting at `from`, returning the value of
    // `want` (or fn.return_id if empty). Uses module.fused_groups to interpret
    // FUSED instructions.
    bool run(const MLIRFunction& fn, const MLIRModule& mod,
             std::string want, TensorBuffer& out);

    const std::map<std::string, TensorBuffer>& values() const { return values_; }
    const std::vector<std::string>& errors() const { return errors_; }

private:
    std::map<std::string, TensorBuffer> values_;
    std::map<std::string, double> scalars_;
    std::vector<std::string> errors_;

    bool eval_instr(const MLIRInstr& instr, const MLIRModule& mod);
    bool eval_fused(const MLIRInstr& instr, const MLIRModule& mod);

    const TensorBuffer* getv(const std::string& id) const;
    // elementwise linear-index op over one or two buffers
    static void elementwise(const TensorBuffer& a, const TensorBuffer* b,
                            const std::string& op, TensorBuffer& out);
    static void activation(const std::string& act, TensorBuffer& m);
    static void layernorm(TensorBuffer& m);
    static void softmax(TensorBuffer& m);
    static void matmul(const TensorBuffer& a, const TensorBuffer& b, TensorBuffer& c);
};

} // namespace ns
