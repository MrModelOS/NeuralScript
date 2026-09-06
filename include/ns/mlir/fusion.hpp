#pragma once
#include "ns/mlir/mlir_compiler.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace ns {

// Kernel fusion pass (Phase 3).
//
// Operates on the high-level MLIR module and groups elementwise epilogues
// (activations, layer-norm, dropout) onto their single-consumer GEMM producer
// into a single `FUSED` op. This models the classic "GEMM + activation" /
// "GEMM + layernorm" fusion used to avoid intermediate device round-trips.
//
// Fusion is only applied when:
//   - the epilogue elementwise op consumes the GEMM result
//   - the GEMM result has exactly one use (no other consumers that would
//     require materializing the middle buffer)
//   - the epilogue chain is a pure elementwise-style tail (no branch/loop)
//
// The pass is safe by construction: a FUSED op is a pure rewriting that
// preserves the same dataflow (the numeric evaluator treats FUSED exactly as
// the composition of its parts).

enum class FusionKind {
    GEMM_ACTIVATION,   // GEMM -> single activation
    GEMM_LAYERNORM,    // GEMM -> layernorm
    GEMM_ELEMENTWISE,  // GEMM -> one elementwise binop (e.g. bias add)
    GEMM_CHAIN         // GEMM -> multiple elementwise tail ops
};

struct FuseDecision {
    size_t start;               // index of the leader (MATMUL)
    size_t end;                 // index of the last fused op (exclusive)
    FusionKind kind;
    std::string leader_result;  // GEMM result id
    std::string fuse_result;    // final result id of the fused run
};

class FusionPass {
public:
    // Run the pass in place; returns number of fusion groups introduced.
    size_t run(MLIRModule& module);

    const std::vector<FuseDecision>& decisions() const { return decisions_; }

    static bool is_fusible_epilogue(MLIROp op);

private:
    std::vector<FuseDecision> decisions_;

    // Count uses of a value id across a function.
    static size_t use_count(const MLIRFunction& fn, const std::string& id);

    // Attempt to fuse the run starting at `start` (must be a MATMUL).
    FuseDecision try_fuse(const MLIRFunction& fn, size_t start);
};

} // namespace ns
