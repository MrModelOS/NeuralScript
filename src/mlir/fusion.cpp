#include "ns/mlir/fusion.hpp"

namespace ns {

// Map a MLIROp to a stable textual op name (used in fused group descriptors).
static std::string op_name(MLIROp op) {
    switch (op) {
        case MLIROp::RELU: return "relu";
        case MLIROp::LEAKY_RELU: return "leaky_relu";
        case MLIROp::SIGMOID: return "sigmoid";
        case MLIROp::TANH: return "tanh";
        case MLIROp::SWISH: return "swish";
        case MLIROp::GELU: return "gelu";
        case MLIROp::SILU: return "silu";
        case MLIROp::IDENTITY: return "identity";
        case MLIROp::SOFTMAX: return "softmax";
        case MLIROp::LAYERNORM: return "layernorm";
        case MLIROp::ELEMENTWISE_BINOP: return "binop";
        default: return "op";
    }
}

bool FusionPass::is_fusible_epilogue(MLIROp op) {
    switch (op) {
        case MLIROp::RELU:
        case MLIROp::LEAKY_RELU:
        case MLIROp::SIGMOID:
        case MLIROp::TANH:
        case MLIROp::SWISH:
        case MLIROp::GELU:
        case MLIROp::SILU:
        case MLIROp::IDENTITY:
        case MLIROp::SOFTMAX:
        case MLIROp::LAYERNORM:
        case MLIROp::ELEMENTWISE_BINOP:
            return true;
        default:
            return false;
    }
}

size_t FusionPass::use_count(const MLIRFunction& fn, const std::string& id) {
    size_t uses = 0;
    for (const auto& instr : fn.instructions) {
        for (const auto& op : instr.operands) {
            if (op == id) uses++;
        }
    }
    return uses;
}

FuseDecision FusionPass::try_fuse(const MLIRFunction& fn, size_t start) {
    const MLIRInstr& leader = fn.instructions[start];
    const std::string& leader_result = leader.result_id;

    // The GEMM result must be consumed exactly once (by the first epilogue)
    // to be safely removed.
    if (use_count(fn, leader_result) != 1) {
        return {start, start, FusionKind::GEMM_CHAIN, leader_result, leader_result};
    }

    size_t end = start + 1;
    std::string consumed = leader_result; // value feeding the current candidate
    FusionKind kind = FusionKind::GEMM_ACTIVATION;
    bool any = false;

    while (end < fn.instructions.size()) {
        const MLIRInstr& cand = fn.instructions[end];
        if (!FusionPass::is_fusible_epilogue(cand.op)) break;
        if (cand.operands.empty() || cand.operands[0] != consumed) break;

        // The candidate result must be consumed by at most once — either by
        // the next fused epilogue, or be the final (terminal) fused output.
        size_t uses = use_count(fn, cand.result_id);
        bool is_last = end + 1 >= fn.instructions.size() ||
                       !FusionPass::is_fusible_epilogue(fn.instructions[end + 1].op) ||
                       fn.instructions[end + 1].operands.empty() ||
                       fn.instructions[end + 1].operands[0] != cand.result_id;
        if (uses > 1 || (!is_last && uses == 0)) break;

        if (cand.op == MLIROp::LAYERNORM) kind = FusionKind::GEMM_LAYERNORM;
        else if (cand.op == MLIROp::ELEMENTWISE_BINOP && any == false && end == start + 1)
            kind = FusionKind::GEMM_ELEMENTWISE;

        consumed = cand.result_id;
        end++;
        any = true;
    }

    if (!any) {
        return {start, start, FusionKind::GEMM_CHAIN, leader_result, leader_result};
    }
    return {start, end, kind, leader_result, consumed};
}

size_t FusionPass::run(MLIRModule& module) {
    decisions_.clear();
    size_t fused_groups = 0;

    for (auto& fn : module.functions) {
        std::vector<MLIRInstr> out;
        out.reserve(fn.instructions.size());
        // Redirect any later consumer of an internal (fused-away) value to the
        // fused result.
        std::unordered_map<std::string, std::string> redirect;
        std::unordered_map<std::string, std::string> fuse_of; // internal -> group

        size_t i = 0;
        while (i < fn.instructions.size()) {
            const MLIRInstr& instr = fn.instructions[i];

            if (instr.op == MLIROp::MATMUL) {
                FuseDecision d = try_fuse(fn, i);
                if (d.end > d.start) {
                    // Build the fused group descriptor.
                    FusedopGroup group;
                    group.result_id = d.fuse_result;
                    group.c = d.leader_result;
                    group.result_type = fn.instructions[d.end - 1].result_type;

                    std::string consumed = d.leader_result;
                    for (size_t j = d.start + 1; j < d.end; j++) {
                        const MLIRInstr& ep = fn.instructions[j];
                        // For binops, use the actual operator symbol so the
                        // evaluator can reproduce the exact elementwise op.
                        std::string nm = (ep.op == MLIROp::ELEMENTWISE_BINOP)
                                             ? ep.attribute : op_name(ep.op);
                        group.ops.push_back(nm);
                        for (size_t k = 1; k < ep.operands.size(); k++) {
                            group.epilogue_operands.push_back(ep.operands[k]);
                            if (ep.op == MLIROp::ELEMENTWISE_BINOP && ep.attribute == "+") {
                                group.has_bias = true;
                                group.bias_operand = ep.operands[k];
                            }
                        }
                        // Redirect future consumers of this internal value to
                        // the final fused result.
                        fuse_of[ep.result_id] = d.fuse_result;
                    }
                    fuse_of[d.leader_result] = d.fuse_result;
                    redirect = fuse_of;
                    module.fused_groups.push_back(std::move(group));
                    fused_groups++;

                    // Emit a single FUSED instruction. Its operands are the
                    // GEMM inputs (A, B) followed by any epilogue operands
                    // (e.g. the bias vector).
                    MLIRInstr fused(MLIROp::FUSED, d.fuse_result);
                    fused.operands = instr.operands; // A, B from the MATMUL leader
                    for (auto& g : module.fused_groups.back().epilogue_operands)
                        fused.operands.push_back(g);
                    fused.result_type = fn.instructions[d.end - 1].result_type;
                    fused.comment = "fused " + std::to_string(d.end - d.start) + " ops";
                    out.push_back(fused);

                    i = d.end;
                    continue;
                }
            }

            // Copy, rewriting operands that were fused away.
            MLIRInstr copy = instr;
            for (auto& op : copy.operands) {
                auto it = redirect.find(op);
                if (it != redirect.end()) op = it->second;
            }
            out.push_back(std::move(copy));
            i++;
        }

        fn.instructions = std::move(out);
    }

    return fused_groups;
}

} // namespace ns
