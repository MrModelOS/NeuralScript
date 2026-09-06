#include "ns/parser/ast.hpp"

// Type-system helpers used by the shape checker.
// This file centralises dtype compatibility and dimension-manipulation
// utilities so later compiler phases (codegen, fusion) share one source.

namespace ns {

// Highest-precedence dtype promotion for elementwise binary ops.
// e.g. float16 + float32 -> float32 (and so on).
Dtype promote_dtype(Dtype a, Dtype b) {
    // Simple ranking: Bool < Int-ish < FP8 < FP16 < FP32 < FP64
    auto rank = [](Dtype d) -> int {
        switch (d) {
            case Dtype::Bool: return 0;
            case Dtype::Int8: return 1;
            case Dtype::Int16: return 2;
            case Dtype::Int32: return 3;
            case Dtype::Int64: return 4;
            case Dtype::FP4: return 5;
            case Dtype::FP8: return 6;
            case Dtype::Float16: return 7;
            case Dtype::Float32: return 8;
            case Dtype::Float64: return 9;
        }
        return 8;
    };
    return rank(a) >= rank(b) ? a : b;
}

} // namespace ns
