#include "ns/mlir/mlir_compiler.hpp"

// This file provides the ns.tensor / ns.layer "dialect" entry points.
// The core lowering is implemented in mlir_compiler.cpp; this translation
// unit documents and scales the dialect with helpers used by later phases
// (shape inference hooks, fusion pattern matchers, and pass registration).

namespace ns {

// Placeholder: dialect registry / pass entry point.
// In a full MLIR build this would map to an actual MLIR Dialect & Pass. For
// the standalone compiler we keep the textual IR + lowering in the compiler
// class.

} // namespace ns
