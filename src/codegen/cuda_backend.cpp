#include "ns/codegen/codegen.hpp"

// CUDA-specific backend extensions beyond the shared CodeGenerator.
// Currently the generic generator in codegen.cpp emits CUDA kernels;
// this translation unit is reserved for target-specific tuning (Tensor
// Core / MMA instruction selection, fp16/fp8 storage, shared-memory
// tiling) that will plug into the shared pipeline.

namespace ns {

} // namespace ns
