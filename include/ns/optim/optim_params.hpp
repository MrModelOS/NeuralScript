#pragma once

#include <cstddef>

namespace ns {
namespace optim {

// Canonical optimizer hyperparameters. This is the SINGLE source of truth for
// the generated AOT training cores (CPU and CUDA emitters) and the
// double-precision reference optimizer (MuonOptimizer). Change a value here
// and every backend stays in sync; do not re-hardcode these literals in
// emitters or kernels.
//
// The AOT cores are fused CG expression templates at codegen time: the CPU and
// CUDA emitters stringify these constants into the generated source, and the
// CUDA kernels receive them as arguments instead of baking them in.

constexpr float kMuonMomentum = 0.95f;   // exp-decay on the Muon momentum buffer
constexpr float kMuonDecay = 0.01f;      // Muon coupled weight-decay coefficient
constexpr float kAdamWBeta1 = 0.9f;      // AdamW first-moment decay
constexpr float kAdamWBeta2 = 0.999f;    // AdamW second-moment decay
constexpr float kAdamWEps = 1e-8f;       // AdamW epsilon (denominator floor)
constexpr float kAdamWDecay = 0.01f;     // AdamW decoupled weight-decay coefficient

// Derived counters: (1 - beta) as EXACT float literals. Computing 1.f-0.9f at
// runtime would round to 0.100000024, a different float than 0.1f; the
// emitters/kernels splice these literals in, so keep them explicit to match
// the validated training schedules bit-for-bit.
constexpr float kOneMinusBeta1 = 0.1f;
constexpr float kOneMinusBeta2 = 0.001f;

} // namespace optim
} // namespace ns