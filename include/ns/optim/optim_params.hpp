#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

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

// ---- Learning-rate schedules (compiled into the AOT runtime driver). ----
// The generated ns_runtime_train_step wrapper scales the caller's `lr` by a
// schedule multiplier that depends only on the monotonically increasing step
// counter (capped in the wrapper, not in the fused CG kernels). Choosing
// kConstant matches the legacy behavior exactly.
enum class LRSchedule : int {
    kConstant = 0,        // multiplier always 1.0
    kCosineWithWarmup = 1 // linear warmup then cosine decay to kLrMinFactor
};

// Global schedule chosen at compile time. Rebuilding changes the compiled
// driver; there is no runtime graph to edit.
constexpr LRSchedule kLRSchedule = LRSchedule::kConstant;

constexpr int64_t kLrWarmupSteps = 100;    // linear ramp 0 -> 1 over this many steps
constexpr int64_t kLrTotalSteps = 10000;   // cosine decay length (per-process)
constexpr float kLrMinFactor = 0.05f;      // floor for the cosine tail

// Multiplier applied by the runtime wrapper: warmup climbs 0..1, then a
// half-cosine decays toward kLrMinFactor. kConstant always returns 1.
inline float lr_scale(int64_t step) {
    if (kLRSchedule == LRSchedule::kConstant) return 1.0f;
    if (step < kLrWarmupSteps)
        return kLrWarmupSteps > 0 ? (float)step / (float)kLrWarmupSteps : 1.0f;
    const int64_t end = kLrTotalSteps > kLrWarmupSteps ? kLrTotalSteps : kLrWarmupSteps + 1;
    const int64_t s = step >= end ? end : step;
    const float t = (float)(s - kLrWarmupSteps) / (float)(end - kLrWarmupSteps);
    return kLrMinFactor + 0.5f * (1.0f - kLrMinFactor) * (1.0f + cosf(3.14159265f * t));
}

} // namespace optim
} // namespace ns