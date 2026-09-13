#pragma once
#include "ns/optim/optim_params.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ns {

// Muon optimizer (Phase 4.1).
//
// Muon: for 2D weight matrices, run SGD-style momentum on the gradient, then
// orthonormalize the momentum buffer via Newton-Schulz iterations before
// applying:
//
//     m   = momentum * m + grad
//     P   = orthonormalize(m)            (Newton-Schulz, ~10 iters)
//     W  -= sqrt(rank(W)) * lr * P       (rank = min(rows, cols))
//
// The Newton-Schulz polynomial uses the coefficients (a=3.4445, b=-4.7750,
// c=2.0315) that appear in the reference Muon implementations.
//
// Non-matrix parameters (biases, vectors, scalars) fall back to AdamW.

// Low-precision storage formats (Phase 4.2).
enum class StorageFormat {
    FP32,
    FP8_E4M3,   // e4m3: 1 sign, 4 exponent, 3 mantissa
    FP8_E5M2,   // e5m2: 1 sign, 5 exponent, 2 mantissa
    FP4         // 1 sign, 2 exponent, 1 mantissa (E2M1-ish value set)
};

struct LowPrecision {
    static uint8_t pack(double v, StorageFormat fmt);
    static double unpack(uint8_t bits, StorageFormat fmt);
    static void pack_buf(const double* in, uint8_t* out, size_t n, StorageFormat fmt);
    static void unpack_buf(const uint8_t* in, double* out, size_t n, StorageFormat fmt);
};

// Newton-Schulz iteration for the matrix-sign / polar factor of G (M x N),
// following the reference Muon algorithm:
//   X = G; if M > N: X = X^T; X /= ||X||_F; if M > N: X = X^T
//   for _ in range(steps):
//       A = X @ X.T
//       X = a*X + (b*A + c*A@A) @ X
class NewtonSchulz {
public:
    // In-place; result converges to an orthogonal-stable matrix with the same
    // shape as G (orthonormal rows or columns depending on orientation).
    static void orthonormalize(double* G, size_t M, size_t N, size_t steps = 10);
};

struct MuonParams {
    double lr = 0.02;              // base learning rate
    double weight_decay = optim::kMuonDecay;
    double muon_momentum = optim::kMuonMomentum;
    double adam_beta1 = optim::kAdamWBeta1;
    double adam_beta2 = optim::kAdamWBeta2;
    double adam_eps = optim::kAdamWEps;
};

struct MuonState {
    std::vector<double> muon_m;  // persistent momentum buffer (matrix params)
    std::vector<double> adam_m;
    std::vector<double> adam_v;
    bool is_matrix = false;
};

class MuonOptimizer {
public:
    explicit MuonOptimizer(MuonParams p = {}) : p_(p) {}

    void add_param(double* data, size_t rows, size_t cols);
    void zero_state();

    // Apply one step to a single registered parameter index using the given
    // gradient. Applies Muon for 2D matrix params, AdamW otherwise.
    void step_param(size_t index, const double* grad);

    // Advance the internal step counter (for bias correction). Call this with
    // the 0-based step number before step_param sequence.
    void set_step(size_t s) { step_ = s; }

    const MuonParams& params() const { return p_; }
    size_t param_count() const { return params_.size(); }
    const MuonState& state(size_t i) const { return states_[i]; }

private:
    MuonParams p_;
    struct Param { double* data; size_t rows, cols, n; };
    std::vector<Param> params_;
    std::vector<MuonState> states_;
    size_t step_ = 0;
};

} // namespace ns