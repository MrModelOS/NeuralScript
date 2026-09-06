// Muon optimizer + low-precision storage tests (Phase 4).
//
// 1. Newton-Schulz orthonormalization: after iterations, rows of G become
//    orthonormal (check G @ G^T ≈ I for a tall matrix).
// 2. Muon on a linear regression: loss decreases over steps.
// 3. Muon momentum orthonormalized update decreases loss on a matrix parameter.
// 4. FP8/FP4 round-trip accuracy bounded by quantization step.
#include "ns/optim/muon.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using namespace ns;

void test_newton_schulz() {
    const size_t M = 6, N = 4; // tall
    std::vector<double> G(M * N, 0.0);
    // Full column rank: identity-like top block + random-ish perturbation.
    double s = 0;
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++) {
            s += 0.7;
            double perturb = 0.1 * std::sin(s * 0.37 + 1.0);
            G[i * N + j] = (i == j ? 1.0 : 0.0) + perturb;
        }

    NewtonSchulz::orthonormalize(G.data(), M, N, 10);

    // For a tall matrix, the columns become orthonormal: G^T G ≈ I_N.
    double worst = 0.0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < M; k++) dot += G[k * N + i] * G[k * N + j];
            double expect = (i == j) ? 1.0 : 0.0;
            worst = std::max(worst, std::fabs(dot - expect));
        }
    }
    std::cout << "newton-schulz orthonorm worst=" << worst << "\n";
    assert(worst < 1e-4);
}

void test_muon_regression() {
    // Fit W (2x3) so that X @ W ≈ Y with X (5x3), Y (5x2).
    // Loss = 0.5 * ||X W - Y||^2. Muon should reduce the loss.
    const size_t R = 5, K = 3, C = 2;
    std::vector<double> X = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        2.0, 1.0, 0.0,
        3.0, 1.0, 4.0,
    };
    std::vector<double> Y = {
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.0, 0.0,
        1.0, 0.0,
    };
    std::vector<double> W(K * C, 0.0);

    auto loss = [&]() {
        double L = 0.0;
        for (size_t i = 0; i < R; i++)
            for (size_t j = 0; j < C; j++) {
                double pred = 0.0;
                for (size_t k = 0; k < K; k++) pred += X[i * K + k] * W[k * C + j];
                double d = pred - Y[i * C + j];
                L += 0.5 * d * d;
            }
        return L;
    };
    auto grad = [&](std::vector<double>& g) {
        g.assign(K * C, 0.0);
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < C; j++) {
                double acc = 0.0;
                for (size_t i = 0; i < R; i++) {
                    double pred = 0.0;
                    for (size_t t = 0; t < K; t++) pred += X[i * K + t] * W[t * C + j];
                    acc += (pred - Y[i * C + j]) * X[i * K + k];
                }
                g[k * C + j] = acc;
            }
    };

    double l0 = loss();
    MuonParams p;
    p.weight_decay = 0.0; // keep pure Muon for loss-decrease test
    MuonOptimizer opt(p);
    opt.add_param(W.data(), K, C);

    std::vector<double> g;
    for (int step = 0; step < 50; step++) {
        opt.set_step(step);
        grad(g);
        opt.step_param(0, g.data());
    }
    double lf = loss();
    std::cout << "muon regression l0=" << l0 << " lf=" << lf << "\n";
    assert(lf < l0);
    // Muon without weight decay should substantially reduce loss on this
    // low-rank-ish problem.
    assert(lf < 0.35 * l0);
}

void test_fp8_roundtrip() {
    const size_t n = 64;
    std::vector<double> in(n);
    double s = 0;
    // Keep inputs bounded away from zero so relative error is meaningful:
    // e4m3 mantissa = 3 bits → step ~2^-4 in [1,2).
    for (auto& v : in) { s += 0.5; v = 0.5 + 0.5 * std::sin(s * 0.11); }
    // FP8 e4m3 has ~3 mantissa bits → relative error ~2^-4 = 6%.
    std::vector<uint8_t> packed(n);
    std::vector<double> out(n);
    LowPrecision::pack_buf(in.data(), packed.data(), n, StorageFormat::FP8_E4M3);
    LowPrecision::unpack_buf(packed.data(), out.data(), n, StorageFormat::FP8_E4M3);
    double max_rel = 0.0;
    for (size_t i = 0; i < n; i++)
        max_rel = std::max(max_rel, std::fabs(out[i] - in[i]) / std::fabs(in[i]));
    std::cout << "fp8 roundtrip max_rel=" << max_rel << "\n";
    assert(max_rel < 0.13);
    // Max absolute error for e4m3 in [1,2) is ~1/16.
    double max_abs = 0.0;
    for (size_t i = 0; i < n; i++)
        max_abs = std::max(max_abs, std::fabs(out[i] - in[i]));
    std::cout << "fp8 roundtrip max_abs=" << max_abs << "\n";
    assert(max_abs < 0.08);

    // FP4: coarse (steps of 0.5); round-trip must stay within the value set.
    LowPrecision::pack_buf(in.data(), packed.data(), n, StorageFormat::FP4);
    LowPrecision::unpack_buf(packed.data(), out.data(), n, StorageFormat::FP4);
    double max4 = 0.0;
    for (size_t i = 0; i < n; i++) {
        double a = std::fabs(out[i]);
        // check each output is in the FP4 magnitude set {0,0.5,1,1.5,2,3,4,6}
        static const double set[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
        bool in_set = false;
        for (int k = 0; k < 8; k++) if (std::fabs(a - set[k]) < 1e-12) in_set = true;
        assert(in_set);
        max4 = std::max(max4, std::fabs(out[i] - in[i]));
    }
    std::cout << "fp4 roundtrip max_abs=" << max4 << "\n";
    // Values in [0.5, 6] with steps ≤ 1 → abs err bounded by ~1.5.
    assert(max4 < 1.6);
}

void test_adamw_bias() {
    // A 1-D bias param should follow the AdamW path.
    std::vector<double> b = {0.0};
    MuonParams p;
    MuonOptimizer opt(p);
    opt.add_param(b.data(), 1, 1); // 1x1: is_matrix = false -> AdamW
    opt.set_step(0);
    std::vector<double> g = {1.0};
    opt.step_param(0, g.data());
    // AdamW does bias-corrected step; b should decrease (lr>0).
    std::cout << "adamw bias b[0]=" << b[0] << "\n";
    assert(b[0] < 0.0);
}

} // namespace

int main() {
    test_newton_schulz();
    test_muon_regression();
    test_fp8_roundtrip();
    test_adamw_bias();
    std::cout << "Optimizer tests PASSED\n";
    return 0;
}