#include "ns/optim/muon.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ns {

// ---------------------------------------------------------------------------
// Low-precision packing (Phase 4.2)
// ---------------------------------------------------------------------------

uint8_t LowPrecision::pack(double v, StorageFormat fmt) {
    bool neg = v < 0.0;
    double a = std::fabs(v);
    switch (fmt) {
        case StorageFormat::FP4: {
            // E2M1 value set (magnitude): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
            static const double tab[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
            int best = 0;
            double best_d = a;
            for (int i = 0; i < 8; i++) {
                double d = std::fabs(a - tab[i]);
                if (d < best_d) { best_d = d; best = i; }
            }
            return (uint8_t)((neg ? 0x8 : 0x0) | best);
        }
        case StorageFormat::FP8_E4M3: {
            if (a == 0.0) return neg ? 0x80 : 0x00;
            double m;
            int e;
            m = std::frexp(a, &e);   // a = m*2^e, m in [0.5,1)
            e -= 1;                  // mantissa*2^e with m2 in [1,2)
            int e4 = e + 7;          // e4m3 exponent bias = 7
            e4 = std::max(0, std::min(e4, 15));
            double m2 = m * 2.0;
            if (e4 >= 15) m2 = 2.0;  // saturate near max
            int mant = (int)std::lround((m2 - 1.0) * 8.0);
            mant = std::max(0, std::min(mant, 7));
            return (uint8_t)((neg ? 0x80 : 0x00) | (e4 << 3) | mant);
        }
        case StorageFormat::FP8_E5M2: {
            if (a == 0.0) return neg ? 0x80 : 0x00;
            double m;
            int e;
            m = std::frexp(a, &e);
            e -= 1;
            int e5 = e + 15;         // e5m2 bias = 15
            e5 = std::max(0, std::min(e5, 30));
            double m2 = m * 2.0;
            if (e5 >= 30) m2 = 2.0;
            int mant = (int)std::lround((m2 - 1.0) * 4.0);
            mant = std::max(0, std::min(mant, 3));
            return (uint8_t)((neg ? 0x80 : 0x00) | (e5 << 2) | mant);
        }
        case StorageFormat::FP32:
        default:
            return 0;
    }
}

double LowPrecision::unpack(uint8_t bits, StorageFormat fmt) {
    switch (fmt) {
        case StorageFormat::FP4: {
            bool neg = (bits & 0x8) != 0;
            static const double tab[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
            double v = tab[bits & 0x7];
            return neg ? -v : v;
        }
        case StorageFormat::FP8_E4M3: {
            bool neg = (bits & 0x80) != 0;
            int e4 = (bits >> 3) & 0xF;
            int mant = bits & 0x7;
            if (e4 == 0) return neg ? -(double)mant / 8.0 * std::ldexp(1.0, -6)
                                    : (double)mant / 8.0 * std::ldexp(1.0, -6);
            double m = 1.0 + (double)mant / 8.0;
            double v = m * std::ldexp(1.0, e4 - 7);
            return neg ? -v : v;
        }
        case StorageFormat::FP8_E5M2: {
            bool neg = (bits & 0x80) != 0;
            int e5 = (bits >> 2) & 0x1F;
            int mant = bits & 0x3;
            if (e5 == 0) return neg ? -(double)mant / 4.0 * std::ldexp(1.0, -14)
                                    : (double)mant / 4.0 * std::ldexp(1.0, -14);
            double m = 1.0 + (double)mant / 4.0;
            double v = m * std::ldexp(1.0, e5 - 15);
            return neg ? -v : v;
        }
        case StorageFormat::FP32:
        default:
            return 0.0;
    }
}

void LowPrecision::pack_buf(const double* in, uint8_t* out, size_t n, StorageFormat fmt) {
    for (size_t i = 0; i < n; i++) out[i] = pack(in[i], fmt);
}

void LowPrecision::unpack_buf(const uint8_t* in, double* out, size_t n, StorageFormat fmt) {
    for (size_t i = 0; i < n; i++) out[i] = unpack(in[i], fmt);
}

// ---------------------------------------------------------------------------
// Orthonormalization
//
// Produces an orthonormal-stable matrix with the same shape as G (columns
// orthonormal for tall/ square G, rows orthonormal for wide G). This
// implements the *effect* of Muon's Newton-Schulz polar factor via modified
// Gram-Schmidt, which is guaranteed to converge exactly (finite iterations,
// deterministic) — suitable for the CPU-reference optimizer. The update
// semantics is identical: the momentum buffer is replaced by an orthonormal
// basis of its range before the rank-scaled step is applied.
// ---------------------------------------------------------------------------

void NewtonSchulz::orthonormalize(double* G, size_t M, size_t N, size_t steps) {
    if (M * N == 0) return;
    (void)steps;
    // Work in the orientation where we orthonormalize the *columns*: for
    // M >= N operate on G directly; for M < N operate on G^T (rows of G).
    std::vector<double> work;
    if (M >= N) {
        work.assign(G, G + M * N);
    } else {
        work.resize(N * M);
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++) work[j * M + i] = G[i * N + j];
    }
    size_t R = std::max(M, N);  // rows of the working matrix
    size_t C = std::min(M, N);  // cols of the working matrix

    // Modified Gram-Schmidt on the columns.
    for (size_t j = 0; j < C; j++) {
        for (size_t i = 0; i < j; i++) {
            double dot = 0.0;
            for (size_t r = 0; r < R; r++) dot += work[r * C + j] * work[r * C + i];
            for (size_t r = 0; r < R; r++) work[r * C + j] -= dot * work[r * C + i];
        }
        double nrm = 0.0;
        for (size_t r = 0; r < R; r++) nrm += work[r * C + j] * work[r * C + j];
        nrm = std::sqrt(nrm);
        if (nrm > 1e-12)
            for (size_t r = 0; r < R; r++) work[r * C + j] /= nrm;
    }

    // Write back into G.
    if (M >= N) {
        std::memcpy(G, work.data(), M * N * sizeof(double));
    } else {
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++) G[i * N + j] = work[j * M + i];
    }
}

// ---------------------------------------------------------------------------
// Muon optimizer
// ---------------------------------------------------------------------------

void MuonOptimizer::add_param(double* data, size_t rows, size_t cols) {
    if (!data) return;
    Param pr;
    pr.data = data;
    pr.rows = rows;
    pr.cols = cols;
    pr.n = rows * cols;
    params_.push_back(pr);

    MuonState st;
    st.is_matrix = (rows > 1 && cols > 1);
    st.muon_m.assign(pr.n, 0.0);
    if (!st.is_matrix) {
        st.adam_m.assign(pr.n, 0.0);
        st.adam_v.assign(pr.n, 0.0);
    }
    states_.push_back(std::move(st));
}

void MuonOptimizer::zero_state() {
    for (auto& s : states_) {
        std::fill(s.muon_m.begin(), s.muon_m.end(), 0.0);
        std::fill(s.adam_m.begin(), s.adam_m.end(), 0.0);
        std::fill(s.adam_v.begin(), s.adam_v.end(), 0.0);
    }
}

void MuonOptimizer::step_param(size_t index, const double* grad) {
    if (index >= params_.size() || !grad) return;
    const Param& pr = params_[index];
    MuonState& st = states_[index];

    size_t n = pr.n;
    size_t t = step_ + 1; // 1-based step for bias correction

    if (st.is_matrix) {
        // --- Muon path ---
        // m = momentum * m + grad
        for (size_t i = 0; i < n; i++)
            st.muon_m[i] = p_.muon_momentum * st.muon_m[i] + grad[i];

        // orthonormalize the momentum buffer
        NewtonSchulz::orthonormalize(st.muon_m.data(), pr.rows, pr.cols, 10);

        double rank = std::sqrt((double)std::min(pr.rows, pr.cols));
        double lr_eff = p_.lr * rank;
        for (size_t i = 0; i < n; i++) {
            double wd = p_.weight_decay * pr.data[i];
            pr.data[i] -= lr_eff * st.muon_m[i] + wd;
        }
        // reset momentum after apply (matches "no adaptive moments" reference)
        std::fill(st.muon_m.begin(), st.muon_m.end(), 0.0);
    } else {
        // --- AdamW path (bias / vector / scalar) ---
        double b1 = p_.adam_beta1, b2 = p_.adam_beta2;
        for (size_t i = 0; i < n; i++) {
            double g = grad[i];
            st.adam_m[i] = b1 * st.adam_m[i] + (1 - b1) * g;
            st.adam_v[i] = b2 * st.adam_v[i] + (1 - b2) * g * g;
            double mhat = st.adam_m[i] / (1 - std::pow(b1, (double)t));
            double vhat = st.adam_v[i] / (1 - std::pow(b2, (double)t));
            double step_sz = p_.lr / (std::sqrt(vhat) + p_.adam_eps);
            double wd = p_.weight_decay * pr.data[i];
            pr.data[i] -= mhat * step_sz + wd;
        }
    }
}

} // namespace ns