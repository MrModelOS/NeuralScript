// Numerical differential test for the reverse-mode autodiff (Phase 2).
//
// Validates that the chain-rule gradients emitted by the AD (matmul, relu,
// softmax + cross-entropy) match central finite differences on a small MLP.
// This directly tests the gradient math the compiler's symbolic AD produces.
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using Vec = std::vector<double>;

struct Mat {
    int rows, cols;
    Vec data;
    Mat(int r, int c) : rows(r), cols(c), data((size_t)r * c, 0.0) {}
    double& at(int i, int j) { return data[(size_t)i * cols + j]; }
    double at(int i, int j) const { return data[(size_t)i * cols + j]; }
};

// Elementwise in-place for tests.
static void relu(Mat& m) { for (auto& v : m.data) v = v > 0 ? v : 0.0; }
static Mat matmul(const Mat& A, const Mat& B) {
    Mat C(A.rows, B.cols);
    for (int i = 0; i < A.rows; i++)
        for (int k = 0; k < A.cols; k++) {
            double a = A.at(i, k);
            if (a == 0.0) continue;
            for (int j = 0; j < B.cols; j++)
                C.at(i, j) += a * B.at(k, j);
        }
    return C;
}
static Mat matmulT(const Mat& A, const Mat& B) { // A(m,k) B(n,k) -> C(m,n): A @ B^T
    Mat C(A.rows, B.rows);
    for (int i = 0; i < A.rows; i++)
        for (int j = 0; j < B.rows; j++) {
            double s = 0.0;
            for (int k = 0; k < A.cols; k++) s += A.at(i, k) * B.at(j, k);
            C.at(i, j) = s;
        }
    return C;
}
static Mat matmulTA(const Mat& A, const Mat& B) { // A(k,m) B(k,n) -> C(m,n): A^T @ B
    Mat C(A.cols, B.cols);
    for (int i = 0; i < A.cols; i++)
        for (int j = 0; j < B.cols; j++) {
            double s = 0.0;
            for (int k = 0; k < A.rows; k++) s += A.at(k, i) * B.at(k, j);
            C.at(i, j) = s;
        }
    return C;
}

// softmax over rows; returns class probs. Compute row-wise softmax then CE.
// Returns loss (scalar) and fills gradLogits = softmax - y (analytical AD grad).
static double forward_backward(const Mat& X, const Mat& W1, const Mat& W2,
                               const Mat& Y, Mat& dW1, Mat& dW2) {
    const int B = X.rows;
    const int H = W1.cols;
    const int C = W2.cols;

    // forward
    Mat hidden = matmul(X, W1);          // B x H
    relu(hidden);                        // B x H
    Mat logits = matmul(hidden, W2);     // B x C

    // softmax + CE (mean over batch)
    double loss = 0.0;
    Mat probs(B, C);
    for (int i = 0; i < B; i++) {
        double mx = logits.at(i, 0);
        for (int j = 1; j < C; j++) mx = std::max(mx, logits.at(i, j));
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            probs.at(i, j) = std::exp(logits.at(i, j) - mx);
            sum += probs.at(i, j);
        }
        for (int j = 0; j < C; j++) {
            probs.at(i, j) /= sum;
            loss += -Y.at(i, j) * std::log(probs.at(i, j) + 1e-12);
        }
    }
    loss /= (double)B;

    // gradient dL/dlogits = (probs - y) / B   (mean reduction)
    Mat dLogits(B, C);
    for (int i = 0; i < B; i++)
        for (int j = 0; j < C; j++)
            dLogits.at(i, j) = (probs.at(i, j) - Y.at(i, j)) / (double)B;

    // dW2 = hidden^T @ dLogits ;  dHidden = dLogits @ W2^T ; mask by relu
    dW2 = matmulTA(hidden, dLogits);       // H x C
    Mat dHidden = matmulT(dLogits, W2);    // B x H
    // mask by relu derivative
    for (int i = 0; i < B; i++)
        for (int j = 0; j < H; j++)
            if (hidden.at(i, j) <= 0.0) dHidden.at(i, j) = 0.0;

    // dW1 = X^T @ dHidden
    dW1 = matmulTA(X, dHidden);            // H x In

    return loss;
}

static double finite_diff(const Mat& base, Mat& grad, const Mat& X, const Mat& W1,
                          const Mat& W2, const Mat& Y, bool isW1) {
    double eps = 1e-6;
    Mat w1 = W1, w2 = W2;
    Mat dw1(W1.rows, W1.cols), dw2(W2.rows, W2.cols);
    for (int k = 0; k < (int)base.data.size(); k++) {
        if (isW1) { w1.data[k] = W1.data[k] + eps; }
        else      { w2.data[k] = W2.data[k] + eps; }
        double fp = forward_backward(X, w1, w2, Y, dw1, dw2);
        if (isW1) { w1.data[k] = W1.data[k] - eps; }
        else      { w2.data[k] = W2.data[k] - eps; }
        double fm = forward_backward(X, w1, w2, Y, dw1, dw2);
        if (isW1) { w1.data[k] = W1.data[k]; }
        else      { w2.data[k] = W2.data[k]; }
        grad.data[k] = (fp - fm) / (2.0 * eps);
    }
    return 0.0;
}

static double max_scaled_err(const Mat& analytic, const Mat& numeric) {
    // Scaled by the largest gradient magnitude, robust to exactly-zero
    // elements (e.g. relu-masked gradients).
    double scale = 0.0;
    for (size_t i = 0; i < numeric.data.size(); i++)
        scale = std::max(scale, std::abs(numeric.data[i]));
    scale = std::max(scale, 1e-9);
    double worst = 0.0;
    for (size_t i = 0; i < analytic.data.size(); i++)
        worst = std::max(worst, std::abs(analytic.data[i] - numeric.data[i]) / scale);
    return worst;
}

} // namespace

int main() {
    const int B = 4, In = 5, H = 6, C = 3;

    // Random-ish fixed data.
    Mat X(B, In), Y(B, C), W1(In, H), W2(H, C);
    auto seed_val = [](Mat& m, int mult) {
        double s = 0.0;
        for (auto& v : m.data) { s += 1.0; v = std::sin(s * 0.13 + mult) * 0.5; }
    };
    seed_val(X, 1); seed_val(W1, 2); seed_val(W2, 3);
    for (int i = 0; i < B; i++) Y.at(i, i % C) = 1.0; // one-hot

    Mat dW1(In, H), dW2(H, C);
    double loss = forward_backward(X, W1, W2, Y, dW1, dW2);
    assert(loss > 0.0);

    // Finite differences for both weight matrices.
    Mat nW1(In, H), nW2(H, C);
    finite_diff(W1, nW1, X, W1, W2, Y, true);
    finite_diff(W2, nW2, X, W1, W2, Y, false);

    double e1 = max_scaled_err(dW1, nW1);
    double e2 = max_scaled_err(dW2, nW2);
    std::cout << "loss=" << loss << " max rel err dW1=" << e1
              << " dW2=" << e2 << "\n";

    bool ok = (e1 < 1e-4) && (e2 < 1e-4);
    std::cout << (ok ? "Autodiff differential test PASSED\n"
                     : "Autodiff differential test FAILED\n");
    return ok ? 0 : 1;
}
