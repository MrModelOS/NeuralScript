#include "ns/optim/optim_params.hpp"
#include "ns/optim/muon.hpp"
#include "ns/training/trainer.hpp"
#include "ns/mlir/mlir_compiler.hpp"
#include "ns/mlir/eval.hpp"
#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include "ns/mlir/fusion.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>

using namespace ns;

// -----------------------------------------------------------------------
// Minimal NumericTrainer usage: just enough to exercise apply_step + lr
// scaling path without generating a real graph (we test lr patching in
// isolation).
// -----------------------------------------------------------------------
static void test_lr_constant() {
    // With kConstant (the default), lr_scale must return 1 for all steps.
    if (optim::kLRSchedule == optim::LRSchedule::kConstant) {
        for (int64_t s : {(int64_t)0, (int64_t)1, (int64_t)50, (int64_t)100,
                            (int64_t)5000, (int64_t)999999}) {
            float v = optim::lr_scale(s);
            assert(std::fabs(v - 1.0f) < 1e-6f && "kConstant lr_scale != 1");
        }
        std::printf("  lr_constant: ok\n");
    } else {
        std::printf("  lr_constant: skipped (non-default schedule)\n");
    }
}

// -----------------------------------------------------------------------
// Independent re-derivation of the cosine-with-warmup schedule, validating
// that the formula encoded in optim_params.hpp is correct.
// -----------------------------------------------------------------------
static void test_cosine_formula() {
    const int64_t W = optim::kLrWarmupSteps;
    const int64_t T = optim::kLrTotalSteps;
    const float   m = optim::kLrMinFactor;

    // Re-implement the formula independently.
    auto expected = [&](int64_t step) -> float {
        if (optim::kLRSchedule == optim::LRSchedule::kConstant) return 1.0f;
        if (W <= 0) return 1.0f;
        if (step < W) return (float)step / (float)W;
        int64_t end = T > W ? T : W + 1;
        int64_t s = step >= end ? end : step;
        float t = (float)(s - W) / (float)(end - W);
        return m + 0.5f * (1.0f - m) * (1.0f + std::cos(3.14159265358979323846 * t));
    };

    if (optim::kLRSchedule == optim::LRSchedule::kConstant) {
        // Still verify consistency under kConstant.
        for (int64_t s : {(int64_t)0, (int64_t)100, T / 2, T, T * 2})
            assert(std::fabs(optim::lr_scale(s) - expected(s)) < 1e-6f);
        std::printf("  cosine_formula (constant fallback): ok\n");
        return;
    }

    // Warmup region: linear 0..1.
    assert(std::fabs(optim::lr_scale(0) - 0.0f) < 1e-6f);
    assert(std::fabs(optim::lr_scale(W / 2) - 0.5f) < 0.01f);
    assert(std::fabs(optim::lr_scale(W) - 1.0f) < 1e-4f);

    // Cosine region: monotonic decreasing (except rounding).
    float prev = 2.0f; // sentinel
    for (int64_t s = W; s <= T; s += std::max<int64_t>(1, (T - W) / 1000)) {
        float v = optim::lr_scale(s);
        assert(v >= m - 1e-5f && "scale below min_factor");
        assert(v <= 1.0f + 1e-5f && "scale above 1");
        assert(v <= prev + 1e-4f && "non-monotonic in cosine region");
        prev = v;
    }

    // At kLrTotalSteps, must reach the floor.
    assert(std::fabs(optim::lr_scale(T) - m) < 1e-4f);

    // Beyond kLrTotalSteps, clamped.
    assert(std::fabs(optim::lr_scale(T + 10000) - m) < 1e-4f);
    assert(std::fabs(optim::lr_scale(999999) - m) < 1e-4f);

    // Consistency check vs independently derived formula.
    for (int64_t s : {(int64_t)0, (int64_t)1, W / 3, W - 1, W, W + 1, T / 2, T - 1, T, T + 1}) {
        float got = optim::lr_scale(s);
        float exp = expected(s);
        assert(std::fabs(got - exp) < 1e-5f);
    }
    std::printf("  cosine_formula: ok\n");
}

// -----------------------------------------------------------------------
// Verify set_lr round-trips correctly and apply_step scales then restores
// the optimizer's base lr.
// -----------------------------------------------------------------------
static void test_set_lr_restore() {
    MuonParams mp;
    mp.lr = 0.1;
    MuonOptimizer opt(mp);
    opt.set_step(0);

    // set_lr persists.
    opt.set_lr(0.2);
    assert(std::fabs(opt.params().lr - 0.2) < 1e-9);
    opt.set_lr(0.1);
    assert(std::fabs(opt.params().lr - 0.1) < 1e-9);

    // Under kConstant, apply_step must not change lr after returning.
    std::string src = R"(
type Bs = Dynamic
network T {
    input:  Tensor[Bs, 2] float32
    output: Tensor[Bs, 1] float32
    layer fc = Dense(in: 2, out: 1, activation: Identity)
    forward(x) { return x -> fc }
    train(x: Tensor[Bs, 2], labels: Tensor[Bs, 1]) -> float32 {
        grad { var loss = mse(x @ fc, labels) }
        return loss
    }
}
)";
    Lexer lex(src); auto toks = lex.tokenize();
    Parser par(toks); Program prog = par.parse_program();
    ShapeChecker chk; chk.check(prog);
    MLIRCompiler mlir; auto mod = mlir.compile(prog);
    FusionPass fuse; fuse.run(mod);

    const MLIRFunction* train_fn = nullptr;
    for (auto& f : mod.functions)
        if (f.is_train) { train_fn = &f; break; }
    if (!train_fn) { std::printf("FAIL: no train function\n"); return; }

    NumericTrainer trainer(*train_fn, mod);
    double xd[] = {1.0, 2.0, -1.0, 3.0};
    trainer.bind_input("x", std::vector<double>(xd, xd + 4), {2, 2});
    trainer.forward();

    MuonParams p2; p2.lr = 0.3;
    MuonOptimizer o2(p2);
    double base = o2.params().lr;
    trainer.apply_step(o2, 0);
    // lr must be restored to base after apply_step.
    assert(std::fabs(o2.params().lr - base) < 1e-9);
    trainer.apply_step(o2, 100);
    assert(std::fabs(o2.params().lr - base) < 1e-9);
    std::printf("  set_lr_restore: ok\n");
}

int main() {
    std::printf("test_lr_schedule:\n");
    test_lr_constant();
    test_cosine_formula();
    test_set_lr_restore();
    std::printf("PASS\n");
    return 0;
}
