# NeuralScript

A small end-to-end compiler for a neural-network DSL: `network {...}` definitions
lower through a typed IR to standalone C++ inference kernels or self-contained
C-ABI runtime drivers with **AOT-compiled training** (backward tape
+ Muon / AdamW weight updates) — no runtime graph, no Python-side numerics.
*Version 1.3.0.*

## Highlights

- DSL → lexer → parser → type/shape checker → MLIR-like dialect → codegen
- Network pipelines lower to `matmul + activation + dropout` chains with
  weight tensors owned by named layers (`fc1_w`, `fc2_w`, ...)
- CPU codegen emits self-contained C++ that compiles without the frontend
- `--runtime` appends a C-ABI driver (`ns_runtime_init` / `ns_eval_infer` /
  `ns_model_layout` / `ns_model_get_weights` / `ns_weight_count_static` /
  `ns_free`) — see `include/ns/runtime/ns_runtime.h`
- In-network `train()` methods compile the backward tape ahead of time and
  share the **same weight blob** as inference (`ns_runtime_train_step`,
  `ns_objective_loss`)
- Numeric-gradient training (`NumericTrainer`) wired into a Muon optimizer (with
  AdamW fallback), verified to converge on real MLPs (XOR: loss → ~0)
- AOT training cores (CPU + CUDA) each get a fused optimizer step generated
  ahead of time; all optimizer hyperparameters live in one place,
`include/ns/optim/optim_params.hpp`

## v1.2: data movement, attention, MoE

The forward pipeline now accepts the full feature-lowering stack beyond
`matmul + activation + dropout`:

- **Data-movement ops** (CPU `ns_*` kernels + CUDA kernels, both static-weight
  and runtime-rows variants): `concat(a, b, axis)`, `transpose(t)`,
  `reshape(t, r, c)`, `slice(t, axis, start, stop)`, `index(t, axis, i0, i1, …)`
  and `scatter(t, axis, upd, i0, i1, …)` — usable directly in `forward()`
  bodies via ordinary function-call syntax.
- **`Embedding(vocab_size, d_model)`** layer: index tensor ([B, seq]) → row
  lookups ([B, seq, d_model]).
- **`Attention(d_model, heads)` / `MultiHeadAttention`** layer: dense
  Q/K/V/O projections + scaled dot-product attention + per-head softmax, with a
  fused CPU kernel and an SM-tiled CUDA core (`ns_attention_core_kernel`).
- **`LayerNorm()`** normalization layer ([…, last] covariance, eps within).
  `attention → layernorm` (residual sandwich) is the intended v1.2 block.
- **`MoE(d_model, num_experts)` / `MixtureOfExperts`** router: gate logits
  `x @ Wg`, softmax over experts, per-token **top-1** dispatch, and the chosen
  expert's `[D,D]` matrix applied to the token scaled by its routing weight —
  fused as `ns_moe_fwd` / `ns_moe_kernel`.
- `concat`/`transpose`/`reshape`/`slice`/`index`/`scatter` lower through the
  parser's function-call path (new instruction forms `ns.concat`,
  `ns.slice`, `ns.index`, `ns.scatter`, …), so they typecheck, fuse and
  codegen like any other v1.2 head.

## v1.3: AOT backprop through the v1.2 layers

The `grad{}` lowering is no longer limited to `matmul + activation + dropout`.
The reverse-mode scan now lowers the forward-only layers above to real
gradient instructions, so they can be used in AOT training bodies:

- **`LayerNorm()`** → `LAYERNORM_GRAD`: recomputed per-row `γ = mean((x−x̄)·d)`,
  `dx = (d − d̄ − (x−x̄)·γ)·1/√(var+eps)` — verified by
  `test_layernorm_train` (XOR + LayerNorm between layers converges to loss ~0,
  4/4).
- **`Embedding()`** → `EMBEDDING_GRAD_W`: scatter-add of `dOut` rows into
  `dW` by the (non-differentiable) token index — verified by
  `test_embedding_train` (token→class, 8/8).
- **`MoE()`** → `MOE_GRAD_X` / `MOE_GRAD_WG` / `MOE_GRAD_WE`: each backward
  kernel recomputes the fused routing (softmax over the top-1 expert, gate-grad
  `p⊙(dp−Σp·dp)`, expert-grad `xᵀ⊗(p·d)`) — verified by `test_moe_train`
  (2-expert token→class, 16/16).
- All four have CPU (`ns_*_grad` / `ns_attention_bwd`) and CUDA kernels
  (`ns_*_grad_kernel` / `ns_attention_grad_kernel`) and forward cases in both
  `ns_train_core` CPU and CUDA emit, so the same train body compiles for either
  backend. The optimizer steps on the MoE gate and expert weights separately
  (`moe_g_w`, `moe_e_w`) and on the four attention projection matrices
  (`attn_q_w`, `attn_k_w`, `attn_v_w`, `attn_o_w`). Attention is verified by
  `test_attention_train` (copy-first-token over a 16-token sequence, 16/16),
  and the full stack trains end-to-end in `test_transformer_train`
  (emb→attn→ln→mlp→CE, copy-first-token, 16/16) — both CPU and CUDA.

## v1.3: LR schedules, checkpoints, examples

- **Compiled-in LR schedule** (`include/ns/optim/optim_params.hpp`): the
  `ns_runtime_train_step` wrapper (CPU + CUDA) applies a schedule multiplier to
  the caller's base `lr` before entering `ns_train_core`, so the decay logic
  is baked into the AOT binary. Two schedules: `kConstant` (default, multiplier
  always 1 — identical to legacy behavior) and `kCosineWithWarmup` (linear warmup
  then half-cosine decay to `kLrMinFactor`). The library-side
  `NumericTrainer::apply_step` scales the MuonOptimizer's `lr` by the same
  `lr_scale()` at each step, keeping the reference and compiled paths in parity.
  Unit-tested by `test_lr_schedule` (formula + `set_lr` round-trip).
- **Checkpoint save/load** (`ns_save_checkpoint`, `ns_load_checkpoint`): emitted
  in every self-contained runtime driver alongside `ns_free`. Format: 4-byte
  magic `NSM1` (weights only) or `NSM2` (weights + MoE liveness mask, see
  below), `size_t` float count, raw host-endian weights. CUDA load re-syncs the
  device weight buffer via `cudaMemcpy`. Verified in `test_runtime` (CPU
  round-trip) and in `test_cuda_codegen` (device re-sync through the CUDA
  attention path).
- **`examples/transformer.ns`**: a minimal copy-first-token transformer
  (emb→attn→ln→mlp→CE, 2240 params) demonstrating the full v1.3 layer stack
  and the `train()` method; usable with `nsc --cpp --runtime` or `--cuda --runtime`.

Core tests: `test_transformer` (embedding→attention→layernorm vs a scalar
reference), `test_datamove` (programmatic embedding→matmul→transpose→concat→
reshape→layernorm), `test_dsl_datamove` (parser path for slice/index/scatter/
concat/transpose/reshape), `test_moe` (router vs a scalar reference); each also
`nvcc -c`-validates the emitted CUDA.

## v1.3: FFN MoE experts and dynamic expert lifecycle

The MoE layer now uses real **FFN experts** and supports growing/shrinking the
live-expert set at runtime, all over a static AOT graph:

- **FFN experts**: each expert is a two-layer `D → H → D` block with GELU
  (`H = 4·D` by default, or set `ffn_dim` explicitly), with weights
  `moe_e1_w [E, D, H]` and `moe_e2_w [E, H, D]` beside the router
  `moe_g_w [D, E]`. Seven fused kernels (`ns_moe_fwd` +
  `ns_moe_grad_x/wg/we1/we2`) implement forward and backward, refactorable
  identically on CPU and CUDA:
  `forward: out[i,j] = p · Σ_h gelu(x·We1[b])·We2[b]`,
  `dOut → x, Wg, We1, We2` via the same recomputed routing.
- **Live expert mask**: an `active[]` liveness byte-vector lives in the model
  context (`moe_active` on CPU, `h_moe_a`/`d_moe_a` on CUDA). `NULL`/zero mask
  means “all alive”; otherwise the router softmax is computed **over live
  experts only**, so routing probabilies renormalize as experts die or are born.
- **`initial_experts` (K0)**: compile-time attribute (`MoE(d_model: 8,
  num_experts: 4, ffn_dim: 32, initial_experts: 1)`). The first `K0` experts
  start alive; the rest are dormant but fully allocated in the fused graph.
- **Expert lifecycle C ABI** (emitted only when the model has a MoE layer;
  one MoE layer per model):
  - `ns_expert_count(m)` — number of live experts.
  - `ns_expert_birth(m, n)` — copy a random live expert's router column + FFN
    slices with ±1e-2 noise into the first `n` dormant slots (up to capacity).
  - `ns_expert_merge(m, a, b)` — average two live experts into `a`, retire `b`.
  - `ns_expert_kill(m, k)` — retire live expert `k` (optional pre-step pruning).
  CPU operates on `ns_model.w` directly; CUDA updates `h_w` then re-syncs the
  `d_wb` / `d_moe_a` device buffers, so the next `ns_runtime_train_step` or
  `ns_eval_infer` sees the new expert set immediately.
- **NSM2 checkpoints**: `ns_save_checkpoint` now writes magic `NSM2` = raw
  weights + `{n_layers=1, capacity, <capacity> mask bytes}` on MoE models;
  `ns_load_checkpoint` restores both and re-syncs the GPU. Legacy `NSM1` files
  still load (all experts assumed alive).
- Verified by `test_moe` (FFN reference, live-only softmax with a mutated
  mask; CPU vs naive + nvcc), `test_moe_lifecycle` (K=1 growth by birth,
  merge/kill, NSM2 round-trip, NSM1 fallback), `test_moe_train`
  (16/16 token→class via AOT backprop through the FFN MoE), and the MOEX net
  in `test_cuda_codegen` (MoE training + lifecycle + checkpoint on device).

## Build

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires a C++20 compiler. Optional sanitizer run:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j && ctest --test-dir build-asan
```

Installed (CPU) suite: 21 tests. Optional CUDA backend integration test —
opt-in so plain `ctest` stays green on machines without a CUDA toolkit or GPU:

```sh
cmake -S . -B build-cuda -DNS_ENABLE_CUDA=ON
cmake --build build-cuda -j && ctest --test-dir build-cuda
```

`test_cuda_codegen` compiles each emitted `.cu` with `nvcc`, then runs an
AOT-training loop on a CUDA-capable device and verifies weight writeback and
`ns_objective_loss`:

- **AdamW path** — XOR MLP (2×16→16×2, weights have `min(r,c)=2 < 8` so the
  hybrid optimizer falls back to AdamW): 400 steps, loss ~0.71 → ~0.00009,
  4/4 accuracy. A host-side CE check confirms the objective value.
- **Muon path** — MUONX (8×8→8×8, weights have `min(r,c)=8 ≥ 8` so the
  optimizer takes the Muon branch: momentum EMA, `ns_orthonom_kernel`
  Gram-Schmidt on device, scaled step): 30000 steps at lr=0.0005 converge to
  8/8 on the identity-on-steps task, with the identical schedule also run
  through the CPU reference so both implementations must agree qualitatively.
  GPU/CPU loss trajectories over a bounded window agree to ~1e-3; the fp
  summation order drifts weights by ~1 after 30k full-batch steps.
- **Attention path** — ATTEND (embedding 8×16 → self-attention d=16, h=4 →
  linear 16×4, all weights Muon): copy-first-token over a 16-token sequence
  fed as a single input (S=N), 12000 steps at lr=0.01, loss
  ~1.386 → ~0.00000, 16/16 device accuracy through `ns_attention_core_kernel`
  + `ns_attention_grad_kernel`.
- **Transformer path** — TRANSFORM (emb 8×16 → attention d=16/h=4 →
  LayerNorm → MLP 16→32→16 → linear 16×4, 2240 params): same first-token task,
  15000 steps at lr=0.007, loss ~1.384 → ~0.00000, 16/16 device accuracy,
  exercising `ns_layernorm_grad_kernel` together with the attention and MLP
  backward paths. Also exercises the checkpoint round-trip: after training,
  `ns_save_checkpoint` → reload into an untrained model → device eval confirms
  accuracy.

Without nvcc or a GPU the test self-skips (exit 77, reported as SKIPPED by
CTest). GPU-heavy runs are separated into `.github/workflows/cuda.yml`
(self-hosted GPU runner); the default PR pipeline is CPU-only. The generated
CUDA is a full runtime, not a stub: device kernels — including a shared-memory
tiled GEMM (16×16 tile, K-strip in padded shared memory, bounds-checked for
arbitrary/unaligned M,K,N) — the C-ABI driver (`ns_runtime_init`,
`ns_runtime_train_step`, `ns_objective_loss`, `ns_eval_infer`,
`ns_model_get_weights`, `ns_free`) and the AOT `ns_train_core`. Device kernel
sources are owned by `src/codegen/cuda_backend.cpp` (the target-specific
layer, spliced by `CodeGenerator::gen_cuda`), the staging ground for MMA /
mixed-precision hardware tuning. A host driving the same ABI can be
AOT-compiled with `nvcc ns_cuda_kernels.cu host.cpp -o app` — no
libNeuralScript runtime needed.

Installable (exports `NeuralScript::ns_core`, headers, `find_package`
config):

```sh
cmake --install build --prefix /opt/ns
cmake -DCMAKE_PREFIX_PATH=/opt/ns /path/to/your/app
```

## Usage

```sh
build/nsc examples/mlp.ns --check    # static shape verification
build/nsc examples/mlp.ns --mlir     # lower to the dialect (forward + _train)
build/nsc examples/mlp.ns --cpp      # standalone C++ inference
build/nsc examples/mlp.ns --cpp --runtime   # ... + C-ABI driver with training
```

## Language: `train()` methods

A network may declare a training method alongside `forward`:

```
train(batch_x: Tensor[Batch, Features],
      batch_y: Tensor[Batch, Classes]) -> float32 {
    grad {
        var hidden = batch_x @ fc1          // layer weights bind directly
        var act    = relu(hidden)           // activation function calls
        var preds  = act @ fc2
        var loss   = cross_entropy(preds, batch_y)
    }
    return loss
}
```

Semantics:

- The `grad { }` body is the forward pass; the compiler runs a reverse-mode
  scan over it and appends gradient + optimizer instructions to the lowered
  `<network>_train` function. The `return` value is the scalar loss.
- Layer names (`fc1`, `fc2`) refer to the layer weight tensors (`fc1_w`,
  `fc2_w`) — one weight set per layer, shared with inference. Top-level
  `train_step`/`infer` functions with duplicated `w1/w2` are not needed.
- Supported in the train body: `@` matrix multiply, activations
  (`relu`, `leaky_relu`, `gelu`, `sigmoid`, `tanh`, `silu`/`swish`,
  `identity`, `dropout`), and `cross_entropy`.
- Optimizer: Muon for matrices with rank ≥ 8 (Newton-Schulz-equivalent
  orthonormalization), bias-corrected AdamW otherwise; weight decay `λ = 0.01`
  is scaled by the learning rate (`θ ← θ − lr·step − λ·lr·θ`). Every
  hyperparameter is a named constant in `include/ns/optim/optim_params.hpp`.

## C-ABI training

With `--runtime`, networks that define `train()` get two extra entry points
(both operating on the same weights-as-inference `ns_model*`):

```c
/* forward + backward + one optimizer step; writes updated weights back */
int ns_runtime_train_step(ns_model* m, const float* input, const float* labels,
                          size_t input_numel, float* loss_out, float lr);

/* forward-only mean cross-entropy loss; never mutates weights */
int ns_objective_loss(ns_model* m, const float* input, const float* labels,
                      size_t input_numel, float* loss_out);

/* copy the whole weight blob out of the model (inference layout) */
int ns_model_get_weights(const ns_model* m, float* out, size_t n);
```

A host training loop is just:

```c
float loss;
for (int epoch = 0; epoch < N; epoch++)
    ns_runtime_train_step(m, features, labels, batch * in_cols, &loss, lr);
```

`labels` are one-hot rows of `in_cols * out_cols` per pattern; the returned
loss is the mean over the batch.

## Layout

| Path | What it is |
| --- | --- |
| `src/lexer`, `src/parser`, `src/typechecker` | frontend |
| `src/mlir` | dialect, compiler (incl. AOT backward lowering), fusion, numerical eval |
| `src/codegen` | CPU + CUDA backends, grad kernels, train core, C-ABI driver emission |
| `src/codegen/cuda_backend.cpp` | CUDA-specific layer: device kernel sources (incl. tiled shared-mem GEMM), host MM tricks, AdamW/Muon/orthonom, runtime utils |
| `src/optim`, `src/training` | Muon/AdamW reference optimizer, `NumericTrainer` |
| `include/ns/optim/optim_params.hpp` | canonical optimizer hyperparameters (single source of truth) |
| `include/ns/runtime/ns_runtime.h` | canonical C-ABI for hosts (training entry points) |
| `tests/` | frontend + codegen + runtime + AOT-training integration tests (21) |

## License

[MIT](LICENSE)