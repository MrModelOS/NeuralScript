# NeuralScript

A small end-to-end compiler for a neural-network DSL: `network {...}` definitions
lower through a typed IR to standalone C++ inference kernels or self-contained
C-ABI runtime drivers with **AOT-compiled training** (backward tape
+ Muon / AdamW weight updates) — no runtime graph, no Python-side numerics.
*Version 1.1.0.*

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

Installed (CPU) suite: 11 tests. Optional CUDA backend integration test —
opt-in so plain `ctest` stays green on machines without a CUDA toolkit or GPU:

```sh
cmake -S . -B build-cuda -DNS_ENABLE_CUDA=ON
cmake --build build-cuda -j && ctest --test-dir build-cuda
```

`test_cuda_codegen` compiles the emitted `.cu` with `nvcc`, then runs two
AOT-training loops on a CUDA-capable device and verifies weight writeback and
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
| `tests/` | frontend + codegen + runtime + AOT-training integration tests (11) |

## License

[MIT](LICENSE)