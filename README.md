# NeuralScript

A small end-to-end compiler and quantized-body compiler for a neural-network DSL:
`samples`-style `network {...}` definitions lower through a typed IR to standalone C
(inference) or self-contained C-ABI runtime drivers, with numeric autodiff that feeds
a Muon / AdamW optimizer for actual training.

## Highlights

- DSL → lexer → parser → type/shape checker → MLIR-like dialect → codegen
- Network pipelines lower to `matmul + activation + dropout` chains with
  weight tensors owned by named layers (`fc1_w`, `fc2_w`, ...)
- CPU codegen emits self-contained C++ that compiles without the frontend
- `--runtime` appends a C-ABI driver (`ns_runtime_init` / `ns_eval_infer` /
  `ns_model_layout` / `ns_free`) — see `include/ns/runtime/ns_runtime.h`
- Numeric reverse-mode autodiff wired into a Muon optimizer (with AdamW fallback),
  verified to converge on real MLPs (XOR: loss → ~0)

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

## Usage

```sh
build/nsc examples/mlp.ns --mlir     # lower to the dialect
build/nsc examples/mlp.ns --cpp      # standalone C++ inference
build/nsc examples/mlp.ns --cpp --runtime   # ... + C-ABI driver
```

## Layout

| Path | What it is |
| --- | --- |
| `src/lexer`, `src/parser`, `src/typechecker` | frontend |
| `src/mlir` | dialect, compiler, fusion, numerical eval (`ModuleEvaluator`) |
| `src/codegen` | CPU + CUDA backends, C-ABI driver emission |
| `src/autodiff`, `src/optim`, `src/training` | reverse-mode autodiff, Muon/AdamW, `NumericTrainer` |
| `include/ns/runtime/ns_runtime.h` | canonical C-ABI for hosts |
| `tests/` | lexer/parser/typechecker + codegen + training integration tests (10) |

## License

[MIT](LICENSE)