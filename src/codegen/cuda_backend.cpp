#include "ns/codegen/codegen.hpp"

// CUDA-specific backend layer. The shared CodeGenerator (codegen.cpp) owns
// instruction-level control flow; every piece of *device* source text that
// this target needs lives here as black-box snippets the generator splices
// into the emitted .cu. Keeping the kernels in one translation unit is the
// staging ground for hardware tuning: MMA instruction selection, fp16/fp8
// storage, smaller/larger shared-memory tile sizes and launch heuristics can
// be swapped here without touching the front end or the lowering passes.

namespace ns {
namespace cuda {

// Launch grid for flat 1D kernels (256 threads/block).
static const char kLaunchMacros[] = R"CUDA(
#define NS_CU_GRID(n) (((unsigned)(n) + 255u) / 256u)
#define NS_LAUNCH1(kern, n, ...) do { kern<<<NS_CU_GRID(n), 256>>>(__VA_ARGS__); } while (0)
// Block-per-tile launch for tiled kernels: grid.x == number of blocks (each
// block owns one output tile), blockDim.x == block size in threads.
#define NS_LAUNCH_BLOCKS(kern, nblocks, ...) do { kern<<<((nblocks) > 0 ? (nblocks) : 1), 256>>>(__VA_ARGS__); } while (0)
)CUDA";

static const char kDeviceHelpers[] = R"CUDA(
__device__ float ns_act2_d(float x, int code) {
  switch (code) {
    case 0: return x;
    case 1: return x > 0.f ? x : 0.f;
    case 2: return x > 0.f ? x : 0.01f * x;
    case 3: return 1.f / (1.f + expf(-x));
    case 4: return tanhf(x);
    case 5: return x / (1.f + expf(-x));
    case 6: { float u = x / 1.41421356f; return 0.5f * x * (1.f + erff(u)); }
    default: return x;
  }
}
__device__ float ns_act_deriv_d(float x, int code) {
  switch (code) {
    case 0: return 1.f;
    case 1: return x > 0.f ? 1.f : 0.f;
    case 2: return x > 0.f ? 1.f : 0.01f;
    case 3: { float s = 1.f / (1.f + expf(-x)); return s * (1.f - s); }
    case 4: { float t = tanhf(x); return 1.f - t * t; }
    case 5: { float s = x / (1.f + expf(-x)); return s + x * (1.f - s); }
    case 6: { float u = x / 1.41421356f; float p = 0.5f * (1.f + erff(u));
              return p + x * expf(-u * u) / 2.50662827f; }
    default: return 1.f;
  }
}
)CUDA";

// Uniform forward kernels.
static const char kForwardKernels[] = R"CUDA(
// Tiled GEMM: 16x16 output tiles, 16-wide K strip in shared memory (padded to
// avoid bank conflicts). Every load/store is bounds-checked, so the kernel is
// safe for arbitrary M/K/N (including tiny M<16 or K<16) and for M that is not
// a multiple of the tile width. Launched via NS_LAUNCH_BLOCKS with one block
// per output tile: grid.x = ceil(M/16) * ceil(N/16).
__global__ void ns_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int K, int N) {
  __shared__ float As[16][17];
  __shared__ float Bs[16][17];
  int nb = (N + 15) >> 4;
  int blk = blockIdx.x;
  int by = blk / nb, bx = blk - by * nb;
  int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
  int row = (by << 4) + ty, col = (bx << 4) + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    int ak = k0 + tx, ai = row;
    As[ty][tx] = (ai < M && ak < K) ? A[ai * K + ak] : 0.f;
    int bk = k0 + ty, bj = col;
    Bs[ty][tx] = (bk < K && bj < N) ? B[bk * N + bj] : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; kk++) acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}
__global__ void ns_act_kernel(const float* __restrict__ in, float* __restrict__ out,
                              int n, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = ns_act2_d(in[idx], code);
}
__global__ void ns_copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx];
}
__global__ void ns_fill_kernel(float* __restrict__ out, int n, float v) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = v;
}
__global__ void ns_transpose2d_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int i = idx / cols, j = idx - i * cols;
  out[j * rows + i] = in[i * cols + j];
}
__global__ void ns_concat2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int rows, int ca, int co) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * co;
  if (idx >= total) return;
  int r = idx / co, c = idx - r * co;
  out[idx] = (c < ca) ? A[r * ca + c] : B[r * (co - ca) + (c - ca)];
}
__global__ void ns_concat0_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int na, int nb, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (na + nb) * D;
  if (idx >= total) return;
  int r = idx / D, c = idx - r * D;
  out[idx] = (r < na) ? A[idx] : B[(r - na) * D + c];
}
__global__ void ns_slice2_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int M, int C, int cs, int ce) {
  int ow = ce - cs;
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * ow;
  if (idx_i >= total) return;
  int r = idx_i / ow, c = idx_i - r * ow;
  out[idx_i] = in[r * C + cs + c];
}
__global__ void ns_slicerows_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int C, int rs, int re) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (re - rs) * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[(rs + r) * C + c];
}
__global__ void ns_index_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                float* __restrict__ out, int M, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * L;
  if (idx_i >= total) return;
  int r = idx_i / L, k = idx_i - r * L;
  out[idx_i] = in[r * C + idx[k]];
}
__global__ void ns_indexrows_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                    float* __restrict__ out, int L, int C) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = L * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[(int)idx[r] * C + c];
}
__global__ void ns_scatter_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                  const float* __restrict__ upd, float* __restrict__ out,
                                  int M, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[idx_i];
  for (int k = 0; k < L; k++)
    if (idx[k] == c) { out[idx_i] = upd[r * L + k]; break; }
}
__global__ void ns_scatterrows_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                      const float* __restrict__ upd, float* __restrict__ out,
                                      int R, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = R * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[idx_i];
  for (int k = 0; k < L; k++)
    if (idx[k] == r) { out[idx_i] = upd[k * C + c]; break; }
}
__global__ void ns_embedding_kernel(const float* __restrict__ W, const float* __restrict__ idx,
                                    float* __restrict__ out, int n, int V, int D) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * D;
  if (idx_i >= total) return;
  int r = idx_i / D, c = idx_i - r * D;
  int k = (int)idx[r];
  if (k < 0 || k >= V) k = 0;
  out[idx_i] = W[k * D + c];
}
// Multi-head attention core: one block per (batch, head). Grid = B*H blocks.
// Shared memory holds the S*S score matrix (dynamic shared memory). Q, K, V
// are already projected on device (each [B*S, D]); Out accumulates the per-head
// context [B*S, D]. Head dimension Dk = D/H, sequence length S, scale=1/sqrt(Dk).
extern "C" __global__ void ns_attention_core_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ Out,
    int S, int D, int H, float scale) {
  extern __shared__ float sc[];   // [S*S]
  int b = blockIdx.x / H, h = blockIdx.x - b * H;
  int Dk = D / H;
  const float* Qb = Q + b * S * D + h * Dk;
  const float* Kb = K + b * S * D + h * Dk;
  const float* Vb = V + b * S * D + h * Dk;
  // scores[i,j] = Qb[i*D + d] . Kb[j*D + d] * scale
  for (int t = threadIdx.x; t < S * S; t += blockDim.x) {
    int i = t / S, j = t - i * S;
    float a = 0.f;
    for (int d = 0; d < Dk; d++)
      a += Qb[i * D + d] * Kb[j * D + d];
    sc[t] = a * scale;
  }
  __syncthreads();
  // row softmax in-place
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float mx = sc[i * S];
    for (int j = 1; j < S; j++) mx = fmaxf(mx, sc[i * S + j]);
    float s = 0.f;
    for (int j = 0; j < S; j++) { sc[i * S + j] = expf(sc[i * S + j] - mx); s += sc[i * S + j]; }
    for (int j = 0; j < S; j++) sc[i * S + j] /= s;
  }
  __syncthreads();
  // context[i,d] = sum_j sc[i,j] * Vb[j*D+d]
  float* Ob = Out + b * S * D + h * Dk;
  for (int i = threadIdx.x; i < S; i += blockDim.x)
    for (int d = 0; d < Dk; d++) {
      float a = 0.f;
      for (int j = 0; j < S; j++) a += sc[i * S + j] * Vb[j * D + d];
      Ob[i * D + d] = a;
    }
}
// Mixture-of-experts router: one thread per token. Router logits = x @ Wg with
// a softmax over experts; the top-1 expert per token is applied to x (its [D,D]
// matrix lives at We + e*D*D) and the result is scaled by the routing weight.
__global__ void ns_moe_kernel(const float* __restrict__ x, const float* __restrict__ Wg,
                              const float* __restrict__ We, float* __restrict__ out,
                              int M, int D, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float lg[256];
  int Ee = E < 256 ? E : 256;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    lg[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) { lg[e] = expf(lg[e] - mx); sum += lg[e]; }
  int best = 0;
  for (int e = 1; e < Ee; e++) if (lg[e] > lg[best]) best = e;
  float p = (sum > 0.f) ? lg[best] / sum : 0.f;
  for (int j = 0; j < D; j++) {
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * We[best * D * D + k * D + j];
    out[i * D + j] = p * a;
  }
}
__global__ void ns_binop_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int n, int nb, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float b = nb > 0 ? B[idx % nb] : 0.f;
  switch (code) {
    case 0: C[idx] = A[idx] + b; break;
    case 1: C[idx] = A[idx] - b; break;
    case 2: C[idx] = A[idx] * b; break;
    case 3: C[idx] = A[idx] / b; break;
  }
}
__global__ void ns_layernorm_kernel(float* __restrict__ m, int n, int last) {
  int rows = n / last;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mean = 0.f, var = 0.f;
  for (int j = 0; j < last; j++) mean += m[r * last + j];
  mean /= last;
  for (int j = 0; j < last; j++) { float d = m[r * last + j] - mean; var += d * d; }
  var /= last;
  float inv = rsqrtf(var + 1e-5f);
  for (int j = 0; j < last; j++) m[r * last + j] = (m[r * last + j] - mean) * inv;
}
__global__ void ns_softmax_kernel(float* __restrict__ m, int n, int last) {
  int rows = n / last;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = m[r * last];
  for (int j = 1; j < last; j++) mx = fmaxf(mx, m[r * last + j]);
  float s = 0.f;
  for (int j = 0; j < last; j++) { m[r * last + j] = expf(m[r * last + j] - mx); s += m[r * last + j]; }
  for (int j = 0; j < last; j++) m[r * last + j] /= s;
}
)CUDA";

// Reverse-mode + optimizer kernels (AOT training).
static const char kTrainKernels[] = R"CUDA(
__global__ void ns_grad_a_kernel(const float* __restrict__ dC, const float* __restrict__ B,
                                 float* __restrict__ dA, int M, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * K;
  for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
    int i = t / K, k = t % K;
    float acc = 0.f;
    for (int j = 0; j < N; j++) acc += dC[i * N + j] * B[k * N + j];
    dA[t] = acc;
  }
}
__global__ void ns_grad_w_kernel(const float* __restrict__ A, const float* __restrict__ dC,
                                 float* __restrict__ dB, int M, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * N;
  for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
    int k = t / N, j = t % N;
    float acc = 0.f;
    for (int i = 0; i < M; i++) acc += A[i * K + k] * dC[i * N + j];
    dB[t] = acc;
  }
}
__global__ void ns_act_grad_kernel(const float* __restrict__ dIn, const float* __restrict__ actIn,
                                   float* __restrict__ dOut, int n, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dOut[idx] = dIn[idx] * ns_act_deriv_d(actIn[idx], code);
}
__global__ void ns_loss_grad_kernel(const float* __restrict__ preds, const float* __restrict__ labels,
                                    float* __restrict__ d, int numel, int C) {
  int rows = numel / C;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = preds[r * C];
  for (int j = 1; j < C; j++) mx = fmaxf(mx, preds[r * C + j]);
  float s = 0.f;
  for (int j = 0; j < C; j++) s += expf(preds[r * C + j] - mx);
  for (int j = 0; j < C; j++) {
    float p = expf(preds[r * C + j] - mx) / s;
    d[r * C + j] = (p - labels[r * C + j]) / (float)rows;
  }
}
__global__ void ns_ce_row_kernel(const float* __restrict__ preds, const float* __restrict__ labels,
                                 float* __restrict__ rowloss, int numel, int C) {
  int rows = numel / C;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = preds[r * C];
  for (int j = 1; j < C; j++) mx = fmaxf(mx, preds[r * C + j]);
  float s = 0.f;
  for (int j = 0; j < C; j++) s += expf(preds[r * C + j] - mx);
  float loss = 0.f;
  for (int j = 0; j < C; j++) {
    float p = expf(preds[r * C + j] - mx) / s;
    if (labels[r * C + j] > 0.f) loss -= logf(p);
  }
  rowloss[r] = loss;
}
__global__ void ns_adamw_kernel(float* __restrict__ w, const float* __restrict__ g,
                                float* __restrict__ am, float* __restrict__ av,
                                int n, float lr, float b1t, float b2t,
                                float beta1, float beta2, float omb1, float omb2,
                                float eps, float decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float gg = g[idx];
  float a = beta1 * am[idx] + omb1 * gg;
  float v = beta2 * av[idx] + omb2 * gg * gg;
  float mh = a / b1t, vh = v / b2t;
  am[idx] = a; av[idx] = v;
  w[idx] -= (lr * mh / (sqrtf(vh) + eps)) + decay * w[idx];
}
__global__ void ns_muon_ema_kernel(float* __restrict__ m, const float* __restrict__ g,
                                   int n, float momentum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) m[idx] = momentum * m[idx] + g[idx];
}
__global__ void ns_muon_step_kernel(float* __restrict__ w, float* __restrict__ m,
                                    int n, float le, float decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { w[idx] -= le * m[idx] + decay * w[idx]; m[idx] = 0.f; }
}
// Modified Gram-Schmidt on G (columns if M>=N, rows otherwise), single block,
// thread 0 only: byte-for-byte mirrors the CPU-reference ns_orthonom, so the
// Muon update is bit-identical regardless of device.
__global__ void ns_orthonom_kernel(float* __restrict__ G, int M, int N) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  bool transposed = M < N;
  int R = transposed ? N : M;
  int C = transposed ? M : N;
  for (int j = 0; j < C; j++) {
    for (int i = 0; i < j; i++) {
      float dot = 0.f;
      for (int k = 0; k < R; k++) {
        float av = transposed ? G[i * R + k] : G[k * N + i];
        float bv = transposed ? G[j * R + k] : G[k * N + j];
        dot += av * bv;
      }
      for (int k = 0; k < R; k++) {
        float av = transposed ? G[i * R + k] : G[k * N + i];
        float v = transposed ? G[j * R + k] : G[k * N + j];
        if (transposed) G[j * R + k] = v - dot * av; else G[k * N + j] = v - dot * av;
      }
    }
    float nrm = 0.f;
    for (int k = 0; k < R; k++) { float v = transposed ? G[j * R + k] : G[k * N + j]; nrm += v * v; }
    nrm = sqrtf(nrm) + 1e-30f;
    for (int k = 0; k < R; k++) {
      if (transposed) G[j * R + k] /= nrm; else G[k * N + j] /= nrm;
    }
  }
}
// Inverted dropout forward (train only): draws per-element uniform samples via
// curand, keeps with probability (1 - rate), scales kept units by 1/(1 - rate).
// The scaled mask is stored so the backward pass can apply the SAME mask
// (dOut[i] = dIn[i] * mask[i]).
__global__ void ns_dropout_fwd_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      float* __restrict__ mask, int n, float rate, unsigned seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  curandState st;
  curand_init(seed, idx, 0, &st);
  float u = curand_uniform(&st);
  float keep = (u >= rate) ? (1.0f / (1.0f - rate)) : 0.0f;
  mask[idx] = keep;
  out[idx] = in[idx] * keep;
}
__global__ void ns_dropout_mul_kernel(const float* __restrict__ dIn, const float* __restrict__ mask,
                                      float* __restrict__ dOut, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dOut[idx] = dIn[idx] * mask[idx];
}
)CUDA";

// Runtime utilities: launcher macros and a lazy grow-only device allocator that
// zero-initializes on (re)allocation (AdamW/Muon moment buffers rely on this
// first-touch zeroing). All pointers live in the per-model NSContext; nothing
// here is file-scope state, so separate models never share device buffers.
static const char kRuntimeUtils[] = R"CUDA(
static int ns_cu_reserve(float** pp, size_t* pc, size_t bytes, int zero) {
  if (!*pp || *pc < bytes) {
    if (*pp) cudaFree(*pp);
    if (bytes == 0) bytes = 1;
    if (cudaMalloc((void**)pp, bytes) != cudaSuccess) return -1;
    *pc = bytes;
    if (zero) cudaMemset(*pp, 0, bytes);
  }
  return 0;
}
)CUDA";

std::string device_helpers_source()  { return std::string(kLaunchMacros) + kDeviceHelpers; }
std::string forward_kernels_source() { return kForwardKernels; }
std::string train_kernels_source()   { return kTrainKernels; }
std::string runtime_utils_source()   { return kRuntimeUtils; }

} // namespace cuda
} // namespace ns