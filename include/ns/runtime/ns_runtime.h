#ifndef NS_RUNTIME_H
#define NS_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of the ABI/interface this header implements. The generated model
 * sources are produced by the matching compiler release.
 */
#define NS_RUNTIME_VERSION_MAJOR 1
#define NS_RUNTIME_VERSION_MINOR 1
#define NS_RUNTIME_VERSION_PATCH 0
#define NS_RUNTIME_VERSION "1.1.0"

/* NeuralScript AOT runtime C-ABI.
 *
 * A compiled model is a self-contained translation unit that implements this
 * interface. The host application links the generated source and drives it
 * exclusively through the functions below.
 *
 * Weights are presented as ONE contiguous, host-owned float blob:
 *   [w0 | w1 | ... | wN-1]  (concatenation in model definition order).
 * ns_eval_infer copies the blob into the model at init; ns_eval_infer then
 * runs the graph without further host I/O.
 */

typedef struct ns_model ns_model;

/* One trainable parameter in the concatenated weight layout. */
typedef struct ns_weight_desc {
    const char* name;   /* layer weight name, e.g. "fc1_w" */
    size_t      offset; /* float offset of the start of this weight in the blob */
    size_t      count;  /* number of floats */
} ns_weight_desc;

/* Layout of the full blob. */
typedef struct ns_weight_layout {
    size_t           num_weights; /* number of ns_weight_desc entries */
    const ns_weight_desc* desc;
} ns_weight_layout;

/* Allocate an empty model and copy `num_floats` floats of weights into it.
 * Returns NULL on allocation failure. num_weights must match the compiled
 * graph's static weight size (see ns_model_weight_count). */
ns_model* ns_runtime_init(const float* weights, size_t num_floats);

/* Run inference:   output[b*j + k] for batch b, class k.
 * input : input_numel floats, row-major, leading dim = batch.
 * output: at least ns_model_output_numel(input_numel) floats.
 * Returns 0 on success, non-zero on error. */
int ns_eval_infer(ns_model* m, const float* input, float* output, size_t input_numel);

/* Number of output floats for a given input size. */
size_t ns_model_output_numel(const ns_model* m, size_t input_numel);

/* Number of floats the compiled graph expects in the weights blob. */
size_t ns_model_weight_count(const ns_model* m);

/* Same as ns_model_weight_count, but does not require a model instance: the
 * generated graph's blob size is fixed at compile time, so a host can size
 * its weight buffer before ns_runtime_init. */
size_t ns_weight_count_static(void);

/* Copy the full weight blob out of the model. Returns 0 on success, -1 if
 * m/out is null or n does not match ns_model_weight_count(m). */
int ns_model_get_weights(const ns_model* m, float* out, size_t n);

/* The concatenated weight layout of the compiled graph. */
const ns_weight_layout* ns_model_layout(const ns_model* m);

/* Free the model and its buffers. */
void ns_free(ns_model* m);

/* Persist the current weight blob (and, for MoE models, the expert liveness
   mask) to `path`. Binary format: 4-byte magic "NSM1" (weights only) or
   "NSM2" (weights + {n_layers, capacity, mask bytes}). Returns 0 on success,
   -1 on error. */
int ns_save_checkpoint(const ns_model* m, const char* path);

/* Restore weights (and the MoE mask) from a checkpoint. NSM1 files load with
   all experts alive. The host-side copy is replaced; device weight state is
   re-synced on CUDA backends. Returns 0 on success, -1 on error. */
int ns_load_checkpoint(ns_model* m, const char* path);

/* ---- MoE expert lifecycle (present when the graph has a MoE layer) ----
 *
 * The capacity (compile-time constant ns_moe_cap, exposed via the weight
 * layout: gate row [D,cap] + per-expert ffn weights) is fixed at compile
 * time, but the LIVENESS of each expert slot is runtime state. The mask is
 * threaded into the fused forward/backward kernels, so only live experts are
 * routed to. Exactly one MoE layer per model is supported.
 *
 * birth(n): turns up to `n` currently-inactive slots live, copying the full
 *           (gate column + both ffn matrices) of a random live expert with a
 *           small perturbation. Returns the new live count.
 * merge(a,b): averages the weights of live experts `a` and `b` into `a` and
 *           deactivates `b`. Both must be live, distinct indices. Returns the
 *           new live count.
 * kill(k): deactivates live expert `k`. Returns the new live count.
 * count(): number of currently live expert slots.
 * All of these return the input count unchanged when the model has no MoE
 * layer or the arguments are invalid. */
size_t ns_expert_count(const ns_model* m);
size_t ns_expert_birth(ns_model* m, int n);
size_t ns_expert_merge(ns_model* m, int a, int b);
size_t ns_expert_kill(ns_model* m, int k);

/* ---- AOT training (present when the network defines a train() method) ----
 *
 * The generated training core shares the SAME weight blob as inference, so a
 * host loop can do: ns_eval_infer(...); ns_runtime_train_step(...); continue.
 *
 * input : batch-major features, input_numel floats (batch * in_cols).
 * labels: batch-major one-hot classes, input_numel / in_cols * out_cols floats.
 * loss_out: receives the mean cross-entropy loss over the batch.
 * lr    : learning rate for the compiled optimizer (Muon for matrices,
 *         AdamW otherwise). Runs forward + backward + one optimizer step and
 *         writes the updated weights back into the model.
 * Returns 0 on success, non-zero on error. */
int ns_runtime_train_step(ns_model* m, const float* input, const float* labels,
                          size_t input_numel, float* loss_out, float lr);

/* Forward-only objective: computes the mean cross-entropy loss WITHOUT
 * updating weights. Returns 0 on success. */
int ns_objective_loss(ns_model* m, const float* input, const float* labels,
                      size_t input_numel, float* loss_out);

#ifdef __cplusplus
}
#endif

#endif /* NS_RUNTIME_H */