#ifndef NS_RUNTIME_H
#define NS_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    size_t      offset; /* byte offset within the weight blob (floats*4) */
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

/* The concatenated weight layout of the compiled graph. */
const ns_weight_layout* ns_model_layout(const ns_model* m);

/* Free the model and its buffers. */
void ns_free(ns_model* m);

#ifdef __cplusplus
}
#endif

#endif /* NS_RUNTIME_H */