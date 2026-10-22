/* nd_quant.h - Cactus-Quants kernels.
 *
 * A CQ tensor stores, per group of `group` weights, a codebook index per
 * weight plus one FP16 group norm. The logical weights are
 *
 *     w_group = (codebook[idx] * norm) @ H,     H = Walsh(group)/sqrt(group)
 *
 * H is symmetric and orthonormal, so for a matvec we never have to
 * materialise w:
 *
 *     <w_group, x_group> = <(cb[idx] * norm) @ H, x_group>
 *                        = <cb[idx] * norm, H @ x_group>
 *
 * i.e. transform the *activation* once per group (O(in log group)) and every
 * output row then costs one codebook lookup and one multiply-add per weight,
 * with no dequantised weights in RAM. That is what makes a 45M-parameter model
 * tractable on an LX7 reading weights straight out of flash.
 */
#ifndef ND_QUANT_H
#define ND_QUANT_H

#include <stdint.h>
#include <string.h>

#include "nd_cact.h"

/* The GEMV inner loop runs ~45M times per token. On the ESP32 the definitions
 * are placed in IRAM so they are not fetched through the instruction cache
 * from flash. Applied at the definition only: repeating it on the prototype
 * makes GCC emit conflicting section names. */
#ifdef ESP_PLATFORM
#include "esp_attr.h"
#define ND_HOT IRAM_ATTR
#else
#define ND_HOT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* IEEE half -> float.
 *
 * Inline and branch-free on the normal-number path: this is called once per
 * norm scale, per MLP diagonal and per confidence probe element, which adds up
 * to >100K calls per token. As an out-of-line call it cost more than the
 * arithmetic it feeds. Subnormals and inf/nan fall back to the slow path. */
float nd_f16_slow(uint16_t h);

static inline float nd_f16(uint16_t h)
{
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t bits;
    float    f;

    if (e == 0u || e == 0x1Fu)
        return nd_f16_slow(h);

    bits = ((uint32_t)(h & 0x8000u) << 16) |
           ((e + (127u - 15u)) << 23) |
           ((uint32_t)(h & 0x3FFu) << 13);
    memcpy(&f, &bits, 4);
    return f;
}

/* exp(x), ~5x faster than libm's and accurate to ~1e-7 relative.
 *
 * silu, the attention gate, the engram alpha, Sinkhorn and the attention
 * softmax together make ~30K exponential calls per token; on an LX7 that is
 * software-emulated and measurable. Range-reduce to 2^k * 2^f with f in
 * [-0.5, 0.5], evaluate 2^f with a degree-5 polynomial, and apply 2^k by
 * assembling the exponent field directly. */
static inline float nd_expf(float x)
{
    float    z, f, p;
    int      k;
    uint32_t bits;
    float    scale;

    if (x > 88.0f)  return 3.4028235e38f;
    if (x < -88.0f) return 0.0f;

    z = x * 1.44269504f;               /* x / ln 2 */
    k = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
    f = z - (float)k;

    p = 0.0013333f;
    p = p * f + 0.0096181f;
    p = p * f + 0.0555041f;
    p = p * f + 0.2402265f;
    p = p * f + 0.6931472f;
    p = p * f + 1.0f;

    bits = (uint32_t)(k + 127) << 23;  /* 2^k */
    memcpy(&scale, &bits, 4);
    return p * scale;
}

/* In-place unnormalised fast Walsh-Hadamard transform. `n` must be a power
 * of two. Apply 1/sqrt(n) yourself if you want the orthonormal H. */
void nd_fwht(float *x, uint32_t n);

/* Scratch an activation needs before nd_cq_gemv: `in_pad` floats. */
uint32_t nd_cq_scratch(const nd_tensor *t);

/* Prepare an activation for one or more matvecs that share a reduction axis:
 * zero-pads x to in_pad and applies the orthonormal H per group.
 * `xh` must hold nd_cq_scratch(t) floats. Reusable across every tensor with
 * the same (shape[1], group). */
void nd_cq_prepare(const nd_tensor *t, const float *x, float *xh);

/* y[0..out) = W @ x, where `xh` came from nd_cq_prepare on the same tensor. */
void nd_cq_gemv_prepared(const nd_cact *c, const nd_tensor *t, const void *blob,
                         const float *xh, float *y);

/* Same, but only rows [r0, r0+nrows). The mHC phi tensors stack every layer
 * into one tensor, so a layer's slice is a row range. */
void nd_cq_gemv_rows(const nd_cact *c, const nd_tensor *t, const void *blob,
                     const float *xh, uint32_t r0, uint32_t nrows, float *y);

/* Convenience: prepare + matvec. `scratch` holds nd_cq_scratch(t) floats. */
void nd_cq_gemv(const nd_cact *c, const nd_tensor *t, const void *blob,
                const float *x, float *scratch, float *y);

/* ---- 2-bit lookup path -------------------------------------------------
 *
 * At 2 bits the per-weight work (extract index, index the codebook, load the
 * activation, multiply-add) is ~6 instructions, and it repeats for every
 * output row. Since the activation is fixed across rows, precompute instead:
 * for each adjacent PAIR of reduction positions, tabulate the 16 possible
 * partial sums. The inner loop then becomes one indexed load and one add per
 * pair - no multiplies at all.
 *
 * The table is 16 floats per pair (in_pad/2 pairs), built once per prepared
 * activation and shared by every tensor that reduces over it.
 */
static inline uint32_t nd_cq_lut_floats(uint32_t in_pad)
{
    return in_pad / 2 * 16;
}

/* ---- optional row-level parallelism ------------------------------------
 *
 * Every GEMV here is embarrassingly parallel across output rows: each row
 * reads its own slice of weights and writes one output. The engine stays
 * single-threaded and portable by default; a platform can install a splitter
 * (the ESP32 build hands half the rows to the second core) and every matvec
 * picks it up. */
typedef void (*nd_row_fn)(void *ctx, uint32_t r0, uint32_t r1);
extern void (*nd_parallel_rows)(nd_row_fn fn, void *ctx, uint32_t nrows);

/* Build the pair table from an already-prepared activation. */
void nd_cq_lut_build(const nd_cact *c, const float *xh, uint32_t in_pad,
                            float *lut);

/* y[0..out) = W @ x for a 2-bit tensor, via the pair table. */
void nd_cq_gemv_lut2(const nd_tensor *t, const void *blob,
                            const float *lut, float *y);

/* A quad table (one byte -> one lookup -> four weights) was tried and removed:
 * it issues fewer instructions but forces group-outer iteration, which uses
 * only 32 bytes of every 64-byte cache line and measured 38% slower on the
 * ESP32-S3. The pair table above is the faster of the two.
 */

/* y[i] = W[ids[i]] . x, for a prepared activation. Used to score only the
 * tokens a grammar currently permits instead of the whole 8192-row vocabulary. */
void nd_cq_gemv_gather(const nd_cact *c, const nd_tensor *t, const void *blob,
                              const float *xh, const uint32_t *ids, uint32_t n,
                              float *y);

/* Batched matmul: Y[t][r] = W[r] . x_t for T activations at once, with
 * `xh` holding T rows of nd_cq_prepare output (stride in_pad) and Y written
 * with stride shape[0]. Each weight group is decoded once and applied to all
 * T rows, instead of once per row as T separate GEMVs would. Whistle's
 * encoder runs every frame of a clip through the same weights, which is
 * where this pays. Rows are split with nd_parallel_rows. */
void nd_cq_gemm_prepared(const nd_cact *c, const nd_tensor *t, const void *blob,
                         const float *xh, uint32_t T, float *y);

/* Batched matmuls run in int8 when this is set (the default) and the tensor
 * is 2- or 4-bit with 128-wide groups: weights decode exactly to int8 codebook
 * levels, activations get one int8 scale per group, and the float group norm
 * is applied after each integer dot. On the ESP32-S3 and ESP32-P4 the dots
 * use the PIE vector unit (16 multiply-adds per instruction); elsewhere a C
 * loop computes the same integers, so the host build checks what the board
 * does. Set to 0 for the float path (tools/check_whistle.py's tight NumPy
 * comparison). */
extern int nd_cq_int8;

/* Compare the vector dot against the C loop on fixed patterns, including the
 * extremes. Returns 0 when they agree. A wrong kernel would not crash, only
 * make transcripts quietly worse, so the firmware runs this at boot. */
int nd_cq_int8_check(void);

/* An activation batch ready for the int8 path: T rows of the Hadamard-domain
 * activation as int8, one scale per 128-wide group. Prepare it once and run
 * every tensor that reduces over it (Q, K, V and gate share one; so do the
 * three mHC phi tensors) - preparing per tensor repeated the transform, the
 * quantisation and a float copy of the batch for each one. */
typedef struct {
    int8_t  *xq;        /* [T][in_pad], 16-byte aligned */
    float   *sx;        /* [T][in_pad/128] */
    uint32_t T, in_pad;
    void    *mem;
} nd_q8_act;

/* Fill `a` from T rows of x (stride shape[1]) for tensors shaped like `t`.
 * Returns 0, or -1 if the tensor cannot use the int8 path or memory ran out. */
int  nd_cq_q8_prepare(const nd_tensor *t, const float *x, uint32_t T, nd_q8_act *a);
void nd_cq_q8_free(nd_q8_act *a);

/* Y[k][r - r0] = W[r] . x_k for r in [r0, r0+nrows), from a prepared batch. */
void nd_cq_q8_gemm(const nd_cact *c, const nd_tensor *t, const void *blob,
                   const nd_q8_act *a, uint32_t r0, uint32_t nrows, float *y);

/* True when nd_cq_q8_* applies to `t` (2/4-bit, group 128, in <= 2048). */
int  nd_cq_q8_ok(const nd_tensor *t);

/* Same, over rows [r0, r0+nrows) only, with Y stride nrows. For the stacked
 * mHC phi tensors, where one layer is a row range. */
void nd_cq_gemm_rows(const nd_cact *c, const nd_tensor *t, const void *blob,
                     const float *xh, uint32_t T, uint32_t r0, uint32_t nrows,
                     float *y);

/* Reconstruct one dequantised output row into `w` (shape[1] floats).
 * Used for validation and for the engram tables, which are gathered by row
 * rather than streamed as a matvec. `scratch` holds in_pad floats. */
void nd_cq_dequant_row(const nd_cact *c, const nd_tensor *t, const void *blob,
                       uint32_t row, float *scratch, float *w);

#ifdef __cplusplus
}
#endif
#endif /* ND_QUANT_H */
