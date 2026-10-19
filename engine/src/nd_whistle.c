#include "nd_whistle.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nd_ledcmd.h"     /* nd_utf8_lower */
#include "nd_quant.h"

uint64_t nd_wprof[ND_WP_COUNT];

#ifdef ND_PROFILE
#ifdef ESP_PLATFORM
#include "esp_timer.h"
#define ND_NOW_US() ((uint64_t)esp_timer_get_time())
#else
#include <time.h>
#define ND_NOW_US() ((uint64_t)(clock() * (1000000.0 / CLOCKS_PER_SEC)))
#endif
#define ND_T0(v) uint64_t v = ND_NOW_US()
#define ND_T1(v, slot) nd_wprof[slot] += ND_NOW_US() - (v)
#else
#define ND_T0(v) ((void)0)
#define ND_T1(v, slot) ((void)0)
#endif

#define EPS          1e-6f
#define SINKHORN     20
#define EG_HIST      16          /* >= (taps-1)*dilation + 1 = 10 */
#define EG_SEED      0x9E3779B9u
#define EG_PRIME     0x01000193u
#define BOS_ID       2u
#define EOS_ID       1u
#define LANG_BASE    8192u       /* en de fr es it nl pl follow the text vocab */

/* Front-end constants, established against the official engine (see
 * tools/whistle_ref.py DEFAULTS): the clip is scaled to peak 1, then
 * ln(mel + LOG_EPS), then each mel channel is normalised over time. */
#define LOG_EPS      6e-7f
#define NORM_EPS     1e-3f

/* Per-layer tensor offsets, in canon order (tools/whistle_cact.py). */
enum { D_NORM_IN, D_Q, D_K, D_V, D_QT, D_KT, D_VT, D_QN, D_KN, D_GATE, D_OUT,
       D_POST, D_AGATE, D_PRE_HADA, D_MLP, D_N = D_MLP + 13 };
enum { X_NORM, X_Q, X_K, X_V, X_GATE, X_OUT, X_QN, X_KN, X_POST, X_GATE_S, X_N };
enum { E_PRE_HADA0, E_NORM_IN, E_POST, E_CONV_NORM, E_CONV_OUT_NORM, E_PRE_HADA,
       E_Q, E_K, E_V, E_GATE, E_OUT, E_QN, E_KN, E_AGATE, E_PW1, E_PW2, E_DW,
       E_MLP0, E_MLP = E_MLP0 + 13, E_N = E_MLP + 13 };
enum { M_D1, M_D2, M_B2, M_D3, M_D4, M_W1A, M_W1B, M_W2A, M_W2B, M_W3A, M_W3B,
       M_CV, M_CU };
enum { H_A_PRE, H_A_POST, H_A_RES, H_B_PRE, H_B_POST, H_B_RES, H_PHI_PRE,
       H_PHI_POST, H_PHI_RES };
enum { G_TABLES, G_KEY, G_VALUE, G_TAPS };
enum { S_W, S_DW1, S_PW1, S_DW2, S_PW2, S_OUT };

static int32_t s_perm1[512], s_perm2[512];

/* ------------------------------------------------------------------ utils */

static const void *dat(const nd_whistle *w, uint32_t i)
{
    return w->p[i];         /* bounds-checked once, at open */
}

/* Element k of a small fp16 (or fp32) tensor. */
static float hv(const nd_whistle *w, uint32_t i, uint32_t k)
{
    if (w->t[i].dtype == ND_DT_FP32)
        return ((const float *)dat(w, i))[k];
    return nd_f16(((const uint16_t *)dat(w, i))[k]);
}

static void hvec(const nd_whistle *w, uint32_t i, float *out, uint32_t n)
{
    uint32_t k;
    for (k = 0; k < n; k++)
        out[k] = hv(w, i, k);
}

static float sigm(float x)
{
    if (x >= 0.0f)
        return 1.0f / (1.0f + nd_expf(-x));
    {
        float e = nd_expf(x);
        return e / (1.0f + e);
    }
}

static float silu(float x) { return x * sigm(x); }

/* No double anywhere hot: the LX7 FPU is single precision, so double is
 * software-emulated. In the first board run a double accumulator here and in
 * the front end made a 2 s clip take 24 s (mel alone 9.2 s). */
static void rms_unit(const float *x, uint32_t n, float *out)
{
    float    s = 0.0f, r;
    uint32_t k;
    for (k = 0; k < n; k++)
        s += x[k] * x[k];
    r = 1.0f / sqrtf(s / (float)n + EPS);
    for (k = 0; k < n; k++)
        out[k] = x[k] * r;
}

/* ZCRMSNorm: (1 + scale) * x / rms(x). `scale` is an fp16 tensor; `off`
 * selects a slice of it (q_norm / k_norm are one vector shared by heads). */
static void zc(const nd_whistle *w, const float *x, uint32_t n, uint32_t si,
               float *out)
{
    uint32_t k;
    rms_unit(x, n, out);
    for (k = 0; k < n; k++)
        out[k] *= 1.0f + hv(w, si, k);
}

static void softmax(float *x, uint32_t n)
{
    float    m = -3.4e38f, s = 0.0f;
    uint32_t k;
    for (k = 0; k < n; k++)
        if (x[k] > m)
            m = x[k];
    for (k = 0; k < n; k++) {
        x[k] = nd_expf(x[k] - m);
        s += x[k];
    }
    for (k = 0; k < n; k++)
        x[k] /= s;
}

/* Half-split rotary on one head of `hd` floats at position `pos`. */
static void rope(const nd_whistle *w, float *z, uint32_t hd, uint32_t pos)
{
    uint32_t half = hd / 2, i;
    for (i = 0; i < half; i++) {
        float a   = (float)pos * w->rope_inv[i];
        float c   = cosf(a), s = sinf(a);
        float z1  = z[i], z2 = z[i + half];
        z[i]        = z1 * c - z2 * s;
        z[i + half] = z2 * c + z1 * s;
    }
}

/* ------------------------------------------------------------- matmuls */

/* Y[T][out] = X[T][in] @ W^T. One row uses the 2-bit pair table (Needle 2's
 * fast path); several rows use the batched kernel, so each weight is decoded
 * once per clip rather than once per frame. */
static void lin_rows(nd_whistle *w, uint32_t ti, const float *X, uint32_t T,
                     uint32_t r0, uint32_t nrows, float *Y)
{
    const nd_tensor *t      = &w->t[ti];
    const void      *b      = dat(w, ti);
    uint32_t         in     = t->shape[1];
    uint32_t         in_pad = nd_cq_in_pad(t);
    uint32_t         k;
    ND_T0(t0);

    if (nd_cq_q8_ok(t)) {
        nd_q8_act a;
        if (nd_cq_q8_prepare(t, X, T, &a) == 0) {
            nd_cq_q8_gemm(&w->c, t, b, &a, r0, nrows, Y);
            nd_cq_q8_free(&a);
            ND_T1(t0, ND_WP_LIN);
            return;
        }
    }
    if (T == 1) {
        nd_cq_prepare(t, X, w->xh);
        if (t->bits == 2 && r0 == 0 && nrows == t->shape[0]) {
            nd_cq_lut_build(&w->c, w->xh, in_pad, w->lut);
            nd_cq_gemv_lut2(t, b, w->lut, Y);
        } else {
            nd_cq_gemv_rows(&w->c, t, b, w->xh, r0, nrows, Y);
        }
        ND_T1(t0, ND_WP_LIN);
        return;
    }
    {
        float *xh = (float *)ND_ALLOC((size_t)T * in_pad * sizeof(float));
        for (k = 0; k < T; k++)
            nd_cq_prepare(t, X + (size_t)k * in, xh + (size_t)k * in_pad);
        nd_cq_gemm_rows(&w->c, t, b, xh, T, r0, nrows, Y);
        ND_FREE(xh);
    }
    ND_T1(t0, ND_WP_LIN);
}

static void lin(nd_whistle *w, uint32_t ti, const float *X, uint32_t T, float *Y)
{
    lin_rows(w, ti, X, T, 0, w->t[ti].shape[0], Y);
}

/* One input, several tensors (Q/K/V/gate; the three mHC phi; cross K/V):
 * prepare and quantise the input once, then run each tensor against it. */
typedef struct {
    nd_q8_act    a;
    const float *X;
    uint32_t     T;
    int          q8;
} shared_in;

static void sh_begin(nd_whistle *w, shared_in *s, uint32_t ti, const float *X, uint32_t T)
{
    ND_T0(t0);
    s->X  = X;
    s->T  = T;
    s->q8 = nd_cq_q8_prepare(&w->t[ti], X, T, &s->a) == 0;
    ND_T1(t0, ND_WP_LIN);
}

static void sh_rows(nd_whistle *w, shared_in *s, uint32_t ti, uint32_t r0,
                    uint32_t nrows, float *Y)
{
    if (s->q8) {
        ND_T0(t0);
        nd_cq_q8_gemm(&w->c, &w->t[ti], dat(w, ti), &s->a, r0, nrows, Y);
        ND_T1(t0, ND_WP_LIN);
    } else {
        lin_rows(w, ti, s->X, s->T, r0, nrows, Y);
    }
}

static void sh_lin(nd_whistle *w, shared_in *s, uint32_t ti, float *Y)
{
    sh_rows(w, s, ti, 0, w->t[ti].shape[0], Y);
}

static void sh_end(shared_in *s)
{
    if (s->q8)
        nd_cq_q8_free(&s->a);
}

/* --------------------------------------------------- Hadamard MLP (Needle 3)
 *
 * Monarch form: three Kronecker 16x32 mixes with fixed permutations between
 * them, diagonal scales, and a rank-8 input-conditioned gain:
 *   cond = 1 + softmax(x @ cond_v) @ cond_u
 *   z = kron1(d1*x)[p1];  z = kron2(silu(d2*cond*z + b2))[p2];  y = d4*kron3(d3*z)
 * kron(z, a, b) = a^T Z b with Z = z viewed as 16x32. */

typedef struct {
    float a[3][256], b[3][1024];
    float d1[512], d2[512], b2[512], d3[512], d4[512];
    float cv[512 * 8], cu[8 * 512];
} mlp_w;

static void mlp_load(const nd_whistle *w, uint32_t base, mlp_w *m)
{
    int s;
    for (s = 0; s < 3; s++) {
        hvec(w, base + M_W1A + 2 * s, m->a[s], 256);
        hvec(w, base + M_W1B + 2 * s, m->b[s], 1024);
    }
    hvec(w, base + M_D1, m->d1, 512);
    hvec(w, base + M_D2, m->d2, 512);
    hvec(w, base + M_B2, m->b2, 512);
    hvec(w, base + M_D3, m->d3, 512);
    hvec(w, base + M_D4, m->d4, 512);
    hvec(w, base + M_CV, m->cv, 512 * 8);
    hvec(w, base + M_CU, m->cu, 8 * 512);
}

/* Inner loops run along contiguous rows (axpy form): the textbook loop
 * order walks b and tmp down columns and serialises every sum. (Holding
 * eight output columns in an accumulator array measured 11% slower on the
 * board: GCC kept the array in memory.) */
static void kron(const float *z, const float *a, const float *b, float *out)
{
    float    tmp[16 * 32];
    uint32_t i, j, k, l;
    for (i = 0; i < 16; i++) {
        float *t = tmp + i * 32;
        memset(t, 0, 32 * sizeof(float));
        for (j = 0; j < 32; j++) {
            float        zz = z[i * 32 + j];
            const float *bj = b + j * 32;
            for (l = 0; l < 32; l++)
                t[l] += zz * bj[l];
        }
    }
    for (k = 0; k < 16; k++) {
        float *o = out + k * 32;
        memset(o, 0, 32 * sizeof(float));
        for (i = 0; i < 16; i++) {
            float        aa = a[i * 16 + k];
            const float *ti = tmp + i * 32;
            for (l = 0; l < 32; l++)
                o[l] += aa * ti[l];
        }
    }
}

static void mlp_row(const mlp_w *m, const float *x, float *out)
{
    float    c8[8], cond[512], z[512], y[512];
    uint32_t k, r;

    memset(c8, 0, sizeof(c8));
    for (k = 0; k < 512; k++) {                 /* contiguous in cond_v */
        float        xk = x[k];
        const float *cv = m->cv + k * 8;
        for (r = 0; r < 8; r++)
            c8[r] += xk * cv[r];
    }
    softmax(c8, 8);
    for (k = 0; k < 512; k++)
        cond[k] = 1.0f;
    for (r = 0; r < 8; r++) {                   /* contiguous in cond_u */
        float        cr = c8[r];
        const float *cu = m->cu + r * 512;
        for (k = 0; k < 512; k++)
            cond[k] += cr * cu[k];
    }
    for (k = 0; k < 512; k++)
        z[k] = m->d1[k] * x[k];
    kron(z, m->a[0], m->b[0], y);
    for (k = 0; k < 512; k++)
        z[k] = silu(m->d2[k] * cond[k] * y[s_perm1[k]] + m->b2[k]);
    kron(z, m->a[1], m->b[1], y);
    for (k = 0; k < 512; k++)
        z[k] = m->d3[k] * y[s_perm2[k]];
    kron(z, m->a[2], m->b[2], y);
    for (k = 0; k < 512; k++)
        out[k] = m->d4[k] * y[k];
}

typedef struct {
    const nd_whistle *w;
    const mlp_w      *m;
    float            *x;
    uint32_t          norm_i;
    float             scale;
} mlp_ctx;

static void mlp_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const mlp_ctx *c = (const mlp_ctx *)vc;
    float          n[512], o[512];
    uint32_t       t, k;
    for (t = r0; t < r1; t++) {
        float *row = c->x + (size_t)t * c->w->d;
        zc(c->w, row, c->w->d, c->norm_i, n);
        mlp_row(c->m, n, o);
        for (k = 0; k < c->w->d; k++)
            row[k] += c->scale * o[k];
    }
}

/* x[T][d] += scale * MLP(zc(x, norm)), rows split across cores. */
static void mlp_residual(nd_whistle *w, float *x, uint32_t T, uint32_t norm_i,
                         uint32_t base, float scale)
{
    /* Internal SRAM: every row reads all 54 KB of it, and from PSRAM it
     * competed with the activations for the shared 64 KB cache. */
    mlp_w  *m = (mlp_w *)ND_ALLOC_FAST(sizeof(mlp_w));
    mlp_ctx c;
    ND_T0(t0);

    mlp_load(w, base, m);
    c.w = w; c.m = m; c.x = x; c.norm_i = norm_i; c.scale = scale;
    if (T > 1)
        nd_parallel_rows(mlp_rows, &c, T);
    else
        mlp_rows(&c, 0, 1);
    ND_FREE_FAST(m);
    ND_T1(t0, ND_WP_MLP);
}

/* ------------------------------------------------------------------ mHC */

static void sinkhorn(float *m, uint32_t n)
{
    uint32_t it, i, j;
    for (it = 0; it < SINKHORN; it++) {
        for (i = 0; i < n; i++) {
            float mx = m[i * n], s = 0.0f;
            for (j = 1; j < n; j++)
                if (m[i * n + j] > mx)
                    mx = m[i * n + j];
            for (j = 0; j < n; j++)
                s += nd_expf(m[i * n + j] - mx);
            s = mx + logf(s);
            for (j = 0; j < n; j++)
                m[i * n + j] -= s;
        }
        for (j = 0; j < n; j++) {
            float mx = m[j], s = 0.0f;
            for (i = 1; i < n; i++)
                if (m[i * n + j] > mx)
                    mx = m[i * n + j];
            for (i = 0; i < n; i++)
                s += nd_expf(m[i * n + j] - mx);
            s = mx + logf(s);
            for (i = 0; i < n; i++)
                m[i * n + j] -= s;
        }
    }
    for (i = 0; i < n * n; i++)
        m[i] = expf(m[i]);
}

typedef void (*block_fn)(nd_whistle *w, uint32_t layer, float *u, uint32_t T,
                         void *ctx);

/* One mHC step over T rows: mix lanes into u, run the block, scatter back.
 * `layer` indexes the parameters; `slot` (its position in the active stack)
 * picks the lane offsets, as the ladder subnetworks do. */
static void mhc_step(nd_whistle *w, uint32_t mhc, uint32_t layer, uint32_t slot,
                     float *lanes, uint32_t T, block_fn fn, void *ctx)
{
    uint32_t n = w->lanes, d = w->d, nd = n * d;
    float   *nx   = (float *)ND_ALLOC((size_t)T * nd * sizeof(float));
    float   *pre  = (float *)ND_ALLOC((size_t)T * n * sizeof(float));
    float   *post = (float *)ND_ALLOC((size_t)T * n * sizeof(float));
    float   *res  = (float *)ND_ALLOC((size_t)T * n * n * sizeof(float));
    float   *u    = (float *)ND_ALLOC((size_t)T * d * sizeof(float));
    float   *u0   = (float *)ND_ALLOC((size_t)T * d * sizeof(float));
    float   *mix  = (float *)ND_ALLOC((size_t)nd * sizeof(float));   /* off the stack: 8 KB */
    float    a_pre  = hv(w, mhc + H_A_PRE, layer);
    float    a_post = hv(w, mhc + H_A_POST, layer);
    float    a_res  = hv(w, mhc + H_A_RES, layer);
    uint32_t hot = slot % n, t, j, k, c;
    ND_T0(t0);

    for (t = 0; t < T; t++)
        rms_unit(lanes + (size_t)t * nd, nd, nx + (size_t)t * nd);
    {
        shared_in si;
        sh_begin(w, &si, mhc + H_PHI_PRE, nx, T);
        sh_rows(w, &si, mhc + H_PHI_PRE, layer * n, n, pre);
        sh_rows(w, &si, mhc + H_PHI_POST, layer * n, n, post);
        sh_rows(w, &si, mhc + H_PHI_RES, layer * n * n, n * n, res);
        sh_end(&si);
    }

    for (t = 0; t < T; t++) {
        float *l = lanes + (size_t)t * nd, *ut = u + (size_t)t * d;
        memset(ut, 0, d * sizeof(float));
        for (j = 0; j < n; j++) {
            float off = j == hot ? 4.0f : -4.0f;
            float h   = sigm(a_pre * pre[t * n + j] + hv(w, mhc + H_B_PRE, layer * n + j) + off);
            for (c = 0; c < d; c++)
                ut[c] += h * l[j * d + c];
        }
    }
    memcpy(u0, u, (size_t)T * d * sizeof(float));
    ND_T1(t0, ND_WP_MHC);
    fn(w, layer, u, T, ctx);
    {
    ND_T0(t1);

    for (t = 0; t < T; t++) {
        float *l = lanes + (size_t)t * nd, *r = res + (size_t)t * n * n;
        for (k = 0; k < n * n; k++)
            r[k] = a_res * r[k] + hv(w, mhc + H_B_RES, layer * n * n + k);
        sinkhorn(r, n);
        for (j = 0; j < n; j++) {
            float off  = j == hot ? 0.0f : -4.0f;
            float hp   = 2.0f * sigm(a_post * post[t * n + j] +
                                     hv(w, mhc + H_B_POST, layer * n + j) + off);
            for (c = 0; c < d; c++) {
                float s = 0.0f;
                for (k = 0; k < n; k++)
                    s += r[j * n + k] * l[k * d + c];
                mix[j * d + c] = s + hp * (u[(size_t)t * d + c] - u0[(size_t)t * d + c]);
            }
        }
        memcpy(l, mix, nd * sizeof(float));
    }
    ND_T1(t1, ND_WP_MHC);
    }
    ND_FREE(nx); ND_FREE(pre); ND_FREE(post); ND_FREE(res); ND_FREE(u); ND_FREE(u0);
    ND_FREE(mix);
}

static void lanes_init(const float *x, uint32_t T, uint32_t n, uint32_t d, float *lanes)
{
    uint32_t t, j;
    for (t = 0; t < T; t++)
        for (j = 0; j < n; j++)
            memcpy(lanes + ((size_t)t * n + j) * d, x + (size_t)t * d, d * sizeof(float));
}

static void lanes_out(const nd_whistle *w, const float *lanes, uint32_t T,
                      uint32_t norm_i, float *out)
{
    uint32_t n = w->lanes, d = w->d, t, j, c;
    float    m[512] = { 0 };   /* d <= 512 is checked at open */
    for (t = 0; t < T; t++) {
        for (c = 0; c < d; c++) {
            float s = 0.0f;
            for (j = 0; j < n; j++)
                s += lanes[((size_t)t * n + j) * d + c];
            m[c] = s / (float)n;
        }
        zc(w, m, d, norm_i, out + (size_t)t * d);
    }
}

/* ------------------------------------------------------------ front end */

static void fft512(const nd_whistle *w, float *re, float *im)
{
    uint32_t n = 512, i, j, len;
    for (i = 1, j = 0; i < n; i++) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        uint32_t step = n / len;               /* twiddle stride in the table */
        for (i = 0; i < n; i += len) {
            for (j = 0; j < len / 2; j++) {
                float wr = w->fft_cos[j * step], wi = w->fft_sin[j * step];
                float ur = re[i + j], ui = im[i + j];
                float vr = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
                float vi = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
                re[i + j] = ur + vr;            im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr;  im[i + j + len / 2] = ui - vi;
            }
        }
    }
}

void nd_whistle_logmel(const nd_whistle *w, const float *x, uint32_t n, float *out)
{
    uint32_t     M = n / w->hop, nb = w->n_fft / 2 + 1, mels = w->mels;
    uint32_t     t, k, b;
    const float *fb = (const float *)dat(w, w->i_melfb);      /* [nb][mels] */
    float        peak = 0.0f, gain, re[512], im[512], pw[257];
    float       *mean, *var;

    for (k = 0; k < n; k++)
        if (fabsf(x[k]) > peak)
            peak = fabsf(x[k]);
    gain = peak > 1e-9f ? 1.0f / peak : 1.0f;

    for (t = 0; t < M; t++) {
        /* Centred frame: n_fft samples around t*hop, zero padded at the
         * edges, periodic Hann over the middle `win` samples. */
        for (k = 0; k < w->n_fft; k++) {
            long s = (long)t * w->hop + (long)k - (long)(w->n_fft / 2);
            re[k] = (s >= 0 && s < (long)n) ? x[s] * gain * w->hann[k] : 0.0f;
            im[k] = 0.0f;
        }
        fft512(w, re, im);
        for (k = 0; k < nb; k++)
            pw[k] = re[k] * re[k] + im[k] * im[k];
        for (b = 0; b < mels; b++) {           /* only the band's own bins */
            float s = 0.0f;
            for (k = w->fb_lo[b]; k < w->fb_hi[b]; k++)
                s += pw[k] * fb[k * mels + b];
            out[(size_t)t * mels + b] = logf(s + LOG_EPS);
        }
    }

    /* Normalise each mel channel over the clip (sample std, ddof=1). */
    mean = (float *)ND_ALLOC(mels * sizeof(float));
    var  = (float *)ND_ALLOC(mels * sizeof(float));
    for (b = 0; b < mels; b++) {
        float s = 0.0f, q = 0.0f;
        for (t = 0; t < M; t++)
            s += out[(size_t)t * mels + b];
        s /= (float)(M ? M : 1);
        for (t = 0; t < M; t++) {
            float dd = out[(size_t)t * mels + b] - s;
            q += dd * dd;
        }
        mean[b] = s;
        var[b]  = M > 1 ? sqrtf(q / (float)(M - 1)) : 0.0f;
    }
    for (t = 0; t < M; t++)
        for (b = 0; b < mels; b++)
            out[(size_t)t * mels + b] = (out[(size_t)t * mels + b] - mean[b]) / (var[b] + NORM_EPS);
    ND_FREE(mean);
    ND_FREE(var);
}

/* ------------------------------------------------------------------ stem */

/* 3x3 'same' conv, stride 2 on both axes, over [T][F][C], for output rows
 * [o0, o0+on) only. The input buffer holds global rows [i0, i0+in); rows
 * outside [0, T) are the zero padding. Kernel [9][C], index kt*3 + kf.
 * in_c 1 broadcasts one input channel to C outputs; otherwise depthwise. */
typedef struct {
    const float *k, *x;
    float       *out;
    uint32_t     i0, T, F, in_c, C, o0;
    int          act;       /* apply SiLU before storing */
} conv_ctx;

/* Output rows [r0, r1) of the chunk. Each output pixel's C channels are
 * summed in a local buffer and stored once: accumulating straight into the
 * PSRAM output re-read and re-wrote it nine times per pixel. */
static void conv_part(void *vc, uint32_t r0, uint32_t r1)
{
    const conv_ctx *cc = (const conv_ctx *)vc;
    uint32_t        Fo = (cc->F + 1) / 2, C = cc->C, r, fo, c, a, b;
    float           acc[128];
    for (r = r0; r < r1; r++) {
        uint32_t to = cc->o0 + r;
        for (fo = 0; fo < Fo; fo++) {
            memset(acc, 0, C * sizeof(float));
            for (a = 0; a < 3; a++) {
                long tt = 2 * (long)to + a - 1;
                if (tt < 0 || tt >= (long)cc->T)
                    continue;
                for (b = 0; b < 3; b++) {
                    long         ff = 2 * (long)fo + b - 1;
                    const float *xi, *kk = cc->k + (a * 3 + b) * C;
                    if (ff < 0 || ff >= (long)cc->F)
                        continue;
                    xi = cc->x + ((size_t)(tt - cc->i0) * cc->F + ff) * cc->in_c;
                    if (cc->in_c == 1) {             /* one channel fans out */
                        float xv = xi[0];
                        for (c = 0; c < C; c++)
                            acc[c] += xv * kk[c];
                    } else {                         /* depthwise */
                        for (c = 0; c < C; c++)
                            acc[c] += xi[c] * kk[c];
                    }
                }
            }
            if (cc->act)
                for (c = 0; c < C; c++)
                    acc[c] = silu(acc[c]);
            memcpy(cc->out + ((size_t)r * Fo + fo) * C, acc, C * sizeof(float));
        }
    }
}

/* 3x3 'same' conv, stride 2 on both axes, over [T][F][C], for output rows
 * [o0, o0+on) only. The input buffer holds global rows [i0, i0+in); rows
 * outside [0, T) are the zero padding. Kernel [9][C], index kt*3 + kf.
 * in_c 1 broadcasts one input channel to C outputs; otherwise depthwise. */
static void conv_rows(const float *k, const float *x, uint32_t i0, uint32_t T,
                      uint32_t F, uint32_t in_c, uint32_t C, uint32_t o0,
                      uint32_t on, float *out, int act)
{
    conv_ctx cc;
    cc.k = k; cc.x = x; cc.out = out; cc.i0 = i0; cc.T = T; cc.F = F;
    cc.in_c = in_c; cc.C = C; cc.o0 = o0; cc.act = act;
    nd_parallel_rows(conv_part, &cc, on);
}

static void silu_inplace(float *x, size_t n)
{
    size_t k;
    for (k = 0; k < n; k++)
        x[k] = silu(x[k]);
}

#define STEM_CHUNK 8      /* encoder frames per pass */

/* [M][mels] log-mel -> [T][d]. Run in chunks of STEM_CHUNK output frames,
 * each computing just the rows of every stage it depends on: the first
 * stage is [M/2][mels/2][128] floats, ~5 MB for a 5 s clip, which would not
 * fit beside everything else in PSRAM. */
static uint32_t stem(nd_whistle *w, const float *lm, uint32_t M, float *out)
{
    uint32_t C = w->t[w->i_stem + S_W].shape[1];          /* 128 */
    uint32_t F0 = w->mels, F1 = (F0 + 1) / 2, F2 = (F1 + 1) / 2, F3 = (F2 + 1) / 2;
    uint32_t T1 = (M + 1) / 2, T2 = (T1 + 1) / 2, T3 = (T2 + 1) / 2;
    uint32_t rows1 = 4 * STEM_CHUNK + 4, a, c, f, t;
    float   *k0 = (float *)ND_ALLOC(9 * C * sizeof(float));
    float   *k1 = (float *)ND_ALLOC(9 * C * sizeof(float));
    float   *k2 = (float *)ND_ALLOC(9 * C * sizeof(float));
    float   *b1 = (float *)ND_ALLOC((size_t)rows1 * F1 * C * sizeof(float));
    float   *b2 = (float *)ND_ALLOC((size_t)rows1 * F1 * C * sizeof(float));
    float   *flat = (float *)ND_ALLOC((size_t)STEM_CHUNK * C * F3 * sizeof(float));

    hvec(w, w->i_stem + S_W, k0, 9 * C);
    hvec(w, w->i_stem + S_DW1, k1, 9 * C);
    hvec(w, w->i_stem + S_DW2, k2, 9 * C);
    for (a = 0; a < T3; a += STEM_CHUNK) {
        uint32_t b   = a + STEM_CHUNK < T3 ? a + STEM_CHUNK : T3;
        uint32_t r2a = a ? 2 * a - 1 : 0, r2b = 2 * b < T2 ? 2 * b : T2;
        uint32_t r1a = r2a ? 2 * r2a - 1 : 0, r1b = 2 * r2b < T1 ? 2 * r2b : T1;
        uint32_t n1 = r1b - r1a, n2 = r2b - r2a, n3 = b - a;

        conv_rows(k0, lm, 0, M, F0, 1, C, r1a, n1, b1, 1);
        conv_rows(k1, b1, r1a, T1, F1, C, C, r2a, n2, b2, 0);
        lin(w, w->i_stem + S_PW1, b2, n2 * F2, b1);
        silu_inplace(b1, (size_t)n2 * F2 * C);
        conv_rows(k2, b1, r2a, T2, F2, C, C, a, n3, b2, 0);
        lin(w, w->i_stem + S_PW2, b2, n3 * F3, b1);
        silu_inplace(b1, (size_t)n3 * F3 * C);
        for (t = 0; t < n3; t++)                         /* channel-major */
            for (f = 0; f < F3; f++)
                for (c = 0; c < C; c++)
                    flat[(size_t)t * C * F3 + c * F3 + f] = b1[((size_t)t * F3 + f) * C + c];
        lin(w, w->i_stem + S_OUT, flat, n3, out + (size_t)a * w->d);
    }
    ND_FREE(k0); ND_FREE(k1); ND_FREE(k2); ND_FREE(b1); ND_FREE(b2); ND_FREE(flat);

    /* Sinusoidal position, gated by sigmoid(pe_gate) (~0.002 in this blob). */
    {
        float g = sigm(hv(w, w->i_pe_gate, 0));
        for (t = 0; t < T3; t++)
            for (c = 0; c < w->d; c += 2) {
                float div = expf(-(float)c * logf(10000.0f) / (float)w->d);
                out[(size_t)t * w->d + c]     += g * sinf((float)t * div);
                out[(size_t)t * w->d + c + 1] += g * cosf((float)t * div);
            }
    }
    return T3;
}

/* --------------------------------------------------------------- encoder */

/* Non-causal GQA self attention over T rows of normed input xn. */
static void enc_attention(nd_whistle *w, uint32_t L, const float *xn, uint32_t T,
                          float *out)
{
    uint32_t nh = w->nh, nkv = w->nkv, qk = w->qk, vd = w->vd, rep = nh / nkv;
    float   *q  = (float *)ND_ALLOC((size_t)T * nh * qk * sizeof(float));
    float   *k  = (float *)ND_ALLOC((size_t)T * nkv * qk * sizeof(float));
    float   *v  = (float *)ND_ALLOC((size_t)T * nkv * vd * sizeof(float));
    float   *o  = (float *)ND_ALLOC((size_t)T * nh * vd * sizeof(float));
    float   *g  = (float *)ND_ALLOC((size_t)T * nh * vd * sizeof(float));
    float   *sc = (float *)ND_ALLOC((size_t)T * sizeof(float));
    float    tmp[64], scale = 1.0f / sqrtf((float)qk);
    uint32_t t, h, s, e;

    shared_in si;
    sh_begin(w, &si, L + E_Q, xn, T);
    sh_lin(w, &si, L + E_Q, q);
    sh_lin(w, &si, L + E_K, k);
    sh_lin(w, &si, L + E_V, v);
    for (t = 0; t < T; t++) {
        for (h = 0; h < nh; h++) {
            float *z = q + ((size_t)t * nh + h) * qk;
            zc(w, z, qk, L + E_QN, tmp);
            memcpy(z, tmp, qk * sizeof(float));
            rope(w, z, qk, t);
        }
        for (h = 0; h < nkv; h++) {
            float *z = k + ((size_t)t * nkv + h) * qk;
            zc(w, z, qk, L + E_KN, tmp);
            memcpy(z, tmp, qk * sizeof(float));
            rope(w, z, qk, t);
        }
    }
    {
    ND_T0(ta);
    for (t = 0; t < T; t++)
        for (h = 0; h < nh; h++) {
            const float *qh = q + ((size_t)t * nh + h) * qk;
            uint32_t     kh = h / rep;
            float       *oh = o + ((size_t)t * nh + h) * vd;
            for (s = 0; s < T; s++) {
                const float *ks = k + ((size_t)s * nkv + kh) * qk;
                float        a  = 0.0f;
                for (e = 0; e < qk; e++)
                    a += qh[e] * ks[e];
                sc[s] = a * scale;
            }
            softmax(sc, T);
            memset(oh, 0, vd * sizeof(float));
            for (s = 0; s < T; s++) {
                const float *vs = v + ((size_t)s * nkv + kh) * vd;
                for (e = 0; e < vd; e++)
                    oh[e] += sc[s] * vs[e];
            }
        }
    ND_T1(ta, ND_WP_ATTN);
    }
    sh_lin(w, &si, L + E_GATE, g);
    sh_end(&si);
    for (t = 0; t < (size_t)T * nh * vd; t++)
        o[t] *= sigm(g[t]);
    lin(w, L + E_OUT, o, T, out);
    ND_FREE(q); ND_FREE(k); ND_FREE(v); ND_FREE(o); ND_FREE(g); ND_FREE(sc);
}

static void enc_block(nd_whistle *w, uint32_t layer, float *x, uint32_t T, void *ctx)
{
    uint32_t L = w->i_enc + layer * E_N, d = w->d, t, c, j;
    float   *xn = (float *)ND_ALLOC((size_t)T * d * sizeof(float));
    float   *o  = (float *)ND_ALLOC((size_t)T * 2 * d * sizeof(float));
    float   *g  = (float *)ND_ALLOC((size_t)T * d * sizeof(float));
    float    tmp[512], dw[9 * 512];
    (void)ctx;

    /* Half-step Hadamard MLP (macaron). */
    mlp_residual(w, x, T, L + E_PRE_HADA0, L + E_MLP0, 0.5f);

    /* Gated attention. */
    for (t = 0; t < T; t++)
        zc(w, x + (size_t)t * d, d, L + E_NORM_IN, xn + (size_t)t * d);
    enc_attention(w, L, xn, T, o);
    {
        float gate = sigm(hv(w, L + E_AGATE, 0));
        for (t = 0; t < T; t++) {
            zc(w, o + (size_t)t * d, d, L + E_POST, tmp);
            for (c = 0; c < d; c++)
                x[(size_t)t * d + c] += gate * tmp[c];
        }
    }

    /* Conv module: pw1 -> GLU -> 9-tap depthwise (centred) -> norm -> SiLU -> pw2. */
    for (t = 0; t < T; t++)
        zc(w, x + (size_t)t * d, d, L + E_CONV_NORM, xn + (size_t)t * d);
    lin(w, L + E_PW1, xn, T, o);                         /* [T][2d] */
    for (t = 0; t < T; t++)
        for (c = 0; c < d; c++)
            g[(size_t)t * d + c] = o[(size_t)t * 2 * d + c] * sigm(o[(size_t)t * 2 * d + d + c]);
    hvec(w, L + E_DW, dw, 9 * d);
    for (t = 0; t < T; t++)
        for (c = 0; c < d; c++) {
            float s = 0.0f;
            for (j = 0; j < 9; j++) {
                long tt = (long)t + (long)j - 4;
                if (tt >= 0 && tt < (long)T)
                    s += dw[j * d + c] * g[(size_t)tt * d + c];
            }
            xn[(size_t)t * d + c] = s;
        }
    for (t = 0; t < T; t++) {
        zc(w, xn + (size_t)t * d, d, L + E_CONV_OUT_NORM, tmp);
        for (c = 0; c < d; c++)
            xn[(size_t)t * d + c] = silu(tmp[c]);
    }
    lin(w, L + E_PW2, xn, T, o);
    for (t = 0; t < (size_t)T * d; t++)
        x[t] += o[t];

    mlp_residual(w, x, T, L + E_PRE_HADA, L + E_MLP, 0.5f);
    ND_FREE(xn); ND_FREE(o); ND_FREE(g);
}

uint32_t nd_whistle_frames(const nd_whistle *w, uint32_t n)
{
    uint32_t m = n / w->hop;
    m = (m + 1) / 2;
    m = (m + 1) / 2;
    return (m + 1) / 2;
}

int nd_whistle_encode(nd_whistle *w, const float *x, uint32_t n, float *out)
{
    uint32_t M = n / w->hop, T, i;
    float   *lm, *s, *lanes;

    if (M == 0)
        return -1;
    {
        ND_T0(t0);
        lm = (float *)ND_ALLOC((size_t)M * w->mels * sizeof(float));
        nd_whistle_logmel(w, x, n, lm);
        ND_T1(t0, ND_WP_MEL);
    }
    {
        ND_T0(t0);
        s = (float *)ND_ALLOC((size_t)nd_whistle_frames(w, n) * w->d * sizeof(float));
        T = stem(w, lm, M, s);
        ND_FREE(lm);
        ND_T1(t0, ND_WP_STEM);
    }
    {
        ND_T0(t0);
        lanes = (float *)ND_ALLOC((size_t)T * w->lanes * w->d * sizeof(float));
        lanes_init(s, T, w->lanes, w->d, lanes);
        for (i = 0; i < w->enc_layers; i++)
            mhc_step(w, w->i_enc_mhc, i, i, lanes, T, enc_block, NULL);
        lanes_out(w, lanes, T, w->i_enc_final, out);
        ND_FREE(lanes);
        ND_FREE(s);
        ND_T1(t0, ND_WP_ENCODER);
    }
    return 0;
}

/* --------------------------------------------------------------- decoder */

typedef struct {
    uint32_t pos;
    float   *ek[4], *ev[4];       /* engram k / v for this token, per site */
} dec_ctx;

static void engram_step(nd_whistle *w, uint32_t tok, uint32_t pos, dec_ctx *dc)
{
    const nd_cact_header4 *h = &w->c.h4;
    uint32_t heads = h->engram_tables / h->num_orders;
    uint32_t stride = h->engram_seed_heads ? h->engram_seed_heads : heads;
    uint32_t sub = h->engram_sub_dim, s, oi, hh, j, col;
    uint32_t slot[16], ok[16];
    float    e[1024], row[256], vraw[512];

    w->hist[pos & 7u] = tok;
    col = 0;
    for (oi = 0; oi < h->num_orders; oi++)
        for (hh = 0; hh < heads; hh++) {
            uint32_t acc = EG_SEED * (oi * stride + hh + 1);
            for (j = 0; j < h->orders[oi]; j++) {
                uint32_t u = pos >= j ? w->hist[(pos - j) & 7u] : 0u;
                acc = (acc ^ u) * EG_PRIME;
            }
            acc ^= acc >> 15;
            slot[col] = acc % h->engram_slots;
            ok[col]   = pos + 1 >= h->orders[oi];
            col++;
        }
    for (s = 0; s < w->n_sites; s++) {
        uint32_t G = w->i_engram + s * 4, tt;
        for (tt = 0; tt < h->engram_tables; tt++) {
            if (ok[tt])
                nd_cq_dequant_row(&w->c, &w->t[G + G_TABLES], dat(w, G + G_TABLES),
                                  tt * h->engram_slots + slot[tt], row, e + tt * sub);
            else
                memset(e + tt * sub, 0, sub * sizeof(float));
        }
        {
            shared_in si;
            sh_begin(w, &si, G + G_KEY, e, 1);
            sh_lin(w, &si, G + G_KEY, dc->ek[s]);
            sh_lin(w, &si, G + G_VALUE, vraw);
            sh_end(&si);
        }
        memcpy(w->eg_hist + ((size_t)s * EG_HIST + (pos & (EG_HIST - 1))) * w->d,
               vraw, w->d * sizeof(float));
        memset(dc->ev[s], 0, w->d * sizeof(float));
        for (j = 0; j < h->engram_conv_taps; j++) {
            uint32_t sh = j * h->engram_dilation, c;
            const float *src;
            if (sh > pos)
                break;
            src = w->eg_hist + ((size_t)s * EG_HIST + ((pos - sh) & (EG_HIST - 1))) * w->d;
            for (c = 0; c < w->d; c++)
                dc->ev[s][c] += hv(w, G + G_TAPS, j * w->d + c) * src[c];
        }
    }
}

static void dec_block(nd_whistle *w, uint32_t layer, float *x, uint32_t T, void *ctx)
{
    dec_ctx *dc = (dec_ctx *)ctx;
    uint32_t L = w->i_dec + layer * D_N, X = w->i_cross + layer * X_N;
    uint32_t d = w->d, nh = w->nh, nkv = w->nkv, qk = w->qk, vd = w->vd;
    uint32_t pos = dc->pos, rep = nh / nkv, s, h, e, j, c;
    uint32_t nq = nh * qk, nk = nkv * qk, nv = nkv * vd, nraw = nq + nk + nv;
    float    xn[512], raw[384 + 96 + 128], q[384], o[512], gt[512], t2[512], tmp[64];
    float   *kc = w->k_cache + ((size_t)layer * w->max_seq + pos) * nk;
    float   *vc = w->v_cache + ((size_t)layer * w->max_seq + pos) * nv;
    float   *sc = w->tmp;
    float    scale = 1.0f / sqrtf((float)qk);
    (void)T;

    /* Engram injection at its sites. */
    for (s = 0; s < w->n_sites; s++)
        if (w->sites[s] == layer) {
            float a[512], b[512], dot = 0.0f, alpha;
            rms_unit(x, d, a);
            rms_unit(dc->ek[s], d, b);
            for (c = 0; c < d; c++)
                dot += a[c] * b[c];
            alpha = sigm(dot / sqrtf((float)d));
            for (c = 0; c < d; c++)
                x[c] += alpha * dc->ev[s][c];
        }

    /* Self attention with causal 3-tap conv on Q, K, V. */
    zc(w, x, d, L + D_NORM_IN, xn);
    {
    shared_in si;
    sh_begin(w, &si, L + D_Q, xn, 1);
    sh_lin(w, &si, L + D_Q, raw);
    sh_lin(w, &si, L + D_K, raw + nq);
    sh_lin(w, &si, L + D_V, raw + nq + nk);
    sh_lin(w, &si, L + D_GATE, gt);
    sh_end(&si);
    }
    {
        float *hist = w->qkv_hist + (size_t)layer * (w->taps - 1) * nraw;
        float  conv[384 + 96 + 128];
        for (e = 0; e < nraw; e++) {
            uint32_t ti = e < nq ? L + D_QT : e < nq + nk ? L + D_KT : L + D_VT;
            uint32_t ci = e < nq ? e : e < nq + nk ? e - nq : e - nq - nk;
            uint32_t width = e < nq ? nq : e < nq + nk ? nk : nv;
            float    acc = hv(w, ti, ci) * raw[e];
            for (j = 1; j < w->taps; j++)
                if (pos >= j)
                    acc += hv(w, ti, j * width + ci) *
                           hist[((pos - j) % (w->taps - 1)) * nraw + e];
            conv[e] = acc;
        }
        memcpy(hist + (pos % (w->taps - 1)) * nraw, raw, nraw * sizeof(float));
        memcpy(raw, conv, nraw * sizeof(float));
    }
    for (h = 0; h < nh; h++) {
        zc(w, raw + h * qk, qk, L + D_QN, q + h * qk);
        rope(w, q + h * qk, qk, pos);
    }
    for (h = 0; h < nkv; h++) {
        zc(w, raw + nq + h * qk, qk, L + D_KN, tmp);
        rope(w, tmp, qk, pos);
        memcpy(kc + h * qk, tmp, qk * sizeof(float));
    }
    memcpy(vc, raw + nq + nk, nv * sizeof(float));
    for (h = 0; h < nh; h++) {
        uint32_t kh = h / rep;
        for (s = 0; s <= pos; s++) {
            const float *ks = w->k_cache + ((size_t)layer * w->max_seq + s) * nk + kh * qk;
            float        a  = 0.0f;
            for (e = 0; e < qk; e++)
                a += q[h * qk + e] * ks[e];
            sc[s] = a * scale;
        }
        softmax(sc, pos + 1);
        for (e = 0; e < vd; e++) {
            float acc = 0.0f;
            for (s = 0; s <= pos; s++)
                acc += sc[s] * w->v_cache[((size_t)layer * w->max_seq + s) * nv + kh * vd + e];
            o[h * vd + e] = acc;
        }
    }
    for (c = 0; c < nh * vd; c++)
        o[c] *= sigm(gt[c]);
    lin(w, L + D_OUT, o, 1, t2);
    zc(w, t2, d, L + D_POST, gt);
    {
        float g = sigm(hv(w, L + D_AGATE, 0));
        for (c = 0; c < d; c++)
            x[c] += g * gt[c];
    }

    /* Gated cross attention over the clip (K, V precomputed per layer). */
    zc(w, x, d, X + X_NORM, xn);
    {
    shared_in si;
    sh_begin(w, &si, X + X_Q, xn, 1);
    sh_lin(w, &si, X + X_Q, raw);
    sh_lin(w, &si, X + X_GATE, gt);
    sh_end(&si);
    }
    for (h = 0; h < nh; h++) {
        const float *K = w->cross_k + ((size_t)layer * nh + h) * w->enc_T * qk;
        const float *V = w->cross_v + ((size_t)layer * nh + h) * w->enc_T * vd;
        zc(w, raw + h * qk, qk, X + X_QN, tmp);
        for (s = 0; s < w->enc_T; s++) {
            float a = 0.0f;
            for (e = 0; e < qk; e++)
                a += tmp[e] * K[(size_t)s * qk + e];
            sc[s] = a * scale;
        }
        softmax(sc, w->enc_T);
        for (e = 0; e < vd; e++) {
            float acc = 0.0f;
            for (s = 0; s < w->enc_T; s++)
                acc += sc[s] * V[(size_t)s * vd + e];
            o[h * vd + e] = acc;
        }
    }
    for (c = 0; c < nh * vd; c++)
        o[c] *= sigm(gt[c]);
    lin(w, X + X_OUT, o, 1, t2);
    zc(w, t2, d, X + X_POST, gt);
    {
        float g = sigm(hv(w, X + X_GATE_S, 0));
        for (c = 0; c < d; c++)
            x[c] += g * gt[c];
    }

    mlp_residual(w, x, 1, L + D_PRE_HADA, L + D_MLP, 1.0f);
}

static int session_begin(nd_whistle *w, const float *enc, uint32_t T)
{
    uint32_t nh = w->nh, qk = w->qk, vd = w->vd, L = w->dec_layers, i, h, s;
    uint32_t nraw = nh * qk + w->nkv * (qk + vd);
    float   *k, *v, tmp[64];
    ND_T0(t0);

    w->enc_T    = T;
    w->cross_k  = (float *)ND_ALLOC((size_t)L * nh * T * qk * sizeof(float));
    w->cross_v  = (float *)ND_ALLOC((size_t)L * nh * T * vd * sizeof(float));
    w->k_cache  = (float *)ND_ALLOC((size_t)L * w->max_seq * w->nkv * qk * sizeof(float));
    w->v_cache  = (float *)ND_ALLOC((size_t)L * w->max_seq * w->nkv * vd * sizeof(float));
    w->qkv_hist = (float *)ND_ALLOC((size_t)L * (w->taps - 1) * nraw * sizeof(float));
    w->eg_hist  = (float *)ND_ALLOC((size_t)w->n_sites * EG_HIST * w->d * sizeof(float));
    w->tmp      = (float *)ND_ALLOC((size_t)(T > w->max_seq ? T : w->max_seq) * sizeof(float));
    if (!w->cross_k || !w->cross_v || !w->k_cache || !w->v_cache ||
        !w->qkv_hist || !w->eg_hist || !w->tmp)
        return -1;
    memset(w->hist, 0, sizeof(w->hist));

    k = (float *)ND_ALLOC((size_t)T * nh * qk * sizeof(float));
    v = (float *)ND_ALLOC((size_t)T * nh * vd * sizeof(float));
    for (i = 0; i < w->depth; i++) {
        uint32_t l = w->active[i], X = w->i_cross + l * X_N;
        shared_in si;
        sh_begin(w, &si, X + X_K, enc, T);
        sh_lin(w, &si, X + X_K, k);
        sh_lin(w, &si, X + X_V, v);
        sh_end(&si);
        for (s = 0; s < T; s++)
            for (h = 0; h < nh; h++) {
                zc(w, k + ((size_t)s * nh + h) * qk, qk, X + X_KN, tmp);
                memcpy(w->cross_k + (((size_t)l * nh + h) * T + s) * qk, tmp, qk * sizeof(float));
                memcpy(w->cross_v + (((size_t)l * nh + h) * T + s) * vd,
                       v + ((size_t)s * nh + h) * vd, vd * sizeof(float));
            }
    }
    ND_FREE(k);
    ND_FREE(v);
    ND_T1(t0, ND_WP_CROSS_KV);
    return 0;
}

static void session_end(nd_whistle *w)
{
    ND_FREE(w->cross_k); ND_FREE(w->cross_v);
    ND_FREE(w->k_cache); ND_FREE(w->v_cache);
    ND_FREE(w->qkv_hist); ND_FREE(w->eg_hist); ND_FREE(w->tmp);
    w->cross_k = w->cross_v = w->k_cache = w->v_cache = NULL;
    w->qkv_hist = w->eg_hist = w->tmp = NULL;
}

/* Feed one token at `pos`; leaves the final hidden state in `hid`. */
static void dec_step(nd_whistle *w, uint32_t tok, uint32_t pos, float *hid)
{
    const nd_tensor *E = &w->t[w->i_emb];
    float            x[512], row[512];
    float           *kv = (float *)ND_ALLOC(8 * 512 * sizeof(float));   /* engram k, v */
    float           *lanes = w->lane;
    dec_ctx          dc;
    uint32_t         i, c;

    dc.pos = pos;
    for (i = 0; i < 4; i++) {
        dc.ek[i] = kv + i * 512;
        dc.ev[i] = kv + (4 + i) * 512;
    }
    nd_cq_dequant_row(&w->c, E, dat(w, w->i_emb), tok, row, x);
    for (c = 0; c < w->d; c++)
        x[c] *= sqrtf((float)w->d);
    engram_step(w, tok, pos, &dc);
    lanes_init(x, 1, w->lanes, w->d, lanes);
    for (i = 0; i < w->depth; i++)
        mhc_step(w, w->i_dec_mhc, w->active[i], i, lanes, 1, dec_block, &dc);
    lanes_out(w, lanes, 1, w->i_dec_final, hid);
    ND_FREE(kv);
}

static void logits_all(nd_whistle *w, const float *hid, float *out)
{
    lin(w, w->i_emb, hid, 1, out);       /* tied head: 8199 x 512, 4-bit */
}

int nd_whistle_logits(nd_whistle *w, const float *enc, uint32_t T,
                      const uint32_t *toks, uint32_t n, float *out)
{
    float    hid[512];
    uint32_t p;
    if (n > w->max_seq || session_begin(w, enc, T) != 0)
        return -1;
    for (p = 0; p < n; p++) {
        dec_step(w, toks[p], p, hid);
        logits_all(w, hid, out + (size_t)p * w->vocab);
    }
    session_end(w);
    return 0;
}

/* Keyword biasing, a small Aho-Corasick in spirit: for every stored phrase,
 * find the longest prefix of it that the transcript currently ends with and
 * favour the phrase's next token (its first token when nothing matches).
 *
 * A phrase already said is left alone, judged on the decoded text: without
 * that, the first-token bonus applies at every step and a word, once said,
 * is favoured again straight after itself ("Blue Blue Blue ..." to the token
 * limit). Token ids are not enough to tell - the model can spell "Blue" with
 * different pieces than the tokenizer encodes it with. */
static int is_alpha_u8(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || c >= 0x80;
}

/* `phrase` (lowercase) occurs as a whole word in `text` (lowercased here,
 * UTF-8 aware: "Weiß", "ŻÓŁTY"). */
static int said(const char *text_in, const char *phrase)
{
    char        text[512];
    size_t      m = strlen(phrase);
    const char *t;
    strncpy(text, text_in, sizeof(text) - 1);
    text[sizeof(text) - 1] = '\0';
    nd_utf8_lower(text);
    for (t = text; *t; t++) {
        if (t != text && is_alpha_u8((unsigned char)t[-1]))
            continue;                              /* not a word start */
        if (!strncmp(t, phrase, m) && !is_alpha_u8((unsigned char)t[m]))
            return 1;
    }
    return 0;
}

static void bias_keywords(const nd_whistle *w, const uint32_t *ids, uint32_t n,
                          float *lg)
{
    uint8_t  done[ND_WH_MAX_KW];      /* phrase group already said */
    uint32_t q, k, j;
    char     text[512];

    if (!w->n_kw)
        return;
    if (nd_tok_decode(&w->tok, ids, n, text, sizeof(text)) < 0)
        text[0] = '\0';
    memset(done, 0, sizeof(done));
    for (q = 0; q < w->n_kw; q++)
        if ((w->kw_lang[q] < 0 || w->kw_lang[q] == w->lang) &&
            said(text, w->kw_text[w->kw_group[q]]))
            done[w->kw_group[q]] = 1;
    for (q = 0; q < w->n_kw; q++) {
        uint32_t m = w->kw_len[q], best = 0;
        if (done[w->kw_group[q]] || (w->kw_lang[q] >= 0 && w->kw_lang[q] != w->lang))
            continue;
        for (k = (m - 1 < n ? m - 1 : n); k > 0; k--) {
            for (j = 0; j < k && ids[n - k + j] == w->kw[q][j]; j++)
                ;
            if (j == k) {
                best = k;
                break;
            }
        }
        lg[w->kw[q][best]] += w->kw_bonus;
    }
}

static int add_keywords(nd_whistle *w, int lang, const char *list)
{
    const char *p = list;
    while (*p) {
        const char *e = p;
        size_t      len;
        int         v;
        while (*e && *e != '\n')
            e++;
        len = (size_t)(e - p);
        if (len && len < 24 && w->n_groups < ND_WH_MAX_KW) {
            uint32_t g = w->n_groups++;
            memcpy(w->kw_text[g], p, len);
            w->kw_text[g][len] = '\0';
            nd_utf8_lower(w->kw_text[g]);
            for (v = 0; v < 2; v++) {
                char     buf[24];
                uint32_t ids[16];
                int      nt;
                memcpy(buf, w->kw_text[g], len + 1);
                if (v == 1) {                     /* capitalised variant */
                    if (buf[0] < 'a' || buf[0] > 'z')
                        break;
                    buf[0] = (char)(buf[0] - 32);
                }
                /* add_dummy: the phrase is a word inside the transcript, so
                 * its first piece carries the word-boundary marker. */
                nt = nd_tok_encode_ex(&w->tok, buf, len, ids, 16, 1);
                /* A capitalised spelling whose first piece is a bare capital
                 * or two ("▁V" + "iol" + "ette") would have that fragment
                 * boosted at every step, and it starts countless other words:
                 * French and Spanish transcripts grew stray "V ... Violette
                 * Violet". Leave those out. (Not lowercase spellings: many
                 * colours start with a short piece - "ro"+"uge", "v"+"ert" -
                 * and dropping them cost 10-25 points of command accuracy.) */
                if (nt > 1 && v == 1) {
                    uint16_t    pl;
                    const char *pc = nd_tok_piece(&w->tok, ids[0], &pl);
                    uint32_t    chars = 0, b;
                    for (b = 0; b < pl; b++)       /* UTF-8 chars, minus the ▁ marker */
                        if (((unsigned char)pc[b] & 0xC0) != 0x80)
                            chars++;
                    if (pl >= 3 && (unsigned char)pc[0] == 0xE2 && (unsigned char)pc[1] == 0x96 &&
                        (unsigned char)pc[2] == 0x81)
                        chars--;
                    if (chars <= 2)
                        nt = 0;
                }
                if (nt > 0 && nt <= ND_WH_KW_TOKENS && w->n_kw < ND_WH_MAX_KW) {
                    memcpy(w->kw[w->n_kw], ids, (size_t)nt * sizeof(uint32_t));
                    w->kw_group[w->n_kw] = (uint16_t)g;
                    w->kw_lang[w->n_kw]  = (int8_t)lang;
                    w->kw_len[w->n_kw++] = (uint32_t)nt;
                }
            }
        }
        p = *e ? e + 1 : e;
    }
    return (int)w->n_kw;
}

int nd_whistle_set_keywords(nd_whistle *w, const char *list, float bonus)
{
    w->n_kw = w->n_groups = 0;
    w->kw_bonus = bonus;
    if (!list || bonus == 0.0f)
        return 0;
    return add_keywords(w, -1, list);
}

int nd_whistle_add_keywords(nd_whistle *w, int lang, const char *list)
{
    if (!list || lang < 0 || lang >= ND_WH_LANGS)
        return (int)w->n_kw;
    return add_keywords(w, lang, list);
}

int nd_whistle_find_speech(const float *x, uint32_t n, float floor_rms,
                           float min_db, uint32_t margin_ms,
                           uint32_t *s0, uint32_t *s1)
{
    const uint32_t W = 320;                       /* 20 ms windows */
    uint32_t       nw = n / W, i, loud = 0, first = 0, last = 0;
    float         *r, thr;

    if (nw < 5)
        return 0;
    r = (float *)ND_ALLOC(nw * sizeof(float));
    if (!r)
        return 0;
    for (i = 0; i < nw; i++) {
        float    e = 0.0f;
        uint32_t k;
        for (k = 0; k < W; k++)
            e += x[i * W + k] * x[i * W + k];
        r[i] = sqrtf(e / (float)W);
    }
    if (floor_rms <= 0.0f) {                      /* 10th-percentile window */
        float   *sorted = (float *)ND_ALLOC(nw * sizeof(float));
        uint32_t a, b;
        memcpy(sorted, r, nw * sizeof(float));
        for (a = 1; a < nw; a++)
            for (b = a; b > 0 && sorted[b - 1] > sorted[b]; b--) {
                float t = sorted[b]; sorted[b] = sorted[b - 1]; sorted[b - 1] = t;
            }
        floor_rms = sorted[nw / 10];
        ND_FREE(sorted);
    }
    thr = (floor_rms > 1e-7f ? floor_rms : 1e-7f) * powf(10.0f, min_db / 20.0f);
    for (i = 0; i < nw; i++)
        if (r[i] > thr) {
            if (!loud)
                first = i;
            last = i;
            loud++;
        }
    ND_FREE(r);
    if (loud < 5)                                 /* under 100 ms of signal */
        return 0;
    {
        uint32_t m = margin_ms * 16;
        *s0 = first * W > m ? first * W - m : 0;
        *s1 = (last + 1) * W + m < n ? (last + 1) * W + m : n;
    }
    return 1;
}

/* Length of a run at the end of ids[0..n): the last k tokens repeated three
 * times back to back, for some k <= 4. Returns k, or 0. */
static uint32_t tail_loop(const uint32_t *ids, uint32_t n)
{
    uint32_t k, j;
    for (k = 1; k <= 4; k++) {
        if (n < 3 * k)
            break;
        for (j = 0; j < 2 * k && ids[n - 1 - j] == ids[n - 1 - j - k]; j++)
            ;
        if (j == 2 * k)
            return k;
    }
    return 0;
}

/* One greedy pass in `lang` over the current session, from BOS. The mean
 * log-prob of the chosen tokens (EOS included) is computed from the logits
 * before keyword bias, so it measures fit, not bias. */
static uint32_t decode_pass(nd_whistle *w, int lang, uint32_t *ids, uint32_t cap,
                            uint32_t budget, float *lg, float *raw, float *score)
{
    float    hid[512], sum = 0.0f;
    uint32_t pos = 0, n = 0, tok, k;

    dec_step(w, BOS_ID, pos++, hid);
    w->lang = lang;
    tok = LANG_BASE + (uint32_t)lang;
    while (pos < w->max_seq) {
        uint32_t best = 0;
        float    mx, se = 0.0f, lse;
        dec_step(w, tok, pos++, hid);
        logits_all(w, hid, lg);
        memcpy(raw, lg, w->vocab * sizeof(float));
        mx = raw[0];
        for (k = 1; k < w->vocab; k++)
            if (raw[k] > mx)
                mx = raw[k];
        for (k = 0; k < w->vocab; k++)
            se += nd_expf(raw[k] - mx);
        lse = mx + logf(se);
        bias_keywords(w, ids, n, lg);
        for (k = 1; k < w->vocab; k++)
            if (lg[k] > lg[best])
                best = k;
        sum += raw[best] - lse;
        if (best == EOS_ID || n >= cap || n >= budget)
            break;
        ids[n++] = best;
        tok = best;
        k = tail_loop(ids, n);
        if (k) {
            n -= 2 * k;
            break;
        }
    }
    *score = sum / (float)(n + 1);
    return n;
}

int nd_whistle_decode(nd_whistle *w, const float *enc, uint32_t T, int lang,
                      uint32_t *ids, uint32_t cap)
{
    /* Greedy decoding has no way out of a loop ("Blue Blue Blue ..."), which
     * keyword biasing makes easier to fall into. Two guards: stop when the
     * tail repeats itself three times (keeping one copy), and never emit more
     * than ~6 tokens per second of audio (an encoder frame is 80 ms). */
    uint32_t budget = 8 + T / 2;
    float   *lg  = (float *)ND_ALLOC(w->vocab * sizeof(float));
    float   *raw = (float *)ND_ALLOC(w->vocab * sizeof(float));
    uint32_t n = 0, k;

    if (lang >= ND_WH_LANGS || !lg || !raw || session_begin(w, enc, T) != 0) {
        ND_FREE(lg);
        ND_FREE(raw);
        return -1;
    }
    {
        ND_T0(t0);
        if (lang >= 0) {
            n = decode_pass(w, lang, ids, cap, budget, lg, raw, &w->score);
        } else {
            /* Detect: the decoder's own ranking of the language tokens after
             * BOS, over the enabled languages. */
            int      order[ND_WH_LANGS], no = 0, i, j;
            float    hid[512];
            uint32_t mask = w->lang_mask ? w->lang_mask : 0x7Fu;
            dec_step(w, BOS_ID, 0, hid);
            logits_all(w, hid, lg);
            for (k = 0; k < ND_WH_LANGS; k++) {
                w->lang_logit[k] = lg[LANG_BASE + k];
                if (mask & (1u << k))
                    order[no++] = (int)k;
            }
            for (i = 1; i < no; i++)
                for (j = i; j > 0 && w->lang_logit[order[j]] > w->lang_logit[order[j - 1]]; j--) {
                    int t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
                }
            n = decode_pass(w, order[0], ids, cap, budget, lg, raw, &w->score);
            w->score_first = w->score;
            if (w->accept && no > 1) {
                char text[512];
                nd_whistle_detok(w, ids, n, text, sizeof(text));
                if (!w->accept(text, w->accept_ctx)) {
                    /* Re-decode in the next languages and take the first
                     * transcript `accept` likes. Otherwise keep the first one:
                     * choosing by score instead turned 4 of 10 plain English
                     * questions into garbled other languages. */
                    uint32_t *alt = (uint32_t *)ND_ALLOC(cap * sizeof(uint32_t));
                    int       best = order[0];
                    for (i = 1; i <= (int)w->retry_langs && i < no && alt; i++) {
                        float    sc;
                        uint32_t m = decode_pass(w, order[i], alt, cap, budget, lg, raw, &sc);
                        nd_whistle_detok(w, alt, m, text, sizeof(text));
                        if (sc >= w->score_first - w->retry_margin &&
                            w->accept(text, w->accept_ctx)) {
                            w->score = sc;
                            best = order[i];
                            n = m;
                            memcpy(ids, alt, m * sizeof(uint32_t));
                            break;
                        }
                    }
                    w->lang = best;
                    ND_FREE(alt);
                }
            }
        }
        ND_T1(t0, ND_WP_DECODER);
    }
    session_end(w);
    ND_FREE(lg);
    ND_FREE(raw);
    return (int)n;
}

int nd_whistle_detok(const nd_whistle *w, const uint32_t *ids, uint32_t n,
                     char *out, uint32_t cap)
{
    int len = nd_tok_decode(&w->tok, ids, n, out, cap);
    int k = 0;
    /* The dummy-prefix rule strips one leading space, but the first piece
     * can carry more (seen on the board: "TXT  flash delight red."). */
    while (len > 0 && k < len && out[k] == ' ')
        k++;
    if (k) {
        memmove(out, out + k, (size_t)(len - k) + 1);
        len -= k;
    }
    return len;
}

/* ------------------------------------------------------------------ open */

size_t nd_whistle_cache_head(nd_whistle *w, void *(*alloc)(size_t))
{
    size_t n = (size_t)w->t[w->i_emb].nbytes;
    void  *p = alloc(n);
    if (!p)
        return 0;
    memcpy(p, w->p[w->i_emb], n);
    w->p[w->i_emb] = p;
    return n;
}

int nd_whistle_set_depth(nd_whistle *w, uint32_t depth)
{
    /* Needle 3's nested ladder: both endpoints first, then bisect the
     * largest gap (leftmost on ties). Depth k keeps the first k picks. */
    uint32_t L = w->dec_layers, order[ND_WH_MAX_LAYERS], sel[ND_WH_MAX_LAYERS];
    uint32_t n = 2, ns = 2, i, j;

    if (depth < 2 || depth > L)
        return -1;
    order[0] = sel[0] = 0;
    order[1] = sel[1] = L - 1;
    while (n < L) {
        uint32_t best = 0, bl = 0;
        for (i = 0; i + 1 < ns; i++) {
            uint32_t gap = sel[i + 1] - sel[i];
            if (gap > 1 && gap > best) {
                best = gap;
                bl = i;
            }
        }
        {
            uint32_t mid = (sel[bl] + sel[bl + 1]) / 2;
            order[n++] = mid;
            for (j = ns; j > bl + 1; j--)
                sel[j] = sel[j - 1];
            sel[bl + 1] = mid;
            ns++;
        }
    }
    for (i = 0; i < depth; i++)
        w->active[i] = order[i];
    for (i = 1; i < depth; i++)                  /* ascending */
        for (j = i; j > 0 && w->active[j - 1] > w->active[j]; j--) {
            uint32_t t = w->active[j];
            w->active[j] = w->active[j - 1];
            w->active[j - 1] = t;
        }
    w->depth = depth;
    return 0;
}

int nd_whistle_open(nd_whistle *w, const void *blob, size_t size)
{
    const nd_cact_header4 *h;
    uint32_t i, x, nt;
    int      rc;

    memset(w, 0, sizeof(*w));
    rc = nd_cact_open(&w->c, blob, size);
    if (rc != 0)
        return rc;
    if (w->c.version != 4)
        return -6;
    h  = &w->c.h4;
    nt = w->c.n;
    w->t = (nd_tensor *)ND_ALLOC(nt * sizeof(nd_tensor));
    if (!w->t)
        return -7;
    w->p = (const void **)ND_ALLOC(nt * sizeof(void *));
    if (!w->p)
        return -7;
    for (i = 0; i < nt; i++)
        if (nd_cact_tensor(&w->c, i, &w->t[i]) != 0 ||
            !(w->p[i] = nd_cact_data(&w->c, &w->t[i])))
            return -8;

    w->d = h->d_model;  w->nh = h->num_heads;  w->nkv = h->num_kv_heads;
    w->qk = h->qk_head_dim;  w->vd = h->v_head_dim;  w->lanes = h->mhc_lanes;
    w->taps = h->qkv_conv_taps;  w->vocab = h->vocab_size;
    w->max_seq = h->max_seq_len;  w->dec_layers = h->num_layers;
    w->n_sites = h->num_sites;  w->rope_theta = h->rope_theta;
    for (i = 0; i < h->qk_head_dim / 2 && i < 32; i++)
        w->rope_inv[i] = powf(w->rope_theta, -(float)(2 * i) / (float)h->qk_head_dim);
    for (i = 0; i < w->n_sites && i < 4; i++)
        w->sites[i] = h->sites[i];
    if (w->d != 512 || w->lanes > ND_MAX_LANES || w->taps < 2 ||
        w->dec_layers > ND_WH_MAX_LAYERS || w->n_sites > 4 || w->qk > 64 || w->nh * w->qk > 384 ||
        w->nh * w->vd > 512 || w->nkv * w->qk > 96 || w->nkv * w->vd > 128)
        return -9;           /* the fixed scratch below is sized for Whistle */

    x = 0;
    w->i_emb = x;                    x += 1;
    w->i_dec = x;                    x += w->dec_layers * D_N;
    w->i_dec_mhc = x;                x += 9;
    w->i_perm1 = x++;
    w->i_perm2 = x++;
    w->i_engram = x;                 x += w->n_sites * 4;
    w->i_dec_final = x++;
    w->i_audio_cfg = x++;
    w->i_cross = x;                  x += w->dec_layers * X_N;
    w->i_pe_gate = x++;
    if (w->t[w->i_audio_cfg].dtype != ND_DT_FP32 || w->t[w->i_audio_cfg].shape[0] != 13)
        return -10;
    w->enc_layers = (uint32_t)hv(w, w->i_audio_cfg, 3);
    w->mels  = (uint32_t)hv(w, w->i_audio_cfg, 8);
    w->n_fft = (uint32_t)hv(w, w->i_audio_cfg, 10);
    w->win   = (uint32_t)hv(w, w->i_audio_cfg, 11);
    w->hop   = (uint32_t)hv(w, w->i_audio_cfg, 12);
    w->i_enc = x;                    x += w->enc_layers * E_N;
    w->i_enc_mhc = x;                x += 9;
    w->i_enc_final = x++;
    w->i_stem = x;                   x += 6;
    w->i_melfb = x++;
    w->i_tok = x++;
    if (x != nt || w->t[w->i_tok].dtype != ND_DT_RAW || w->n_fft != 512 ||
        w->t[w->i_melfb].shape[0] != w->n_fft / 2 + 1 || w->t[w->i_melfb].shape[1] != w->mels)
        return -11;

    for (i = 0; i < 256; i++) {
        w->fft_cos[i] = (float)cos(-2.0 * M_PI * i / 512.0);
        w->fft_sin[i] = (float)sin(-2.0 * M_PI * i / 512.0);
    }
    for (i = 0; i < 512; i++) {
        uint32_t off = (w->n_fft - w->win) / 2;
        w->hann[i] = (i >= off && i < off + w->win)
            ? 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)(i - off) / (float)w->win) : 0.0f;
    }
    if (w->mels > 128)
        return -11;
    {
        const float *fb = (const float *)dat(w, w->i_melfb);
        uint32_t     b, k, nb = w->n_fft / 2 + 1;
        for (b = 0; b < w->mels; b++) {
            w->fb_lo[b] = (uint16_t)nb;
            w->fb_hi[b] = 0;
            for (k = 0; k < nb; k++)
                if (fb[k * w->mels + b] != 0.0f) {
                    if (k < w->fb_lo[b])
                        w->fb_lo[b] = (uint16_t)k;
                    w->fb_hi[b] = (uint16_t)(k + 1);
                }
            if (w->fb_lo[b] > w->fb_hi[b])
                w->fb_lo[b] = w->fb_hi[b];
        }
    }
    for (i = 0; i < 512; i++) {
        s_perm1[i] = (int32_t)hv(w, w->i_perm1, i);
        s_perm2[i] = (int32_t)hv(w, w->i_perm2, i);
    }
    if (nd_tok_init(&w->tok, dat(w, w->i_tok), (size_t)w->t[w->i_tok].nbytes) != 0)
        return -12;

    w->lane  = (float *)ND_ALLOC((size_t)w->lanes * w->d * sizeof(float));
    w->xh    = (float *)ND_ALLOC(2048 * sizeof(float));
    w->lut   = (float *)ND_ALLOC(nd_cq_lut_floats(2048) * sizeof(float));
    if (!w->lane || !w->xh || !w->lut)
        return -7;
    w->retry_langs = 2;
    /* Tuned on 1,020 command clips and 137 sentences without one
     * (tools/eval_languages.py): 0.2 nats keeps most rescues and adds no
     * false commands; a looser margin let garbage like "żół białe żół" in. */
    w->retry_margin = 0.2f;
    w->lang_mask   = 0x7Fu;
    return nd_whistle_set_depth(w, w->dec_layers);
}

void nd_whistle_close(nd_whistle *w)
{
    nd_tok_free(&w->tok);
    ND_FREE(w->t);
    ND_FREE((void *)w->p);
    ND_FREE(w->lane);
    ND_FREE(w->xh);
    ND_FREE(w->lut);
    memset(w, 0, sizeof(*w));
}
