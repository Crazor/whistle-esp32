/* nd_bench: where Whistle's decoder spends its time on the ESP32-P4.
 *
 * Loads the model from the "model" partition into PSRAM like whistle_demo,
 * then times the engine's own kernels on the model's own tensors with the
 * cycle counter and prints one table per question:
 *
 *   1. one decoder layer's matmuls at T = 1, 2, 4, 8, 16, 64, per token
 *   2. the int8 path split: unpack only, dot only, both (2- and 4-bit rows)
 *   3. the same rows cold in PSRAM, warm in the L2 cache, and in SRAM
 *   4. PSRAM bandwidth (32-bit and PIE loads, one and two cores, memcpy,
 *      async DMA) and the latency of one cache miss
 *   5. the output layer: all 8199 rows, and subsets via nd_cq_gemv_gather
 *   6. one whole decoder step at depth 4 and 8, split by phase
 *
 * No microphone, no LED: flash the app at 0x10000 next to the model, open the
 * console (tools/mon.py PORT 120 --reset) and wait for "DONE".
 *
 * "Cold" means the weights were evicted from both cache levels first, which
 * is how the decoder meets them: a token streams megabytes through a 256 KB
 * L2 cache, so nothing it read for the last token is still there.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_async_memcpy.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "hal/cache_ll.h"
#include "sdkconfig.h"

#include "nd_quant.h"
#include "nd_simd.h"
#include "nd_whistle.h"

#define MHZ            CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ
#define FLASH_MAP_END  0x1000000     /* see whistle_demo: the cache maps 16 MB */
#define EVICT_BYTES    (1024 * 1024) /* 4x the L2 cache */
#define BW_BYTES       (4 * 1024 * 1024)
#define REPS           3

/* Per-layer tensor offsets, as in nd_whistle.c (tools/whistle_cact.py). */
enum { D_Q = 1, D_K = 2, D_V = 3, D_GATE = 9, D_OUT = 10, D_N = 27 };
enum { X_Q = 1, X_GATE = 4, X_OUT = 5, X_N = 10 };
enum { H_PHI_PRE = 6, H_PHI_POST = 7, H_PHI_RES = 8 };
enum { G_KEY = 1, G_VALUE = 2 };

static nd_whistle s_w;
static uint8_t   *s_evict;

static inline uint32_t cyc(void) { return esp_cpu_get_cycle_count(); }
static double us(uint32_t c) { return (double)c / MHZ; }

/* Read EVICT_BYTES of unrelated PSRAM, one word per 64-byte L1 line: both
 * cache levels then hold that buffer and none of the weights. */
static void evict(void)
{
    volatile const uint32_t *p = (volatile const uint32_t *)s_evict;
    uint32_t                 i, s = 0;
    for (i = 0; i < EVICT_BYTES / 4; i += 16)
        s += p[i];
    (void)s;
}

static uint32_t s_rng = 12345;
static float frand(void)                 /* roughly N(0, 1) */
{
    float s = 0.0f;
    int   k;
    for (k = 0; k < 4; k++) {
        s_rng = s_rng * 1103515245u + 12345u;
        s += (float)(s_rng >> 8) / 16777216.0f;
    }
    return (s - 2.0f) * 1.7320508f;
}

static void fill(float *x, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++)
        x[i] = frand();
}

/* ------------------------------------------------- second-core row worker
 * As whistle_demo: every matmul hands the second half of its rows to core 1. */

static SemaphoreHandle_t s_go, s_done;
static nd_row_fn         s_fn;
static void             *s_ctx;
static uint32_t          s_r0, s_r1;

static void worker_task(void *arg)
{
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_go, portMAX_DELAY);
        s_fn(s_ctx, s_r0, s_r1);
        xSemaphoreGive(s_done);
    }
}

static void rows_dual_core(nd_row_fn fn, void *ctx, uint32_t nrows)
{
    uint32_t half = nrows / 2;
    if (half < 2 || !s_go) {
        fn(ctx, 0, nrows);
        return;
    }
    s_fn  = fn;
    s_ctx = ctx;
    s_r0  = half;
    s_r1  = nrows;
    xSemaphoreGive(s_go);
    fn(ctx, 0, half);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

static void rows_serial(nd_row_fn fn, void *ctx, uint32_t nrows) { fn(ctx, 0, nrows); }

static void worker_start(void)
{
    s_go   = xSemaphoreCreateBinary();
    s_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker_task, "nd_worker", 16384, NULL,
                            configMAX_PRIORITIES - 2, NULL, 1);
    nd_parallel_rows = rows_dual_core;
}

static void nop_rows(void *ctx, uint32_t r0, uint32_t r1) { (void)ctx; (void)r0; (void)r1; }

/* ------------------------------------------------------- model loading */

static void *model_to_psram(const esp_partition_t *part, const void *blob, size_t avail,
                            size_t *size)
{
    nd_cact   c;
    nd_tensor t;
    size_t    end = 0;
    uint32_t  i;
    void     *p;

    if (nd_cact_open(&c, blob, avail) != 0)
        return NULL;
    for (i = 0; i < c.n; i++)
        if (nd_cact_tensor(&c, i, &t) == 0 && t.offset + t.nbytes > end)
            end = (size_t)(t.offset + t.nbytes);
    if (end == 0 || end > part->size)
        return NULL;
    p = heap_caps_aligned_alloc(ND_CACT_ALIGN, end, MALLOC_CAP_SPIRAM);
    if (!p || esp_partition_read(part, 0, p, end) != ESP_OK) {
        heap_caps_free(p);
        return NULL;
    }
    *size = end;
    return p;
}

static int model_open(void)
{
    const esp_partition_t      *part;
    esp_partition_mmap_handle_t handle;
    const void                 *blob = NULL;
    void                       *copy;
    size_t                      size;
    int                         rc;

    part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) {
        printf("ERR no_model_partition\n");
        return -1;
    }
    size = part->size;
    if (part->address + size > FLASH_MAP_END)
        size = part->address < FLASH_MAP_END ? FLASH_MAP_END - part->address : 0;
    if (esp_partition_mmap(part, 0, size, ESP_PARTITION_MMAP_DATA, &blob, &handle) != ESP_OK) {
        printf("ERR mmap_failed\n");
        return -1;
    }
    copy = model_to_psram(part, blob, size, &size);
    if (!copy) {
        printf("ERR model_not_in_psram (the bench needs the model in PSRAM)\n");
        return -1;
    }
    esp_partition_munmap(handle);
    rc = nd_whistle_open(&s_w, copy, size);
    if (rc != 0) {
        printf("ERR whistle_open rc=%d\n", rc);
        return -1;
    }
    printf("model bytes=%u dec_layers=%u vocab=%u psram_free=%u internal_free=%u "
           "internal_largest=%u\n",
           (unsigned)size, (unsigned)s_w.dec_layers, (unsigned)s_w.vocab,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    return 0;
}

/* --------------------------------------------- 1. one layer's matmuls at T */

typedef struct {
    const char *name;
    uint32_t    ti, r0, nrows;
    int         shared;        /* reuses the previous entry's prepared input */
} mm;

static uint32_t layer_mms(uint32_t layer, mm *m)
{
    uint32_t L = s_w.i_dec + layer * D_N, X = s_w.i_cross + layer * X_N;
    uint32_t H = s_w.i_dec_mhc, n = s_w.lanes, k = 0;
    m[k++] = (mm){ "mhc phi_pre",  H + H_PHI_PRE,  layer * n,     n,     0 };
    m[k++] = (mm){ "mhc phi_post", H + H_PHI_POST, layer * n,     n,     1 };
    m[k++] = (mm){ "mhc phi_res",  H + H_PHI_RES,  layer * n * n, n * n, 1 };
    m[k++] = (mm){ "self q",       L + D_Q,    0, s_w.t[L + D_Q].shape[0],    0 };
    m[k++] = (mm){ "self k",       L + D_K,    0, s_w.t[L + D_K].shape[0],    1 };
    m[k++] = (mm){ "self v",       L + D_V,    0, s_w.t[L + D_V].shape[0],    1 };
    m[k++] = (mm){ "self gate",    L + D_GATE, 0, s_w.t[L + D_GATE].shape[0], 1 };
    m[k++] = (mm){ "self out",     L + D_OUT,  0, s_w.t[L + D_OUT].shape[0],  0 };
    m[k++] = (mm){ "cross q",      X + X_Q,    0, s_w.t[X + X_Q].shape[0],    0 };
    m[k++] = (mm){ "cross gate",   X + X_GATE, 0, s_w.t[X + X_GATE].shape[0], 1 };
    m[k++] = (mm){ "cross out",    X + X_OUT,  0, s_w.t[X + X_OUT].shape[0],  0 };
    return k;
}

static uint32_t mm_weights(const mm *m) { return m->nrows * s_w.t[m->ti].shape[1]; }
static uint32_t mm_bytes(const mm *m)
{
    const nd_tensor *t = &s_w.t[m->ti];
    return m->nrows * (nd_cq_row_bytes(t) + nd_cq_groups(t) * 2);
}

/* Run a list of matmuls like the decoder does (prepare once per shared input,
 * then each tensor), on T rows of X. Returns cycles; per-entry cycles in c. */
static uint32_t run_mms(const mm *m, uint32_t n, const float *X, uint32_t T, float *Y,
                        uint32_t *c)
{
    nd_q8_act a = { 0 };
    uint32_t  i, t0 = cyc(), tp;
    for (i = 0; i < n; i++) {
        const nd_tensor *t = &s_w.t[m[i].ti];
        tp = cyc();
        if (!m[i].shared) {
            if (a.mem)
                nd_cq_q8_free(&a);
            nd_cq_q8_prepare(t, X, T, &a);
        }
        nd_cq_q8_gemm(&s_w.c, t, s_w.p[m[i].ti], &a, m[i].r0, m[i].nrows, Y);
        if (c)
            c[i] = cyc() - tp;
    }
    nd_cq_q8_free(&a);
    return cyc() - t0;
}

static void bench_layer_T(void)
{
    static const uint32_t Ts[] = { 1, 2, 4, 8, 16, 64 };
    mm        m[16];
    uint32_t  n = layer_mms(s_w.active[0], m), i, k, r, w = 0, b = 0;
    float    *X = (float *)heap_caps_malloc(64 * 2048 * sizeof(float), MALLOC_CAP_SPIRAM);
    float    *Y = (float *)heap_caps_malloc(64 * 512 * sizeof(float), MALLOC_CAP_SPIRAM);
    uint32_t  c[16], best[16];
    double    t1 = 0.0;

    fill(X, 64 * 2048);
    for (i = 0; i < n; i++) {
        w += mm_weights(&m[i]);
        b += mm_bytes(&m[i]);
    }
    printf("\n## 1. One decoder layer's matmuls (layer %u, cold), per token\n",
           (unsigned)s_w.active[0]);
    printf("%u matmuls, %u weights, %u KB packed\n", (unsigned)n, (unsigned)w, (unsigned)(b / 1024));
    printf("%4s %10s %12s %10s %9s %9s\n", "T", "us/call", "us/token", "vs T=1", "cyc/w", "MB/s");
    for (k = 0; k < sizeof(Ts) / sizeof(Ts[0]); k++) {
        uint32_t T = Ts[k], tot = UINT32_MAX;
        for (r = 0; r < REPS; r++) {
            uint32_t cc;
            evict();
            cc = run_mms(m, n, X, T, Y, c);
            if (cc < tot) {
                tot = cc;
                memcpy(best, c, sizeof(best));
            }
        }
        if (T == 1)
            t1 = us(tot);
        printf("%4u %10.0f %12.1f %9.2fx %9.2f %9.1f\n", (unsigned)T, us(tot), us(tot) / T,
               t1 / (us(tot) / T), (double)tot / T / w, b / us(tot));
        if (T == 1) {
            printf("     %-14s %8s %8s %8s %6s\n", "matmul", "rows", "in", "us", "cyc/w");
            for (i = 0; i < n; i++)
                printf("     %-14s %8u %8u %8.1f %6.2f\n", m[i].name, (unsigned)m[i].nrows,
                       (unsigned)s_w.t[m[i].ti].shape[1], us(best[i]),
                       (double)best[i] / mm_weights(&m[i]));
        }
    }
    heap_caps_free(X);
    heap_caps_free(Y);
}

/* ------------------------------------- 2./3. unpack vs dot, SRAM vs PSRAM
 *
 * A copy of the engine's int8 row loop (nd_quant.c gemm8_rows, T = 1) that
 * can skip either half: unpack only (decode the row to int8, no dot), dot only
 * (rows already unpacked, from w8), or both. Same tables, same nd_simd dot. */

enum { K_BOTH, K_UNPACK, K_DOT };

typedef struct {
    const uint8_t  *packed;
    const uint16_t *norms;
    const int8_t   *w8;         /* [rows][in_pad] pre-unpacked, for K_DOT */
    const int8_t   *xq;
    const float    *sx;
    float          *y;
    uint32_t        rowb, in_pad, ngroup, bits, mode;
    uint32_t        tab2[256];
    uint16_t        tab4[256];
    float           inv_sw;
    int8_t         *wq[2];
} kctx;

static void IRAM_ATTR k_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const kctx *c  = (const kctx *)vc;
    int8_t     *wq = c->wq[r0 == 0 ? 0 : 1];
    float       nrm[16];
    uint32_t    r, j, gi;

    for (r = r0; r < r1; r++) {
        const uint8_t  *p  = c->packed + (size_t)r * c->rowb;
        const uint16_t *nm = c->norms + (size_t)r * c->ngroup;
        const int8_t   *w  = wq;
        float           acc = 0.0f;
        if (c->mode == K_DOT) {
            w = c->w8 + (size_t)r * c->in_pad;
        } else if (c->bits == 2) {
            for (j = 0; j < c->rowb; j++)
                ((uint32_t *)(void *)wq)[j] = c->tab2[p[j]];
        } else {
            for (j = 0; j < c->rowb; j++)
                ((uint16_t *)(void *)wq)[j] = c->tab4[p[j]];
        }
        if (c->mode == K_UNPACK)
            continue;
        for (gi = 0; gi < c->ngroup; gi++)
            nrm[gi] = nd_f16(nm[gi]) * c->inv_sw;
        for (gi = 0; gi < c->ngroup; gi++)
            acc += nrm[gi] * c->sx[gi] * (float)nd_dot128_s8(w + gi * 128, c->xq + gi * 128);
        c->y[r] = acc;
    }
}

static int8_t q8(float v) { return (int8_t)(int)(v + (v >= 0.0f ? 0.5f : -0.5f)); }

/* Point k at `nrows` rows of tensor ti (packed bytes + norms at `blob`). */
static void k_setup(kctx *k, uint32_t ti, const void *blob, uint32_t nrows_total,
                    const nd_q8_act *a, float *y)
{
    const nd_tensor *t  = &s_w.t[ti];
    const float     *cb = nd_cact_codebook(&s_w.c, t->bits);
    uint32_t         nl = 1u << t->bits, i;
    float            mx = 0.0f, sw;
    int8_t           lv[16];

    for (i = 0; i < nl; i++)
        if (fabsf(cb[i]) > mx)
            mx = fabsf(cb[i]);
    sw = 127.0f / mx;
    for (i = 0; i < nl; i++)
        lv[i] = q8(cb[i] * sw);
    for (i = 0; i < 256; i++) {
        k->tab2[i] = (uint32_t)(uint8_t)lv[i & 3] | (uint32_t)(uint8_t)lv[(i >> 2) & 3] << 8 |
                     (uint32_t)(uint8_t)lv[(i >> 4) & 3] << 16 |
                     (uint32_t)(uint8_t)lv[(i >> 6) & 3] << 24;
        if (t->bits == 4)
            k->tab4[i] = (uint16_t)((uint8_t)lv[i & 15] | (uint16_t)(uint8_t)lv[i >> 4] << 8);
    }
    k->inv_sw = 1.0f / sw;
    k->rowb   = nd_cq_row_bytes(t);
    k->in_pad = nd_cq_in_pad(t);
    k->ngroup = k->in_pad / 128;
    k->bits   = t->bits;
    k->packed = (const uint8_t *)blob;
    k->norms  = (const uint16_t *)(const void *)(k->packed + (size_t)nrows_total * k->rowb);
    k->xq     = a->xq;
    k->sx     = a->sx;
    k->y      = y;
}

/* Copy rows [0, nrows) of tensor ti (packed rows, then their norms) into one
 * buffer laid out like a tensor of nrows rows. */
static void *rows_copy(uint32_t ti, uint32_t nrows, uint32_t caps)
{
    const nd_tensor *t    = &s_w.t[ti];
    uint32_t         rowb = nd_cq_row_bytes(t), ng = nd_cq_groups(t);
    const uint8_t   *src  = (const uint8_t *)s_w.p[ti];
    uint8_t         *d    = (uint8_t *)heap_caps_aligned_alloc(128, (size_t)nrows * (rowb + 2 * ng), caps);
    if (!d)
        return NULL;
    memcpy(d, src, (size_t)nrows * rowb);
    memcpy(d + (size_t)nrows * rowb, src + (size_t)t->shape[0] * rowb, (size_t)nrows * ng * 2);
    return d;
}

static uint32_t k_time(kctx *k, uint32_t nrows, int cold)
{
    uint32_t r, best = UINT32_MAX;
    for (r = 0; r < REPS; r++) {
        uint32_t c;
        if (cold)
            evict();
        else
            nd_parallel_rows(k_rows, k, nrows);          /* warm it */
        c = cyc();
        nd_parallel_rows(k_rows, k, nrows);
        c = cyc() - c;
        if (c < best)
            best = c;
    }
    return best;
}

static void bench_split_one(const char *name, uint32_t ti, uint32_t nrows)
{
    const nd_tensor *t = &s_w.t[ti];
    uint32_t  in_pad = nd_cq_in_pad(t), r, w = nrows * in_pad;
    float    *x = (float *)malloc(in_pad * sizeof(float));
    float    *y = (float *)heap_caps_malloc(nrows * sizeof(float), MALLOC_CAP_INTERNAL);
    float    *y2 = (float *)heap_caps_malloc(nrows * sizeof(float), MALLOC_CAP_INTERNAL);
    void     *sram = rows_copy(ti, nrows, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    void     *ps   = rows_copy(ti, nrows, MALLOC_CAP_SPIRAM);
    int8_t   *w8p  = (int8_t *)heap_caps_aligned_alloc(16, (size_t)w, MALLOC_CAP_SPIRAM);
    int8_t   *w8s  = (int8_t *)heap_caps_aligned_alloc(16, (size_t)w, MALLOC_CAP_INTERNAL);
    int8_t   *wq   = (int8_t *)heap_caps_aligned_alloc(16, 2 * in_pad, MALLOC_CAP_INTERNAL);
    kctx     *k    = (kctx *)heap_caps_malloc(sizeof(kctx), MALLOC_CAP_INTERNAL);
    nd_q8_act a;
    uint32_t  c_lib, c_both, c_un, c_dot, c_dot8p, c_dot8s, c_sram, c_warm, c_both1;
    int       agree;

    if (!x || !y || !y2 || !ps || !w8p || !wq || !k) {
        printf("%s: out of memory\n", name);
        goto out;
    }
    fill(x, t->shape[1]);
    nd_cq_q8_prepare(t, x, 1, &a);
    k->wq[0] = wq;
    k->wq[1] = wq + in_pad;

    /* The engine's own call on the same rows (in the model), cold. */
    {
        uint32_t best = UINT32_MAX;
        for (r = 0; r < REPS; r++) {
            uint32_t c;
            evict();
            c = cyc();
            nd_cq_q8_gemm(&s_w.c, t, s_w.p[ti], &a, 0, nrows, y2);
            c = cyc() - c;
            if (c < best)
                best = c;
        }
        c_lib = best;
    }
    k_setup(k, ti, ps, nrows, &a, y);
    k->mode = K_BOTH;
    c_both = k_time(k, nrows, 1);
    for (r = 0, agree = 1; r < nrows; r++)       /* same integers; float rounding only */
        if (fabsf(y[r] - y2[r]) > 1e-5f * (fabsf(y2[r]) + 1e-3f))
            agree = 0;
    nd_parallel_rows = rows_serial;
    c_both1 = k_time(k, nrows, 1);
    nd_parallel_rows = rows_dual_core;
    k->mode = K_UNPACK;
    c_un = k_time(k, nrows, 1);
    k->mode = K_BOTH;
    c_warm = k_time(k, nrows, 0);

    /* Pre-unpack every row (the "int8 at boot" layout). */
    for (r = 0; r < nrows; r++) {
        uint32_t j;
        const uint8_t *p = (const uint8_t *)ps + (size_t)r * k->rowb;
        int8_t        *d = w8p + (size_t)r * in_pad;
        if (t->bits == 2)
            for (j = 0; j < k->rowb; j++)
                memcpy(d + 4 * j, &k->tab2[p[j]], 4);
        else
            for (j = 0; j < k->rowb; j++)
                memcpy(d + 2 * j, &k->tab4[p[j]], 2);
    }
    k->mode = K_DOT;
    k->w8 = w8p;
    c_dot8p = k_time(k, nrows, 1);
    c_dot = k_time(k, nrows, 0);
    c_dot8s = 0;
    if (w8s) {
        memcpy(w8s, w8p, w);
        k->w8 = w8s;
        c_dot8s = k_time(k, nrows, 1);
    }
    c_sram = 0;
    if (sram) {
        k_setup(k, ti, sram, nrows, &a, y);
        k->mode = K_BOTH;
        c_sram = k_time(k, nrows, 1);
    }

    printf("\n%s: %u rows x %u, %u-bit, %u KB packed (%s)\n", name, (unsigned)nrows,
           (unsigned)in_pad, (unsigned)t->bits, (unsigned)(nrows * (k->rowb + 2 * k->ngroup) / 1024),
           agree ? "bench copy matches the engine" : "BENCH COPY DIFFERS FROM THE ENGINE");
    printf("  %-44s %9s %7s\n", "variant (two cores unless noted)", "us", "cyc/w");
#define ROW(label, c) printf("  %-44s %9.1f %7.2f\n", label, us(c), (double)(c) / w)
    ROW("engine nd_cq_q8_gemm, packed in PSRAM, cold", c_lib);
    ROW("unpack + dot, packed in PSRAM, cold", c_both);
    ROW("unpack + dot, packed in PSRAM, cold, 1 core", c_both1);
    ROW("unpack only, packed in PSRAM, cold", c_un);
    ROW("unpack + dot, packed in PSRAM, warm (L2)", c_warm);
    if (sram)
        ROW("unpack + dot, packed in SRAM", c_sram);
    else
        printf("  %-44s (no SRAM for the copy)\n", "unpack + dot, packed in SRAM");
    ROW("dot only, int8 rows in PSRAM, cold", c_dot8p);
    ROW("dot only, int8 rows in PSRAM, warm (L2)", c_dot);
    if (w8s)
        ROW("dot only, int8 rows in SRAM", c_dot8s);
    else
        printf("  %-44s (no SRAM for the copy)\n", "dot only, int8 rows in SRAM");
#undef ROW
    nd_cq_q8_free(&a);
out:
    free(x);
    heap_caps_free(y);
    heap_caps_free(y2);
    heap_caps_free(sram);
    heap_caps_free(ps);
    heap_caps_free(w8p);
    heap_caps_free(w8s);
    heap_caps_free(wq);
    heap_caps_free(k);
}

static void bench_split(void)
{
    uint32_t L = s_w.i_dec + s_w.active[0] * D_N;
    printf("\n## 2./3. Unpack vs dot, PSRAM vs SRAM (T = 1)\n");
    bench_split_one("self q (2-bit)", L + D_Q, 128);
    bench_split_one("self out (2-bit)", L + D_OUT, 128);
    bench_split_one("mhc phi_res (4-bit)", s_w.i_dec_mhc + H_PHI_RES, 32);
    bench_split_one("output layer (4-bit)", s_w.i_emb, 256);
}

/* --------------------------------------------------- 4. PSRAM bandwidth */

typedef struct {
    const uint8_t *p;
    uint32_t       chunk;
    int            pie;
    volatile uint32_t sink;
} bw_ctx;

static void IRAM_ATTR bw_rows(void *vc, uint32_t r0, uint32_t r1)
{
    bw_ctx  *c = (bw_ctx *)vc;
    uint32_t s = 0;
    if (c->pie) {
        register const uint8_t *p __asm__("a4") = c->p + (size_t)r0 * c->chunk;
        const uint8_t          *e = c->p + (size_t)r1 * c->chunk;
        while (p < e)       /* PIE address operands must be x8-x15 */
            __asm__ volatile(
                "esp.vld.128.ip q0, %0, 16\n" "esp.vld.128.ip q1, %0, 16\n"
                "esp.vld.128.ip q2, %0, 16\n" "esp.vld.128.ip q3, %0, 16\n"
                "esp.vld.128.ip q4, %0, 16\n" "esp.vld.128.ip q5, %0, 16\n"
                "esp.vld.128.ip q6, %0, 16\n" "esp.vld.128.ip q7, %0, 16\n"
                : "+r"(p) :: "memory");
    } else {
        const uint32_t *p = (const uint32_t *)(const void *)(c->p + (size_t)r0 * c->chunk);
        const uint32_t *e = (const uint32_t *)(const void *)(c->p + (size_t)r1 * c->chunk);
        for (; p < e; p += 8)
            s += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
    }
    c->sink += s;
}

static SemaphoreHandle_t s_dma_done;
static bool IRAM_ATTR dma_cb(async_memcpy_handle_t h, async_memcpy_event_t *e, void *arg)
{
    BaseType_t woke = pdFALSE;
    (void)h; (void)e; (void)arg;
    xSemaphoreGiveFromISR(s_dma_done, &woke);
    return woke == pdTRUE;
}

static void bench_bandwidth(void)
{
    uint8_t *buf = (uint8_t *)heap_caps_aligned_alloc(128, BW_BYTES, MALLOC_CAP_SPIRAM);
    uint8_t *dst = (uint8_t *)heap_caps_aligned_alloc(128, 32 * 1024,
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    bw_ctx   c;
    uint32_t i, cc;
    double   mb = BW_BYTES / 1e6;

    printf("\n## 4. PSRAM bandwidth (%u MB, cold)\n", (unsigned)(BW_BYTES >> 20));
    printf("hardware prefetch (autoload): L1 data %s, L2 %s\n",
           cache_ll_l1_is_dcache_autoload_enabled(CACHE_LL_ID_ALL) ? "on" : "off",
           cache_ll_l2_is_cache_autoload_enabled(CACHE_LL_ID_ALL) ? "on" : "off");
    if (!buf || !dst) {
        printf("out of memory\n");
        goto out;
    }
    memset(buf, 0x5A, BW_BYTES);
    c.p = buf;
    c.chunk = 4096;
    c.sink = 0;
    for (i = 0; i < 4; i++) {
        static const char *names[] = { "32-bit loads, 1 core", "32-bit loads, 2 cores",
                                       "PIE 128-bit loads, 1 core", "PIE 128-bit loads, 2 cores" };
        c.pie = i >= 2;
        nd_parallel_rows = (i & 1) ? rows_dual_core : rows_serial;
        evict();
        cc = cyc();
        nd_parallel_rows(bw_rows, &c, BW_BYTES / c.chunk);
        cc = cyc() - cc;
        printf("  %-34s %8.1f MB/s\n", names[i], mb / (us(cc) / 1e6));
    }
    nd_parallel_rows = rows_dual_core;

    evict();
    cc = cyc();
    for (i = 0; i < BW_BYTES; i += 32 * 1024)
        memcpy(dst, buf + i, 32 * 1024);
    cc = cyc() - cc;
    printf("  %-34s %8.1f MB/s\n", "memcpy to SRAM, 32 KB chunks", mb / (us(cc) / 1e6));

    {
        async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
        async_memcpy_handle_t h = NULL;
        esp_err_t             e;
        cfg.dma_burst_size = 64;
        s_dma_done = xSemaphoreCreateBinary();
        e = esp_async_memcpy_install_gdma_axi(&cfg, &h);
        if (e == ESP_OK) {
            cc = cyc();
            for (i = 0; i < BW_BYTES && e == ESP_OK; i += 32 * 1024) {
                e = esp_async_memcpy(h, dst, buf + i, 32 * 1024, dma_cb, NULL);
                if (e == ESP_OK)
                    xSemaphoreTake(s_dma_done, portMAX_DELAY);
            }
            cc = cyc() - cc;
            if (e == ESP_OK)
                printf("  %-34s %8.1f MB/s\n", "async DMA to SRAM, 32 KB chunks", mb / (us(cc) / 1e6));
            esp_async_memcpy_uninstall(h);
        }
        if (e != ESP_OK)
            printf("  async DMA: %s\n", esp_err_to_name(e));
        vSemaphoreDelete(s_dma_done);
    }

    /* Latency of one L2 miss: chase pointers through 128-byte lines in a
     * shuffled order, so neither cache nor prefetch can help. */
    {
        uint32_t  nl = BW_BYTES / 128, hops = 8192, *perm, j;
        uint32_t *p;
        perm = (uint32_t *)heap_caps_malloc(nl * sizeof(uint32_t), MALLOC_CAP_SPIRAM);
        if (perm) {
            for (i = 0; i < nl; i++)
                perm[i] = i;
            for (i = nl - 1; i > 0; i--) {
                s_rng = s_rng * 1103515245u + 12345u;
                j = (s_rng >> 8) % (i + 1);
                uint32_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
            }
            for (i = 0; i < nl; i++)
                *(uint32_t **)(void *)(buf + (size_t)perm[i] * 128) =
                    (uint32_t *)(void *)(buf + (size_t)perm[(i + 1) % nl] * 128);
            heap_caps_free(perm);
            evict();
            p = (uint32_t *)(void *)buf;
            cc = cyc();
            for (i = 0; i < hops; i++)
                p = *(uint32_t **)(void *)p;
            cc = cyc() - cc;
            printf("  %-34s %8.0f ns (%.0f cycles; %p)\n", "one cache miss (pointer chase)",
                   us(cc) * 1000.0 / hops, (double)cc / hops, (void *)p);
        }
    }
out:
    heap_caps_free(buf);
    heap_caps_free(dst);
}

/* ------------------------------------------------------ 5. output layer */

static void bench_head(void)
{
    const nd_tensor *t = &s_w.t[s_w.i_emb];
    uint32_t  V = t->shape[0], in_pad = nd_cq_in_pad(t), i, r, n;
    float    *x  = (float *)malloc(in_pad * sizeof(float));
    float    *xh = (float *)malloc(in_pad * sizeof(float));
    float    *y  = (float *)heap_caps_malloc(V * sizeof(float), MALLOC_CAP_SPIRAM);
    uint32_t *ids = (uint32_t *)heap_caps_malloc(V * sizeof(uint32_t), MALLOC_CAP_SPIRAM);
    static const uint32_t subsets[] = { 500, 1000, 2000, 4000 };
    uint32_t  best;

    printf("\n## 5. Output layer: %u x %u, %u-bit, %u KB\n", (unsigned)V, (unsigned)t->shape[1],
           (unsigned)t->bits, (unsigned)(t->nbytes / 1024));
    fill(x, t->shape[1]);
    best = UINT32_MAX;
    for (r = 0; r < REPS; r++) {
        nd_q8_act a;
        uint32_t  c;
        evict();
        c = cyc();
        nd_cq_q8_prepare(t, x, 1, &a);
        nd_cq_q8_gemm(&s_w.c, t, s_w.p[s_w.i_emb], &a, 0, V, y);
        nd_cq_q8_free(&a);
        c = cyc() - c;
        if (c < best)
            best = c;
    }
    printf("  %-40s %9.0f us  %.2f cyc/w\n", "all rows, int8 path, 2 cores", us(best),
           (double)best / ((double)V * in_pad));
    nd_cq_prepare(t, x, xh);
    for (i = 0; i < V; i++)                        /* spread over the vocabulary */
        ids[i] = (uint32_t)(((uint64_t)i * 2654435761u) % V);
    for (n = 0; n < sizeof(subsets) / sizeof(subsets[0]); n++) {
        uint32_t k = subsets[n];
        best = UINT32_MAX;
        for (r = 0; r < REPS; r++) {
            uint32_t c;
            evict();
            c = cyc();
            nd_cq_gemv_gather(&s_w.c, t, s_w.p[s_w.i_emb], xh, ids, k, y);
            c = cyc() - c;
            if (c < best)
                best = c;
        }
        printf("  gather %5u rows, float path, 1 core %13.0f us  %.2f cyc/w\n", (unsigned)k,
               us(best), (double)best / ((double)k * in_pad));
    }
    free(x);
    free(xh);
    heap_caps_free(y);
    heap_caps_free(ids);
}

/* ------------------------------------------------- 6. whole decoder step */

static void bench_step(uint32_t depth)
{
    enum { FRAMES = 63, STEPS = 16 };     /* ~5 s of audio, 16 tokens */
    float    *enc, *out;
    uint32_t  toks[STEPS + 1], i;
    uint64_t  p1[ND_WP_COUNT], p2[ND_WP_COUNT];
    int64_t   t1, t2;

    if (nd_whistle_set_depth(&s_w, depth) != 0) {
        printf("\ndepth %u: not in this model (%u decoder layers)\n", (unsigned)depth,
               (unsigned)s_w.dec_layers);
        return;
    }
    enc = (float *)heap_caps_malloc(FRAMES * s_w.d * sizeof(float), MALLOC_CAP_SPIRAM);
    out = (float *)heap_caps_malloc((size_t)(STEPS + 1) * s_w.vocab * sizeof(float),
                                    MALLOC_CAP_SPIRAM);
    if (!enc || !out) {
        printf("out of memory\n");
        goto done;
    }
    fill(enc, FRAMES * s_w.d);
    toks[0] = 2;                                  /* BOS, <en>, then text */
    toks[1] = 8192;
    for (i = 2; i <= STEPS; i++)
        toks[i] = 300 + 37 * i;

    /* One token, then STEPS + 1: the difference is STEPS decoder steps
     * without the per-clip cross K/V. */
    memset(nd_wprof, 0, sizeof(nd_wprof));
    t1 = esp_timer_get_time();
    nd_whistle_logits(&s_w, enc, FRAMES, toks, 1, out);
    t1 = esp_timer_get_time() - t1;
    memcpy(p1, nd_wprof, sizeof(p1));
    memset(nd_wprof, 0, sizeof(nd_wprof));
    t2 = esp_timer_get_time();
    nd_whistle_logits(&s_w, enc, FRAMES, toks, STEPS + 1, out);
    t2 = esp_timer_get_time() - t2;
    memcpy(p2, nd_wprof, sizeof(p2));
    {
        double tot = (double)(t2 - t1) / STEPS / 1000.0, rest = tot;
        static const struct { int slot; const char *name; } ph[] = {
            { ND_WP_LIN, "matmuls" }, { ND_WP_HEAD, "output layer" }, { ND_WP_MLP, "MLP" },
            { ND_WP_MHC, "mHC (without its matmuls)" }, { ND_WP_ATTN, "attention" },
        };
        printf("\n## 6. Decoder step, depth %u (layers", (unsigned)depth);
        for (i = 0; i < depth; i++)
            printf(" %u", (unsigned)s_w.active[i]);
        printf("), %u frames, mean of %u steps\n", FRAMES, STEPS);
        for (i = 0; i < sizeof(ph) / sizeof(ph[0]); i++) {
            double ms = (double)(p2[ph[i].slot] - p1[ph[i].slot]) / STEPS / 1000.0;
            rest -= ms;
            printf("  %-28s %7.1f ms  %4.0f%%\n", ph[i].name, ms, 100.0 * ms / tot);
        }
        printf("  %-28s %7.1f ms  %4.0f%%\n", "rest (engram, norms, ...)", rest, 100.0 * rest / tot);
        printf("  %-28s %7.1f ms\n", "total per token", tot);
    }
done:
    heap_caps_free(enc);
    heap_caps_free(out);
}

/* ---------------------------------------------------------------- misc */

static void bench_dispatch(void)
{
    uint32_t i, c;
    c = cyc();
    for (i = 0; i < 1000; i++)
        nd_parallel_rows(nop_rows, NULL, 100);
    c = cyc() - c;
    printf("\n## Two-core hand-off: %.1f us per matmul call (empty rows)\n", us(c) / 1000.0);
}

static void bench_task(void *arg)
{
    int64_t t0 = esp_timer_get_time();
    (void)arg;

    bench_dispatch();
    bench_layer_T();
    bench_split();
    bench_bandwidth();
    bench_head();
    bench_step(4);
    bench_step(8);
    printf("\nDONE in %.1f s\n", (esp_timer_get_time() - t0) / 1e6);
    fflush(stdout);
    vTaskDelete(NULL);
}

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(500));
    printf("\n# nd_bench: Whistle decoder kernels on the ESP32-P4, %d MHz\n", MHZ);
    worker_start();
    if (nd_cq_int8_check() != 0) {
        printf("ERR int8_kernel_mismatch\n");
        return;
    }
    if (model_open() != 0)
        return;
    s_evict = (uint8_t *)heap_caps_malloc(EVICT_BYTES, MALLOC_CAP_SPIRAM);
    if (!s_evict) {
        printf("ERR no PSRAM for the eviction buffer\n");
        return;
    }
    memset(s_evict, 1, EVICT_BYTES);
    xTaskCreatePinnedToCore(bench_task, "bench", 48 * 1024, NULL, 5, NULL, 0);
}
