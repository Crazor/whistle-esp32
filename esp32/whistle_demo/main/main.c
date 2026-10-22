/* Whistle on ESP32-P4: hold the button, say a command, the LED does it.
 *
 * Speech recognition runs entirely on the chip. The depth-4 Whistle model
 * (tools/whistle_strip.py, 14.5 MB) lives in the "model" partition and is
 * copied at boot into the 32 MB PSRAM, which streams far faster than the QIO
 * flash mapping. Without room for it there, it is used memory-mapped.
 *
 * Board: Waveshare ESP32-P4-WIFI6 (pins in components/board/board.h)
 *   mic      onboard, through the ES8311 codec
 *   button   BOOT (GPIO35)
 *   LED      external WS2812 on LED_GPIO
 *
 * LED: dim blue for the second it takes to load, then off (listening);
 * white while recording, a dim pulse while transcribing, then the command;
 * two red blinks when nothing was heard, one amber blink for no colour.
 *
 * Out, one line each (the same EVT/ACT vocabulary as needle_demo):
 *   EVT ...     status, levels and per-phase timings
 *   TXT ...     the transcript
 *   ACT led color=<c> mode=<solid|flash> duration=<s>   or   ACT none
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "board.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/rmt_tx.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "nd_ledcmd.h"
#include "nd_quant.h"
#include "nd_whistle.h"

#define RMT_RESOLUTION 10000000      /* 10 MHz -> 100 ns per tick */

#define SAMPLE_RATE    16000
#define MAX_SECS       8             /* the cross-attention cache grows with it */
#define CHUNK          160           /* 10 ms per I2S read */
#define DEBOUNCE_MS    20
#define HEAD_DROP_MS   50            /* press transient: the engine normalises to
                                      * the clip peak, so one click would turn
                                      * every word down */
#define TAIL_DROP_MS   30            /* release transient */
#define FLOOR_WIN      320           /* 20 ms noise-floor windows */
#define FLOOR_HIST     250           /* 5 s of idle history */
#define GATE_DB        12.0f         /* speech must clear the floor by this */
#define KW_BONUS       3.0f          /* tuned on 360 clips: tools/eval_whistle.py */

static const char *const COLORS[] = { "red", "green", "blue", "yellow", "purple", "white" };
static const char *const LANGS[] = { "en", "de", "fr", "es", "it", "nl", "pl" };
#define N_COLORS (sizeof(COLORS) / sizeof(COLORS[0]))
static const uint8_t RGB[][3] = {
    { 64, 0, 0 }, { 0, 64, 0 }, { 0, 0, 64 }, { 48, 40, 0 }, { 40, 0, 56 }, { 40, 40, 40 },
};

static nd_whistle s_w;

/* A transcript is a command if it names a colour, in any of the seven
 * languages. When the first language Whistle detects gives none, it tries the
 * next ones (nd_whistle.h, accept). */
static int has_command(const char *text, void *ctx)
{
    nd_led_cmd c;
    (void)ctx;
    return nd_led_parse(text, COLORS, N_COLORS, 3.0f, &c);
}

static void *psram_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM); }

/* ------------------------------------------------------------------ WS2812
 *
 * One task owns the RMT channel. The inference keeps both cores busy for
 * seconds, so the processing pulse is animated here, at a priority above the
 * inference task, rather than by the code doing the work. */

enum { LED_SOLID, LED_PULSE };
static rmt_channel_handle_t s_chan;
static rmt_encoder_handle_t s_encoder;
static volatile int         s_led_mode = LED_SOLID;
static volatile uint8_t     s_led_rgb[3];

static void led_write(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t grb[3] = { g, r, b };          /* WS2812 wire order */
    rmt_transmit_config_t tx = { .loop_count = 0 };
    rmt_transmit(s_chan, s_encoder, grb, sizeof(grb), &tx);
    rmt_tx_wait_all_done(s_chan, portMAX_DELAY);
}

static void led_task(void *arg)
{
    uint32_t tick = 0;
    uint8_t  last[3] = { 1, 1, 1 };
    (void)arg;
    for (;;) {
        uint8_t c[3];
        if (s_led_mode == LED_PULSE) {
            uint32_t ph = tick % 50;                     /* 1.5 s breath */
            uint8_t  v  = (uint8_t)(1 + (ph < 25 ? ph : 50 - ph) * 12 / 25);
            c[0] = c[1] = c[2] = v;
        } else {
            c[0] = s_led_rgb[0]; c[1] = s_led_rgb[1]; c[2] = s_led_rgb[2];
        }
        if (memcmp(c, last, 3)) {
            led_write(c[0], c[1], c[2]);
            memcpy(last, c, 3);
        }
        tick++;
        vTaskDelay(pdMS_TO_TICKS(30));
    }
}

static void led_init(void)
{
    rmt_tx_channel_config_t chan_cfg = {
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .gpio_num          = LED_GPIO,
        .mem_block_symbols = 64,
        .resolution_hz     = RMT_RESOLUTION,
        .trans_queue_depth = 4,
    };
    rmt_bytes_encoder_config_t enc_cfg = {
        .bit0  = { .level0 = 1, .duration0 = 3, .level1 = 0, .duration1 = 9 },
        .bit1  = { .level0 = 1, .duration0 = 9, .level1 = 0, .duration1 = 3 },
        .flags = { .msb_first = 1 },
    };
    ESP_ERROR_CHECK(rmt_new_tx_channel(&chan_cfg, &s_chan));
    ESP_ERROR_CHECK(rmt_new_bytes_encoder(&enc_cfg, &s_encoder));
    ESP_ERROR_CHECK(rmt_enable(s_chan));
    xTaskCreatePinnedToCore(led_task, "led", 3072, NULL, 6, NULL, 0);
}

static void led_set(uint8_t r, uint8_t g, uint8_t b)
{
    s_led_rgb[0] = r; s_led_rgb[1] = g; s_led_rgb[2] = b;
    s_led_mode = LED_SOLID;
}

static void led_blink(uint8_t r, uint8_t g, uint8_t b, int times, int on_ms, int off_ms)
{
    int i;
    for (i = 0; i < times; i++) {
        led_set(r, g, b);
        vTaskDelay(pdMS_TO_TICKS(on_ms));
        led_set(0, 0, 0);
        vTaskDelay(pdMS_TO_TICKS(off_ms));
    }
}

static void led_command(const nd_led_cmd *c)
{
    const uint8_t *k = RGB[c->color];
    if (c->flash) {
        /* "every N seconds" sets the cycle; otherwise 0.4 s */
        float per    = c->period > 0.1f ? c->period : 0.4f;
        int   cycles = (int)(c->seconds / per + 0.5f);
        int   half   = (int)(per * 500.0f);
        led_blink(k[0], k[1], k[2], cycles < 1 ? 1 : cycles, half, half);
    } else {
        led_set(k[0], k[1], k[2]);
        vTaskDelay(pdMS_TO_TICKS((int)(c->seconds * 1000.0f)));
        led_set(0, 0, 0);
    }
}

/* ------------------------------------------------- second-core GEMV worker
 * As needle_demo: every matvec / batched matmul splits its output rows. */

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

static void worker_start(void)
{
    s_go   = xSemaphoreCreateBinary();
    s_done = xSemaphoreCreateBinary();
    /* 16 KB: the Hadamard MLP's rows run here too, ~10 KB of scratch deep. */
    xTaskCreatePinnedToCore(worker_task, "nd_worker", 16384, NULL,
                            configMAX_PRIORITIES - 2, NULL, 1);
    nd_parallel_rows = rows_dual_core;
}

/* --------------------------------------------------------------------- I2S */

static i2s_chan_handle_t s_rx;

static void mic_init(void) { board_mic_init(SAMPLE_RATE, CHUNK, 0, &s_rx); }

/* Noise floor from the last 5 s of idle audio: the 20th-percentile 20 ms
 * window. The mic runs between presses, so this is known before anyone
 * speaks - better than guessing it from a clip that may be all speech. */
static float s_floor_hist[FLOOR_HIST];
static int   s_floor_n, s_floor_i;

static float noise_floor(void)
{
    static float tmp[FLOOR_HIST];
    int          i, j, n = s_floor_n;
    if (n < 25)
        return 0.0f;                       /* < 0.5 s seen: let the clip decide */
    memcpy(tmp, s_floor_hist, n * sizeof(float));
    for (i = 1; i < n; i++)
        for (j = i; j > 0 && tmp[j - 1] > tmp[j]; j--) {
            float t = tmp[j]; tmp[j] = tmp[j - 1]; tmp[j - 1] = t;
        }
    return tmp[n / 5];
}

/* --------------------------------------------------------------- inference */

static void transcribe(float *x, uint32_t n)
{
    static const char *PN[ND_WP_COUNT] = { "mel", "stem", "encoder", "cross_kv", "decoder",
                                           "~matmul", "~mlp", "~attn", "~mhc" };
    uint32_t s0 = 0, s1 = n, T, ids[96], i;
    float    floor_rms = noise_floor(), peak = 0.0f, *enc;
    int      nt;
    char     text[512];
    int64_t  t0 = esp_timer_get_time(), t_enc, t_dec;
    nd_led_cmd cmd;

    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > peak)
            peak = fabsf(x[i]);
    printf("EVT clip ms=%u peak_dbfs=%.1f floor_dbfs=%.1f\n", (unsigned)(n / 16),
           peak > 0 ? 20.0f * log10f(peak) : -120.0f,
           floor_rms > 0 ? 20.0f * log10f(floor_rms) : -120.0f);

    if (!nd_whistle_find_speech(x, n, floor_rms, GATE_DB, 300, &s0, &s1)) {
        printf("EVT no_speech (nothing %.0f dB above the noise floor)\n", GATE_DB);
        printf("TXT \nACT none\n");
        fflush(stdout);
        led_blink(64, 0, 0, 2, 150, 150);
        return;
    }
    x += s0;
    n  = s1 - s0;
    T  = nd_whistle_frames(&s_w, n);
    printf("EVT speech ms=%u frames=%u\n", (unsigned)(n / 16), (unsigned)T);
    fflush(stdout);

    s_led_mode = LED_PULSE;
    memset(nd_wprof, 0, sizeof(nd_wprof));
    enc = (float *)heap_caps_malloc((size_t)T * s_w.d * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!enc || nd_whistle_encode(&s_w, x, n, enc) != 0) {
        printf("ERR encode\n");
        heap_caps_free(enc);
        led_set(0, 0, 0);
        return;
    }
    t_enc = esp_timer_get_time();
    nt = nd_whistle_decode(&s_w, enc, T, -1 /* detect */, ids, sizeof(ids) / sizeof(ids[0]));
    t_dec = esp_timer_get_time();
    heap_caps_free(enc);
    led_set(0, 0, 0);
    if (nt < 0) {
        printf("ERR decode\n");
        return;
    }
    nd_whistle_detok(&s_w, ids, (uint32_t)nt, text, sizeof(text));

    printf("EVT lang %s\n", LANGS[s_w.lang]);
    printf("TXT %s\n", text);
    printf("EVT time total_ms=%.0f encode_ms=%.0f decode_ms=%.0f tokens=%d audio_ms=%u\n",
           (t_dec - t0) / 1000.0, (t_enc - t0) / 1000.0, (t_dec - t_enc) / 1000.0, nt,
           (unsigned)(n / 16));
    for (i = 0; i < ND_WP_COUNT; i++)
        printf("EVT prof %-9s %8.0f ms\n", PN[i], nd_wprof[i] / 1000.0);

    if (nt == 0 || !text[0]) {
        printf("ACT none\n");
        fflush(stdout);
        led_blink(64, 0, 0, 2, 150, 150);
    } else if (nd_led_parse(text, COLORS, N_COLORS, 3.0f, &cmd)) {
        if (cmd.period > 0.0f)
            printf("ACT led color=%s mode=flash duration=%.1f period=%.1f\n",
                   COLORS[cmd.color], cmd.seconds, cmd.period);
        else
            printf("ACT led color=%s mode=%s duration=%.1f\n", COLORS[cmd.color],
                   cmd.flash ? "flash" : "solid", cmd.seconds);
        fflush(stdout);
        led_command(&cmd);
    } else {
        printf("ACT none\n");               /* heard something, but no colour */
        fflush(stdout);
        led_blink(40, 16, 0, 1, 300, 100);   /* one amber blink */
    }
}

/* -------------------------------------------------------------------- main */

static void whistle_task(void *arg)
{
    const uint32_t cap = SAMPLE_RATE * MAX_SECS;
    float         *clip = (float *)heap_caps_malloc(cap * sizeof(float), MALLOC_CAP_SPIRAM);
    int32_t        raw[CHUNK];
    uint32_t       n = 0;
    int            recording = 0, stable = 1, last = 1;
    int64_t        changed_at = 0, t_start = 0;
    float          win_e = 0.0f;
    uint32_t       win_n = 0;
    (void)arg;

    if (!clip) {
        printf("ERR clip_alloc\n");
        vTaskDelete(NULL);
    }
#ifdef HAVE_SELFTEST
    {   /* Boot self-test: a known clip through the whole pipeline, so a build
         * proves itself (and prints its timings) before anyone speaks. */
        extern const uint8_t st_start[] asm("_binary_selftest_raw_start");
        extern const uint8_t st_end[] asm("_binary_selftest_raw_end");
        uint32_t k, ns = (uint32_t)(st_end - st_start) / 2;
        for (k = 0; k < ns && k < cap; k++)
            clip[k] = (float)(int16_t)(st_start[2 * k] | st_start[2 * k + 1] << 8) / 32768.0f;
        printf("EVT selftest clip (expect: flash blue for two seconds)\n");
        transcribe(clip, k);
        fflush(stdout);
    }
#endif
    printf("\nEVT ==================================================\n");
    printf("EVT  READY - hold BOOT (GPIO%d) and say a command\n", BUTTON_GPIO);
    printf("EVT  e.g. \"flash the light blue\", \"allume la lumière verte\"\n");
    printf("EVT  languages: en de fr es it nl pl (detected)\n");
    printf("EVT ==================================================\n\n");
    fflush(stdout);

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;

        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK)
            continue;
        m = got / sizeof(int32_t);

        lvl = gpio_get_level(BUTTON_GPIO);
        if (lvl != last) {
            last = lvl;
            changed_at = esp_timer_get_time();
        } else if (lvl != stable && esp_timer_get_time() - changed_at >= DEBOUNCE_MS * 1000) {
            stable = lvl;
            if (stable == 0) {
                recording = 1;
                n = 0;
                t_start = esp_timer_get_time();
                led_set(40, 40, 40);
                printf("EVT recording\n");
                fflush(stdout);
            } else if (recording) {
                uint32_t head = HEAD_DROP_MS * 16, tail = TAIL_DROP_MS * 16;
                recording = 0;
                led_set(0, 0, 0);
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                if (n > head + tail + 1600)
                    transcribe(clip + head, n - head - tail);
                else {
                    printf("EVT too_short\nTXT \nACT none\n");
                    led_blink(64, 0, 0, 2, 150, 150);
                }
                fflush(stdout);
                /* Inference stalled the reads; restart the floor estimate on
                 * fresh audio rather than on whatever piled up in the DMA. */
                win_e = 0.0f;
                win_n = 0;
                continue;
            }
        }

        for (i = 0; i < m; i++) {
            /* ES8311: the sample MSB-aligned in the 32-bit slot. */
            float v = (float)(raw[i] >> 16) / 32768.0f;
            if (recording) {
                if (n < cap)
                    clip[n++] = v;
            } else {
                win_e += v * v;
                if (++win_n == FLOOR_WIN) {
                    s_floor_hist[s_floor_i] = sqrtf(win_e / FLOOR_WIN);
                    s_floor_i = (s_floor_i + 1) % FLOOR_HIST;
                    if (s_floor_n < FLOOR_HIST)
                        s_floor_n++;
                    win_e = 0.0f;
                    win_n = 0;
                }
            }
        }
        if (recording == 1 && n >= cap) {
            printf("EVT clip_full secs=%d (release to transcribe)\n", MAX_SECS);
            fflush(stdout);
            recording = 2;
        }
    }
}

/* Copy the .cact at the start of the mapped partition into PSRAM. The blob
 * records no total size, so it ends where its last tensor does. Returns the
 * copy and its size, or NULL to keep using the mapping. */
static void *model_to_psram(const void *blob, size_t avail, size_t *size)
{
    nd_cact   c;
    nd_tensor t;
    size_t    end = 0;
    uint32_t  i;
    void     *p;
    int64_t   t0 = esp_timer_get_time();

    if (nd_cact_open(&c, blob, avail) != 0)
        return NULL;
    for (i = 0; i < c.n; i++)
        if (nd_cact_tensor(&c, i, &t) == 0 && t.offset + t.nbytes > end)
            end = (size_t)(t.offset + t.nbytes);
    if (end == 0 || end > avail)
        return NULL;                        /* let nd_whistle_open report it */
    p = heap_caps_aligned_alloc(ND_CACT_ALIGN, end, MALLOC_CAP_SPIRAM);
    if (!p) {
        printf("EVT model_in_flash (no %u bytes of PSRAM free)\n", (unsigned)end);
        return NULL;
    }
    memcpy(p, blob, end);
    printf("EVT model_in_psram bytes=%u ms=%.0f\n", (unsigned)end,
           (esp_timer_get_time() - t0) / 1000.0);
    *size = end;
    return p;
}

void app_main(void)
{
    const esp_partition_t      *part;
    esp_partition_mmap_handle_t handle;
    const void                 *blob = NULL;
    void                       *copy;
    size_t                      size;
    int                         rc;

    vTaskDelay(pdMS_TO_TICKS(500));
    led_init();
    led_set(0, 0, 8);                       /* dim blue: booting */
    worker_start();
    board_button_init();
    mic_init();
    if (nd_cq_int8_check() != 0) {
        /* Transcripts would still come out, just worse: say so loudly and
         * use the float path. */
        printf("ERR int8_kernel_mismatch (falling back to float)\n");
        nd_cq_int8 = 0;
    }

    part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) {
        printf("ERR no_model_partition\n");
        return;
    }
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &blob, &handle) != ESP_OK) {
        printf("ERR mmap_failed size=%u\n", (unsigned)part->size);
        return;
    }
    size = part->size;
    copy = model_to_psram(blob, size, &size);
    if (copy) {
        esp_partition_munmap(handle);
        blob = copy;
    }
    rc = nd_whistle_open(&s_w, blob, size);
    if (rc != 0) {
        /* -1..-5: no valid .cact at 0x210000; -6: a .cact that is not v4
         * (e.g. needle2.cact is still flashed there). */
        printf("ERR whistle_open rc=%d (flash whistle_d4.cact to 0x%06x)\n", rc,
               (unsigned)part->address);
        led_blink(64, 0, 0, 5, 300, 300);
        return;
    }
    {   /* Colour and blink words for all seven languages; each list only
         * biases a transcript in its own language. */
        int k;
        nd_whistle_set_keywords(&s_w, NULL, 0.0f);
        s_w.kw_bonus = KW_BONUS;
        for (k = 0; k < ND_WH_LANGS; k++)
            nd_whistle_add_keywords(&s_w, k, nd_led_keywords(k));
        s_w.accept = has_command;
    }
    if (!copy) {
        int64_t t = esp_timer_get_time();
        size_t  n = nd_whistle_cache_head(&s_w, psram_alloc);
        printf("EVT head_in_psram bytes=%u ms=%.0f\n", (unsigned)n,
               (esp_timer_get_time() - t) / 1000.0);
    }
    printf("EVT ready model=whistle depth=%u enc_layers=%u d_model=%u vocab=%u "
           "psram_free=%u internal_free=%u\n",
           (unsigned)s_w.depth, (unsigned)s_w.enc_layers, (unsigned)s_w.d, (unsigned)s_w.vocab,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    fflush(stdout);
    led_set(0, 0, 0);

    /* The decoder keeps a few KB of scratch per call level; give it room. */
    xTaskCreatePinnedToCore(whistle_task, "whistle", 48 * 1024, NULL, 5, NULL, 0);
}
