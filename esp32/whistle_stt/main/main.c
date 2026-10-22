/* whistle_stt: hold the button, speak, get the text on the console.
 *
 * The speech-to-text front end of a local voice assistant, and the bench the
 * transcription is measured on: tools/whistle_collect.py records phrases
 * through it, tools/whistle_bench.py replays them at any decoder depth.
 *
 * Board: Waveshare ESP32-P4-WIFI6 (pins in components/board/board.h)
 *   mic      onboard, through the ES8311 codec
 *   button   BOOT (GPIO35)
 *
 * The model (whistle.cact, or whistle_d4.cact from tools/whistle_strip.py)
 * lives in the "model" partition and is copied at boot into PSRAM, which
 * streams far faster than the QIO flash mapping.
 *
 * Out, one line each:
 *   EVT ...     status (booting, ready, recording, transcribing, error ...),
 *               levels and per-phase timings
 *   TXT ...     the transcript
 *   RES ...     the same result as one line of key=value pairs, for tools
 *   CFG ...     the current settings, in answer to `set`
 *   PCM ...     with `dump on`: the clip just transcribed, base64 s16le
 *   OK / ERR    answers to console commands
 *
 * In, one command per line (applied between clips):
 *   set [depth=<n>] [lang=auto|<l>|<l>,<l>...]
 *   dump on|off
 *   clip n=<samples> floor_dbfs=<x|none>, base64 s16le lines, clip end
 *   run
 */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "board.h"
#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "nd_quant.h"
#include "nd_whistle.h"

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
#define FLASH_MAP_END  0x1000000     /* the cache maps only the first 16 MB of
                                      * quad flash (24-bit addresses), though
                                      * the model partition runs to 32 MB */
#define RX_BUF         8192          /* console receive buffer: the loop reads
                                      * it every 10 ms, and a clip upload sends
                                      * a line per OK */
#define LINE_MAX       2048

static const char *const LANGS[] = { "en", "de", "fr", "es", "it", "nl", "pl" };

static nd_whistle s_w;

/* Settings, changed only between clips by the task that transcribes. */
static int      s_lang_force = -1;          /* one language: no detection */
static char     s_lang_name[32] = "auto";   /* as `set` took it */
static int      s_dump;

static void *psram_alloc(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM); }

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

/* ----------------------------------------------------------------- console */

/* UART0 through the USB bridge. The driver's receive buffer holds what
 * arrives while a clip is transcribed; the hardware FIFO alone (128 bytes)
 * fills in 1.4 ms at 921600 baud. */
static void console_init(void)
{
    ESP_ERROR_CHECK(uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, RX_BUF, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* Base64 -> bytes. Returns the count, or -1 on a character outside the
 * alphabet or a length that is not a multiple of 4. */
static int b64_decode(const char *s, uint8_t *out, int cap)
{
    static int8_t rev[256];
    int           n = 0, len = (int)strlen(s), i, k;

    if (!rev['B'])
        for (i = 0; i < 256; i++) {
            const char *p = i ? strchr(B64, i) : NULL;
            rev[i] = p ? (int8_t)(p - B64) : -1;
        }
    if (len % 4)
        return -1;
    for (i = 0; i < len; i += 4) {
        uint32_t v = 0;
        int      pad = 0;
        for (k = 0; k < 4; k++) {
            int c = (uint8_t)s[i + k];
            if (c == '=' && i + 4 == len && k >= 2) {
                pad++;
                v <<= 6;
            } else if (rev[c] < 0 || pad) {
                return -1;
            } else {
                v = v << 6 | (uint32_t)rev[c];
            }
        }
        if (n + 3 - pad > cap)
            return -1;
        out[n++] = (uint8_t)(v >> 16);
        if (pad < 2) out[n++] = (uint8_t)(v >> 8);
        if (pad < 1) out[n++] = (uint8_t)v;
    }
    return n;
}

/* The clip as transcribe() got it, s16le mono in base64, 96 bytes a line as
 * mic_test's PCM lines, then the floor it was gated against. */
static void dump_pcm(const float *x, uint32_t n, float floor_rms)
{
    uint8_t  b[96];
    char     line[4 + 128 + 1];
    uint32_t i = 0;

    while (i < n) {
        int nb = 0, k, o = 0;
        for (; i < n && nb < (int)sizeof(b); i++) {
            int v = (int)lrintf(x[i] * 32768.0f);
            v = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
            b[nb++] = (uint8_t)v;
            b[nb++] = (uint8_t)(v >> 8);
        }
        memcpy(line, "PCM ", 4);
        o = 4;
        for (k = 0; k < nb; k += 3) {
            uint32_t v = (uint32_t)b[k] << 16;
            int      m = nb - k;
            if (m > 1) v |= (uint32_t)b[k + 1] << 8;
            if (m > 2) v |= b[k + 2];
            line[o++] = B64[(v >> 18) & 63];
            line[o++] = B64[(v >> 12) & 63];
            line[o++] = m > 1 ? B64[(v >> 6) & 63] : '=';
            line[o++] = m > 2 ? B64[v & 63] : '=';
        }
        line[o] = '\0';
        puts(line);
        /* Lets the idle task run during a long dump; the console blocks
         * rather than dropping bytes. */
        fflush(stdout);
        vTaskDelay(1);
    }
    if (floor_rms > 0)
        printf("PCM end rate=%d ch=1 n=%u floor_dbfs=%.4f\n", SAMPLE_RATE, (unsigned)n,
               20.0f * log10f(floor_rms));
    else
        printf("PCM end rate=%d ch=1 n=%u floor_dbfs=none\n", SAMPLE_RATE, (unsigned)n);
    fflush(stdout);
}

static void print_cfg(void)
{
    uint32_t i;
    printf("CFG depth=%u layers=", (unsigned)s_w.depth);
    for (i = 0; i < s_w.depth; i++)
        printf(i ? ",%u" : "%u", (unsigned)s_w.active[i]);
    printf(" lang=%s model_layers=%u dump=%s\n", s_lang_name, (unsigned)s_w.dec_layers,
           s_dump ? "on" : "off");
    fflush(stdout);
}

/* "auto", "de" or "de,en" -> a forced language (-1: detect) and the mask
 * detection chooses from. Returns 0, or -1 for anything else. */
static int parse_lang(const char *v, int *force, uint32_t *mask)
{
    char  buf[sizeof(s_lang_name)];
    char *tok, *save;
    int   count = 0, last = -1;

    *force = -1;
    *mask  = 0x7Fu;
    if (!strcmp(v, "auto"))
        return 0;
    if (strlen(v) >= sizeof(buf))
        return -1;
    strcpy(buf, v);
    *mask = 0;
    for (tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        int k;
        for (k = 0; k < ND_WH_LANGS && strcmp(tok, LANGS[k]); k++)
            ;
        if (k == ND_WH_LANGS)
            return -1;
        if (!(*mask & (1u << k)))
            count++;
        *mask |= 1u << k;
        last = k;
    }
    if (!count)
        return -1;
    if (count == 1) {
        *force = last;
        *mask  = 0x7Fu;
    }
    return 0;
}

static void cmd_set(char *args)
{
    char       *tok, *save;
    long        depth = -1;
    const char *lang = NULL;
    int         force = 0;
    uint32_t    mask = 0;

    /* Check everything before changing anything. */
    for (tok = strtok_r(args, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
        if (!strncmp(tok, "depth=", 6)) {
            char *end;
            depth = strtol(tok + 6, &end, 10);
            if (*end || end == tok + 6 || depth < 2 || depth > (long)s_w.dec_layers) {
                printf("ERR set depth must be 2..%u\n", (unsigned)s_w.dec_layers);
                fflush(stdout);
                return;
            }
        } else if (!strncmp(tok, "lang=", 5)) {
            lang = tok + 5;
            if (parse_lang(lang, &force, &mask) != 0) {
                printf("ERR set lang must be auto, one of en de fr es it nl pl, or a "
                       "comma-separated list of them\n");
                fflush(stdout);
                return;
            }
        } else {
            printf("ERR set unknown key %s (depth, lang)\n", tok);
            fflush(stdout);
            return;
        }
    }
    if (depth > 0)
        nd_whistle_set_depth(&s_w, (uint32_t)depth);
    if (lang) {
        s_lang_force  = force;
        s_w.lang_mask = mask;
        strcpy(s_lang_name, lang);
    }
    print_cfg();
}

/* --------------------------------------------------------------- inference */

/* Transcribe x[0..n) against a noise floor (0: estimate it from the clip). */
static void transcribe(float *x, uint32_t n, float floor_rms)
{
    static const char *PN[ND_WP_COUNT] = { "mel", "stem", "encoder", "cross_kv", "decoder",
                                           "~matmul", "~mlp", "~attn", "~mhc" };
    uint32_t     n0 = n, s0 = 0, s1 = n, T = 0, ids[96], i;
    float        peak = 0.0f, *enc;
    int          nt = 0;
    char         text[512];
    const char  *detected = "none";
    int64_t      t0 = esp_timer_get_time(), t_enc = t0, t_dec = t0;

    text[0] = '\0';
    for (i = 0; i < n; i++)
        if (fabsf(x[i]) > peak)
            peak = fabsf(x[i]);
    printf("EVT clip ms=%u peak_dbfs=%.1f floor_dbfs=%.1f\n", (unsigned)(n / 16),
           peak > 0 ? 20.0f * log10f(peak) : -120.0f,
           floor_rms > 0 ? 20.0f * log10f(floor_rms) : -120.0f);

    if (!nd_whistle_find_speech(x, n, floor_rms, GATE_DB, 300, &s0, &s1)) {
        printf("EVT no_speech (nothing %.0f dB above the noise floor)\n", GATE_DB);
        printf("TXT \n");
        s0 = s1 = 0;
        goto result;
    }
    x += s0;
    n  = s1 - s0;
    T  = nd_whistle_frames(&s_w, n);
    printf("EVT transcribing speech_ms=%u frames=%u\n", (unsigned)(n / 16), (unsigned)T);
    fflush(stdout);

    memset(nd_wprof, 0, sizeof(nd_wprof));
    enc = (float *)heap_caps_malloc((size_t)T * s_w.d * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!enc || nd_whistle_encode(&s_w, x, n, enc) != 0) {
        printf("EVT error encode\n");
        heap_caps_free(enc);
        nt = -1;
        goto result;
    }
    t_enc = esp_timer_get_time();
    nt = nd_whistle_decode(&s_w, enc, T, s_lang_force, ids, sizeof(ids) / sizeof(ids[0]));
    t_dec = esp_timer_get_time();
    heap_caps_free(enc);
    if (nt < 0) {
        printf("EVT error decode\n");
        goto result;
    }
    nd_whistle_detok(&s_w, ids, (uint32_t)nt, text, sizeof(text));
    detected = LANGS[s_w.lang];

    printf("EVT lang %s\n", detected);
    printf("TXT %s\n", text);
    printf("EVT time total_ms=%.0f encode_ms=%.0f decode_ms=%.0f tokens=%d audio_ms=%u\n",
           (t_dec - t0) / 1000.0, (t_enc - t0) / 1000.0, (t_dec - t_enc) / 1000.0, nt,
           (unsigned)(n / 16));
    for (i = 0; i < ND_WP_COUNT; i++)
        printf("EVT prof %-9s %8.0f ms\n", PN[i], nd_wprof[i] / 1000.0);

result:
    printf("RES depth=%u lang=%s lang_detected=%s audio_ms=%u speech_ms=%u encode_ms=%.0f "
           "decode_ms=%.0f total_ms=%.0f tokens=%d text=\"",
           (unsigned)s_w.depth, s_lang_name, nt < 0 ? "error" : detected, (unsigned)(n0 / 16),
           (unsigned)((s1 - s0) / 16), (t_enc - t0) / 1000.0, (t_dec - t_enc) / 1000.0,
           (t_dec - t0) / 1000.0, nt < 0 ? 0 : nt);
    for (i = 0; text[i]; i++) {
        if (text[i] == '"' || text[i] == '\\')
            putchar('\\');
        putchar(text[i] == '\n' || text[i] == '\r' ? ' ' : text[i]);
    }
    printf("\"\n");
    fflush(stdout);
}

/* -------------------------------------------------------------------- main */

/* A clip sent over the console for `run`. */
static float   *s_up;
static uint32_t s_up_n, s_up_want;
static float    s_up_floor;
static int      s_uploading, s_up_ok;

/* One console line. Returns 1 if it ran a transcription. */
static int command(char *line, int recording)
{
    if (s_uploading) {
        if (!strcmp(line, "clip end")) {
            s_uploading = 0;
            s_up_ok = s_up_n == s_up_want;
            if (s_up_ok)
                printf("OK clip n=%u\n", (unsigned)s_up_n);
            else
                printf("ERR clip got %u of %u samples\n", (unsigned)s_up_n,
                       (unsigned)s_up_want);
        } else {
            static uint8_t b[LINE_MAX];
            int            nb = b64_decode(line, b, sizeof(b)), k;
            if (nb < 0 || nb % 2 || s_up_n + nb / 2 > s_up_want) {
                s_uploading = 0;
                printf("ERR clip bad line after %u samples (upload dropped)\n",
                       (unsigned)s_up_n);
            } else {
                for (k = 0; k < nb; k += 2)
                    s_up[s_up_n++] = (float)(int16_t)(b[k] | b[k + 1] << 8) / 32768.0f;
                printf("OK\n");
            }
        }
        fflush(stdout);
        return 0;
    }
    if (!strcmp(line, "set") || !strncmp(line, "set ", 4)) {
        cmd_set(line + 3);
    } else if (!strcmp(line, "dump on") || !strcmp(line, "dump off")) {
        s_dump = line[6] == 'n';
        print_cfg();
    } else if (!strncmp(line, "clip ", 5)) {
        char *tok, *save;
        long  n = -1;
        float fl = -1.0f;
        for (tok = strtok_r(line + 5, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
            if (!strncmp(tok, "n=", 2))
                n = strtol(tok + 2, NULL, 10);
            else if (!strcmp(tok, "floor_dbfs=none"))
                fl = 0.0f;
            else if (!strncmp(tok, "floor_dbfs=", 11))
                fl = powf(10.0f, strtof(tok + 11, NULL) / 20.0f);
        }
        if (n <= 0 || n > SAMPLE_RATE * MAX_SECS || fl < 0) {
            printf("ERR clip needs n=1..%d floor_dbfs=<dBFS|none>\n", SAMPLE_RATE * MAX_SECS);
        } else {
            s_uploading = 1;
            s_up_ok     = 0;
            s_up_n      = 0;
            s_up_want   = (uint32_t)n;
            s_up_floor  = fl;
            printf("OK\n");
        }
    } else if (!strcmp(line, "run")) {
        if (recording)
            printf("ERR run while recording\n");
        else if (!s_up_ok)
            printf("ERR run needs a clip (clip n=... then clip end)\n");
        else {
            transcribe(s_up, s_up_n, s_up_floor);
            return 1;
        }
    } else {
        printf("ERR unknown command: %s (set, dump, clip, run)\n", line);
    }
    fflush(stdout);
    return 0;
}

/* Drain the console. Returns 1 if a command ran a transcription. */
static int poll_console(int recording)
{
    static char line[LINE_MAX];
    static int  len, overflow;
    char        buf[256];
    int         got, i, ran = 0;

    while ((got = read(fileno(stdin), buf, sizeof(buf))) > 0) {
        for (i = 0; i < got; i++) {
            char c = buf[i];
            if (c == '\n' || c == '\r') {
                line[len] = '\0';
                if (overflow) {
                    printf("ERR line longer than %d bytes\n", LINE_MAX - 1);
                    fflush(stdout);
                    s_uploading = 0;
                } else if (len) {
                    ran |= command(line, recording);
                }
                len = overflow = 0;
            } else if (len < LINE_MAX - 1) {
                line[len++] = c;
            } else {
                overflow = 1;
            }
        }
    }
    return ran;
}

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

    s_up = (float *)heap_caps_malloc(cap * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!clip || !s_up) {
        printf("EVT error clip_alloc\n");
        vTaskDelete(NULL);
    }
    printf("\nEVT ==================================================\n");
    printf("EVT  READY - hold BOOT (GPIO%d) and speak\n", BUTTON_GPIO);
    printf("EVT  commands: set depth=<n> lang=<l>, dump on|off, clip, run\n");
    printf("EVT ==================================================\n\n");
    print_cfg();

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;

        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got, portMAX_DELAY) != ESP_OK)
            continue;
        m = got / sizeof(int32_t);

        if (poll_console(recording)) {
            /* As after a live clip: start the floor on fresh audio. */
            win_e = 0.0f;
            win_n = 0;
            continue;
        }

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
                printf("EVT recording\n");
                fflush(stdout);
            } else if (recording) {
                uint32_t head = HEAD_DROP_MS * 16, tail = TAIL_DROP_MS * 16;
                recording = 0;
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                if (n > head + tail + 1600) {
                    float floor_rms = noise_floor();
                    transcribe(clip + head, n - head - tail, floor_rms);
                    if (s_dump)
                        dump_pcm(clip + head, n - head - tail, floor_rms);
                } else {
                    printf("EVT too_short\nTXT \n");
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

/* Copy the .cact at the start of the partition into PSRAM. The blob records
 * no total size, so it ends where its last tensor does. The mapping only has
 * to hold the header and directory: the copy goes through esp_partition_read,
 * which reaches past FLASH_MAP_END. Returns the copy and its size, or NULL to
 * keep using the mapping. */
static void *model_to_psram(const esp_partition_t *part, const void *blob, size_t avail,
                            size_t *size)
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
    if (end == 0 || end > part->size)
        return NULL;                        /* let nd_whistle_open report it */
    p = heap_caps_aligned_alloc(ND_CACT_ALIGN, end, MALLOC_CAP_SPIRAM);
    if (!p) {
        printf("EVT model_in_flash (no %u bytes of PSRAM free)\n", (unsigned)end);
        return NULL;
    }
    if (esp_partition_read(part, 0, p, end) != ESP_OK) {
        printf("EVT error model_read_failed bytes=%u\n", (unsigned)end);
        heap_caps_free(p);
        return NULL;
    }
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
    console_init();
    printf("EVT booting\n");
    fflush(stdout);
    worker_start();
    board_button_init();
    mic_init();
    if (nd_cq_int8_check() != 0) {
        /* Transcripts would still come out, just worse: say so loudly and
         * use the float path. */
        printf("EVT error int8_kernel_mismatch (falling back to float)\n");
        nd_cq_int8 = 0;
    }

    part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) {
        printf("EVT error no_model_partition\n");
        return;
    }
    size = part->size;
    if (part->address + size > FLASH_MAP_END)
        size = part->address < FLASH_MAP_END ? FLASH_MAP_END - part->address : 0;
    if (esp_partition_mmap(part, 0, size, ESP_PARTITION_MMAP_DATA, &blob, &handle) != ESP_OK) {
        printf("EVT error mmap_failed size=%u\n", (unsigned)size);
        return;
    }
    copy = model_to_psram(part, blob, size, &size);
    if (copy) {
        esp_partition_munmap(handle);
        blob = copy;
    }
    rc = nd_whistle_open(&s_w, blob, size);
    if (rc != 0) {
        /* -1..-5: no valid .cact at 0x210000; -6: a .cact that is not v4
         * (e.g. needle2.cact is still flashed there). */
        printf("EVT error whistle_open rc=%d (flash whistle.cact or whistle_d4.cact to "
               "0x%06x)\n", rc, (unsigned)part->address);
        return;
    }
    /* No keyword bias and no `accept`: a plain transcript, so no retries in
     * other languages either. Bias words (device and room names) come later. */
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

    /* The decoder keeps a few KB of scratch per call level; give it room. */
    xTaskCreatePinnedToCore(whistle_task, "whistle", 48 * 1024, NULL, 5, NULL, 0);
}
