/* mic_test - hardware check for the Whistle push-to-talk front end.
 *
 * Captures 16 kHz mono audio from an INMP441 over I2S while the button on
 * GPIO7 is held, and on release prints the clip length plus peak and RMS
 * levels. The onboard WS2812 is white while recording.
 *
 *   INMP441  SCK=GPIO4  WS=GPIO5  SD=GPIO6  L/R=GND (left slot)
 *   button   GPIO7 to GND, internal pull-up (0 = pressed)
 *
 * The I2S channel runs continuously and samples are only kept while the
 * button is down. The INMP441 needs ~85 ms after its clock starts before the
 * output is valid, so starting the clock on every press would put a startup
 * transient at the head of every clip.
 *
 * Both I2S slots are captured. The mic should only ever drive the left one;
 * the right slot is a wiring check. An INMP441 releases SD outside its slot,
 * so a loose L/R wire shows up as audio jumping to the right channel while
 * the left reads whatever the floating line picks up -- in testing, exact bit
 * patterns like 0xAAAA and rail-to-rail jumps. SD has a pull-down (as the
 * datasheet recommends), so an undriven slot reads as exact zeros instead.
 *
 * In (optional): 'p' / 'r' on the console press and release a virtual
 * button, so the capture path can be exercised without touching the board.
 *
 * Out, one line each:
 *   EVT ...                     status
 *   CLIP ch=L|R ms=.. samples=.. peak=.. peak_dbfs=.. rms=.. rms_dbfs=.. dc=..
 *        zeros=..  jumps=..     fraction of exact-zero samples, and of
 *                               sample-to-sample steps over half scale
 *   HB ...                      heartbeat every 5 s: loop is alive
 *   PCM <base64>  ...  PCM end  the clip as interleaved stereo s16le, for
 *                               tools/mic_capture.py
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "driver/rmt_tx.h"
#include <fcntl.h>
#include <unistd.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MIC_SCK_GPIO   4
#define MIC_WS_GPIO    5
#define MIC_SD_GPIO    6
#define BUTTON_GPIO    7
#define ONBOARD_GPIO   48
#define RMT_RESOLUTION 10000000      /* 10 MHz -> 100 ns per tick */

#define SAMPLE_RATE    16000
#define MAX_SECS       30
#define CHUNK          160           /* 10 ms per read */
#define DEBOUNCE_MS    20
#define HEARTBEAT_MS   5000

/* The INMP441 sends 24 significant bits, MSB-aligned in a 32-bit slot, so the
 * top 16 bits are the 16-bit PCM sample. */
#define PCM_SHIFT      16

static rmt_channel_handle_t s_chan;
static rmt_encoder_handle_t s_encoder;
static i2s_chan_handle_t    s_rx;

/* ------------------------------------------------------------------ WS2812 */

static void led_init(void)
{
    rmt_tx_channel_config_t chan_cfg = {
        .clk_src           = RMT_CLK_SRC_DEFAULT,
        .gpio_num          = ONBOARD_GPIO,
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
}

static void led_set(uint8_t r, uint8_t g, uint8_t b)
{
    uint8_t grb[3] = { g, r, b };          /* WS2812 wire order */
    rmt_transmit_config_t tx = { .loop_count = 0 };
    rmt_transmit(s_chan, s_encoder, grb, sizeof(grb), &tx);
    rmt_tx_wait_all_done(s_chan, portMAX_DELAY);
}

/* --------------------------------------------------------------------- I2S */

static void mic_init(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0,
                                                            I2S_ROLE_MASTER);
    i2s_std_config_t  std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_SCK_GPIO,
            .ws   = MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = MIC_SD_GPIO,
        },
    };

    /* L/R tied to GND: the mic drives the slot where WS is low. The right
     * slot is captured too, as a wiring check. */
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;

    chan_cfg.dma_frame_num = CHUNK;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &std_cfg));
    /* After init: the I2S driver reconfigures the pin. */
    ESP_ERROR_CHECK(gpio_pulldown_en(MIC_SD_GPIO));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
}

static void button_init(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BUTTON_GPIO,
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
}

/* ------------------------------------------------------------------- output */

static double dbfs(double v)
{
    return v > 0.0 ? 20.0 * log10(v / 32768.0) : -120.0;
}

/* Stats for one channel of an interleaved stereo clip of n frames. */
static void report(const int16_t *pcm, size_t n, int ch)
{
    double sum = 0.0, sq = 0.0, mean, rms;
    int    peak = 0;
    size_t i, zeros = 0, jumps = 0;

    if (n == 0) {
        printf("CLIP ch=%c ms=0 samples=0\n", ch ? 'R' : 'L');
        return;
    }
    for (i = 0; i < n; i++) {
        int v = pcm[2 * i + ch], a = abs(v);
        if (a > peak)
            peak = a;
        sum += v;
        zeros += (v == 0);
        /* Real audio never steps more than half scale between samples. */
        if (i && abs(v - pcm[2 * (i - 1) + ch]) > 16384)
            jumps++;
    }
    mean = sum / n;
    /* RMS around the mean: the INMP441 has a small DC offset, which would
     * otherwise put a floor under the "silent" figure. */
    for (i = 0; i < n; i++) {
        double d = pcm[2 * i + ch] - mean;
        sq += d * d;
    }
    rms = sqrt(sq / n);

    printf("CLIP ch=%c ms=%u samples=%u peak=%d peak_dbfs=%.1f rms=%.1f "
           "rms_dbfs=%.1f dc=%.1f zeros=%.3f jumps=%.4f\n",
           ch ? 'R' : 'L', (unsigned)(n * 1000 / SAMPLE_RATE), (unsigned)n,
           peak, dbfs(peak), rms, dbfs(rms), mean,
           (double)zeros / n, (double)jumps / n);
}

static void dump_pcm(const int16_t *pcm, size_t n)
{
    static const char B64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const uint8_t *p   = (const uint8_t *)pcm;
    size_t         len = 2 * n * sizeof(int16_t), i;
    char           line[4 + 128 + 2];
    int            k = 0;

    /* 96 input bytes -> 128 base64 chars per line. */
    for (i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        int      m = (int)(len - i);

        if (m > 1) v |= (uint32_t)p[i + 1] << 8;
        if (m > 2) v |= p[i + 2];
        line[k++] = B64[(v >> 18) & 63];
        line[k++] = B64[(v >> 12) & 63];
        line[k++] = m > 1 ? B64[(v >> 6) & 63] : '=';
        line[k++] = m > 2 ? B64[v & 63] : '=';
        if (k == 128 || i + 3 >= len) {
            line[k] = '\0';
            printf("PCM %s\n", line);
            k = 0;
            /* 1 ms per 128-char line, ~128 KB/s: slow enough that the host's
             * polling keeps up and the driverless console drops nothing.
             * Also lets the idle task run during a long dump. */
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    printf("PCM end rate=%d ch=2\n", SAMPLE_RATE);
}

/* ----------------------------------------------------------------- console */

/* The console stays on the default driverless USB-Serial/JTAG path, which
 * never blocks: it drops bytes if the host is slow to poll. Installing the
 * driver (blocking writes) was tried and hung the loop in fflush() straight
 * after a dump, leaving the button dead. Dropped bytes are handled instead by
 * pacing the dump (dump_pcm) and by mic_capture.py tolerating bad lines. */
static void console_init(void)
{
    /* Non-blocking stdin, for the 'p' / 'r' virtual button. */
    fcntl(fileno(stdin), F_SETFL, fcntl(fileno(stdin), F_GETFL) | O_NONBLOCK);
}

/* --------------------------------------------------------------------- main */

void app_main(void)
{
    const size_t cap = (size_t)SAMPLE_RATE * MAX_SECS;
    int16_t     *clip;
    int32_t      raw[2 * CHUNK];
    size_t       n = 0;
    int          recording = 0, stable = 1, last = 1, vpress = 0;
    int64_t      changed_at = 0, t_start = 0, t_hb = 0;
    unsigned     reads = 0, timeouts = 0;

    vTaskDelay(pdMS_TO_TICKS(500));
    console_init();
    led_init();
    led_set(0, 0, 0);
    button_init();
    mic_init();

    clip = heap_caps_malloc(2 * cap * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (!clip) {
        printf("ERR clip_alloc bytes=%u\n", (unsigned)(2 * cap * sizeof(int16_t)));
        return;
    }
    printf("EVT mic_test ready rate=%d max_secs=%d psram_free=%u\n",
           SAMPLE_RATE, MAX_SECS,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("EVT hold the button on GPIO%d and speak\n", BUTTON_GPIO);
    fflush(stdout);

    for (;;) {
        size_t got = 0, i, m;
        int    lvl;

        uint8_t cmd;

        /* Blocks for one 10 ms DMA frame; this also paces the loop. A
         * timeout here means the I2S clock or DMA has stopped. */
        if (i2s_channel_read(s_rx, raw, sizeof(raw), &got,
                             pdMS_TO_TICKS(500)) != ESP_OK) {
            printf("ERR i2s_timeout count=%u\n", ++timeouts);
            fflush(stdout);
            continue;
        }
        reads++;
        m = got / (2 * sizeof(int32_t));          /* stereo frames */

        while (read(fileno(stdin), &cmd, 1) == 1) {
            if (cmd == 'p') vpress = 1;
            if (cmd == 'r') vpress = 0;
        }
        lvl = vpress ? 0 : gpio_get_level(BUTTON_GPIO);

        if (esp_timer_get_time() - t_hb >= HEARTBEAT_MS * 1000) {
            t_hb = esp_timer_get_time();
            printf("HB reads=%u timeouts=%u button=%d stable=%d rec=%d\n",
                   reads, timeouts, gpio_get_level(BUTTON_GPIO), stable,
                   recording);
            fflush(stdout);
        }
        if (lvl != last) {
            last       = lvl;
            changed_at = esp_timer_get_time();
        } else if (lvl != stable &&
                   esp_timer_get_time() - changed_at >= DEBOUNCE_MS * 1000) {
            stable = lvl;
            if (stable == 0) {
                recording = 1;
                n         = 0;
                t_start   = esp_timer_get_time();
                led_set(40, 40, 40);
                printf("EVT recording\n");
                fflush(stdout);
            } else if (recording) {
                recording = 0;
                led_set(0, 0, 0);
                printf("EVT released held_ms=%u\n",
                       (unsigned)((esp_timer_get_time() - t_start) / 1000));
                report(clip, n, 0);
                report(clip, n, 1);
                dump_pcm(clip, n);
                fflush(stdout);
            }
        }

        if (recording) {
            for (i = 0; i < m && n < cap; i++, n++) {
                clip[2 * n]     = (int16_t)(raw[2 * i] >> PCM_SHIFT);
                clip[2 * n + 1] = (int16_t)(raw[2 * i + 1] >> PCM_SHIFT);
            }
            if (n == cap && recording == 1) {
                printf("EVT clip_full secs=%d (release to finish)\n", MAX_SECS);
                fflush(stdout);
                recording = 2;      /* keep LED on, stop appending */
            }
        }
    }
}
