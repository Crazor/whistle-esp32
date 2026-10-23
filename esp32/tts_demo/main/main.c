/* Pico text to speech on ESP32-P4, out of the board's ES8311 codec.
 *
 * Type a line on the console and the board says it. SVOX Pico (nd_tts.h)
 * runs on the chip; the lingware lives in the "lang" FAT partition
 * (partitions.csv, tools/pico_lang.py) and is copied to PSRAM when a voice
 * opens.
 *
 *   ES8311   I2C SDA=GPIO7 SCL=GPIO8, I2S MCLK=13 BCLK=12 WS=10 DOUT=9
 *   speaker  NS4150B amplifier, enabled by GPIO53
 *
 * The pins are in board.h (Waveshare ESP32-P4-WIFI6). Espressif's
 * ESP32-P4-Function-EV-Board uses the same ones.
 *
 * In, one line each:
 *   :de          switch voice (en de fr es it, or en-GB); 0.1-0.3 s
 *   :vol 70      speaker volume, 0-100
 *   :tone [hz]   one second of a sine (default 440 Hz), through the codec
 *                driver; :rawtone writes it to I2S directly instead
 *   :pa 0|1      speaker amplifier off/on by hand (GPIO53)
 *   :regs        dump the ES8311's registers
 *   anything     spoken in the current voice. Pico markup works inline:
 *                <speed level="80">slower</speed>, <pitch level="120">
 *
 * Out, one line each (the EVT vocabulary of the other demos):
 *   EVT ...      status, and for each line spoken its length and how long
 *                synthesis took
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_vfs_fat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#else
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#endif

#include "board.h"
#include "nd_tts.h"

#define LANG_MOUNT     "/lang"
#define PREBUF         (ND_TTS_RATE / 5)  /* 200 ms queued before the first
                                           * write, so a slow start of
                                           * synthesis does not stutter */
#define VOLUME         70

static esp_codec_dev_handle_t s_spk;
static const audio_codec_if_t *s_codec_if;   /* for :regs */
static i2s_chan_handle_t      s_tx;          /* for :rawtone */
static nd_tts                 s_tts;
static void                  *s_mem;

/* ------------------------------------------------------------------ codec */

/* As Espressif's board support package for the EV board sets it up: the
 * chip drives I2S (MCLK = 256 fs), the codec is the I2C-controlled slave,
 * and esp_codec_dev switches the speaker amplifier with the codec. */
static esp_codec_dev_handle_t speaker_init(void)
{
    i2c_master_bus_handle_t i2c;
    i2s_chan_handle_t       tx;
    i2c_master_bus_config_t bus = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = BOARD_I2C_SDA_GPIO,
        .scl_io_num = BOARD_I2C_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_std_config_t  std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(ND_TTS_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = BOARD_I2S_MCLK_GPIO,
            .bclk = BOARD_I2S_BCLK_GPIO,
            .ws = BOARD_I2S_WS_GPIO,
            .dout = BOARD_I2S_DOUT_GPIO,
            .din = I2S_GPIO_UNUSED,
        },
    };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus, &i2c));
    /* Silence rather than the last buffer on repeat when synthesis falls
     * behind or a line ends. */
    chan.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan, &tx, NULL));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(tx, &std));
    ESP_ERROR_CHECK(i2s_channel_enable(tx));

    audio_codec_i2s_cfg_t      i2s_cfg = { .port = I2S_NUM_0, .tx_handle = tx };
    /* esp_codec_dev takes the 8-bit address (0x30 for the ES8311's 0x18). */
    audio_codec_i2c_cfg_t      i2c_cfg = { .port = I2C_NUM_0, .addr = BOARD_ES8311_ADDR << 1,
                                           .bus_handle = i2c };
    const audio_codec_data_if_t *data_if = audio_codec_new_i2s_data(&i2s_cfg);
    const audio_codec_ctrl_if_t *ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    const audio_codec_gpio_if_t *gpio_if = audio_codec_new_gpio();
    es8311_codec_cfg_t es = {
        .ctrl_if = ctrl_if,
        .gpio_if = gpio_if,
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC,
        .pa_pin = BOARD_PA_GPIO,
        .use_mclk = true,
        .hw_gain = { .pa_voltage = 5.0, .codec_dac_voltage = 3.3 },
    };
    const audio_codec_if_t *codec_if = es8311_codec_new(&es);
    esp_codec_dev_cfg_t     dev_cfg = {
        .dev_type = ESP_CODEC_DEV_TYPE_OUT,
        .codec_if = codec_if,
        .data_if = data_if,
    };
    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel = 1,
        .sample_rate = ND_TTS_RATE,
    };
    esp_codec_dev_handle_t dev;

    if (!data_if || !ctrl_if || !gpio_if || !codec_if || !(dev = esp_codec_dev_new(&dev_cfg))) {
        printf("EVT error ES8311 not found on I2C (SDA=%d SCL=%d)\n", BOARD_I2C_SDA_GPIO, BOARD_I2C_SCL_GPIO);
        return NULL;
    }
    if (esp_codec_dev_open(dev, &fs) != ESP_CODEC_DEV_OK) {
        printf("EVT error ES8311 did not open\n");
        return NULL;
    }
    esp_codec_dev_set_out_vol(dev, VOLUME);
    /* Readable as well as driven, so :pa and the boot line can report it. */
    gpio_set_direction(BOARD_PA_GPIO, GPIO_MODE_INPUT_OUTPUT);
    gpio_set_level(BOARD_PA_GPIO, 1);
    s_codec_if = codec_if;
    s_tx = tx;
    printf("EVT codec open rate=%d vol=%d pa_gpio=%d pa_level=%d\n", ND_TTS_RATE, VOLUME,
           BOARD_PA_GPIO, gpio_get_level(BOARD_PA_GPIO));
    return dev;
}

/* ----------------------------------------------------------------- speech */

typedef struct {
    int16_t  pre[PREBUF];
    size_t   npre;
    int      started;
    int64_t  write_us;      /* time spent waiting on the codec, not Pico */
    int      peak;          /* largest |sample|, to tell silence from sound */
    int      write_err;     /* last failed codec write, 0 if none */
} player;

static void out(player *p, const int16_t *pcm, size_t n)
{
    int64_t t = esp_timer_get_time();
    int     err = esp_codec_dev_write(s_spk, (void *)pcm, (int)(n * sizeof(*pcm)));
    size_t  i;
    if (err != ESP_CODEC_DEV_OK)
        p->write_err = err;
    for (i = 0; i < n; i++) {
        int a = pcm[i] < 0 ? -pcm[i] : pcm[i];
        if (a > p->peak)
            p->peak = a;
    }
    p->write_us += esp_timer_get_time() - t;
}

/* nd_tts sink: queue the first PREBUF samples, then stream. The codec write
 * blocks while the DMA buffers are full, which paces Pico to the speaker. */
static int play(const int16_t *pcm, size_t n, void *ctx)
{
    player *p = (player *)ctx;
    if (!p->started) {
        size_t k = n < PREBUF - p->npre ? n : PREBUF - p->npre;
        memcpy(p->pre + p->npre, pcm, k * sizeof(*pcm));
        p->npre += k;
        pcm += k;
        n -= k;
        if (p->npre < PREBUF)
            return 0;
        p->started = 1;
        out(p, p->pre, p->npre);
    }
    if (n)
        out(p, pcm, n);
    return 0;
}

static void say(const char *text)
{
    static player p;
    int64_t       t0 = esp_timer_get_time();
    long          n;

    p.npre = 0;
    p.started = 0;
    p.write_us = 0;
    p.peak = 0;
    p.write_err = 0;
    n = nd_tts_speak(&s_tts, text, play, &p);
    if (!p.started && p.npre)
        out(&p, p.pre, p.npre);             /* shorter than the prebuffer */
    if (n < 0) {
        printf("EVT error %s\n", s_tts.err);
        return;
    }
    /* Pico's own time: the whole call less the waits on the codec. */
    double synth_ms = (esp_timer_get_time() - t0 - p.write_us) / 1000.0;
    double speech_ms = n * 1000.0 / ND_TTS_RATE;
    printf("EVT tts lang=%s speech_ms=%.0f synth_ms=%.0f rtf=%.2f peak=%d write_err=%d\n",
           s_tts.lang, speech_ms, synth_ms, speech_ms > 0 ? synth_ms / speech_ms : 0.0, p.peak,
           p.write_err);
}

/* One second of a sine at half scale, through esp_codec_dev or, with `raw`,
 * straight into the I2S channel, to find where sound stops. */
static void tone(int hz, int raw)
{
    static int16_t buf[ND_TTS_RATE / 10];
    player         p = { 0 };
    int            i, k, err = 0;
    size_t         done;

    if (hz <= 0)
        hz = 440;
    for (k = 0; k < 10; k++) {
        for (i = 0; i < ND_TTS_RATE / 10; i++)
            buf[i] = (int16_t)(16000.0f * sinf(6.2831853f * hz * (k * (ND_TTS_RATE / 10) + i) / ND_TTS_RATE));
        if (raw)
            err = i2s_channel_write(s_tx, buf, sizeof(buf), &done, 1000) != ESP_OK ? -1 : err;
        else
            out(&p, buf, ND_TTS_RATE / 10);
    }
    printf("EVT tone hz=%d via=%s err=%d pa_level=%d\n", hz, raw ? "i2s" : "codec",
           raw ? err : p.write_err, gpio_get_level(BOARD_PA_GPIO));
}

static int open_voice(const char *lang)
{
    char    prev[sizeof(s_tts.lang)];
    int64_t t0 = esp_timer_get_time();

    if (!nd_tts_locale(lang)) {
        printf("EVT error no Pico voice for '%s' (en de fr es it en-GB)\n", lang);
        return -1;
    }
    strcpy(prev, s_tts.engine ? s_tts.lang : "");
    nd_tts_close(&s_tts);               /* Pico takes the whole memory area */
    if (nd_tts_open(&s_tts, s_mem, ND_TTS_MEM_SIZE, LANG_MOUNT, lang)) {
        printf("EVT error %s\n", s_tts.err);
        if (prev[0] && !nd_tts_open(&s_tts, s_mem, ND_TTS_MEM_SIZE, LANG_MOUNT, prev))
            printf("EVT voice %s (kept)\n", s_tts.lang);
        return -1;
    }
    printf("EVT voice %s load_ms=%lld\n", s_tts.lang, (long long)(esp_timer_get_time() - t0) / 1000);
    return 0;
}

/* ---------------------------------------------------------------- console */

static void console_init(void)
{
    setvbuf(stdin, NULL, _IONBF, 0);
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    usb_serial_jtag_vfs_use_driver();
#else
    ESP_ERROR_CHECK(uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 1024, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
#endif
}

/* One line from the console, without its line ending. */
static int read_line(char *buf, size_t size)
{
    size_t n;
    if (!fgets(buf, (int)size, stdin))
        return -1;
    n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
        buf[--n] = 0;
    return (int)n;
}

void app_main(void)
{
    static char             line[1024];
    esp_vfs_fat_mount_config_t mount = { .max_files = 2 };
    esp_err_t               err;

    console_init();
    printf("EVT boot tts_demo\n");

    s_spk = speaker_init();
    err = esp_vfs_fat_spiflash_mount_ro(LANG_MOUNT, "lang", &mount);
    if (err != ESP_OK)
        printf("EVT error lang partition: %s (flash it with idf.py lang-flash)\n", esp_err_to_name(err));
    s_mem = heap_caps_malloc(ND_TTS_MEM_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_spk || err != ESP_OK || !s_mem || open_voice("en")) {
        printf("EVT halted\n");
        return;
    }
    printf("EVT ready\n");
    say("Ready.");

    for (;;) {
        int n = read_line(line, sizeof(line));
        if (n <= 0) {
            if (n < 0)
                vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (!strncmp(line, ":vol ", 5)) {
            int v = atoi(line + 5);
            esp_codec_dev_set_out_vol(s_spk, v < 0 ? 0 : v > 100 ? 100 : v);
            printf("EVT volume %d\n", v);
        } else if (!strncmp(line, ":tone", 5)) {
            tone(atoi(line + 5), 0);
        } else if (!strncmp(line, ":rawtone", 8)) {
            tone(atoi(line + 8), 1);
        } else if (!strncmp(line, ":pa ", 4)) {
            gpio_set_level(BOARD_PA_GPIO, atoi(line + 4) ? 1 : 0);
            printf("EVT pa level=%d\n", gpio_get_level(BOARD_PA_GPIO));
        } else if (!strcmp(line, ":regs")) {
            if (s_codec_if && s_codec_if->dump_reg)
                s_codec_if->dump_reg(s_codec_if);
        } else if (line[0] == ':') {
            open_voice(line + 1);
        } else if (s_tts.engine) {
            printf("EVT say %s\n", line);
            say(line);
        } else {
            printf("EVT error no voice open; pick one with :en :de :fr :es :it\n");
        }
    }
}
