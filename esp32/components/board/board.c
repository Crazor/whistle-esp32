/* board.c - ES8311 microphone and BOOT key on the Waveshare ESP32-P4-WIFI6.
 *
 * Only the codec's ADC is used. The register sequence is esp_codec_dev's
 * ES8311 open + ADC start, cut down to the one clock setup needed here: codec
 * as I2S slave, its internal clock from the MCLK pin at 256 x the sample rate. Written
 * against the new i2c_master driver rather than pulling in esp_codec_dev,
 * which would be the only managed component in the repo.
 */
#include "board.h"

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static i2c_master_dev_handle_t s_codec;

static void wr(uint8_t reg, uint8_t val)
{
    uint8_t b[2] = { reg, val };
    ESP_ERROR_CHECK(i2c_master_transmit(s_codec, b, sizeof(b), 100));
}

static void es8311_adc_init(uint8_t gain)
{
    i2c_master_bus_handle_t     bus;
    i2c_master_bus_config_t     bus_cfg = {
        .i2c_port = -1, .sda_io_num = BOARD_I2C_SDA_GPIO, .scl_io_num = BOARD_I2C_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = 1,
    };
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = BOARD_ES8311_ADDR,
        .scl_speed_hz = 100000,
    };
    uint8_t b[2] = { 0x44, 0x08 };

    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &bus));
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus, &dev_cfg, &s_codec));

    /* I2C noise immunity. The codec sometimes NAKs the very first write, so
     * that one is sent twice and its error ignored. */
    i2c_master_transmit(s_codec, b, sizeof(b), 100);
    wr(0x44, 0x08);

    wr(0x01, 0x30);
    wr(0x02, 0x00);          /* pre-divider 1, multiplier 1x */
    wr(0x03, 0x10);          /* single speed, ADC OSR 16 */
    wr(0x16, gain);          /* ADC digital gain */
    wr(0x04, 0x10);
    wr(0x05, 0x00);          /* ADC and DAC clock dividers 1 */
    wr(0x0B, 0x00);
    wr(0x0C, 0x00);
    wr(0x10, 0x1F);
    wr(0x11, 0x7F);
    wr(0x00, 0x80);          /* power on, slave serial port */
    wr(0x01, 0x3F);          /* clocks on, internal MCLK from the MCLK pin */
    wr(0x06, 0x03);          /* BCLK not inverted (divider unused as slave) */
    wr(0x07, 0x00);
    wr(0x08, 0xFF);          /* LRCK = MCLK / 256 */
    wr(0x09, 0x10);          /* serial in: I2S, 32-bit */
    wr(0x0A, 0x10);          /* serial out: I2S, 32-bit, unmuted */
    wr(0x13, 0x10);
    wr(0x1B, 0x0A);
    wr(0x1C, 0x6A);          /* ADC equaliser bypass, digital DC removal */
    wr(0x0D, 0x01);          /* analog power up */

    /* start the ADC */
    wr(0x14, 0x1A);          /* analog mic on MIC1P, PGA +30 dB */
    wr(0x17, 0xBF);          /* ADC volume 0 dB */
    wr(0x0E, 0x02);          /* PGA and ADC modulator on */
    wr(0x15, 0x00);
    vTaskDelay(pdMS_TO_TICKS(50));
    wr(0x45, 0x00);
}

void board_mic_init(uint32_t sample_rate, uint32_t frame_num, int stereo,
                    i2s_chan_handle_t *rx)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_std_config_t  std_cfg = {
        /* MCLK = 256 x fs, the ratio the codec setup above expects */
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                        stereo ? I2S_SLOT_MODE_STEREO
                                                               : I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = BOARD_I2S_MCLK_GPIO, .bclk = BOARD_I2S_BCLK_GPIO,
            .ws = BOARD_I2S_WS_GPIO, .dout = I2S_GPIO_UNUSED, .din = BOARD_I2S_DIN_GPIO,
        },
    };
    std_cfg.slot_cfg.slot_mask = stereo ? I2S_STD_SLOT_BOTH : I2S_STD_SLOT_LEFT;
    chan_cfg.dma_frame_num = frame_num;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, rx));
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(*rx, &std_cfg));
    /* The codec runs off MCLK, so the clocks go first and it is set up on a
     * running bus. */
    ESP_ERROR_CHECK(i2s_channel_enable(*rx));
    es8311_adc_init(BOARD_MIC_GAIN_6DB);
}

void board_button_init(void)
{
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BUTTON_GPIO, .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE, .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
}
