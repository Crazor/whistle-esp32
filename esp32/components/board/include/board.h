/* board.h - the Waveshare ESP32-P4-WIFI6 as the Whistle apps use it.
 *
 *   mic     the onboard microphone, through the ES8311 codec's ADC
 *           I2S  MCLK=GPIO13 BCLK=GPIO12 WS=GPIO10 DIN=GPIO11 (codec ASDOUT)
 *           I2C  SDA=GPIO7 SCL=GPIO8, ES8311 at 0x18
 *   speaker the codec's DAC: I2S DOUT=GPIO9 (codec DSDIN), and the NS4150B
 *           amplifier, enabled by GPIO53 (PA_CTRL, high = on)
 *   button  the BOOT key, GPIO35 (0 = pressed)
 *   LED     an external WS2812 on LED_GPIO; the board has none of its own
 *
 * The header follows the Raspberry Pi Pico pinout and Waveshare documents it
 * only as a picture: check the silkscreen before wiring the LED, and change
 * LED_GPIO if 23 is not where you want it. Avoid GPIO7-13 (codec), 14-19 and
 * 54 (the ESP32-C6 link), 24-27 (USB), 35-38 (BOOT, console) and 39-45 (TF).
 */
#ifndef BOARD_H
#define BOARD_H

#include "driver/i2s_std.h"

#define BOARD_I2S_MCLK_GPIO  13
#define BOARD_I2S_BCLK_GPIO  12
#define BOARD_I2S_WS_GPIO    10
#define BOARD_I2S_DIN_GPIO   11
#define BOARD_I2S_DOUT_GPIO  9
#define BOARD_PA_GPIO        53
#define BOARD_I2C_SDA_GPIO   7
#define BOARD_I2C_SCL_GPIO   8
#define BOARD_ES8311_ADDR    0x18

#define BUTTON_GPIO          35
#define LED_GPIO             23

/* Digital gain after the codec's +30 dB analog PGA, in 6 dB steps (0..7).
 * Clipping hurts recognition more than a low level does, because the engine
 * normalises the clip anyway: tune it with esp32/mic_test, keeping speech
 * peaks well under 0 dBFS. */
#define BOARD_MIC_GAIN_6DB   2

/* Configure the ES8311 for its microphone and start an I2S master
 * channel reading it. Samples come in 32-bit slots, the 16-bit PCM value in
 * the top half, the same layout an INMP441 gives. The codec's mono ADC is in
 * the left slot; `stereo` also captures the right one. */
void board_mic_init(uint32_t sample_rate, uint32_t frame_num, int stereo,
                    i2s_chan_handle_t *rx);

/* The BOOT key as an input with its pull-up. */
void board_button_init(void);

#endif
