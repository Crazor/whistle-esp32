# Pins and the onboard ES8311 microphone of the Waveshare ESP32-P4-WIFI6,
# shared by whistle_demo and mic_test.
idf_component_register(SRCS "board.c"
                       INCLUDE_DIRS "include"
                       REQUIRES esp_driver_i2s esp_driver_i2c esp_driver_gpio)
