idf_component_register(SRCS "main.c"
                       REQUIRES board esp_driver_i2s esp_driver_rmt esp_driver_gpio
                                esp_timer)
