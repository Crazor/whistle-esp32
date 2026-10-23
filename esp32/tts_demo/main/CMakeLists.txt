idf_component_register(SRCS "main.c"
                       REQUIRES picotts board esp_driver_i2s esp_driver_i2c esp_driver_gpio
                                esp_driver_uart esp_driver_usb_serial_jtag fatfs vfs
                                esp_timer)

# The lingware (tools/pico_lang.py -> model/pico) goes into the "lang" FAT
# partition. `idf.py lang-flash` writes it; it only needs doing once, like the
# Whistle model, so a plain `idf.py flash` leaves it alone.
set(LANG_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../model/pico")
if(EXISTS "${LANG_DIR}")
    fatfs_create_rawflash_image(lang "${LANG_DIR}")
else()
    message(WARNING "${LANG_DIR} is missing: run tools/pico_lang.py to build the lang partition")
endif()
