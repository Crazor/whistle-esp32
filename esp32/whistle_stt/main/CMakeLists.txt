idf_component_register(SRCS "main.c"
                       REQUIRES needle board esp_driver_i2s esp_driver_gpio esp_driver_uart
                                esp_partition esp_timer)

# Per-phase timings (EVT prof) are part of what this app measures.
idf_component_get_property(nd_lib needle COMPONENT_LIB)
target_compile_definitions(${nd_lib} PUBLIC ND_PROFILE)
