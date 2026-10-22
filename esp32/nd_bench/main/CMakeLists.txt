idf_component_register(SRCS "main.c"
                       REQUIRES needle esp_partition esp_timer esp_hw_support esp_mm)

# The decoder-step breakdown reads the engine's per-phase counters.
idf_component_get_property(nd_lib needle COMPONENT_LIB)
target_compile_definitions(${nd_lib} PUBLIC ND_PROFILE)
