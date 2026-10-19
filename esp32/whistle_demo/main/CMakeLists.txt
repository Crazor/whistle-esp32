# Optional boot self-test: transcribe a built-in clip at power-on and print
# the timings, to check a new build without pressing anything. Off by
# default - it adds ~8 s to startup and flashes the LED. Enable with
#   idf.py -DWHISTLE_SELFTEST=1 build
# plus main/selftest.raw (16 kHz mono s16le; not committed, it is test audio).
set(ST "${CMAKE_CURRENT_LIST_DIR}/selftest.raw")
if(WHISTLE_SELFTEST AND EXISTS "${ST}")
    set(ST_EMBED EMBED_FILES "selftest.raw")
endif()

idf_component_register(SRCS "main.c"
                       ${ST_EMBED}
                       REQUIRES needle esp_driver_i2s esp_driver_rmt esp_driver_gpio
                                esp_partition esp_timer)
if(WHISTLE_SELFTEST AND EXISTS "${ST}")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE HAVE_SELFTEST)
endif()

# Per-phase timings (mel, stem, encoder, cross K/V, decoder) are part of what
# this demo prints, so profile the engine in this app. It is five timer reads
# per clip in nd_whistle.c; the Needle demo keeps it off.
idf_component_get_property(nd_lib needle COMPONENT_LIB)
target_compile_definitions(${nd_lib} PUBLIC ND_PROFILE)
