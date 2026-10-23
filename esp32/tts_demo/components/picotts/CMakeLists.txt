# SVOX Pico (../../../../third_party/picotts) and its wrapper (../../../../tts),
# built as an ESP-IDF component. Other apps can use it by adding this
# directory's parent to EXTRA_COMPONENT_DIRS.
set(REPO "${CMAKE_CURRENT_LIST_DIR}/../../../..")
file(GLOB PICO_SRC "${REPO}/third_party/picotts/*.c")

idf_component_register(SRCS ${PICO_SRC} "${REPO}/tts/src/nd_tts.c"
                       INCLUDE_DIRS "${REPO}/tts/include"
                       PRIV_INCLUDE_DIRS "${REPO}/third_party/picotts")

# Pico is 2009 code; its own warnings are not ours to fix.
set_source_files_properties(${PICO_SRC} PROPERTIES COMPILE_OPTIONS "-w")
