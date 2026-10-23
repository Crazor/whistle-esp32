cmake_minimum_required(VERSION 3.16)
# The board component holds the ESP32-P4-WIFI6's pins, codec and speaker.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tts_demo)
