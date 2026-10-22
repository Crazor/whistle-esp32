cmake_minimum_required(VERSION 3.16)
# The engine component lives with the Needle demo; both apps build it. The
# board component holds the ESP32-P4-WIFI6's pins and its codec microphone.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../needle_demo/components"
                         "${CMAKE_CURRENT_LIST_DIR}/../components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(whistle_demo)
