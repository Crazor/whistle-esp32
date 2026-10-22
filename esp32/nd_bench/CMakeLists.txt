cmake_minimum_required(VERSION 3.16)
# The engine component lives with the Needle demo, as for whistle_demo.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../needle_demo/components")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(nd_bench)
