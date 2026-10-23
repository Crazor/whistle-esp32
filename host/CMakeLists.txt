cmake_minimum_required(VERSION 3.16)
project(needle_host C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(ENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../engine)

file(GLOB ENGINE_SRC ${ENGINE_DIR}/src/*.c)

add_library(needle_engine STATIC ${ENGINE_SRC})
target_include_directories(needle_engine PUBLIC ${ENGINE_DIR}/include)
target_link_libraries(needle_engine PUBLIC m)

add_executable(nd_dump nd_dump.c)
target_link_libraries(nd_dump PRIVATE needle_engine)

add_executable(nd_gtest nd_gtest.c)
target_link_libraries(nd_gtest PRIVATE needle_engine)

add_executable(wh_run wh_run.c)
target_link_libraries(wh_run PRIVATE needle_engine)

add_executable(ledcmd_test ledcmd_test.c)
target_link_libraries(ledcmd_test PRIVATE needle_engine)

# Text to speech: SVOX Pico (third_party/picotts) and its wrapper (tts/).
# Pico is 2009 code; its own warnings are not ours to fix.
set(PICO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/picotts)
set(TTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tts)
file(GLOB PICO_SRC ${PICO_DIR}/*.c)

add_library(picotts STATIC ${PICO_SRC})
target_include_directories(picotts PUBLIC ${PICO_DIR})
target_compile_options(picotts PRIVATE -w)
target_link_libraries(picotts PUBLIC m)

add_library(nd_tts STATIC ${TTS_DIR}/src/nd_tts.c ${TTS_DIR}/src/nd_comp.c)
target_include_directories(nd_tts PUBLIC ${TTS_DIR}/include)
target_link_libraries(nd_tts PUBLIC picotts)

add_executable(tts_run tts_run.c)
target_link_libraries(tts_run PRIVATE nd_tts)
