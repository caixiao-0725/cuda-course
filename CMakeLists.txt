cmake_minimum_required(VERSION 3.0)
project(CUDA_COURSE)

find_package(CUDA REQUIRED)

message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS ${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(COMMON_ROOT ${CMAKE_SOURCE_DIR}/common)
include_directories(${COMMON_ROOT})

add_subdirectory(reduce)
add_subdirectory(pinned_zero)
add_subdirectory(cache)
add_subdirectory(shuffle)