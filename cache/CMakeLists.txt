cmake_minimum_required(VERSION 3.0)

cuda_add_executable(cache main.cu)
target_link_libraries(cache ${CUDA_LIBRARIES})