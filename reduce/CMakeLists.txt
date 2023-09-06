cmake_minimum_required(VERSION 3.0)

cuda_add_executable(reduce main.cu)
target_link_libraries(reduce ${CUDA_LIBRARIES})