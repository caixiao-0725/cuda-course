cmake_minimum_required(VERSION 3.0)

cuda_add_executable(pinned_zero main.cu)
target_link_libraries(pinned_zero ${CUDA_LIBRARIES})