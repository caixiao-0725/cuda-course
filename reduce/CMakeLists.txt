cmake_minimum_required(VERSION 3.0)

cuda_add_executable(CUDA_EQUATION main.cu)
target_link_libraries(CUDA_EQUATION ${CUDA_LIBRARIES})