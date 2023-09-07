cmake_minimum_required(VERSION 3.0)

cuda_add_executable(shuffle shuffle.cu)
target_link_libraries(shuffle ${CUDA_LIBRARIES})

cuda_add_executable(shuffle_sync shuffle_sync.cu)
