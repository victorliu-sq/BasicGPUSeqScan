cmake_minimum_required(VERSION 3.29)
project(BasicGPUSeqScan CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(BasicGPUSeqScan main.cu)

set_target_properties(BasicGPUSeqScan PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
