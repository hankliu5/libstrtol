cmake_minimum_required(VERSION 3.5)
project(libstrtol C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS -Wall -O3)
add_executable(atoi_compare main.c libstrtol.c)
TARGET_LINK_LIBRARIES(atoi_compare KMCUDA)
target_compile_options (atoi_compare PRIVATE ${CMAKE_CXX_FLAGS})
