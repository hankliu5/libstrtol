cmake_minimum_required(VERSION 3.5)
project(libstrtol C)

set(CMAKE_C_STANDARD 11)

add_library(libstrtol SHARED libstrtol.c libstrtol.h)
add_executable(atoi_compare main.c libstrtol.c)