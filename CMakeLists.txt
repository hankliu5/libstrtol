cmake_minimum_required(VERSION 3.5)
project(libstrtol C)

set(CMAKE_C_STANDARD 11)

add_executable(atoi_compare main.c libstrtol.c)