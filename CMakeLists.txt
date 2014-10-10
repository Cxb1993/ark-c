cmake_minimum_required(VERSION 2.8.4)
project(ark_c)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

set(SOURCE_FILES ark.c headers/bulk.h headers/forces.h headers/stress.h headers/constants.h)

add_executable(ark_c ${SOURCE_FILES})