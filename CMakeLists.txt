cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 23)

add_executable(NeuralNetwork main.c matrix.c matrix.h actvfunc.c actvfunc.h nnetwork.c nnetwork.h mlearning.c mlearning.h)
