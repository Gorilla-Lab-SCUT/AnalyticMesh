cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(polylib)

find_package(Python3 3.6 REQUIRED Interpreter Development)
find_package(pybind11 REQUIRED)

set(SOURCES "src/polylib.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${pybind11_INCLUDE_DIRS})
