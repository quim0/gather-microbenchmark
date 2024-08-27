cmake_minimum_required(VERSION 3.16)

project(
    gathermicrobench
    VERSION 0.1
    LANGUAGES CXX
    )

set(CMAKE_CXX_STANDARD 17)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_subdirectory(${CMAKE_SOURCE_DIR}/external/perf-cpp)
add_subdirectory(${PROJECT_NAME})