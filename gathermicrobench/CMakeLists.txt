set(gathermicrobench_SOURCE
    microbenchmark.cpp
)

add_executable(gathermicrobench ${gathermicrobench_SOURCE})

target_include_directories(gathermicrobench PRIVATE ${CMAKE_SOURCE_DIR}/external/perf-cpp/include)
target_include_directories(gathermicrobench PRIVATE ${CMAKE_SOURCE_DIR}/)
target_include_directories(gathermicrobench PRIVATE ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(gathermicrobench PRIVATE perf-cpp)

target_compile_options(gathermicrobench PRIVATE -march=native)

set_target_properties(gathermicrobench
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)