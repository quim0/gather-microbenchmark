CC=g++
SOURCES=gathermicrobench/include/cxxopts.hpp  gathermicrobench/kernels/sse.h gathermicrobench/kernels/avx512.h gathermicrobench/kernels/avx2.h gathermicrobench/kernels/scalar.h gathermicrobench/benchmark_params.hpp gathermicrobench/generate_data.hpp gathermicrobench/run_benchmark.hpp gathermicrobench/microbenchmark.cpp gathermicrobench/include/libperf-cpp.a
CFLAGS=-Wall -Wextra -I. -march=native  -std=c++17

all:
	$(CC) $(CFLAGS) -O3 $(SOURCES) -o microbenchmark

fast: 
	$(CC) $(CFLAGS) -Ofast $(SOURCES) -o microbenchmark
	
debug:
	$(CC) $(CFLAGS) -g $(SOURCES) -o microbenchmark

clean:
	rm -f microbenchmark
