#include <iostream>
#include <functional>

#include "include/perfcpp/event_counter.h"
#include "generate_data.hpp"
#include "kernels/sse.h"
#include "kernels/avx512.h"
#include "kernels/avx2.h"
#include "kernels/scalar.h"


std::function<void(int32_t*, uint64_t, int)> select_simd_benchmark_32bits(bench_params_t params)
{
  
    switch (params.bench_algo) 
    {
        case RANDOM:   
        if (params.simd_type == REG_512BIT) return avx2_gather32_kernel;
        if (params.simd_type == REG_256BIT) return avx2_gather32_kernel;
        else                                return sse_gather32_kernel;
        case STRIDE: 
        if (params.simd_type == REG_512BIT) return avx2_gather32_stride_kernel;
        if (params.simd_type == REG_256BIT) return avx2_gather32_stride_kernel;
        else                                return sse_gather32_stride_kernel;
        case STRIDE_2EQUAL:    
        if (params.simd_type == REG_512BIT) return avx2_gather32_stride_2equal_kernel;
        if (params.simd_type == REG_256BIT) return avx2_gather32_stride_2equal_kernel;
        else                                return sse_gather32_stride_2equal_kernel;
        case STRIDE_4EQUAL:    
        if (params.simd_type == REG_512BIT) return avx2_gather32_stride_4equal_kernel;
        if (params.simd_type == REG_256BIT) return avx2_gather32_stride_4equal_kernel;
        else                                return sse_gather32_all_same_kernel;
        case ALL_SAME:         
        if (params.simd_type == REG_512BIT) return avx2_gather32_all_same_kernel;
        if (params.simd_type == REG_256BIT) return avx2_gather32_all_same_kernel;
        else                                return sse_gather32_all_same_kernel;
        case LOAD:             
        if (params.simd_type == REG_512BIT) return avx2_256_loadu;
        if (params.simd_type == REG_256BIT) return avx2_256_loadu;
        else                                return sse_128_loadu_32;
        case SCALAR_RANDOM:                 return scalar_gather32_kernel;
        default:                            return sse_gather32_kernel;
    }
}

std::vector<std::function<void(int32_t*, uint64_t, int)>> select_simd_all_benchmark_32bits(bench_params_t params)
{
    std::vector<std::function<void(int32_t*, uint64_t, int)>> benchmarks_r128 = { 
        sse_gather32_kernel, sse_gather32_stride_kernel, sse_gather32_stride_2equal_kernel, 
        sse_gather32_all_same_kernel, sse_gather32_all_same_kernel, sse_128_loadu_32, scalar_gather32_kernel
    }; 

    std::vector<std::function<void(int32_t*, uint64_t, int)>> benchmarks_r256 = { 
        avx2_gather32_kernel, avx2_gather32_stride_kernel, avx2_gather32_stride_2equal_kernel, 
        avx2_gather32_stride_4equal_kernel, avx2_gather32_all_same_kernel, avx2_256_loadu, scalar_gather32_kernel
    }; 

    std::vector<std::function<void(int32_t*, uint64_t, int)>> benchmarks_r512 = { 
        avx2_gather32_kernel, avx2_gather32_stride_kernel, avx2_gather32_stride_2equal_kernel, 
        avx2_gather32_stride_4equal_kernel, avx2_gather32_all_same_kernel, avx2_256_loadu, scalar_gather32_kernel
    }; 
    
    switch (params.simd_type)
    {
        case REG_128BIT: return benchmarks_r128; 
        case REG_256BIT: return benchmarks_r256;
        case REG_512BIT: return benchmarks_r512;
        default:         return benchmarks_r128;
    }
}


void benchmark_run_64bits(bench_params_t params)
{
    const int64_t     iters = params.iters; 
    const bench_algo_t algo = params.bench_algo; 
    const int64_t data_size = get_data_size(params); 

    if (algo == ALL)
    {
        std::vector<std::function<void(int32_t*, uint64_t, int)>> all_benchs = select_simd_all_benchmark_32bits(params);
        int index = 0; 
        
        for (auto benchmark_gather : all_benchs)
        {
            int32_t* data     = new int32_t[data_size];
            init_data_32bits(data, data_size, params);
            
            auto counter_definitions = perf::CounterDefinition{};
            auto event_counter       = perf::EventCounter{counter_definitions};
            if (!event_counter.add({ "instructions",
                                    "cycles",
                                    "branches",
                                    "cycles-per-instruction" })) {
                std::cerr << "Could not add performance counters." << std::endl;
            }
            
            event_counter.start();
            for (int i = 0; i < iters; i++)
            {
                benchmark_gather(data, data_size, params.stride); 
            }
            event_counter.stop();

            const auto result = event_counter.result();
            std::cout << "\nRunning benchmark: " << benchmark_names[index] << std::endl; 
            for (const auto& [counter_name, counter_value] : result) {
                std::cout << counter_value << " " << counter_name << std::endl;
            }
            index++;
            delete data;
        }
    }
    else 
    {
        int32_t* data     = new int32_t[data_size];
        init_data_32bits(data, data_size, params);

        std::function<void(int32_t*, uint64_t, int)> benchmark_gather = select_simd_benchmark_32bits(params);    
        
        auto counter_definitions = perf::CounterDefinition{};
        auto event_counter       = perf::EventCounter{counter_definitions};
        if (!event_counter.add({ "instructions",
                                "cycles",
                                "branches",
                                "cycles-per-instruction" })) {
            std::cerr << "Could not add performance counters." << std::endl;
        }

        event_counter.start();
        for (int i = 0; i < iters; i++)
        {
            benchmark_gather(data, data_size, params.stride); 
        }
        event_counter.stop();

        const auto result = event_counter.result();
        for (const auto& [counter_name, counter_value] : result) {
            std::cout << counter_value << " " << counter_name << std::endl;
        }
        delete data;
    }
}

void benchmark_run_32bits(bench_params_t params)
{
    const int64_t     iters = params.iters; 
    const bench_algo_t algo = params.bench_algo; 
    const int64_t data_size = get_data_size(params); 

    if (algo == ALL)
    {
        std::vector<std::function<void(int32_t*, uint64_t, int)>> all_benchs = select_simd_all_benchmark_32bits(params);
        int index = 0; 
        
        for (auto benchmark_gather : all_benchs)
        {
            int32_t* data     = new int32_t[data_size];
            init_data_32bits(data, data_size, params);
            
            auto counter_definitions = perf::CounterDefinition{};
            auto event_counter       = perf::EventCounter{counter_definitions};
            if (!event_counter.add({ "instructions",
                                    "cycles",
                                    "branches",
                                    "cycles-per-instruction" })) {
                std::cerr << "Could not add performance counters." << std::endl;
            }
            
            event_counter.start();
            for (int i = 0; i < iters; i++)
            {
                benchmark_gather(data, data_size, params.stride); 
            }
            event_counter.stop();

            const auto result = event_counter.result();
            std::cout << "\nRunning benchmark: " << benchmark_names[index] << std::endl; 
            for (const auto& [counter_name, counter_value] : result) {
                std::cout << counter_value << " " << counter_name << std::endl;
            }
            index++;
            delete data;
        }
    }
    else 
    {
        int32_t* data     = new int32_t[data_size];
        init_data_32bits(data, data_size, params);

        std::function<void(int32_t*, uint64_t, int)> benchmark_gather = select_simd_benchmark_32bits(params);    
        
        auto counter_definitions = perf::CounterDefinition{};
        auto event_counter       = perf::EventCounter{counter_definitions};
        if (!event_counter.add({ "instructions",
                                "cycles",
                                "branches",
                                "cycles-per-instruction" })) {
            std::cerr << "Could not add performance counters." << std::endl;
        }

        event_counter.start();
        for (int i = 0; i < iters; i++)
        {
            benchmark_gather(data, data_size, params.stride); 
        }
        event_counter.stop();

        const auto result = event_counter.result();
        for (const auto& [counter_name, counter_value] : result) {
            std::cout << counter_value << " " << counter_name << std::endl;
        }
        delete data;
    }
}

void benchmark_run(bench_params_t params)
{
    if (params.data_type == INT32_DATA) benchmark_run_32bits(params);
    else                                benchmark_run_64bits(params);
}
