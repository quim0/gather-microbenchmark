#include <iostream>
#include <perfcpp/event_counter.h>
#include <gathermicrobench/generate_data.hpp>
#include <gathermicrobench/kernels/scalar.h>

#if __AVX2__
    #include <gathermicrobench/kernels/sse.h>
    #include <gathermicrobench/kernels/avx2.h>
    #if __AVX512F__ && __AVX512VL__
        #include <gathermicrobench/kernels/avx512.h>
    #endif
#else
    #ifdef __ARM_FEATURE_SVE
        #include <gathermicrobench/kernels/sve.h>
    #endif
#endif

void benchmark_32bits(const int32_t* data, int64_t data_size, bench_params_t params)
{
    int32_t benchmark = (int32_t) params.bench_algo;
    int32_t stride    = params.stride;
    #if __AVX2__
        switch (params.simd_type)
        {
            case SCALAR: 
                if (params.bench_mode == THROUGHPUT) scalar_kernel_throughput(data, data_size, benchmark, stride); 
                else                                 scalar_kernel_latency(data, data_size, benchmark, stride);
                return; 
            case REG_128BIT: 
                if (params.bench_mode == THROUGHPUT) sse_gather32_bench_throughput(data, data_size, benchmark, stride); 
                else                                 sse_gather32_bench_latency(data, data_size, benchmark, stride);
                return; 
            case REG_256BIT:   
                if (params.bench_mode == THROUGHPUT) avx2_gather32_bench_throughput(data, data_size, benchmark, stride); 
                else                                 avx2_gather32_bench_latency(data, data_size, benchmark, stride);
                return; 
            case REG_512BIT: 
                #if __AVX512F__ && __AVX512VL__
                    if (params.bench_mode == THROUGHPUT) avx512_gather32_bench_throughput(data, data_size, benchmark, stride); 
                    else                                 avx512_gather32_bench_latency(data, data_size, benchmark, stride);
                    return; 
                #else
                    #pragma message("AVX512 missing using AVX2 instead")
                    if (params.bench_mode == THROUGHPUT) avx2_gather32_bench_throughput(data, data_size, benchmark, stride); 
                    else                                 avx2_gather32_bench_latency(data, data_size, benchmark, stride);
                    return; 
                #endif
            default: 
                if (params.bench_mode == THROUGHPUT) sse_gather32_bench_throughput(data, data_size, benchmark, stride); 
                else                                 sse_gather32_bench_latency(data, data_size, benchmark, stride);
                return; 
        }
    #else
        #ifdef __ARM_FEATURE_SVE
            if (params.simd_type == SCALAR)
            {
                if (params.bench_mode == THROUGHPUT) scalar_kernel_throughput(data, data_size, benchmark, stride); 
                else                                 scalar_kernel_latency(data, data_size, benchmark, stride);
                return; 
            }
            else
            {
                if (params.bench_mode == THROUGHPUT) sve_gather32_bench_throughput(data, data_size, benchmark, stride);
                else                                 sve_gather32_bench_latency(data, data_size, benchmark, stride);
            } 
            return;
        #endif
    #endif
}

void* allocate_array(bench_params_t params, const int64_t data_size)
{
    if (params.is_aligned == ALIGNED) return std::aligned_alloc(32, data_size * sizeof(int32_t));
    void* raw_data = std::malloc(data_size * sizeof(int32_t));
    return static_cast<void*>(static_cast<char*>(raw_data) + 1);
}

void free_array(void* data, bench_params_t params, const int64_t data_size)
{
    if (params.is_aligned == ALIGNED)
    {
        std::free(data);
        return;
    }
    void* raw_data = static_cast<void*>(static_cast<char*>(data) - 1);
    std::free(raw_data);
}

void benchmark_run_32bits(bench_params_t params)
{
    const bench_algo_t algo   = params.bench_algo; 
    const int64_t     iters   = params.iters;
    const int64_t data_size   = get_data_size(params); 
    const int64_t total_iters = iters * data_size; 
    #ifdef __ARM_FEATURE_SVE
        size_t num_lanes = svcntw();
        params.simd_type = (params.simd_type == SCALAR) ? SCALAR     : 
                           ( num_lanes == 4 )           ? REG_128BIT : 
                           ( num_lanes == 8 )           ? REG_256BIT : REG_512BIT;
    #endif
    auto config = perf::Config{};
    config.max_groups(6U);
    config.max_counters_per_group(1U);

    auto counter_definitions = perf::CounterDefinition{};
    auto event_counter       = perf::EventCounter{counter_definitions, config};
    if (!event_counter.add({ "instructions",
                              "cycles",
                              "cache-misses",
                              "cache-references",
                              "L1-dcache-load-misses",
                              "L1-dcache-loads"}, perf::EventCounter::Schedule::Separate)) {
        std::cerr << "Could not add performance counters." << std::endl;
    }

    if (algo == ALL)
    {
        for(auto bench: benchmark_v) 
        {
            params.bench_algo = bench; 
            int32_t* data     = static_cast<int32_t*>(allocate_array(params, data_size));
            init_data_32bits(data, data_size, params);
            
            event_counter.start();
            benchmark_32bits(data, total_iters, params);
            event_counter.stop();
            
            std::cout << "Results for " << benchmark_names[bench] << "\n";
            const auto result = event_counter.result();
            for (const auto& [counter_name, counter_value] : result) {
                std::cout << counter_value << " " << counter_name << std::endl;
            }
            std::cout << std::endl;
            free_array(data, params, data_size);
        }
        params.bench_algo = ALL; 
    }
    else
    {
        int32_t* data     = static_cast<int32_t*>(allocate_array(params, data_size));
        init_data_32bits(data, data_size, params);

        event_counter.start();
        benchmark_32bits(data, total_iters, params);
        event_counter.stop();

        std::cout << "Results for " << benchmark_names[params.bench_algo] << "\n";
        const auto result = event_counter.result();
        for (const auto& [counter_name, counter_value] : result) {
            std::cout << counter_value << " " << counter_name << std::endl;
        }
        std::cout << std::endl;
        free_array(data, params, data_size);
    }
}

void benchmark_run(bench_params_t params)
{
    #if __AVX2__
        std::cout << "Start running benchmark\n";
        if (params.data_type == INT32_DATA) benchmark_run_32bits(params);
        else                                benchmark_run_32bits(params);
    #else
        #if __ARM_FEATURE_SVE
                if (params.data_type == INT32_DATA) benchmark_run_32bits(params);
                else                                benchmark_run_32bits(params);
        #else
            std::cout << "NO AVX2, AVX512 or SVE support, couldnt run benchmark\n";
        #endif
    #endif
}
