#include "include/cxxopts.hpp"
#include "run_benchmark.hpp"

int string_to_value(std::string value, std::string param)
{    
    if (param == "memory-size")
    {
        if      (value == "L1_SIZE") return L1_SIZE;
        else if (value == "L2_SIZE") return L2_SIZE;
        else if (value == "L3_SIZE") return L3_SIZE;
        else if (value == "MM_SIZE") return MM_SIZE;
        else
        {
           std::cout << "Unknown memory-size: Initializing the memory-size to L1_SIZE" << std::endl;
           return L1_SIZE;
        }
    } 

    if (param == "simd-size")
    {
        if      (value == "REG_128BIT") return REG_128BIT;
        else if (value == "REG_256BIT") return REG_256BIT;
        else if (value == "REG_512BIT") return REG_512BIT;
        else 
        {
           std::cout << "Unknown simd-size: Initializing the simd-size to REG_128BIT" << std::endl;
           return REG_128BIT;
        }
    } 

    if (param == "benchmark")
    {
        if      (value == "RANDOM")           return RANDOM;
        else if (value == "STRIDE")           return STRIDE;
        else if (value == "STRIDE_2EQUAL")    return STRIDE_2EQUAL; 
        else if (value == "STRIDE_4EQUAL")    return STRIDE_4EQUAL; 
        else if (value == "ALL_SAME")         return ALL_SAME; 
        else if (value == "ALL")              return ALL; 
        else if (value == "SCALAR_RANDOM")    return SCALAR_RANDOM; 
        else if (value == "LOAD")             return LOAD; 
        else 
        {
           std::cout << "Unknown benchmark: Initializing the memory-size to RANDOM" << std::endl;
           return RANDOM;
        }
    } 

    if (param == "data-type")
    {
        if      (value == "INT32_DATA") return INT32_DATA;
        else if (value == "INT64_DATA") return INT64_DATA;
        else 
        {
           std::cout << "Unknown data-type: Initializing the memory-size to INT32_DATA" << std::endl;
           return INT32_DATA;
        }
    } 

    return 0;
}



void parse_arguments_benchmark(int argc, char** argv, bench_params_t* params)
{
    if (argc <= 1) return;

    cxxopts::Options options("microbenchmark", "Performance tests on the SIMD gather instruction. This microbenchmark has different parameters that allow us to determine the behavior of the gather and measure its performance. The parameters allow to choose, the type of memory access pattern, where the data will be stored (L1 cache, L2, RAM,... ), the type of vector instruction (128, 256 or 512 bits) and the type of data (32 or 64 bits).");

    options.add_options()
        ("i,iterations", "Iterations to be executed", cxxopts::value<int64_t>())
        ("m,memory-size", "Memory Size (L1_SIZE, L2_SIZE, L3_SIZE, MM_SIZE)", cxxopts::value<std::string>())
        ("s,simd-size", "SIMD register size (REG_128BIT, REG_256BIT, REG_512BIT)", cxxopts::value<std::string>())
        ("b,benchmark", "Benchmark (RANDOM, STRIDE, ALL, ...)", cxxopts::value<std::string>())
        ("d,data-type", "Data type (INT32_DATA, INT64_DATA)", cxxopts::value<std::string>())
        ("S,stride", "Stride (1, 2 or 4): If benchmark is not stride this field is ignored", cxxopts::value<int>())
        ("h,help", "Print usage")
    ;

    try
    {
        auto result = options.parse(argc, argv);
    
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(EXIT_SUCCESS);
        }
        
        if (result.count("iterations"))
        {
            params->iters = result["iterations"].as<int64_t>();
        }

        if (result.count("memory-size"))
        {
            std::string mm_size = result["memory-size"].as<std::string>(); 
            params->mm_type = (memory_size_t)string_to_value(mm_size, "memory-size");
        }

        if (result.count("simd-size"))
        {
            std::string simd_type = result["simd-size"].as<std::string>(); 
            params->simd_type = (simd_size_t)string_to_value(simd_type, "simd-size");
        }

        if (result.count("benchmark"))
        {
            std::string bench = result["benchmark"].as<std::string>(); 
            params->bench_algo = (bench_algo_t)string_to_value(bench, "benchmark");
        }

        if (result.count("data-type"))
        {
            std::string data_type = result["data-type"].as<std::string>(); 
            params->data_type = (data_bytes_t)string_to_value(data_type, "data-type");
        }

        if (result.count("stride"))
        {
            int stride = result["stride"].as<int>();
            if (!(stride == 4 || stride == 2 || stride == 1))
            {
                std::cout << "stride set to 1 because an attempt has been made to initialize with a value other than 1, 2 or 4." << std::endl;
            }
            stride = (stride == 4) ? 4 : (stride == 2) ? 2 : 1; 
            params->stride = stride;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    bench_params_t params = bench_default_params(); 
    parse_arguments_benchmark(argc, argv, &params);
    benchmark_run(params);
    return 0;
}
