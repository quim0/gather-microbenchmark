#include <string>
#include <vector>
#include <cinttypes>

typedef enum {
    L1_SIZE, 
    L2_SIZE, 
    L3_SIZE, 
    MM_SIZE
} memory_size_t; 


typedef enum {
    RANDOM,
    STRIDE,
    STRIDE_2EQUAL,
    STRIDE_4EQUAL,
    ALL_SAME,
    LOAD,
    SCALAR_RANDOM,
    SCALAR_RANDOM_NOSPILLING,
    ALL
} bench_algo_t; 


typedef enum {
    INT32_DATA, 
    INT64_DATA
} data_bytes_t;


typedef enum {
    REG_128BIT, 
    REG_256BIT,
    REG_512BIT
} simd_size_t;


typedef struct {
    memory_size_t mm_type; 
    bench_algo_t  bench_algo; 
    data_bytes_t  data_type;
    simd_size_t   simd_type; 
    int64_t       iters;
    int           stride; 
} bench_params_t; 


const std::vector<std::string> benchmark_names = {
    "RANDOM", "STRIDE", "STRIDE_2EQUAL", "STRIDE_4EQUAL", "ALL_SAME", "LOAD", "SCALAR_RANDOM", "SCALAR_RANDOM_NOSPILLING"
};


bench_params_t bench_default_params(void)
{
    return (bench_params_t){
        .mm_type     = L1_SIZE,
        .bench_algo  = ALL_SAME,
        .data_type   = INT32_DATA,
        .simd_type   = REG_128BIT,
        .iters       = 750000,
        .stride      = 1, 
    };
}