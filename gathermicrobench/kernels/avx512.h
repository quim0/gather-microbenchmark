#ifndef AVX512_GATHER_BENCH_H_
#define AVX512_GATHER_BENCH_H_

#if __AVX512F__ && __AVX512VL__
#include <stdint.h>
#include <immintrin.h>
#include "kernels_common.h"

__m512i _mm512_set_epi32_benchmark(const int32_t* data, int start, int benchmark, int stride)
{
    int start16 = start << 4; 
    switch(benchmark)
    {
        case /*RANDOM*/ 0:        return _mm512_loadu_si512((__m512i*)&data[start16]); 
        case /*STRIDE*/ 1:        return _mm512_set_epi32(
                                        stride*15+start16, stride*14+start16, stride*13+start16, stride*12+start16, stride*11+start16, stride*10+start16, 
                                        stride*9+start16,  stride*8+start16,  stride*7+start16,  stride*6+start16,  stride*5+start16,  stride*4+start16, 
                                        stride*3+start16,  stride*2+start16,    stride+start16,  start16
                                    );
        case /*STRIDE_2EQUAL*/ 2: return _mm512_set_epi32(
                                        7+start16, 7+start16, 6+start16, 6+start16, 5+start16, 5+start16, 
                                        4+start16, 4+start16, 3+start16, 3+start16, 2+start16, 2+start16, 
                                        1+start16, 1+start16,   start16,   start16
                                    ); 
        case /*STRIDE_4EQUAL*/ 3: return _mm512_set_epi32(
                                        3+start16, 3+start16, 3+start16, 3+start16, 2+start16, 2+start16, 
                                        2+start16, 2+start16, 1+start16, 1+start16, 1+start16, 1+start16, 
                                          start16,   start16,   start16,   start16
                                    ); 
        case /*ALL_SAME*/ 4:      return _mm512_set1_epi32(start);
        default:                  return _mm512_loadu_si512((__m512i*)&data[start16]);
    }
}

void _mm512_loadu_kernel_throughput (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i random_simd1,  random_simd2;
    __m512i random_simd3,  random_simd4;
    __m512i random_simd5,  random_simd6;
    __m512i random_simd7,  random_simd8;
    __m512i random_simd9,  random_simd10;
    __m512i random_simd11, random_simd12;
    __m512i random_simd13;
    int32_t index1  =   0, index2  =  16; 
    int32_t index3  =  32, index4  =  48; 
    int32_t index5  =  64, index6  =  80; 
    int32_t index7  =  96, index8  = 112;
    int32_t index9  = 128, index10 = 144; 
    int32_t index11 = 160, index12 = 176; 
    int32_t index13 = 192;  
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm512_loadu_si512((__m512i*)&data[index1]);
        random_simd2  = _mm512_loadu_si512((__m512i*)&data[index2]);
        random_simd3  = _mm512_loadu_si512((__m512i*)&data[index3]);
        random_simd4  = _mm512_loadu_si512((__m512i*)&data[index4]);
        random_simd5  = _mm512_loadu_si512((__m512i*)&data[index5]);
        random_simd6  = _mm512_loadu_si512((__m512i*)&data[index6]);
        random_simd7  = _mm512_loadu_si512((__m512i*)&data[index7]);
        random_simd8  = _mm512_loadu_si512((__m512i*)&data[index8]);
        random_simd9  = _mm512_loadu_si512((__m512i*)&data[index9]);
        random_simd10 = _mm512_loadu_si512((__m512i*)&data[index10]);
        random_simd11 = _mm512_loadu_si512((__m512i*)&data[index11]);
        random_simd12 = _mm512_loadu_si512((__m512i*)&data[index12]);
        random_simd13 = _mm512_loadu_si512((__m512i*)&data[index13]);
        index1  = data[index1];  index2  = data[index2]; 
        index3  = data[index3];  index4  = data[index4];
        index5  = data[index5];  index6  = data[index6]; 
        index7  = data[index7];  index8  = data[index8];  
        index5  = data[index5];  index6  = data[index6]; 
        index7  = data[index7];  index8  = data[index8]; 
        index9  = data[index9];  index10 = data[index10];  
        index11 = data[index11]; index12 = data[index12]; 
        index13 = data[index13];
        do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
        do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
        do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
        do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
        do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
        do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
        do_not_optimize(random_simd13); 
    }
    unused(stride); 
    do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13); 
}

void _mm512_gather32_kernel_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    __m512i random_simd1  = _mm512_set_epi32_benchmark(data,  0, benchmark, stride);
    __m512i random_simd2  = _mm512_set_epi32_benchmark(data,  1, benchmark, stride);
    __m512i random_simd3  = _mm512_set_epi32_benchmark(data,  2, benchmark, stride);
    __m512i random_simd4  = _mm512_set_epi32_benchmark(data,  3, benchmark, stride);
    __m512i random_simd5  = _mm512_set_epi32_benchmark(data,  4, benchmark, stride);
    __m512i random_simd6  = _mm512_set_epi32_benchmark(data,  5, benchmark, stride);
    __m512i random_simd7  = _mm512_set_epi32_benchmark(data,  6, benchmark, stride);
    __m512i random_simd8  = _mm512_set_epi32_benchmark(data,  7, benchmark, stride);
    __m512i random_simd9  = _mm512_set_epi32_benchmark(data,  8, benchmark, stride);
    __m512i random_simd10 = _mm512_set_epi32_benchmark(data,  9, benchmark, stride);
    __m512i random_simd11 = _mm512_set_epi32_benchmark(data, 10, benchmark, stride);
    __m512i random_simd12 = _mm512_set_epi32_benchmark(data, 11, benchmark, stride);
    __m512i random_simd13 = _mm512_set_epi32_benchmark(data, 12, benchmark, stride);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm512_i32gather_epi32 (random_simd1,  (int const*)&data[0], sizeof(int));
        random_simd2  = _mm512_i32gather_epi32 (random_simd2,  (int const*)&data[0], sizeof(int));
        random_simd3  = _mm512_i32gather_epi32 (random_simd3,  (int const*)&data[0], sizeof(int));
        random_simd4  = _mm512_i32gather_epi32 (random_simd4,  (int const*)&data[0], sizeof(int));
        random_simd5  = _mm512_i32gather_epi32 (random_simd5,  (int const*)&data[0], sizeof(int));
        random_simd6  = _mm512_i32gather_epi32 (random_simd6,  (int const*)&data[0], sizeof(int));
        random_simd7  = _mm512_i32gather_epi32 (random_simd7,  (int const*)&data[0], sizeof(int));
        random_simd8  = _mm512_i32gather_epi32 (random_simd8,  (int const*)&data[0], sizeof(int));
        random_simd9  = _mm512_i32gather_epi32 (random_simd9,  (int const*)&data[0], sizeof(int));
        random_simd10 = _mm512_i32gather_epi32 (random_simd10, (int const*)&data[0], sizeof(int));
        random_simd11 = _mm512_i32gather_epi32 (random_simd11, (int const*)&data[0], sizeof(int));
        random_simd12 = _mm512_i32gather_epi32 (random_simd12, (int const*)&data[0], sizeof(int));
        random_simd13 = _mm512_i32gather_epi32 (random_simd13, (int const*)&data[0], sizeof(int));
    }
    do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13); 
}

void _mm512_gather32_kernel_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    __m512i random_simd1  = _mm512_set_epi32_benchmark(data,  0, benchmark, stride);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm512_i32gather_epi32 (random_simd1,  (int const*)&data[0], sizeof(int));
    }
    do_not_optimize(random_simd1);
}

void _mm512_loadu_kernel_latency (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i random_simd1;
    int32_t index1 =  0;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm512_loadu_si512((__m512i*)&data[index1]);
        index1  = data[index1];
        do_not_optimize(random_simd1);
    }
    unused(stride); 
    do_not_optimize(random_simd1);
}

void avx512_gather32_bench_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) _mm512_loadu_kernel_throughput(data, data_size, stride);
    else                _mm512_gather32_kernel_throughput(data, data_size, benchmark, stride);
}

void avx512_gather32_bench_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) _mm512_loadu_kernel_latency(data, data_size, stride);
    else                _mm512_gather32_kernel_latency(data, data_size, benchmark, stride);
}

#endif // __AVX512F__
#endif // AVX512_GATHER_BENCH_H_
