#ifndef AVX2_GATHER_BENCH_H_
#define AVX2_GATHER_BENCH_H_

#if __AVX2__
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include "kernels_common.h"

__m256i _mm256_set_epi32_benchmark(const int32_t* data, int start, int benchmark, int stride)
{
    int start8 = start << 3; 
    switch(benchmark)
    {
        case /*RANDOM*/ 0:        return _mm256_lddqu_si256((__m256i*)&data[start8]); 
        case /*STRIDE*/ 1:        return _mm256_set_epi32(
                                        stride*7+start8, stride*6+start8, stride*5+start8, stride*4+start8, 
                                        stride*3+start8, stride*2+start8, stride+start8,   start8
                                  );
        case /*STRIDE_2EQUAL*/ 2: return _mm256_set_epi32(start8+3, start8+3, start8+2, start8+2, start8+1, start8+1, start8, start8);
        case /*STRIDE_4EQUAL*/ 3: return _mm256_set_epi32(start8+1, start8+1, start8+1, start8+1, start8, start8, start8, start8);
        case /*ALL_SAME*/ 4:      return _mm256_set1_epi32(start);
        default:                  return _mm256_lddqu_si256((__m256i*)&data[start8]);
    }
}

void _mm256_loadu_kernel_throughput (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i random_simd1,  random_simd2;
    __m256i random_simd3,  random_simd4;
    __m256i random_simd5,  random_simd6;
    __m256i random_simd7,  random_simd8;
    __m256i random_simd9,  random_simd10;
    __m256i random_simd11, random_simd12;
    __m256i random_simd13;
    int32_t index1  =  0, index2  =  8; 
    int32_t index3  = 16, index4  = 24; 
    int32_t index5  = 32, index6  = 40; 
    int32_t index7  = 48, index8  = 56; 
    int32_t index9  = 64, index10 = 72; 
    int32_t index11 = 80, index12 = 88; 
    int32_t index13 = 96; 
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm256_lddqu_si256((__m256i*)&data[index1]);
        random_simd2  = _mm256_lddqu_si256((__m256i*)&data[index2]);
        random_simd3  = _mm256_lddqu_si256((__m256i*)&data[index3]);
        random_simd4  = _mm256_lddqu_si256((__m256i*)&data[index4]);
        random_simd5  = _mm256_lddqu_si256((__m256i*)&data[index5]);
        random_simd6  = _mm256_lddqu_si256((__m256i*)&data[index6]);
        random_simd7  = _mm256_lddqu_si256((__m256i*)&data[index7]);
        random_simd8  = _mm256_lddqu_si256((__m256i*)&data[index8]);
        random_simd9  = _mm256_lddqu_si256((__m256i*)&data[index9]);
        random_simd10 = _mm256_lddqu_si256((__m256i*)&data[index10]);
        random_simd11 = _mm256_lddqu_si256((__m256i*)&data[index11]);
        random_simd12 = _mm256_lddqu_si256((__m256i*)&data[index12]);
        random_simd13 = _mm256_lddqu_si256((__m256i*)&data[index13]);
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

void _mm256_gather32_kernel_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    __m256i random_simd1  = _mm256_set_epi32_benchmark(data,  0, benchmark, stride);
    __m256i random_simd2  = _mm256_set_epi32_benchmark(data,  1, benchmark, stride);
    __m256i random_simd3  = _mm256_set_epi32_benchmark(data,  2, benchmark, stride);
    __m256i random_simd4  = _mm256_set_epi32_benchmark(data,  3, benchmark, stride);
    __m256i random_simd5  = _mm256_set_epi32_benchmark(data,  4, benchmark, stride);
    __m256i random_simd6  = _mm256_set_epi32_benchmark(data,  5, benchmark, stride);
    __m256i random_simd7  = _mm256_set_epi32_benchmark(data,  6, benchmark, stride);
    __m256i random_simd8  = _mm256_set_epi32_benchmark(data,  7, benchmark, stride);
    __m256i random_simd9  = _mm256_set_epi32_benchmark(data,  8, benchmark, stride);
    __m256i random_simd10 = _mm256_set_epi32_benchmark(data,  9, benchmark, stride);
    __m256i random_simd11 = _mm256_set_epi32_benchmark(data, 10, benchmark, stride);
    __m256i random_simd12 = _mm256_set_epi32_benchmark(data, 11, benchmark, stride);
    __m256i random_simd13 = _mm256_set_epi32_benchmark(data, 12, benchmark, stride);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
        random_simd9  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd9, sizeof(int));
        random_simd10 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd10, sizeof(int));
        random_simd11 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd11, sizeof(int));
        random_simd12 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd12, sizeof(int));
        random_simd13 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd13, sizeof(int));
    }
    do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
}

void _mm256_gather32_kernel_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    __m256i random_simd1  = _mm256_set_epi32_benchmark(data,  0, benchmark, stride);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
    }
    do_not_optimize(random_simd1);
}

void _mm256_loadu_kernel_latency (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i random_simd1;
    int32_t index1 = 0;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_lddqu_si256((__m256i*)&data[index1]);
        index1 = data[index1]; 
        do_not_optimize(random_simd1);
    }
    unused(stride); 
    do_not_optimize(random_simd1);
}

void avx2_gather32_bench_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) _mm256_loadu_kernel_throughput(data, data_size, stride);
    else                _mm256_gather32_kernel_throughput(data, data_size, benchmark, stride);
}

void avx2_gather32_bench_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) _mm256_loadu_kernel_latency(data, data_size, stride);
    else                _mm256_gather32_kernel_latency(data, data_size, benchmark, stride);
}
#endif 
#endif