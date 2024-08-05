#ifndef AVX2_GATHER_BENCH_H_
#define AVX2_GATHER_BENCH_H_

#include <inttypes.h>
#include <stdlib.h>
#include <immintrin.h>

static __attribute__((always_inline)) void avx2_256_loadu (
    int const* data, 
    uint64_t data_size
    ) 
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+=8) 
    {
        res = _mm256_lddqu_si256((__m256i*)&data[i]);
    }
}

static __attribute__((always_inline)) void avx2_gather32_kernel (
    const int* const data,
    const uint64_t data_size
    ) 
{
    __m256i volatile random_simd1 = _mm256_lddqu_si256(&data[0]);
    __m256i volatile random_simd2 = _mm256_lddqu_si256(&data[8]);
    __m256i volatile random_simd3 = _mm256_lddqu_si256(&data[16]);
    __m256i volatile random_simd4 = _mm256_lddqu_si256(&data[24]);

    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}

static __attribute__((always_inline)) void avx2_gather32_stride_field_2 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        // TODO
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 2); 
    }
}


static __attribute__((always_inline)) void avx2_gather32_stride_field_4 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        // TODO
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 4); 
    }
}

#endif // AVX2_GATHER_BENCH_H_