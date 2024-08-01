#include "avx2_256_32bits.h"


void static inline __attribute__((always_inline)) avx2_256_loadu (
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


void static inline __attribute__((always_inline)) avx2_gather32_random (
    int const* data,
    uint64_t data_size
    ) 
{
    __m256i volatile res; 
    __m256i volatile res; 
    /*
        random_simd generator aligned: 
    */
    __m256i random_simd; 

     for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd, 1);

    }
}


void static inline __attribute__((always_inline)) avx2_gather32_random_u (
    int const* data,
    uint64_t data_size
    ) 
{
    __m256i volatile res; 
    /*
        random_simd generator unaligned: 
    */
    __m256i random_simd; 

     for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd, 1);

    }
}


void static inline __attribute__((always_inline)) avx2_gather32_stride (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 1); 
    }
}


void static inline __attribute__((always_inline)) avx2_gather32_same_index (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    ) 
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+=8) 
    {
        res = _mm256_i32gather_epi32 ((int const*)&data[rand() % data_size], vindex, 1);
    } 
}


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_1 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 1); 
    }
}


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_2 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 2); 
    }
}


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_4 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    )
{
    __m256i volatile res; 
    for (uint64_t i = 0; i < data_size; i+= 8) 
    {
        res = _mm256_i32gather_epi32((int const*)&data[rand() % data_size], vindex, 4); 
    }
}







