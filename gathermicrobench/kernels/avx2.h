#ifndef AVX2_GATHER_BENCH_H_
#define AVX2_GATHER_BENCH_H_


#if __AVX2__
#include <stdint.h>
#include <immintrin.h>
#include "kernels_common.h"

static void avx2_256_loadu (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i random_simd1, random_simd2;
    __m256i random_simd3, random_simd4;
    __m256i random_simd5, random_simd6;
    __m256i random_simd7, random_simd8;
    int32_t index1 = 0, index2 = 8; 
    int32_t index3 = 16, index4 = 24; 
    int32_t index5 = 32, index6 = 40; 
    int32_t index7 = 48, index8 = 56; 
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_lddqu_si256((__m256i*)&data[index1]);
        random_simd2 = _mm256_lddqu_si256((__m256i*)&data[index2]);
        random_simd3 = _mm256_lddqu_si256((__m256i*)&data[index3]);
        random_simd4 = _mm256_lddqu_si256((__m256i*)&data[index4]);
        random_simd5 = _mm256_lddqu_si256((__m256i*)&data[index5]);
        random_simd6 = _mm256_lddqu_si256((__m256i*)&data[index6]);
        random_simd7 = _mm256_lddqu_si256((__m256i*)&data[index7]);
        random_simd8 = _mm256_lddqu_si256((__m256i*)&data[index8]);
        index1 = data[index1]; index2 = data[index2]; 
        index3 = data[index3]; index4 = data[index4];
        index5 = data[index5]; index6 = data[index6]; 
        index7 = data[index7]; index8 = data[index8];  
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}

static void avx2_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i random_simd1 = _mm256_lddqu_si256((__m256i*)&data[0]);
    __m256i random_simd2 = _mm256_lddqu_si256((__m256i*)&data[8]);
    __m256i random_simd3 = _mm256_lddqu_si256((__m256i*)&data[16]);
    __m256i random_simd4 = _mm256_lddqu_si256((__m256i*)&data[24]);
    __m256i random_simd5 = _mm256_lddqu_si256((__m256i*)&data[32]);
    __m256i random_simd6 = _mm256_lddqu_si256((__m256i*)&data[40]);
    __m256i random_simd7 = _mm256_lddqu_si256((__m256i*)&data[48]);
    __m256i random_simd8 = _mm256_lddqu_si256((__m256i*)&data[56]);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}

static void avx2_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    
    __m256i random_simd1 = _mm256_set_epi32(
        stride*7, stride*6, stride*5, stride*4, stride*3, stride*2, stride, 0
    );
    
    __m256i random_simd2 = _mm256_set_epi32(
        stride*7+8, stride*6+8, stride*5+8, stride*4+8, stride*3+8, stride*2+8, stride+8, 8
    );
        
    __m256i random_simd3 = _mm256_set_epi32(
        stride*7+16, stride*6+16, stride*5+16, stride*4+16, stride*3+16, stride*2+16, stride+16, 16
    );
    
    __m256i random_simd4 = _mm256_set_epi32(
        stride*7+24, stride*6+24, stride*5+24, stride*4+24, stride*3+24, stride*2+24, stride+24, 24
    );

    __m256i random_simd5 = _mm256_set_epi32(
        stride*7+32, stride*6+32, stride*5+32, stride*4+32, stride*3+32, stride*2+32, stride+32, 32
    );
    
    __m256i random_simd6 = _mm256_set_epi32(
        stride*7+40, stride*6+40, stride*5+40, stride*4+40, stride*3+40, stride*2+40, stride+40, 40
    );
        
    __m256i random_simd7 = _mm256_set_epi32(
        stride*7+48, stride*6+48, stride*5+48, stride*4+48, stride*3+48, stride*2+48, stride+48, 48  
    );

    __m256i random_simd8 = _mm256_set_epi32(
        stride*7+56, stride*6+56, stride*5+56, stride*4+56, stride*3+56, stride*2+56, stride+56, 56
    );

    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}

static void avx2_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{

    __m256i random_simd1 = _mm256_set_epi32(
        stride*3, stride*3, stride*2, stride*2, stride, stride, 0, 0
    );
    
    __m256i random_simd2 = _mm256_set_epi32(
        stride*3+8, stride*3+8, stride*2+8, stride*2+8, stride+8, stride+8, 8, 8
    );
        
    __m256i random_simd3 = _mm256_set_epi32(
        stride*3+16,stride*3+16, stride*2+16, stride*2+16, stride+16, stride+16, 16, 16
    );
    
    __m256i random_simd4 = _mm256_set_epi32(
        stride*3+24, stride*3+24, stride*2+24, stride*2+24, stride+24, stride+24, 24, 24
    );
    
    __m256i random_simd5 = _mm256_set_epi32(
        stride*3+32, stride*3+32, stride*2+32, stride*2+32, stride+32, stride+32, 32, 32
    );

    __m256i random_simd6 = _mm256_set_epi32(
        stride*3+40, stride*3+40, stride*2+40, stride*2+40, stride+40, stride+40, 40, 40
    );

    __m256i random_simd7 = _mm256_set_epi32(
        stride*3+48, stride*3+48, stride*2+48, stride*2+48, stride+48, stride+48, 48, 48
    );

     __m256i random_simd8 = _mm256_set_epi32(
        stride*3+56, stride*3+56, stride*2+56, stride*2+56, stride+56, stride+56, 56, 56
    );

    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}


static void avx2_gather32_stride_4equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m256i random_simd1 = _mm256_set_epi32(
        stride, stride, stride, stride, 0, 0, 0, 0
    );
    
    __m256i random_simd2 = _mm256_set_epi32(
        stride+8, stride+8, stride+8, stride+8, 8, 8, 8, 8
    );
        
    __m256i random_simd3 = _mm256_set_epi32(
        stride+16, stride+16, stride+16, stride+16, 16, 16, 16, 16
    );
    
    __m256i random_simd4 = _mm256_set_epi32(
        stride+24, stride+24, stride+24, stride+24, 24, 24, 24, 24
    );

    __m256i random_simd5 = _mm256_set_epi32(
        stride+32, stride+32, stride+32, stride+32, 32, 32, 32, 32
    );

    __m256i random_simd6 = _mm256_set_epi32(
        stride+40, stride+40, stride+40, stride+40, 40, 40, 40, 40
    );

    __m256i random_simd7 = _mm256_set_epi32(
        stride+48, stride+48, stride+48, stride+48, 48, 48, 48, 48
    );

    __m256i random_simd8 = _mm256_set_epi32(
        stride+56, stride+56, stride+56, stride+56, 56, 56, 56, 56
    );
    
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}


static void avx2_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m256i random_simd1 = _mm256_set1_epi32(0);
    __m256i random_simd2 = _mm256_set1_epi32(1);
    __m256i random_simd3 = _mm256_set1_epi32(2);
    __m256i random_simd4 = _mm256_set1_epi32(3);
    __m256i random_simd5 = _mm256_set1_epi32(4);
    __m256i random_simd6 = _mm256_set1_epi32(5);
    __m256i random_simd7 = _mm256_set1_epi32(6);
    __m256i random_simd8 = _mm256_set1_epi32(7);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}
#endif // __AVX2__
#endif // AVX2_GATHER_BENCH_H_
