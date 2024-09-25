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
    __m256i random_simd9, random_simd10;
    __m256i random_simd11, random_simd12;
    __m256i random_simd13;
    int32_t index1 = 0, index2 = 8; 
    int32_t index3 = 16, index4 = 24; 
    int32_t index5 = 32, index6 = 40; 
    int32_t index7 = 48, index8 = 56; 
    int32_t index9 = 64, index10 = 72; 
    int32_t index11 = 80, index12 = 88; 
    int32_t index13 = 96; 
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
        random_simd9 = _mm256_lddqu_si256((__m256i*)&data[index9]);
        random_simd10 = _mm256_lddqu_si256((__m256i*)&data[index10]);
        random_simd11 = _mm256_lddqu_si256((__m256i*)&data[index11]);
        random_simd12 = _mm256_lddqu_si256((__m256i*)&data[index12]);
        random_simd13 = _mm256_lddqu_si256((__m256i*)&data[index13]);
        index1 = data[index1]; index2 = data[index2]; 
        index3 = data[index3]; index4 = data[index4];
        index5 = data[index5]; index6 = data[index6]; 
        index7 = data[index7]; index8 = data[index8];  
        index5 = data[index5]; index6 = data[index6]; 
        index7 = data[index7]; index8 = data[index8]; 
        index9 = data[index9]; index10 = data[index10];  
        index11 = data[index11]; index12 = data[index12]; 
        index13 = data[index13];
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13); 
}


static void avx2_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i random_simd1  = _mm256_lddqu_si256((__m256i*)&data[0]);
    __m256i random_simd2  = _mm256_lddqu_si256((__m256i*)&data[8]);
    __m256i random_simd3  = _mm256_lddqu_si256((__m256i*)&data[16]);
    __m256i random_simd4  = _mm256_lddqu_si256((__m256i*)&data[24]);
    __m256i random_simd5  = _mm256_lddqu_si256((__m256i*)&data[32]);
    __m256i random_simd6  = _mm256_lddqu_si256((__m256i*)&data[40]);
    __m256i random_simd7  = _mm256_lddqu_si256((__m256i*)&data[48]);
    __m256i random_simd8  = _mm256_lddqu_si256((__m256i*)&data[56]);
    __m256i random_simd9  = _mm256_lddqu_si256((__m256i*)&data[64]);
    __m256i random_simd10 = _mm256_lddqu_si256((__m256i*)&data[72]);
    __m256i random_simd11 = _mm256_lddqu_si256((__m256i*)&data[80]);
    __m256i random_simd12 = _mm256_lddqu_si256((__m256i*)&data[88]);
    __m256i random_simd13 = _mm256_lddqu_si256((__m256i*)&data[96]);
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
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
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
    __m256i random_simd9 = _mm256_set_epi32(
        stride*7+64, stride*6+64, stride*5+64, stride*4+64, stride*3+64, stride*2+64, stride+64, 64
    );
    __m256i random_simd10 = _mm256_set_epi32(
        stride*7+72, stride*6+72, stride*5+72, stride*4+72, stride*3+72, stride*2+72, stride+72, 72
    );
    __m256i random_simd11 = _mm256_set_epi32(
        stride*7+80, stride*6+80, stride*5+80, stride*4+80, stride*3+80, stride*2+80, stride+80, 80
    );
    __m256i random_simd12 = _mm256_set_epi32(
        stride*7+88, stride*6+88, stride*5+88, stride*4+88, stride*3+88, stride*2+88, stride+88, 88
    );
    __m256i random_simd13 = _mm256_set_epi32(
        stride*7+96, stride*6+96, stride*5+96, stride*4+96, stride*3+96, stride*2+96, stride+96, 96
    );
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
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
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
    __m256i random_simd9 = _mm256_set_epi32(
        stride*3+64, stride*3+64, stride*2+64, stride*2+64, stride+64, stride+64, 64, 64
    );
    __m256i random_simd10 = _mm256_set_epi32(
        stride*3+72, stride*3+72, stride*2+72, stride*2+72, stride+72, stride+72, 72, 72
    );
    __m256i random_simd11 = _mm256_set_epi32(
        stride*3+80, stride*3+80, stride*2+80, stride*2+80, stride+80, stride+80, 80, 80
    );
    __m256i random_simd12 = _mm256_set_epi32(
        stride*3+88, stride*3+88, stride*2+88, stride*2+88, stride+88, stride+88, 88, 88
    );
    __m256i random_simd13 = _mm256_set_epi32(
        stride*3+96, stride*3+96, stride*2+96, stride*2+96, stride+96, stride+96, 96, 96
    );
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
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
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
    __m256i random_simd9 = _mm256_set_epi32(
        stride+64, stride+64, stride+64, stride+64, 64, 64, 64, 64
    );
    __m256i random_simd10 = _mm256_set_epi32(
        stride+72, stride+72, stride+72, stride+72, 72, 72, 72, 72
    );
    __m256i random_simd11 = _mm256_set_epi32(
        stride+80, stride+80, stride+80, stride+80, 80, 80, 80, 80
    );
    __m256i random_simd12 = _mm256_set_epi32(
        stride+88, stride+88, stride+88, stride+88, 88, 88, 88, 88
    );
    __m256i random_simd13 = _mm256_set_epi32(
        stride+96, stride+96, stride+96, stride+96, 96, 96, 96, 96
    );
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
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
}


static void avx2_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m256i random_simd1  = _mm256_set1_epi32(0);
    __m256i random_simd2  = _mm256_set1_epi32(1);
    __m256i random_simd3  = _mm256_set1_epi32(2);
    __m256i random_simd4  = _mm256_set1_epi32(3);
    __m256i random_simd5  = _mm256_set1_epi32(4);
    __m256i random_simd6  = _mm256_set1_epi32(5);
    __m256i random_simd7  = _mm256_set1_epi32(6);
    __m256i random_simd8  = _mm256_set1_epi32(7);
    __m256i random_simd9  = _mm256_set1_epi32(8);
    __m256i random_simd10 = _mm256_set1_epi32(9);
    __m256i random_simd11 = _mm256_set1_epi32(10);
    __m256i random_simd12 = _mm256_set1_epi32(11);
    __m256i random_simd13 = _mm256_set1_epi32(12);
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
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9); do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
}
#endif // __AVX2__
#endif // AVX2_GATHER_BENCH_H_
