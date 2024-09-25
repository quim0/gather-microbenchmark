#ifndef SSE_GATHER_BENCH_H_
#define SSE_GATHER_BENCH_H_

#if __AVX2__
#include <stdint.h>
#include <immintrin.h>
#include "kernels_common.h"

static void sse_128_loadu_32 (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m128i random_simd1,  random_simd2;
    __m128i random_simd3,  random_simd4;
    __m128i random_simd5,  random_simd6;
    __m128i random_simd7,  random_simd8;
    __m128i random_simd9,  random_simd10;
    __m128i random_simd11, random_simd12;
    __m128i random_simd13;
    int32_t index1 = 0,  index2 = 4;
    int32_t index3 = 8,  index4 = 12; 
    int32_t index5 = 16, index6 = 20;
    int32_t index7 = 24, index8 = 28;
    int32_t index9 = 32, index10 = 36;
    int32_t index11 = 40, index12 = 44;
    int32_t index13 = 48;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm_lddqu_si128((__m128i*)&data[index1]);
        random_simd2  = _mm_lddqu_si128((__m128i*)&data[index2]);
        random_simd3  = _mm_lddqu_si128((__m128i*)&data[index3]);
        random_simd4  = _mm_lddqu_si128((__m128i*)&data[index4]);
        random_simd5  = _mm_lddqu_si128((__m128i*)&data[index5]);
        random_simd6  = _mm_lddqu_si128((__m128i*)&data[index6]);
        random_simd7  = _mm_lddqu_si128((__m128i*)&data[index7]);
        random_simd8  = _mm_lddqu_si128((__m128i*)&data[index8]);
        random_simd9  = _mm_lddqu_si128((__m128i*)&data[index9]);
        random_simd10 = _mm_lddqu_si128((__m128i*)&data[index10]);
        random_simd11 = _mm_lddqu_si128((__m128i*)&data[index11]);
        random_simd12 = _mm_lddqu_si128((__m128i*)&data[index12]);
        random_simd13 = _mm_lddqu_si128((__m128i*)&data[index13]);
        index1  = data[index1];  index2  = data[index2]; 
        index3  = data[index3];  index4  = data[index4];
        index5  = data[index5];  index6  = data[index6]; 
        index7  = data[index7];  index8  = data[index8];  
        index5  = data[index5];  index6  = data[index6]; 
        index7  = data[index7];  index8  = data[index8]; 
        index9  = data[index9];  index10 = data[index10];  
        index11 = data[index11]; index12 = data[index12]; 
        index13 = data[index13];
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3); do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5); do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7); do_not_optimize(random_simd8); 
}

static void sse_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m128i random_simd1  = _mm_lddqu_si128((__m128i*)&data[0]);
    __m128i random_simd2  = _mm_lddqu_si128((__m128i*)&data[4]);
    __m128i random_simd3  = _mm_lddqu_si128((__m128i*)&data[8]);
    __m128i random_simd4  = _mm_lddqu_si128((__m128i*)&data[12]);
    __m128i random_simd5  = _mm_lddqu_si128((__m128i*)&data[16]);
    __m128i random_simd6  = _mm_lddqu_si128((__m128i*)&data[20]);
    __m128i random_simd7  = _mm_lddqu_si128((__m128i*)&data[24]);
    __m128i random_simd8  = _mm_lddqu_si128((__m128i*)&data[28]);
    __m128i random_simd9  = _mm_lddqu_si128((__m128i*)&data[32]);
    __m128i random_simd10 = _mm_lddqu_si128((__m128i*)&data[36]);
    __m128i random_simd11 = _mm_lddqu_si128((__m128i*)&data[40]);
    __m128i random_simd12 = _mm_lddqu_si128((__m128i*)&data[44]);
    __m128i random_simd13 = _mm_lddqu_si128((__m128i*)&data[48]);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
        random_simd9  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd9, sizeof(int));
        random_simd10 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd10, sizeof(int));
        random_simd11 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd11, sizeof(int));
        random_simd12 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd12, sizeof(int));
        random_simd13 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd13, sizeof(int));
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

static void sse_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m128i random_simd1 = _mm_set_epi32(
        stride*3, stride*2, stride, 0
    );
    __m128i random_simd2 = _mm_set_epi32(
        stride*3+4, stride*2+4, stride+4, 4
    );   
    __m128i random_simd3 = _mm_set_epi32(
        stride*3+8, stride*2+8, stride+8, 8
    );
    __m128i random_simd4 = _mm_set_epi32(
        stride*3+12, stride*2+12, stride+12, 12
    );
    __m128i random_simd5 = _mm_set_epi32(
        stride*3+16, stride*2+16, stride+16, 16
    );
    __m128i random_simd6 = _mm_set_epi32(
        stride*3+20, stride*2+20, stride+20, 20
    );
    __m128i random_simd7 = _mm_set_epi32(
        stride*3+24, stride*2+24, stride+24, 24
    );
    __m128i random_simd8 = _mm_set_epi32(
        stride*3+28, stride*2+28, stride+28, 28
    );
    __m128i random_simd9 = _mm_set_epi32(
        stride*3+32, stride*2+32, stride+32, 32
    );
    __m128i random_simd10 = _mm_set_epi32(
        stride*3+36, stride*2+36, stride+36, 36
    );
    __m128i random_simd11 = _mm_set_epi32(
        stride*3+40, stride*2+40, stride+40, 40
    );
    __m128i random_simd12 = _mm_set_epi32(
        stride*3+44, stride*2+44, stride+44, 44
    );
    __m128i random_simd13 = _mm_set_epi32(
        stride*3+48, stride*2+48, stride+48, 48
    );
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
        random_simd9  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd9, sizeof(int));
        random_simd10 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd10, sizeof(int));
        random_simd11 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd11, sizeof(int));
        random_simd12 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd12, sizeof(int));
        random_simd13 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd13, sizeof(int));
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

static void sse_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m128i random_simd1 = _mm_set_epi32(
        stride, stride, 0, 0
    );
    __m128i random_simd2 = _mm_set_epi32(
        stride+4, stride+4, 4, 4
    );   
    __m128i random_simd3 = _mm_set_epi32(
        stride+8, stride+8, stride+8, 8
    );
    __m128i random_simd4 = _mm_set_epi32(
        stride+12, stride+12, 12, 12
    );
    __m128i random_simd5 = _mm_set_epi32(
        stride+16, stride+16, 16, 16
    );
    __m128i random_simd6 = _mm_set_epi32(
        stride+20, stride+20, 20, 20
    );
    __m128i random_simd7 = _mm_set_epi32(
        stride+24, stride+24, 24, 24
    );
    __m128i random_simd8 = _mm_set_epi32(
        stride+28, stride+28, 28, 28
    );
    __m128i random_simd9 = _mm_set_epi32(
        stride+32, stride+32, 32, 32
    );
    __m128i random_simd10 = _mm_set_epi32(
        stride+36, stride+36, 36, 36
    );
    __m128i random_simd11 = _mm_set_epi32(
        stride+40, stride+40, 40, 40
    );
    __m128i random_simd12 = _mm_set_epi32(
        stride+44, stride+44, 44, 44
    );
    __m128i random_simd13 = _mm_set_epi32(
        stride+48, stride+48, 48, 48
    );
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
        random_simd9  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd9, sizeof(int));
        random_simd10 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd10, sizeof(int));
        random_simd11 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd11, sizeof(int));
        random_simd12 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd12, sizeof(int));
        random_simd13 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd13, sizeof(int));
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

static void sse_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m128i random_simd1  = _mm_set1_epi32(0);
    __m128i random_simd2  = _mm_set1_epi32(1);
    __m128i random_simd3  = _mm_set1_epi32(2);
    __m128i random_simd4  = _mm_set1_epi32(3);
    __m128i random_simd5  = _mm_set1_epi32(4);
    __m128i random_simd6  = _mm_set1_epi32(5);
    __m128i random_simd7  = _mm_set1_epi32(6);
    __m128i random_simd8  = _mm_set1_epi32(7);
    __m128i random_simd9  = _mm_set1_epi32(8);
    __m128i random_simd10 = _mm_set1_epi32(9);
    __m128i random_simd11 = _mm_set1_epi32(10);
    __m128i random_simd12 = _mm_set1_epi32(11);
    __m128i random_simd13 = _mm_set1_epi32(12);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
        random_simd9  = _mm_i32gather_epi32 ((int const*)&data[0], random_simd9, sizeof(int));
        random_simd10 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd10, sizeof(int));
        random_simd11 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd11, sizeof(int));
        random_simd12 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd12, sizeof(int));
        random_simd13 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd13, sizeof(int));
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
#endif // SSE_GATHER_BENCH_H_