#ifndef AVX512_GATHER_BENCH_H_
#define AVX512_GATHER_BENCH_H_

#if __AVX2__
#include <stdint.h>
#include <immintrin.h>
#include "kernels_common.h"

static void avx_512_loadu (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i random_simd1, random_simd2;
    __m512i random_simd3, random_simd4;
    __m512i random_simd5, random_simd6;
    __m512i random_simd7, random_simd8;
    int32_t index1 = 0,  index2 = 16; 
    int32_t index3 = 32, index4 = 48; 
    int32_t index5 = 64, index6 = 80; 
    int32_t index7 = 96, index8 = 112; 
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_loadu_si512((__m512i*)&data[index1]);
        random_simd2 = _mm512_loadu_si512((__m512i*)&data[index2]);
        random_simd3 = _mm512_loadu_si512((__m512i*)&data[index3]);
        random_simd4 = _mm512_loadu_si512((__m512i*)&data[index4]);
        random_simd5 = _mm512_loadu_si512((__m512i*)&data[index5]);
        random_simd6 = _mm512_loadu_si512((__m512i*)&data[index6]);
        random_simd7 = _mm512_loadu_si512((__m512i*)&data[index7]);
        random_simd8 = _mm512_loadu_si512((__m512i*)&data[index8]);
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

static void avx512_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i random_simd1 = _mm512_loadu_si512((__m512i*)&data[0]);
    __m512i random_simd2 = _mm512_loadu_si512((__m512i*)&data[16]);
    __m512i random_simd3 = _mm512_loadu_si512((__m512i*)&data[32]);
    __m512i random_simd4 = _mm512_loadu_si512((__m512i*)&data[48]);
    __m512i random_simd5 = _mm512_loadu_si512((__m512i*)&data[64]);
    __m512i random_simd6 = _mm512_loadu_si512((__m512i*)&data[80]);
    __m512i random_simd7 = _mm512_loadu_si512((__m512i*)&data[96]);
    __m512i random_simd8 = _mm512_loadu_si512((__m512i*)&data[112]);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_i32gather_epi32 (random_simd1, (int const*)&data[0], sizeof(int));
        random_simd2 = _mm512_i32gather_epi32 (random_simd2, (int const*)&data[0], sizeof(int));
        random_simd3 = _mm512_i32gather_epi32 (random_simd3, (int const*)&data[0], sizeof(int));
        random_simd4 = _mm512_i32gather_epi32 (random_simd4, (int const*)&data[0], sizeof(int));
        random_simd5 = _mm512_i32gather_epi32 (random_simd5, (int const*)&data[0], sizeof(int));
        random_simd6 = _mm512_i32gather_epi32 (random_simd6, (int const*)&data[0], sizeof(int));
        random_simd7 = _mm512_i32gather_epi32 (random_simd7, (int const*)&data[0], sizeof(int));
        random_simd8 = _mm512_i32gather_epi32 (random_simd8, (int const*)&data[0], sizeof(int));
    }
    unused(stride); 
    do_not_optimize(random_simd1); do_not_optimize(random_simd2);
    do_not_optimize(random_simd3); do_not_optimize(random_simd4);
    do_not_optimize(random_simd5); do_not_optimize(random_simd6);
    do_not_optimize(random_simd7); do_not_optimize(random_simd8);
}

static void avx512_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    
    __m512i random_simd1 = _mm512_set_epi32(
        stride*15, stride*14, stride*13, stride*12, stride*11, stride*10, stride*9, stride*8,
         stride*7,  stride*6,  stride*5,  stride*4,  stride*3,  stride*2,   stride, 0
    );
    
    __m512i random_simd2 = _mm512_set_epi32(
        stride*15+16, stride*14+16, stride*13+16, stride*12+16, stride*11+16, stride*10+16, 
         stride*9+16,  stride*8+16,  stride*7+16,  stride*6+16,  stride*5+16,  stride*4+16, 
         stride*3+16,  stride*2+16,    stride+16,  16
    );


    __m512i random_simd3 = _mm512_set_epi32(
        stride*15+32, stride*14+32, stride*13+32, stride*12+32, stride*11+32, stride*10+32, 
         stride*9+32,  stride*8+32,  stride*7+32,  stride*6+32,  stride*5+32,  stride*4+32, 
         stride*3+32,  stride*2+32,    stride+32,  32
    );


    __m512i random_simd4 = _mm512_set_epi32(
        stride*15+48, stride*14+48, stride*13+48, stride*12+48, stride*11+48, stride*10+48, 
         stride*9+48,  stride*8+48,  stride*7+48,  stride*6+48,  stride*5+48,  stride*4+48, 
         stride*3+48,  stride*2+48,    stride+48,  48
    );


    __m512i random_simd5 = _mm512_set_epi32(
        stride*15+64, stride*14+64, stride*13+64, stride*12+64, stride*11+64, stride*10+64, 
         stride*9+64,  stride*8+64,  stride*7+64,  stride*6+64,  stride*5+64,  stride*4+64, 
         stride*3+64,  stride*2+64,    stride+64,  64
    );


    __m512i random_simd6 = _mm512_set_epi32(
        stride*15+80, stride*14+80, stride*13+80, stride*12+80, stride*11+80, stride*10+80, 
         stride*9+80,  stride*8+80,  stride*7+80,  stride*6+80, stride*5+80,  stride*4+80, 
         stride*3+80,  stride*2+80,    stride+80,  80
    );


    __m512i random_simd7 = _mm512_set_epi32(
        stride*15+96, stride*14+96, stride*13+96, stride*12+96, stride*11+96, stride*10+96, 
         stride*9+96,  stride*8+96,  stride*7+96,  stride*6+96, stride*5+96,  stride*4+96, 
         stride*3+96,  stride*2+96,    stride+96,  96
    );


    __m512i random_simd8 = _mm512_set_epi32(
        stride*15+112, stride*14+112, stride*13+112, stride*12+112, stride*11+112, stride*10+112, 
         stride*9+112,  stride*8+112,  stride*7+112,  stride*6+112,  stride*5+112,  stride*4+112, 
         stride*3+112,  stride*2+112,    stride+112,  112
    );


    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_i32gather_epi32 (random_simd1, (int const*)&data[0], sizeof(int));
        random_simd2 = _mm512_i32gather_epi32 (random_simd2, (int const*)&data[0], sizeof(int));
        random_simd3 = _mm512_i32gather_epi32 (random_simd3, (int const*)&data[0], sizeof(int));
        random_simd4 = _mm512_i32gather_epi32 (random_simd4, (int const*)&data[0], sizeof(int));
        random_simd5 = _mm512_i32gather_epi32 (random_simd5, (int const*)&data[0], sizeof(int));
        random_simd6 = _mm512_i32gather_epi32 (random_simd6, (int const*)&data[0], sizeof(int));
        random_simd7 = _mm512_i32gather_epi32 (random_simd7, (int const*)&data[0], sizeof(int));
        random_simd8 = _mm512_i32gather_epi32 (random_simd8, (int const*)&data[0], sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2);
    do_not_optimize(random_simd3); do_not_optimize(random_simd4);
    do_not_optimize(random_simd5); do_not_optimize(random_simd6);
    do_not_optimize(random_simd7); do_not_optimize(random_simd8);
}

static void avx512_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{

    __m512i random_simd1 = _mm512_set_epi32(
        stride*7, stride*7, stride*6, stride*6, stride*5, stride*5, stride*4, stride*4,
        stride*3, stride*3, stride*2, stride*2, stride, stride, 0, 0
    );
    
    __m512i random_simd2 = _mm512_set_epi32(
        stride*7+16, stride*7+16, stride*6+16, stride*6+16, stride*5+16, stride*5+16, 
        stride*4+16, stride*4+16, stride*3+16,  stride*3+16, stride*2+16,  stride*2+16, 
        stride+16,   stride+16,    16,  16
    ); 


    __m512i random_simd3 = _mm512_set_epi32(
        stride*7+32, stride*7+32, stride*6+32, stride*6+32, stride*5+32, stride*5+32, 
         stride*4+32,  stride*4+32,  stride*3+32,  stride*3+32,  stride*2+32,  stride*2+32, 
         stride+32,  stride+32,    32,  32
    ); 


    __m512i random_simd4 = _mm512_set_epi32(
        stride*7+48, stride*7+48, stride*6+48, stride*6+48, stride*5+48, stride*5+48, 
         stride*4+48,  stride*4+48,  stride*3+48,  stride*3+48,  stride*2+48,  stride*2+48, 
         stride+48,  stride+48,    48,  48
    ); 


    __m512i random_simd5 = _mm512_set_epi32(
        stride*7+64, stride*7+64, stride*6+64, stride*6+64, stride*5+64, stride*5+64, 
        stride*4+64, stride*4+64, stride*3+64, stride*3+64, stride*2+64, stride*2+64, 
        stride+64,   stride+64, 64,  64
    ); 


    __m512i random_simd6 = _mm512_set_epi32(
        stride*7+80, stride*7+80, stride*6+80, stride*6+80, stride*5+80, stride*5+80, 
        stride*4+80, stride*4+80, stride*3+80, stride*3+80, stride*2+80, stride*2+80, 
        stride+80,   stride+80,    80,  80
    ); 


    __m512i random_simd7 = _mm512_set_epi32(
        stride*7+96, stride*7+96, stride*6+96, stride*6+96, stride*5+96, stride*5+96, 
        stride*4+96, stride*4+96, stride*3+96, stride*3+96, stride*2+96, stride*2+96, 
        stride+96,   stride+96, 96,  96
    ); 


    __m512i random_simd8 = _mm512_set_epi32(
        stride*7+112, stride*7+112, stride*6+112, stride*6+112, stride*5+112, stride*5+112, 
        stride*4+112, stride*4+112, stride*3+112, stride*3+112, stride*2+112, stride*2+112, 
        stride+112,   stride+112,  112,  112
    ); 

    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_i32gather_epi32 (random_simd1, (int const*)&data[0], sizeof(int));
        random_simd2 = _mm512_i32gather_epi32 (random_simd2, (int const*)&data[0], sizeof(int));
        random_simd3 = _mm512_i32gather_epi32 (random_simd3, (int const*)&data[0], sizeof(int));
        random_simd4 = _mm512_i32gather_epi32 (random_simd4, (int const*)&data[0], sizeof(int));
        random_simd5 = _mm512_i32gather_epi32 (random_simd5, (int const*)&data[0], sizeof(int));
        random_simd6 = _mm512_i32gather_epi32 (random_simd6, (int const*)&data[0], sizeof(int));
        random_simd7 = _mm512_i32gather_epi32 (random_simd7, (int const*)&data[0], sizeof(int));
        random_simd8 = _mm512_i32gather_epi32 (random_simd8, (int const*)&data[0], sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2);
    do_not_optimize(random_simd3); do_not_optimize(random_simd4);
    do_not_optimize(random_simd5); do_not_optimize(random_simd6);
    do_not_optimize(random_simd7); do_not_optimize(random_simd8);
}


static void avx512_gather32_stride_4equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m512i random_simd1 = _mm512_set_epi32(
        stride*3, stride*3, stride*3, stride*3, stride*2, stride*2, stride*2, stride*2,
        stride, stride, stride, stride, 0, 0, 0, 0
    );
    
    __m512i random_simd2 = _mm512_set_epi32(
        stride*3+16, stride*3+16, stride*3+16, stride*3+16, stride*2+16, stride*2+16, 
        stride*2+16,  stride*2+16,  stride+16,  stride+16,  stride+16,  stride+16, 
        16, 16, 16,  16
    ); 


    __m512i random_simd3 = _mm512_set_epi32(
        stride*3+32, stride*3+32, stride*3+32, stride*3+32, stride*2+32, stride*2+32, 
        stride*2+32,  stride*2+32,  stride+32,  stride+32,  stride+32,  stride+32, 
        32, 32, 32,  32
    ); 


    __m512i random_simd4 = _mm512_set_epi32(
        stride*3+48, stride*3+48, stride*3+48, stride*3+48, stride*2+48, stride*2+48, 
        stride*2+48,  stride*2+48,  stride+48,  stride+48,  stride+48,  stride+48, 
         48,  48,    48,  48
    ); 


    __m512i random_simd5 = _mm512_set_epi32(
        stride*3+64, stride*3+64, stride*3+64, stride*3+64, stride*2+64, stride*2+64, 
        stride*2+64,  stride*2+64,  stride+64,  stride+64,  stride+64,  stride+64, 
         64,  64,    64,  64
    ); 


    __m512i random_simd6 = _mm512_set_epi32(
        stride*3+80, stride*3+80, stride*3+80, stride*3+80, stride*2+80, stride*2+80, 
        stride*2+80,  stride*2+80,  stride+80,  stride+80,  stride+80,  stride+80, 
         80,  80,    80,  80
    ); 


    __m512i random_simd7 = _mm512_set_epi32(
        stride*3+96, stride*3+96, stride*3+96, stride*3+96, stride*2+96, stride*2+96, 
        stride*2+96, stride*2+96, stride+96,  stride+96,  stride+96,  stride+96, 
        96,  96,    96,  96
    ); 


    __m512i random_simd8 = _mm512_set_epi32(
        stride*3+112, stride*3+112, stride*3+112, stride*3+112, stride*2+112, stride*2+112, 
        stride*2+112, stride*2+112, stride+112,  stride+112,  stride+112, stride+112, 
        112,  112, 112,  112
    ); 

    
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_i32gather_epi32 (random_simd1, (int const*)&data[0], sizeof(int));
        random_simd2 = _mm512_i32gather_epi32 (random_simd2, (int const*)&data[0], sizeof(int));
        random_simd3 = _mm512_i32gather_epi32 (random_simd3, (int const*)&data[0], sizeof(int));
        random_simd4 = _mm512_i32gather_epi32 (random_simd4, (int const*)&data[0], sizeof(int));
        random_simd5 = _mm512_i32gather_epi32 (random_simd5, (int const*)&data[0], sizeof(int));
        random_simd6 = _mm512_i32gather_epi32 (random_simd6, (int const*)&data[0], sizeof(int));
        random_simd7 = _mm512_i32gather_epi32 (random_simd7, (int const*)&data[0], sizeof(int));
        random_simd8 = _mm512_i32gather_epi32 (random_simd8, (int const*)&data[0], sizeof(int));
    }
    do_not_optimize(random_simd1); do_not_optimize(random_simd2);
    do_not_optimize(random_simd3); do_not_optimize(random_simd4);
    do_not_optimize(random_simd5); do_not_optimize(random_simd6);
    do_not_optimize(random_simd7); do_not_optimize(random_simd8);
}


static void avx512_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m512i random_simd1 = _mm512_set1_epi32(0);
    __m512i random_simd2 = _mm512_set1_epi32(1);
    __m512i random_simd3 = _mm512_set1_epi32(2);
    __m512i random_simd4 = _mm512_set1_epi32(3);
    __m512i random_simd5 = _mm512_set1_epi32(4);
    __m512i random_simd6 = _mm512_set1_epi32(5);
    __m512i random_simd7 = _mm512_set1_epi32(6);
    __m512i random_simd8 = _mm512_set1_epi32(7);
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1 = _mm512_i32gather_epi32 (random_simd1, (int const*)&data[0], sizeof(int));
        random_simd2 = _mm512_i32gather_epi32 (random_simd2, (int const*)&data[0], sizeof(int));
        random_simd3 = _mm512_i32gather_epi32 (random_simd3, (int const*)&data[0], sizeof(int));
        random_simd4 = _mm512_i32gather_epi32 (random_simd4, (int const*)&data[0], sizeof(int));
        random_simd5 = _mm512_i32gather_epi32 (random_simd5, (int const*)&data[0], sizeof(int));
        random_simd6 = _mm512_i32gather_epi32 (random_simd6, (int const*)&data[0], sizeof(int));
        random_simd7 = _mm512_i32gather_epi32 (random_simd7, (int const*)&data[0], sizeof(int));
        random_simd8 = _mm512_i32gather_epi32 (random_simd8, (int const*)&data[0], sizeof(int));
    }
    unused(stride);
    do_not_optimize(random_simd1); do_not_optimize(random_simd2);
    do_not_optimize(random_simd3); do_not_optimize(random_simd4);
    do_not_optimize(random_simd5); do_not_optimize(random_simd6);
    do_not_optimize(random_simd7); do_not_optimize(random_simd8);
}

#endif // __AVX512F__
#endif // AVX512_GATHER_BENCH_H_
