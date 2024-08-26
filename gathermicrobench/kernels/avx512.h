#ifndef AVX512_GATHER_BENCH_H_
#define AVX512_GATHER_BENCH_H_

#if __AVX2__
#include <stdint.h>
#include <immintrin.h>

static void avx_512_loadu (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i volatile random_simd1, random_simd2;
    __m512i volatile random_simd3, random_simd4;
    __m512i volatile random_simd5, random_simd6;
    __m512i volatile random_simd7, random_simd8;
    int32_t index1 = 0, index2 = 1; 
    int32_t index3 = 2, index4 = 3; 
    int32_t index5 = 4, index6 = 5; 
    int32_t index7 = 6, index8 = 7; 
    for (uint64_t i = 64; i < data_size; i+=16) 
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
    (void)stride; 
    (void)random_simd1; (void)random_simd2; 
    (void)random_simd3; (void)random_simd4; 
    (void)random_simd5; (void)random_simd6; 
    (void)random_simd7; (void)random_simd8; 
}

static void avx512_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m512i volatile random_simd1 = _mm512_loadu_si512((__m512i*)&data[0]);
    __m512i volatile random_simd2 = _mm512_loadu_si512((__m512i*)&data[16]);
    __m512i volatile random_simd3 = _mm512_loadu_si512((__m512i*)&data[32]);
    __m512i volatile random_simd4 = _mm512_loadu_si512((__m512i*)&data[48]);
    __m512i volatile random_simd5 = _mm512_loadu_si512((__m512i*)&data[64]);
    __m512i volatile random_simd6 = _mm512_loadu_si512((__m512i*)&data[80]);
    __m512i volatile random_simd7 = _mm512_loadu_si512((__m512i*)&data[96]);
    __m512i volatile random_simd8 = _mm512_loadu_si512((__m512i*)&data[112]);
    for (uint64_t i = 64; i < data_size; i+=16) 
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
    (void)stride; 
}

static void avx512_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    
    __m512i volatile random_simd1 = _mm512_set_epi32(
        data[stride*15], data[stride*14], data[stride*13], data[stride*12], data[stride*11], data[stride*10], data[stride*9], data[stride*8],
         data[stride*7],  data[stride*6],  data[stride*5],  data[stride*4],  data[stride*3],  data[stride*2],   data[stride], data[0]
    );
    
    __m512i volatile random_simd2 = _mm512_set_epi32(
        data[stride*15+16], data[stride*14+16], data[stride*13+16], data[stride*12+16], data[stride*11+16], data[stride*10+16], 
         data[stride*9+16],  data[stride*8+16],  data[stride*7+16],  data[stride*6+16],  data[stride*5+16],  data[stride*4+16], 
         data[stride*3+16],  data[stride*2+16],    data[stride+16],  data[16]
    );


    __m512i volatile random_simd3 = _mm512_set_epi32(
        data[stride*15+32], data[stride*14+32], data[stride*13+32], data[stride*12+32], data[stride*11+32], data[stride*10+32], 
         data[stride*9+32],  data[stride*8+32],  data[stride*7+32],  data[stride*6+32],  data[stride*5+32],  data[stride*4+32], 
         data[stride*3+32],  data[stride*2+32],    data[stride+32],  data[32]
    );


    __m512i volatile random_simd4 = _mm512_set_epi32(
        data[stride*15+48], data[stride*14+48], data[stride*13+48], data[stride*12+48], data[stride*11+48], data[stride*10+48], 
         data[stride*9+48],  data[stride*8+48],  data[stride*7+48],  data[stride*6+48],  data[stride*5+48],  data[stride*4+48], 
         data[stride*3+48],  data[stride*2+48],    data[stride+48],  data[48]
    );


    __m512i volatile random_simd5 = _mm512_set_epi32(
        data[stride*15+64], data[stride*14+64], data[stride*13+64], data[stride*12+64], data[stride*11+64], data[stride*10+64], 
         data[stride*9+64],  data[stride*8+64],  data[stride*7+64],  data[stride*6+64],  data[stride*5+64],  data[stride*4+64], 
         data[stride*3+64],  data[stride*2+64],    data[stride+64],  data[64]
    );


    __m512i volatile random_simd6 = _mm512_set_epi32(
        data[stride*15+80], data[stride*14+80], data[stride*13+80], data[stride*12+80], data[stride*11+80], data[stride*10+80], 
         data[stride*9+80],  data[stride*8+80],  data[stride*7+80],  data[stride*6+80],  data[stride*5+80],  data[stride*4+80], 
         data[stride*3+80],  data[stride*2+80],    data[stride+80],  data[80]
    );


    __m512i volatile random_simd7 = _mm512_set_epi32(
        data[stride*15+96], data[stride*14+96], data[stride*13+96], data[stride*12+96], data[stride*11+96], data[stride*10+96], 
         data[stride*9+96],  data[stride*8+96],  data[stride*7+96],  data[stride*6+96],  data[stride*5+96],  data[stride*4+96], 
         data[stride*3+96],  data[stride*2+96],    data[stride+96],  data[96]
    );


    __m512i volatile random_simd8 = _mm512_set_epi32(
        data[stride*15+112], data[stride*14+112], data[stride*13+112], data[stride*12+112], data[stride*11+112], data[stride*10+112], 
         data[stride*9+112],  data[stride*8+112],  data[stride*7+112],  data[stride*6+112],  data[stride*5+112],  data[stride*4+112], 
         data[stride*3+112],  data[stride*2+112],    data[stride+112],  data[112]
    );


    for (uint64_t i = 64; i < data_size; i+=16) 
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
}

static void avx512_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{

    __m512i volatile random_simd1 = _mm512_set_epi32(
        data[stride*7], data[stride*7], data[stride*6], data[stride*6], data[stride*5], data[stride*5], data[stride*4], data[stride*4],
        data[stride*3], data[stride*3], data[stride*2], data[stride*2], data[stride], data[stride], data[0], data[0]
    );
    
    __m512i volatile random_simd2 = _mm512_set_epi32(
        data[stride*7+16], data[stride*7+16], data[stride*6+16], data[stride*6+16], data[stride*5+16], data[stride*5+16], 
         data[stride*4+16],  data[stride*4+16],  data[stride*3+16],  data[stride*3+16],  data[stride*2+16],  data[stride*2+16], 
         data[stride+16],  data[stride+16],    data[16],  data[16]
    ); 


    __m512i volatile random_simd3 = _mm512_set_epi32(
        data[stride*7+32], data[stride*7+32], data[stride*6+32], data[stride*6+32], data[stride*5+32], data[stride*5+32], 
         data[stride*4+32],  data[stride*4+32],  data[stride*3+32],  data[stride*3+32],  data[stride*2+32],  data[stride*2+32], 
         data[stride+32],  data[stride+32],    data[32],  data[32]
    ); 


    __m512i volatile random_simd4 = _mm512_set_epi32(
        data[stride*7+48], data[stride*7+48], data[stride*6+48], data[stride*6+48], data[stride*5+48], data[stride*5+48], 
         data[stride*4+48],  data[stride*4+48],  data[stride*3+48],  data[stride*3+48],  data[stride*2+48],  data[stride*2+48], 
         data[stride+48],  data[stride+48],    data[48],  data[48]
    ); 


    __m512i volatile random_simd5 = _mm512_set_epi32(
        data[stride*7+64], data[stride*7+64], data[stride*6+64], data[stride*6+64], data[stride*5+64], data[stride*5+64], 
         data[stride*4+64],  data[stride*4+64],  data[stride*3+64],  data[stride*3+64],  data[stride*2+64],  data[stride*2+64], 
         data[stride+64],  data[stride+64],    data[64],  data[64]
    ); 


    __m512i volatile random_simd6 = _mm512_set_epi32(
        data[stride*7+80], data[stride*7+80], data[stride*6+80], data[stride*6+80], data[stride*5+80], data[stride*5+80], 
         data[stride*4+80],  data[stride*4+80],  data[stride*3+80],  data[stride*3+80],  data[stride*2+80],  data[stride*2+80], 
         data[stride+80],  data[stride+80],    data[80],  data[80]
    ); 


    __m512i volatile random_simd7 = _mm512_set_epi32(
        data[stride*7+96], data[stride*7+96], data[stride*6+96], data[stride*6+96], data[stride*5+96], data[stride*5+96], 
         data[stride*4+96],  data[stride*4+96],  data[stride*3+96],  data[stride*3+96],  data[stride*2+96],  data[stride*2+96], 
         data[stride+96],  data[stride+96],    data[96],  data[96]
    ); 


    __m512i volatile random_simd8 = _mm512_set_epi32(
        data[stride*7+112], data[stride*7+112], data[stride*6+112], data[stride*6+112], data[stride*5+112], data[stride*5+112], 
         data[stride*4+112],  data[stride*4+112],  data[stride*3+112],  data[stride*3+112],  data[stride*2+112],  data[stride*2+112], 
         data[stride+112],  data[stride+112],    data[112],  data[112]
    ); 

    for (uint64_t i = 64; i < data_size; i+=16) 
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
}


static void avx512_gather32_stride_4equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m512i volatile random_simd1 = _mm512_set_epi32(
        data[stride*3], data[stride*3], data[stride*3], data[stride*3], data[stride*2], data[stride*2], data[stride*2], data[stride*2],
        data[stride], data[stride], data[stride], data[stride], data[0], data[0], data[0], data[0]
    );
    
    __m512i volatile random_simd2 = _mm512_set_epi32(
        data[stride*3+16], data[stride*3+16], data[stride*3+16], data[stride*3+16], data[stride*2+16], data[stride*2+16], 
         data[stride*2+16],  data[stride*2+16],  data[stride+16],  data[stride+16],  data[stride+16],  data[stride+16], 
         data[16],  data[16],    data[16],  data[16]
    ); 


    __m512i volatile random_simd3 = _mm512_set_epi32(
        data[stride*3+32], data[stride*3+32], data[stride*3+32], data[stride*3+32], data[stride*2+32], data[stride*2+32], 
         data[stride*2+32],  data[stride*2+32],  data[stride+32],  data[stride+32],  data[stride+32],  data[stride+32], 
         data[32],  data[32],    data[32],  data[32]
    ); 


    __m512i volatile random_simd4 = _mm512_set_epi32(
        data[stride*3+48], data[stride*3+48], data[stride*3+48], data[stride*3+48], data[stride*2+48], data[stride*2+48], 
         data[stride*2+48],  data[stride*2+48],  data[stride+48],  data[stride+48],  data[stride+48],  data[stride+48], 
         data[48],  data[48],    data[48],  data[48]
    ); 


    __m512i volatile random_simd5 = _mm512_set_epi32(
        data[stride*3+64], data[stride*3+64], data[stride*3+64], data[stride*3+64], data[stride*2+64], data[stride*2+64], 
         data[stride*2+64],  data[stride*2+64],  data[stride+64],  data[stride+64],  data[stride+64],  data[stride+64], 
         data[64],  data[64],    data[64],  data[64]
    ); 


    __m512i volatile random_simd6 = _mm512_set_epi32(
        data[stride*3+80], data[stride*3+80], data[stride*3+80], data[stride*3+80], data[stride*2+80], data[stride*2+80], 
         data[stride*2+80],  data[stride*2+80],  data[stride+80],  data[stride+80],  data[stride+80],  data[stride+80], 
         data[80],  data[80],    data[80],  data[80]
    ); 


    __m512i volatile random_simd7 = _mm512_set_epi32(
        data[stride*3+96], data[stride*3+96], data[stride*3+96], data[stride*3+96], data[stride*2+96], data[stride*2+96], 
         data[stride*2+96],  data[stride*2+96],  data[stride+96],  data[stride+96],  data[stride+96],  data[stride+96], 
         data[96],  data[96],    data[96],  data[96]
    ); 


    __m512i volatile random_simd8 = _mm512_set_epi32(
        data[stride*3+112], data[stride*3+112], data[stride*3+112], data[stride*3+112], data[stride*2+112], data[stride*2+112], 
         data[stride*2+112],  data[stride*2+112],  data[stride+112],  data[stride+112],  data[stride+112],  data[stride+112], 
         data[112],  data[112],    data[112],  data[112]
    ); 

    
    for (uint64_t i = 64; i < data_size; i+=16) 
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
}


static void avx512_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    (void)stride; 
    __m512i volatile random_simd1 = _mm512_set1_epi32(data[0]);
    __m512i volatile random_simd2 = _mm512_set1_epi32(data[1]);
    __m512i volatile random_simd3 = _mm512_set1_epi32(data[2]);
    __m512i volatile random_simd4 = _mm512_set1_epi32(data[3]);
    __m512i volatile random_simd5 = _mm512_set1_epi32(data[4]);
    __m512i volatile random_simd6 = _mm512_set1_epi32(data[5]);
    __m512i volatile random_simd7 = _mm512_set1_epi32(data[6]);
    __m512i volatile random_simd8 = _mm512_set1_epi32(data[7]);
    for (uint64_t i = 64; i < data_size; i+=16) 
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
}

#endif // __AVX512F__
#endif // AVX512_GATHER_BENCH_H_
