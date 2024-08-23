#ifndef SSE_GATHER_BENCH_H_
#define SSE_GATHER_BENCH_H_

#include <stdint.h>
#include <immintrin.h>

static void sse_128_loadu_32 (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m128i volatile random_simd1, random_simd2;
    __m128i volatile random_simd3, random_simd4;
    __m128i volatile random_simd5, random_simd6;
    __m128i volatile random_simd7, random_simd8;
    int32_t index1 = 0, index2 = 1;
    int32_t index3 = 2, index4 = 3; 
    int32_t index5 = 4, index6 = 5;
    int32_t index7 = 6, index8 = 7;
    for (uint64_t i = 64; i < data_size; i+=4) 
    {
        random_simd1 = _mm_lddqu_si128((__m128i*)&data[index1]);
        random_simd2 = _mm_lddqu_si128((__m128i*)&data[index2]);
        random_simd3 = _mm_lddqu_si128((__m128i*)&data[index3]);
        random_simd4 = _mm_lddqu_si128((__m128i*)&data[index4]);
        random_simd5 = _mm_lddqu_si128((__m128i*)&data[index5]);
        random_simd6 = _mm_lddqu_si128((__m128i*)&data[index6]);
        random_simd7 = _mm_lddqu_si128((__m128i*)&data[index7]);
        random_simd8 = _mm_lddqu_si128((__m128i*)&data[index8]);
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

static void sse_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m128i volatile random_simd1 = _mm_lddqu_si128((__m128i*)&data[0]);
    __m128i volatile random_simd2 = _mm_lddqu_si128((__m128i*)&data[8]);
    __m128i volatile random_simd3 = _mm_lddqu_si128((__m128i*)&data[16]);
    __m128i volatile random_simd4 = _mm_lddqu_si128((__m128i*)&data[24]);
    __m128i volatile random_simd5 = _mm_lddqu_si128((__m128i*)&data[32]);
    __m128i volatile random_simd6 = _mm_lddqu_si128((__m128i*)&data[40]);
    __m128i volatile random_simd7 = _mm_lddqu_si128((__m128i*)&data[48]);
    __m128i volatile random_simd8 = _mm_lddqu_si128((__m128i*)&data[56]);
    for (uint64_t i = 32; i < data_size; i+=4) 
    {
        random_simd1 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
    (void)stride; 
}

static void sse_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    
    __m128i volatile random_simd1 = _mm_set_epi32(
        data[stride*3], data[stride*2], data[stride], data[0]
    );
    
    __m128i volatile random_simd2 = _mm_set_epi32(
        data[stride*3+8], data[stride*2+8], data[stride+8], data[8]
    );
        
    __m128i volatile random_simd3 = _mm_set_epi32(
        data[stride*3+16], data[stride*2+16], data[stride+16], data[16]
    );
    
    __m128i volatile random_simd4 = _mm_set_epi32(
        data[stride*3+24], data[stride*2+24], data[stride+24], data[24]
    );

    __m128i volatile random_simd5 = _mm_set_epi32(
        data[stride*3+32], data[stride*2+32], data[stride+32], data[32]
    );

    __m128i volatile random_simd6 = _mm_set_epi32(
        data[stride*3+40], data[stride*2+40], data[stride+40], data[40]
    );

    __m128i volatile random_simd7 = _mm_set_epi32(
        data[stride*3+48], data[stride*2+48], data[stride+48], data[48]
    );

    __m128i volatile random_simd8 = _mm_set_epi32(
        data[stride*3+56], data[stride*2+56], data[stride+56], data[56]
    );

    for (uint64_t i = 64; i < data_size; i+=4) 
    {
        random_simd1 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
}

static void sse_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{

    __m128i volatile random_simd1 = _mm_set_epi32(
        data[stride], data[stride], data[0], data[0]
    );
    
    __m128i volatile random_simd2 = _mm_set_epi32(
    data[stride+8], data[stride+8], data[8], data[8]
    );
        
    __m128i volatile random_simd3 = _mm_set_epi32(
        data[stride+16], data[stride+16], data[16], data[16]
    );
    
    __m128i volatile random_simd4 = _mm_set_epi32(
        data[stride+24], data[stride+24], data[24], data[24]
    );

    __m128i volatile random_simd5 = _mm_set_epi32(
        data[stride+32], data[stride+32], data[32], data[32]
    );
    
    __m128i volatile random_simd6 = _mm_set_epi32(
       data[stride+40], data[stride+40], data[40], data[40]
    );
        
    __m128i volatile random_simd7 = _mm_set_epi32(
        data[stride+48], data[stride+48], data[48], data[48]
    );
    
    __m128i volatile random_simd8 = _mm_set_epi32(
        data[stride+56], data[stride+56], data[56], data[56]
    );

    for (uint64_t i = 64; i < data_size; i+=4) 
    {
        random_simd1 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
}

static void sse_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    (void)stride; 

    __m128i volatile random_simd1 = _mm_set1_epi32(data[0]);
    __m128i volatile random_simd2 = _mm_set1_epi32(data[1]);
    __m128i volatile random_simd3 = _mm_set1_epi32(data[2]);
    __m128i volatile random_simd4 = _mm_set1_epi32(data[3]);
    __m128i volatile random_simd5 = _mm_set1_epi32(data[4]);
    __m128i volatile random_simd6 = _mm_set1_epi32(data[5]);
    __m128i volatile random_simd7 = _mm_set1_epi32(data[6]);
    __m128i volatile random_simd8 = _mm_set1_epi32(data[7]);

    for (uint64_t i = 64; i < data_size; i+=4) 
    {
        random_simd1 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
        random_simd5 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd5, sizeof(int));
        random_simd6 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd6, sizeof(int));
        random_simd7 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd7, sizeof(int));
        random_simd8 = _mm_i32gather_epi32 ((int const*)&data[0], random_simd8, sizeof(int));
    }
}


#endif // SSE_GATHER_BENCH_H_