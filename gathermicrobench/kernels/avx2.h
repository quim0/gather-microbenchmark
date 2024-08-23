#ifndef AVX2_GATHER_BENCH_H_
#define AVX2_GATHER_BENCH_H_

#include <stdint.h>
#include <immintrin.h>

static void avx2_256_loadu (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i volatile random_simd1, random_simd2;
    __m256i volatile random_simd3, random_simd4;
    __m256i volatile random_simd5, random_simd6;
    __m256i volatile random_simd7, random_simd8;
    int32_t index1 = 0, index2 = 1; 
    int32_t index3 = 2, index4 = 3; 
    int32_t index5 = 4, index6 = 5; 
    int32_t index7 = 6, index8 = 7; 
    for (uint64_t i = 64; i < data_size; i+=8) 
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
    (void)stride; 
    (void)random_simd1; (void)random_simd2; 
    (void)random_simd3; (void)random_simd4; 
    (void)random_simd5; (void)random_simd6; 
    (void)random_simd7; (void)random_simd8; 
}

static void avx2_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    __m256i volatile random_simd1 = _mm256_lddqu_si256((__m256i*)&data[0]);
    __m256i volatile random_simd2 = _mm256_lddqu_si256((__m256i*)&data[8]);
    __m256i volatile random_simd3 = _mm256_lddqu_si256((__m256i*)&data[16]);
    __m256i volatile random_simd4 = _mm256_lddqu_si256((__m256i*)&data[24]);
    __m256i volatile random_simd5 = _mm256_lddqu_si256((__m256i*)&data[32]);
    __m256i volatile random_simd6 = _mm256_lddqu_si256((__m256i*)&data[40]);
    __m256i volatile random_simd7 = _mm256_lddqu_si256((__m256i*)&data[48]);
    __m256i volatile random_simd8 = _mm256_lddqu_si256((__m256i*)&data[56]);
    for (uint64_t i = 64; i < data_size; i+=8) 
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
    (void)stride; 
}

static void avx2_gather32_stride_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    
    __m256i volatile random_simd1 = _mm256_set_epi32(
        data[stride*7], data[stride*6], data[stride*5], data[stride*4], data[stride*3], data[stride*2], data[stride], data[0]
    );
    
    __m256i volatile random_simd2 = _mm256_set_epi32(
        data[stride*7+8], data[stride*6+8], data[stride*5+8], data[stride*4+8], data[stride*3+8], data[stride*2+8], data[stride+8], data[8]
    );
        
    __m256i volatile random_simd3 = _mm256_set_epi32(
        data[stride*7+16], data[stride*6+16], data[stride*5+16], data[stride*4+16], data[stride*3+16], data[stride*2+16], data[stride+16], data[16]
    );
    
    __m256i volatile random_simd4 = _mm256_set_epi32(
        data[stride*7+24], data[stride*6+24], data[stride*5+24], data[stride*4+24], data[stride*3+24], data[stride*2+24], data[stride+24], data[24]
    );

    __m256i volatile random_simd5 = _mm256_set_epi32(
        data[stride*7+32], data[stride*6+32], data[stride*5+32], data[stride*4+32], data[stride*3+32], data[stride*2+32], data[stride+32], data[32]
    );
    
    __m256i volatile random_simd6 = _mm256_set_epi32(
        data[stride*7+40], data[stride*6+40], data[stride*5+40], data[stride*4+40], data[stride*3+40], data[stride*2+40], data[stride+40], data[40]
    );
        
    __m256i volatile random_simd7 = _mm256_set_epi32(
        data[stride*7+48], data[stride*6+48], data[stride*5+48], data[stride*4+48], data[stride*3+48], data[stride*2+48], data[stride+48], data[48]
    );
    
    __m256i volatile random_simd8 = _mm256_set_epi32(
        data[stride*7+56], data[stride*6+56], data[stride*5+56], data[stride*4+56], data[stride*3+56], data[stride*2+56], data[stride+56], data[56]
    );

    for (uint64_t i = 64; i < data_size; i+=8) 
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
}

static void avx2_gather32_stride_2equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{

    __m256i volatile random_simd1 = _mm256_set_epi32(
        data[stride*3], data[stride*3], data[stride*2], data[stride*2], data[stride], data[stride], data[0], data[0]
    );
    
    __m256i volatile random_simd2 = _mm256_set_epi32(
        data[stride*3+8], data[stride*3+8], data[stride*2+8], data[stride*2+8], data[stride+8], data[stride+8], data[8], data[8]
    );
        
    __m256i volatile random_simd3 = _mm256_set_epi32(
        data[stride*3+16], data[stride*3+16], data[stride*2+16], data[stride*2+16], data[stride+16], data[stride+16], data[16], data[16]
    );
    
    __m256i volatile random_simd4 = _mm256_set_epi32(
        data[stride*3+24], data[stride*3+24], data[stride*2+24], data[stride*2+24], data[stride+24], data[stride+24], data[24], data[24]
    );
    
    __m256i volatile random_simd5 = _mm256_set_epi32(
        data[stride*3+32], data[stride*3+32], data[stride*2+32], data[stride*2+32], data[stride+32], data[stride+32], data[32], data[32]
    );

    __m256i volatile random_simd6 = _mm256_set_epi32(
        data[stride*3+40], data[stride*3+40], data[stride*2+40], data[stride*2+40], data[stride+40], data[stride+40], data[40], data[40]
    );

    __m256i volatile random_simd7 = _mm256_set_epi32(
        data[stride*3+48], data[stride*3+48], data[stride*2+48], data[stride*2+48], data[stride+48], data[stride+48], data[48], data[48]
    );

     __m256i volatile random_simd8 = _mm256_set_epi32(
        data[stride*3+56], data[stride*3+56], data[stride*2+56], data[stride*2+56], data[stride+56], data[stride+56], data[56], data[56]
    );

    for (uint64_t i = 64; i < data_size; i+=8) 
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
}


static void avx2_gather32_stride_4equal_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    __m256i volatile random_simd1 = _mm256_set_epi32(
        data[stride], data[stride], data[stride], data[stride], data[0], data[0], data[0], data[0]
    );
    
    __m256i volatile random_simd2 = _mm256_set_epi32(
        data[stride+8], data[stride+8], data[stride+8], data[stride+8], data[8], data[8], data[8], data[8]
    );
        
    __m256i volatile random_simd3 = _mm256_set_epi32(
        data[stride+16], data[stride+16], data[stride+16], data[stride+16], data[16], data[16], data[16], data[16]
    );
    
    __m256i volatile random_simd4 = _mm256_set_epi32(
        data[stride+24], data[stride+24], data[stride+24], data[stride+24], data[24], data[24], data[24], data[24]
    );

    __m256i volatile random_simd5 = _mm256_set_epi32(
        data[stride+32], data[stride+32], data[stride+32], data[stride+32], data[32], data[32], data[32], data[32]
    );

    __m256i volatile random_simd6 = _mm256_set_epi32(
        data[stride+40], data[stride+40], data[stride+40], data[stride+40], data[40], data[40], data[40], data[40]
    );

    __m256i volatile random_simd7 = _mm256_set_epi32(
        data[stride+48], data[stride+48], data[stride+48], data[stride+48], data[48], data[48], data[48], data[48]
    );

    __m256i volatile random_simd8 = _mm256_set_epi32(
        data[stride+56], data[stride+56], data[stride+56], data[stride+56], data[56], data[56], data[56], data[56]
    );
    
    for (uint64_t i = 64; i < data_size; i+=8) 
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
}


static void avx2_gather32_all_same_kernel (
    const int32_t* const data,
    const uint64_t data_size, 
    const int stride
    ) 
{
    (void)stride; 
    __m256i volatile random_simd1 = _mm256_set1_epi32(data[0]);
    __m256i volatile random_simd2 = _mm256_set1_epi32(data[1]);
    __m256i volatile random_simd3 = _mm256_set1_epi32(data[2]);
    __m256i volatile random_simd4 = _mm256_set1_epi32(data[3]);
    __m256i volatile random_simd5 = _mm256_set1_epi32(data[4]);
    __m256i volatile random_simd6 = _mm256_set1_epi32(data[5]);
    __m256i volatile random_simd7 = _mm256_set1_epi32(data[6]);
    __m256i volatile random_simd8 = _mm256_set1_epi32(data[7]);
    for (uint64_t i = 64; i < data_size; i+=8) 
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
}


#endif // AVX2_GATHER_BENCH_H_
