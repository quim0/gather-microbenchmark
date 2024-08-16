#ifndef AVX2_GATHER_BENCH_H_
#define AVX2_GATHER_BENCH_H_

#include <stdint.h>
#include <immintrin.h>

static __attribute__((always_inline)) void avx2_256_loadu (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    (void)stride; 
    __m256i volatile random_simd1;
    __m256i volatile random_simd2;
    __m256i volatile random_simd3;
    __m256i volatile random_simd4;
    int32_t index1 = 0, index2 = 1, index3 = 2, index4 = 3; 
    for (uint64_t i = 0; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_lddqu_si256((__m256i*)&data[index1]);
        random_simd2 = _mm256_lddqu_si256((__m256i*)&data[index2]);
        random_simd3 = _mm256_lddqu_si256((__m256i*)&data[index3]);
        random_simd4 = _mm256_lddqu_si256((__m256i*)&data[index4]);
        index1 = data[index1]; 
        index2 = data[index2]; 
        index3 = data[index3]; 
        index4 = data[index4]; 
    }
    (void)random_simd1; 
    (void)random_simd2; 
    (void)random_simd3; 
    (void)random_simd4; 
}

static __attribute__((always_inline)) void avx2_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    (void)stride; 
    __m256i volatile random_simd1 = _mm256_lddqu_si256((__m256i*)&data[0]);
    __m256i volatile random_simd2 = _mm256_lddqu_si256((__m256i*)&data[8]);
    __m256i volatile random_simd3 = _mm256_lddqu_si256((__m256i*)&data[16]);
    __m256i volatile random_simd4 = _mm256_lddqu_si256((__m256i*)&data[24]);

    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}

static __attribute__((always_inline)) void avx2_gather32_stride_kernel (
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

    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}

static __attribute__((always_inline)) void avx2_gather32_stride_2equal_kernel (
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

    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}


static __attribute__((always_inline)) void avx2_gather32_stride_4equal_kernel (
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


    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}


static __attribute__((always_inline)) void avx2_gather32_all_same_kernel (
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

    for (uint64_t i = 32; i < data_size; i+=8) 
    {
        random_simd1 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd1, sizeof(int));
        random_simd2 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd2, sizeof(int));
        random_simd3 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd3, sizeof(int));
        random_simd4 = _mm256_i32gather_epi32 ((int const*)&data[0], random_simd4, sizeof(int));
    }
}


#endif // AVX2_GATHER_BENCH_H_