#include <inttypes.h>
#include <stdlib.h>
#include <immintrin.h>


void static inline __attribute__((always_inline)) avx2_256_loadu (
    int const* data, 
    uint64_t data_size
    ); 


void static inline __attribute__((always_inline)) avx2_gather32_random_u (
    int const* data,
    uint64_t data_size
    ); 


void static inline __attribute__((always_inline)) avx2_gather32_random (
    int const* data,
    uint64_t data_size
    ); 


void static inline __attribute__((always_inline)) avx2_gather32_stride (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    );


void static inline __attribute__((always_inline)) avx2_gather32_same_index (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    ); 


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_1 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    );


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_2 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    );


void static inline __attribute__((always_inline)) avx2_gather32_stride_field_4 (
    int const* data, 
    __m256i vindex, 
    uint64_t data_size
    );

