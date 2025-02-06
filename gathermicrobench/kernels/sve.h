#ifdef __ARM_FEATURE_SVE
#include <stdint.h>
#include <arm_sve.h>
#include "kernels_common.h"

svint32_t svset_s32(const int32_t* data, int start, int benchmark, int stride)
{
    int32_t start8 = start << __builtin_ctz(svcntw()); 
    svbool_t mask  = svptrue_b32();
    int32_t buff_2equal[16] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
    int32_t buff_4equal[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    svint32_t start_reg     = svdup_s32(start8);
    svint32_t stride_4equal = svld1_s32(mask, &buff_4equal[0]); 
    svint32_t stride_2equal = svld1_s32(mask, &buff_2equal[0]); 
    switch(benchmark)
    {
        case /*RANDOM*/ 0:        return svld1_s32(mask, &data[start8]); 
        case /*STRIDE*/ 1:        return svindex_s32(start8, stride);
        case /*STRIDE_2EQUAL*/ 2: return svadd_s32_x(mask, stride_2equal, start_reg);
        case /*STRIDE_4EQUAL*/ 3: return svadd_s32_x(mask, stride_4equal, start_reg);
        case /*ALL_SAME*/ 4:      return svdup_s32(start);
        default:                  return svld1_s32(mask, &data[start8]); 
    }
}

void sve_ld1_throughput (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    svbool_t mask    = svptrue_b32();
    size_t num_lanes = svcntw(); 
    svint32_t random_simd1,  random_simd2;
    svint32_t random_simd3,  random_simd4;
    svint32_t random_simd5,  random_simd6;
    svint32_t random_simd7,  random_simd8;
    svint32_t random_simd9,  random_simd10;
    svint32_t random_simd11, random_simd12;
    svint32_t random_simd13;
    int32_t index1  = 0,              index2  = 1  * num_lanes; 
    int32_t index3  = 2  * num_lanes, index4  = 3  * num_lanes; 
    int32_t index5  = 4  * num_lanes, index6  = 5  * num_lanes; 
    int32_t index7  = 6  * num_lanes, index8  = 7  * num_lanes;
    int32_t index9  = 8  * num_lanes, index10 = 9  * num_lanes; 
    int32_t index11 = 10 * num_lanes, index12 = 11 * num_lanes; 
    int32_t index13 = 12 * num_lanes; 
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = svld1_s32(mask, &data[index1]); 
        random_simd2  = svld1_s32(mask, &data[index2]); 
        random_simd3  = svld1_s32(mask, &data[index3]); 
        random_simd4  = svld1_s32(mask, &data[index4]); 
        random_simd5  = svld1_s32(mask, &data[index5]); 
        random_simd6  = svld1_s32(mask, &data[index6]); 
        random_simd7  = svld1_s32(mask, &data[index7]); 
        random_simd8  = svld1_s32(mask, &data[index8]); 
        random_simd9  = svld1_s32(mask, &data[index9]); 
        random_simd10 = svld1_s32(mask, &data[index10]); 
        random_simd11 = svld1_s32(mask, &data[index11]); 
        random_simd12 = svld1_s32(mask, &data[index12]); 
        random_simd13 = svld1_s32(mask, &data[index13]); 
        index1  = svlasta_s32(mask, random_simd1);
        index2  = svlasta_s32(mask, random_simd2);
        index3  = svlasta_s32(mask, random_simd3);
        index4  = svlasta_s32(mask, random_simd4);
        index5  = svlasta_s32(mask, random_simd5);
        index6  = svlasta_s32(mask, random_simd6);
        index7  = svlasta_s32(mask, random_simd7);
        index8  = svlasta_s32(mask, random_simd8);
        index9  = svlasta_s32(mask, random_simd9);
        index10 = svlasta_s32(mask, random_simd10);
        index11 = svlasta_s32(mask, random_simd11);
        index12 = svlasta_s32(mask, random_simd12);
        index13 = svlasta_s32(mask, random_simd13);
    }
    unused(stride); 
    do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13); 
}

void sve_gather32_kernel_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    svbool_t mask = svptrue_b32();
    svint32_t random_simd1  = svset_s32(data,  0, benchmark, stride);
    svint32_t random_simd2  = svset_s32(data,  1, benchmark, stride);
    svint32_t random_simd3  = svset_s32(data,  2, benchmark, stride);
    svint32_t random_simd4  = svset_s32(data,  3, benchmark, stride);
    svint32_t random_simd5  = svset_s32(data,  4, benchmark, stride);
    svint32_t random_simd6  = svset_s32(data,  5, benchmark, stride);
    svint32_t random_simd7  = svset_s32(data,  6, benchmark, stride);
    svint32_t random_simd8  = svset_s32(data,  7, benchmark, stride);
    svint32_t random_simd9  = svset_s32(data,  8, benchmark, stride);
    svint32_t random_simd10 = svset_s32(data,  9, benchmark, stride);
    svint32_t random_simd11 = svset_s32(data, 10, benchmark, stride);
    svint32_t random_simd12 = svset_s32(data, 11, benchmark, stride);
    svint32_t random_simd13 = svset_s32(data, 12, benchmark, stride);
    for (int i = 0 ; i < data_size; i++)
    {
        random_simd1  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd1);
        random_simd2  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd2);
        random_simd3  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd3);
        random_simd4  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd4);
        random_simd5  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd5);
        random_simd6  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd6);
        random_simd7  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd7);
        random_simd8  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd8);
        random_simd9  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd9);
        random_simd10 = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd10);
        random_simd11 = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd11);
        random_simd12 = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd12);
        random_simd13 = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd13);
    }
    unused(stride); 
    do_not_optimize(random_simd1);  do_not_optimize(random_simd2); 
    do_not_optimize(random_simd3);  do_not_optimize(random_simd4); 
    do_not_optimize(random_simd5);  do_not_optimize(random_simd6); 
    do_not_optimize(random_simd7);  do_not_optimize(random_simd8); 
    do_not_optimize(random_simd9);  do_not_optimize(random_simd10); 
    do_not_optimize(random_simd11); do_not_optimize(random_simd12); 
    do_not_optimize(random_simd13);
}

void sve_gather32_kernel_latency(
    const int32_t* const data, 
    const uint64_t data_size,
    const int benchmark,
    const int stride
)
{
    svbool_t mask = svptrue_b32();
    svint32_t random_simd1  = svset_s32(data,  0, benchmark, stride);
    for (int i = 0 ; i < data_size; i++)
    {
        random_simd1  = svld1_gather_s32index_s32(mask, (int32_t*)data, random_simd1);
    }
    unused(stride); 
    do_not_optimize(random_simd1);
}

void sve_ld1_latencty(
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    svbool_t mask = svptrue_b32();
    svint32_t random_simd1;
    int32_t index1 = 0;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        random_simd1  = svld1_s32(mask, &data[index1]); 
        index1  = data[index1];  
        do_not_optimize(random_simd1); 
    }
    unused(stride); 
    do_not_optimize(random_simd1); 
}

void sve_gather32_bench_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) sve_ld1_throughput(data, data_size, stride);
    else                sve_gather32_kernel_throughput(data, data_size, benchmark, stride);
}

void sve_gather32_bench_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    )
{
    if (benchmark == 5) sve_ld1_latencty(data, data_size, stride);
    else                sve_gather32_kernel_latency(data, data_size, benchmark, stride);
}
#endif
