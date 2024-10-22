#ifndef SCALAR_H_
#define SCALAR_H_

#include <stdint.h>
#include "kernels_common.h"

void scalar_kernel_throughput(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    ) 
{
    int32_t gather1_index1, gather1_index2, gather1_index3, gather1_index4;
    int32_t gather2_index1, gather2_index2, gather2_index3, gather2_index4; 
    int32_t gather3_index1, gather3_index2, gather3_index3, gather3_index4;
    switch(benchmark)
    {
        case /*RANDOM*/ 0:
        case /*LOAD*/   5:
        default:
            gather1_index1 = 0; gather1_index2 = 1; gather1_index3 = 2;  gather1_index4 = 3;
            gather2_index1 = 4; gather2_index2 = 5; gather2_index3 = 6;  gather2_index4 = 7;
            gather3_index1 = 8; gather3_index2 = 9; gather3_index3 = 10; gather3_index4 = 11;
            break;
        case /*STRIDE*/ 1:
            gather1_index1 = 0;           gather1_index2 = stride; 
            gather1_index3 = stride*2;    gather1_index4 = stride*3;
            gather2_index1 = 4;           gather2_index2 = 4+stride; 
            gather2_index3 = 4+stride*2;  gather2_index4 = 4+stride*3;
            gather3_index1 = 8;           gather3_index2 = 8+stride; 
            gather3_index3 = 8+stride*2;  gather3_index4 = 8+stride*3;
            break;
        case /*STRIDE_2EQUAL*/ 2:
            gather1_index1 = 0; gather1_index2 = 0; gather1_index3 = 1; gather1_index4 = 1;
            gather2_index1 = 4; gather2_index2 = 4; gather2_index3 = 5; gather2_index4 = 5;
            gather3_index1 = 8; gather3_index2 = 8; gather3_index3 = 9; gather3_index4 = 9;
            break;
        case /*STRIDE_4EQUAL*/ 3:
        case /*ALL_SAME*/ 4:
            gather1_index1 = 0; gather1_index2 = 0; gather1_index3 = 0; gather1_index4 = 0;
            gather1_index1 = 1; gather2_index2 = 1; gather2_index3 = 1; gather2_index4 = 1;
            gather1_index1 = 2; gather3_index2 = 2; gather3_index3 = 2; gather3_index4 = 2;
            break;
    }
    for (uint64_t i = 0; i < data_size; i++) 
    {
        //gather1
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        //gather2
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        //gather3
        gather3_index1 = data[gather3_index1]; gather3_index2 = data[gather3_index2];
        gather3_index3 = data[gather3_index3]; gather3_index4 = data[gather3_index4];
    }
    //gather1
    do_not_optimize_scalar(gather1_index1); do_not_optimize_scalar(gather1_index2);
    do_not_optimize_scalar(gather1_index3); do_not_optimize_scalar(gather1_index4);
    //gather2
    do_not_optimize_scalar(gather2_index1); do_not_optimize_scalar(gather2_index2);
    do_not_optimize_scalar(gather2_index3); do_not_optimize_scalar(gather2_index4);
    //gather3
    do_not_optimize_scalar(gather3_index1); do_not_optimize_scalar(gather3_index2);
    do_not_optimize_scalar(gather3_index3); do_not_optimize_scalar(gather3_index4);
}

void scalar_kernel_latency(
    const int32_t* const data,
    const uint64_t data_size,
    const int benchmark,
    const int stride
    ) 
{
    int32_t gather1_index1 = 0;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        //gather1
        gather1_index1 = data[gather1_index1];
    }
    unused(stride);
    do_not_optimize_scalar(gather1_index1);
}
#endif