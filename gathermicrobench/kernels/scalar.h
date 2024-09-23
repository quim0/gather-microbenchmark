#ifndef SCALAR_H_
#define SCALAR_H_

#include <stdint.h>
#include "kernels_common.h"

void scalar_gather32_no_register_spilling_kernel(
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    int32_t gather1_index1 =  0, gather1_index2 =  1, gather1_index3 =  2, gather1_index4 =  3;
    int32_t gather1_index5 =  4, gather1_index6 =  5, gather1_index7 =  6, gather1_index8 =  7; 
    int32_t gather2_index1 =  8, gather2_index2 =  9, gather2_index3 = 10, gather2_index4 = 11;
    for (uint64_t i = 0; i < data_size; i++) 
    {
        //gather1
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        gather1_index5 = data[gather1_index5]; gather1_index6 = data[gather1_index6];
        gather1_index7 = data[gather1_index7]; gather1_index8 = data[gather1_index8];

        //gather2
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        gather1_index2 = data[gather1_index2]; gather2_index1 = data[gather2_index1];
        gather1_index1 = data[gather1_index1]; gather2_index2 = data[gather2_index2];

        //gather3
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        gather1_index5 = data[gather1_index5]; gather1_index6 = data[gather1_index6];
        gather1_index7 = data[gather1_index7]; gather1_index8 = data[gather1_index8];

        //gather4
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        gather1_index2 = data[gather1_index2]; gather2_index1 = data[gather2_index1];
        gather1_index1 = data[gather1_index1]; gather2_index2 = data[gather2_index2];

        //gather5
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        gather1_index5 = data[gather1_index5]; gather1_index6 = data[gather1_index6];
        gather1_index7 = data[gather1_index7]; gather1_index8 = data[gather1_index8];

        //gather6
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        gather1_index2 = data[gather1_index2]; gather2_index1 = data[gather2_index1];
        gather1_index1 = data[gather1_index1]; gather2_index2 = data[gather2_index2];

        //gather7
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        gather1_index5 = data[gather1_index5]; gather1_index6 = data[gather1_index6];
        gather1_index7 = data[gather1_index7]; gather1_index8 = data[gather1_index8];

        //gather8
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        gather1_index2 = data[gather1_index2]; gather2_index1 = data[gather2_index1];
        gather1_index1 = data[gather1_index1]; gather2_index2 = data[gather2_index2];
    }
    unused(stride);
    do_not_optimize_scalar(gather1_index1); do_not_optimize_scalar(gather1_index2);
    do_not_optimize_scalar(gather1_index3); do_not_optimize_scalar(gather1_index4);
    do_not_optimize_scalar(gather1_index5); do_not_optimize_scalar(gather1_index6);
    do_not_optimize_scalar(gather1_index7); do_not_optimize_scalar(gather1_index8);
    do_not_optimize_scalar(gather2_index1); do_not_optimize_scalar(gather2_index2);
    do_not_optimize_scalar(gather2_index3); do_not_optimize_scalar(gather2_index4);
}

static void scalar_gather32_spilling_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    int32_t gather1_index1 =  0, gather1_index2 =  1, gather1_index3 =  2, gather1_index4 =  3;
    int32_t gather1_index5 =  4, gather1_index6 =  5, gather1_index7 =  6, gather1_index8 =  7;
    
    int32_t gather2_index1 =  8, gather2_index2 =  9, gather2_index3 = 10, gather2_index4 = 11;
    int32_t gather2_index5 = 12, gather2_index6 = 13, gather2_index7 = 14, gather2_index8 = 15;

    int32_t gather3_index1 = 16, gather3_index2 = 17, gather3_index3 = 18, gather3_index4 = 19;
    int32_t gather3_index5 = 20, gather3_index6 = 21, gather3_index7 = 22, gather3_index8 = 23;

    int32_t gather4_index1 = 24, gather4_index2 = 25, gather4_index3 = 26, gather4_index4 = 27;
    int32_t gather4_index5 = 28, gather4_index6 = 29, gather4_index7 = 30, gather4_index8 = 31;

    int32_t gather5_index1 = 32, gather5_index2 = 33, gather5_index3 = 34, gather5_index4 = 35;
    int32_t gather5_index5 = 36, gather5_index6 = 37, gather5_index7 = 38, gather5_index8 = 39;
    
    int32_t gather6_index1 = 40, gather6_index2 = 41, gather6_index3 = 42, gather6_index4 = 43;
    int32_t gather6_index5 = 44, gather6_index6 = 45, gather6_index7 = 46, gather6_index8 = 47;
    
    int32_t gather7_index1 = 48, gather7_index2 = 49, gather7_index3 = 50, gather7_index4 = 51;
    int32_t gather7_index5 = 52, gather7_index6 = 53, gather7_index7 = 54, gather7_index8 = 55;
   
    int32_t gather8_index1 = 56, gather8_index2 = 57, gather8_index3 = 58, gather8_index4 = 59;
    int32_t gather8_index5 = 60, gather8_index6 = 61, gather8_index7 = 62, gather8_index8 = 63;


    for (uint64_t i = 0; i < data_size; i++) 
    {
        //gather1
        gather1_index1 = data[gather1_index1]; gather1_index2 = data[gather1_index2];
        gather1_index3 = data[gather1_index3]; gather1_index4 = data[gather1_index4];
        gather1_index5 = data[gather1_index5]; gather1_index6 = data[gather1_index6];
        gather1_index7 = data[gather1_index7]; gather1_index8 = data[gather1_index8];

        //gather2
        gather2_index1 = data[gather2_index1]; gather2_index2 = data[gather2_index2];
        gather2_index3 = data[gather2_index3]; gather2_index4 = data[gather2_index4];
        gather2_index5 = data[gather2_index5]; gather2_index6 = data[gather2_index6];
        gather2_index7 = data[gather2_index7]; gather2_index8 = data[gather2_index8];

        //gather3
        gather3_index1 = data[gather3_index1]; gather3_index2 = data[gather3_index2];
        gather3_index3 = data[gather3_index3]; gather3_index4 = data[gather3_index4];
        gather3_index5 = data[gather3_index5]; gather3_index6 = data[gather3_index6];
        gather3_index7 = data[gather3_index7]; gather3_index8 = data[gather3_index8];

        //gather4
        gather4_index1 = data[gather4_index1]; gather4_index2 = data[gather4_index2];
        gather4_index3 = data[gather4_index3]; gather4_index4 = data[gather4_index4];
        gather4_index5 = data[gather4_index5]; gather4_index6 = data[gather4_index6];
        gather4_index7 = data[gather4_index7]; gather4_index8 = data[gather4_index8];
        
        //gather5
        gather5_index1 = data[gather5_index1]; gather5_index2 = data[gather5_index2];
        gather5_index3 = data[gather5_index3]; gather5_index4 = data[gather5_index4];
        gather5_index5 = data[gather5_index5]; gather5_index6 = data[gather5_index6];
        gather5_index7 = data[gather5_index7]; gather5_index8 = data[gather5_index8];

        //gather6
        gather6_index1 = data[gather6_index1]; gather6_index2 = data[gather6_index2];
        gather6_index3 = data[gather6_index3]; gather6_index4 = data[gather6_index4];
        gather6_index5 = data[gather6_index5]; gather6_index6 = data[gather6_index6];
        gather6_index7 = data[gather6_index7]; gather6_index8 = data[gather6_index8];

        //gather7
        gather7_index1 = data[gather7_index1]; gather7_index2 = data[gather7_index2];
        gather7_index3 = data[gather7_index3]; gather7_index4 = data[gather7_index4];
        gather7_index5 = data[gather7_index5]; gather7_index6 = data[gather7_index6];
        gather7_index7 = data[gather7_index7]; gather7_index8 = data[gather7_index8];

        //gather8
        gather8_index1 = data[gather8_index1]; gather8_index2 = data[gather8_index2];
        gather8_index3 = data[gather8_index3]; gather8_index4 = data[gather8_index4];
        gather8_index5 = data[gather8_index5]; gather8_index6 = data[gather8_index6];
        gather8_index7 = data[gather8_index7]; gather8_index8 = data[gather8_index8];
    } 
    unused(stride);
    do_not_optimize_scalar(gather1_index1); do_not_optimize_scalar(gather1_index2);
    do_not_optimize_scalar(gather1_index3); do_not_optimize_scalar(gather1_index4);
    do_not_optimize_scalar(gather1_index5); do_not_optimize_scalar(gather1_index6);
    do_not_optimize_scalar(gather1_index7); do_not_optimize_scalar(gather1_index8);

    do_not_optimize_scalar(gather2_index1); do_not_optimize_scalar(gather2_index2);
    do_not_optimize_scalar(gather2_index3); do_not_optimize_scalar(gather2_index4);
    do_not_optimize_scalar(gather2_index5); do_not_optimize_scalar(gather2_index6);
    do_not_optimize_scalar(gather2_index7); do_not_optimize_scalar(gather2_index8);

    do_not_optimize_scalar(gather3_index1); do_not_optimize_scalar(gather3_index2);
    do_not_optimize_scalar(gather3_index3); do_not_optimize_scalar(gather3_index4);
    do_not_optimize_scalar(gather3_index5); do_not_optimize_scalar(gather3_index6);
    do_not_optimize_scalar(gather3_index7); do_not_optimize_scalar(gather3_index8);

    do_not_optimize_scalar(gather4_index1); do_not_optimize_scalar(gather4_index2);
    do_not_optimize_scalar(gather4_index3); do_not_optimize_scalar(gather4_index4);
    do_not_optimize_scalar(gather4_index5); do_not_optimize_scalar(gather4_index6);
    do_not_optimize_scalar(gather4_index7); do_not_optimize_scalar(gather4_index8);

    do_not_optimize_scalar(gather5_index1); do_not_optimize_scalar(gather5_index2);
    do_not_optimize_scalar(gather5_index3); do_not_optimize_scalar(gather5_index4);
    do_not_optimize_scalar(gather5_index5); do_not_optimize_scalar(gather5_index6);
    do_not_optimize_scalar(gather5_index7); do_not_optimize_scalar(gather5_index8);

    do_not_optimize_scalar(gather6_index1); do_not_optimize_scalar(gather6_index2);
    do_not_optimize_scalar(gather6_index3); do_not_optimize_scalar(gather6_index4);
    do_not_optimize_scalar(gather6_index5); do_not_optimize_scalar(gather6_index6);
    do_not_optimize_scalar(gather6_index7); do_not_optimize_scalar(gather6_index8);

    do_not_optimize_scalar(gather7_index1); do_not_optimize_scalar(gather7_index2);
    do_not_optimize_scalar(gather7_index3); do_not_optimize_scalar(gather7_index4);
    do_not_optimize_scalar(gather7_index5); do_not_optimize_scalar(gather7_index6);
    do_not_optimize_scalar(gather7_index7); do_not_optimize_scalar(gather7_index8);

    do_not_optimize_scalar(gather8_index1); do_not_optimize_scalar(gather8_index2);
    do_not_optimize_scalar(gather8_index3); do_not_optimize_scalar(gather8_index4);
    do_not_optimize_scalar(gather8_index5); do_not_optimize_scalar(gather8_index6);
    do_not_optimize_scalar(gather8_index7); do_not_optimize_scalar(gather8_index8);
}

#endif