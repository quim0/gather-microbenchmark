#ifndef SCALAR_H_
#define SCALAR_H_

#include <stdint.h>

static __attribute__((always_inline)) void scalar_gather32_kernel (
    const int32_t* const data,
    const uint64_t data_size,
    const int stride
    ) 
{
    (void)stride; 
    int32_t volatile gather1_index1 =  0, gather1_index2 =  1, gather1_index3 =  2, gather1_index4 =  3;
    int32_t volatile gather1_index5 =  4, gather1_index6 =  5, gather1_index7 =  6, gather1_index8 =  7;
    
    int32_t volatile gather2_index1 =  8, gather2_index2 =  9, gather2_index3 = 10, gather2_index4 = 11;
    int32_t volatile gather2_index5 = 12, gather2_index6 = 13, gather2_index7 = 14, gather2_index8 = 15;

    int32_t volatile gather3_index1 = 16, gather3_index2 = 17, gather3_index3 = 18, gather3_index4 = 19;
    int32_t volatile gather3_index5 = 20, gather3_index6 = 21, gather3_index7 = 22, gather3_index8 = 23;

    int32_t volatile gather4_index1 = 24, gather4_index2 = 25, gather4_index3 = 26, gather4_index4 = 27;
    int32_t volatile gather4_index5 = 28, gather4_index6 = 29, gather4_index7 = 30, gather4_index8 = 31;

    for (uint64_t i = 0; i < data_size; i+=8) 
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
    } 
}

#endif