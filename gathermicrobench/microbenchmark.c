// TODO: MIT
#include <gathermicrobench/microarch.h>
#include <gathermicrobench/generate_data.h>

#include <gathermicrobench/kernels/avx2.h>
#include <gathermicrobench/kernels/sse.h>

#include <stdio.h>

// https://github.com/jmuehlig/perf-cpp

int main (int argc, char* argv[]) 
{
    //args 
    const int data_size = L1_CACHE_SIZE;
    const int iters     = 1000; 

    init_rand(0);

    // Generate data
    int* data = aligned_alloc(sizeof(int), data_size * sizeof(int));
    if (data == NULL) 
    {
        printf("Error: malloc failed\n");
        return -1; 
    }

    generate_data(data, data_size, RANDOM_UNALIGNED);

    // start counters
    for (int i = 0; i < iters; i++)
    { 
        avx2_gather32_kernel(data, data_size);
    }
    // stop counters

    free(data);
    return 0; 
}

