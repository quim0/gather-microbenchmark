
#include "kernels/avx2_256_32bits.h"
#include "kernels/sse_128_32bits.h"

int main (int argc, char* argv[]) 
{
    
    //args 
    const int data_size = atoi(argv[1]); 
    const int iters     = 10; 

    //inicializacion data
    int* data = (int*)malloc(data_size * sizeof(int)); 
    for (int i = 0; i < data_size; i++) {   data[i] = i;  }


    //microbenchmark
    __m256i vindex; 
    for (int i = 0; i < iters; i++)
    { 
    }

    return 0; 
}

