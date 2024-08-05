#ifndef GENERATE_DATA_H_
#define GENERATE_DATA_H_

#include <stdlib.h>

typedef enum {
    RANDOM_ALIGNED, // Random pattern algined to the data type
    RANDOM_UNALIGNED
    // ...
} data_pattern_t;

void init_rand(unsigned int seed);
void generate_data(int* data, size_t data_size, data_pattern_t pattern);
void generate_data_strided(int* data, size_t data_size, size_t stride);

#endif // GENERATE_DATA_H_
