#include <gathermicrobench/generate_data.h>
#include <stdint.h>

void init_rand(unsigned int seed)
{
    srand(seed);
}

void generate_data(int* data, size_t data_size, data_pattern_t pattern)
{
    // TODO
    switch (pattern)
    {
        case RANDOM_UNALIGNED:
            for (size_t i = 0; i < data_size; i++)
            {
                data[i] = (rand() % (int)data_size);
            }
            break;
    }
}

void generate_data_strided(int* data, size_t data_size, size_t stride)
{
}