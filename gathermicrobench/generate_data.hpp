#include <gathermicrobench/benchmark_params.hpp>
#include <algorithm>

#define SIZE_12KB   12000
#define SIZE_1MB   1000000
#define SIZE_10MB  10000000
#define SIZE_256MB 256000000


int64_t get_data_size(bench_params_t params)
{
    const memory_size_t  mm_type = params.mm_type;
    const data_bytes_t data_type = params.data_type;
    const int64_t  bytes_of_data = (data_type == INT32_DATA) ? sizeof(int32_t) : sizeof(int64_t);

    switch(mm_type)
    {
        case L1_SIZE: return SIZE_12KB   / bytes_of_data; 
        case L2_SIZE: return SIZE_1MB   / bytes_of_data; 
        case L3_SIZE: return SIZE_10MB  / bytes_of_data; 
        case MM_SIZE: return SIZE_256MB / bytes_of_data; 
        default:      return SIZE_12KB   / bytes_of_data; 
    }
}


void generate_data_random_32bits(int32_t* data, int64_t data_size)
{
    int32_t* data_aux = new int32_t[data_size];
    
    for (int64_t i = 0; i < data_size; i++) 
    {
        data_aux[i] = i;
    }
    
    std::srand(0);
    std::random_shuffle(data_aux, data_aux+data_size);
    
    data[data_aux[data_size-1]] = data_aux[0];
    for (int64_t i = 0; i < data_size-1; i++)
    {
        data[data_aux[i]] = data_aux[i+1];
    }
    
    delete data_aux;
}


void generate_data_stride_32bits(int32_t* data, int64_t data_size, bench_params_t params)
{
    const int stride = params.stride; 
    int stride_size  = stride * 7 + 1; 
    int v_size       = data_size / stride_size; 

    std::vector<std::vector<int32_t>> v_start; 
    std::vector<int32_t> index; 
    
    for (int i = 0; i < v_size; i++)
    {
        std::vector<int32_t> v_aux; 
        for (int j = i * stride_size; j < (i+1)*stride_size; j++){
            v_aux.push_back(j);
        }
        v_start.push_back(v_aux); 
          index.push_back(i);
    }

    std::vector<std::vector<int32_t>> result(v_start); 
    size_t index_size = index.size() - 1; 

    std::srand(0);
    std::random_shuffle(index.begin(), index.end()); 

    result[ index[index_size] ] = v_start[index[0]];  
    for (size_t i = 0; i < index_size; i++)
    {
        result[index[i]] = v_start[index[i+1]];
    }

    for (size_t i = 0; i < result.size(); i++)
    {
        for (size_t j = 0; j < result[0].size(); j++)
        {
            data[i*result[0].size() + j] = result[i][j];
        }
    }
}


void generate_data_random_64bits(int64_t* data, int64_t data_size)
{
    int64_t* data_aux = new int64_t[data_size];
    
    for (int64_t i = 0; i < data_size; i++) 
    {
        data_aux[i] = i;
    }
    
    std::srand(0);
    std::random_shuffle(data_aux, data_aux+data_size);
    
    data[data_aux[data_size-1]] = data_aux[0];
    for (int64_t i = 0; i < data_size-1; i++)
    {
        data[data_aux[i]] = data_aux[i+1];
    }
    
    delete data_aux;
}


void generate_data_stride_64bits(int64_t* data, int64_t data_size, bench_params_t params)
{
    const int stride = params.stride; 
    const int multi  = (params.simd_type == REG_512BIT) ?  15 : (params.simd_type == REG_256BIT) ? 7 : 3; 
    int stride_size  = stride * multi + 1; 
    int v_size       = data_size / stride_size; 

    std::vector<std::vector<int64_t>> v_start; 
    std::vector<int64_t> index; 
    
    for (int i = 0; i < v_size; i++)
    {
        std::vector<int64_t> v_aux; 
        for (int j = i * stride_size; j < (i+1)*stride_size; j++){
            v_aux.push_back(j);
        }
        v_start.push_back(v_aux); 
          index.push_back(i);
    }

    std::vector<std::vector<int64_t>> result(v_start); 
    size_t index_size = index.size() - 1; 

    std::srand(0);
    std::random_shuffle(index.begin(), index.end()); 

    result[ index[index_size] ] = v_start[index[0]];  
    for (size_t i = 0; i < index_size; i++)
    {
        result[index[i]] = v_start[index[i+1]];
    }

    for (size_t i = 0; i < result.size(); i++)
    {
        for (size_t j = 0; j < result[0].size(); j++)
        {
            data[i*result[0].size() + j] = result[i][j];
        }
    }
}


void init_data_32bits(int32_t* data, int64_t data_size, bench_params_t params)
{ 
    switch (params.bench_algo) 
    {
        case RANDOM:           generate_data_random_32bits(data, data_size);         break;
        case STRIDE:           generate_data_stride_32bits(data, data_size, params); break;
        case STRIDE_2EQUAL:    generate_data_stride_32bits(data, data_size, params); break;
        case STRIDE_4EQUAL:    generate_data_stride_32bits(data, data_size, params); break;
        case ALL_SAME:         generate_data_random_32bits(data, data_size);         break;
        case LOAD:             generate_data_random_32bits(data, data_size);         break;
        case SCALAR_RANDOM:    generate_data_random_32bits(data, data_size);         break;
        default:               generate_data_random_32bits(data, data_size);         break;
    }
}


void init_data_64bits(int64_t* data, int64_t data_size, bench_params_t params)
{ 
    switch (params.bench_algo) 
    {
        case RANDOM:           generate_data_random_64bits(data, data_size);         break;
        case STRIDE:           generate_data_stride_64bits(data, data_size, params); break;
        case STRIDE_2EQUAL:    generate_data_stride_64bits(data, data_size, params); break;
        case STRIDE_4EQUAL:    generate_data_stride_64bits(data, data_size, params); break;
        case ALL_SAME:         generate_data_random_64bits(data, data_size);         break;
        case LOAD:             generate_data_random_64bits(data, data_size);         break;
        case SCALAR_RANDOM:    generate_data_random_64bits(data, data_size);         break;
        default:               generate_data_random_64bits(data, data_size);         break;
    }
}
