
#include <cmath>

#include "utilities.h"
#include "states.h"
#include "hash.h"

///////////////////////////////////////////////////////////////////////////////////////////////

// FNV-1a hash (32bit)
__host__ __device__ uint32_t hash1(const u_char *__restrict__ data, int numBytes)
{
    uint32_t hash = 0x811C9DC5;
    while (numBytes--)
        hash = (hash ^ (*data++)) * 0x01000193;
    return hash;
}

// FNV-1 hash (32bit)
__host__ __device__ uint32_t hash2(const u_char *__restrict__ data, int numBytes)
{
    uint32_t hash = 0x811C9DC5;
    while (numBytes--)
        hash = (hash * 0x01000193) ^ (*data++);
    return hash;
}

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void fillTable_kernel(uint32_t *__restrict__ data, int len, uint32_t value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len)
        data[tid] = value;
}

__host__ void fillTable(uint32_t *data, int len, uint32_t value)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, len);
    const int block_num = int(std::ceil(len / float(thread_num)));
    fillTable_kernel<<<block_num, thread_num>>>(data, len, value);
    CUDASYNC();
}

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initTable_kernel(const bool *__restrict__ src, uint32_t *__restrict__ table, u_char *__restrict__ temp, status *__restrict__ vec,
                                 int states_num, int state_len, int uchar_len,
                                 int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < states_num)
    {
        vec[tid] = try_insert(&src[tid * state_len], state_len,
                              table,
                              &temp[tid * uchar_len], uchar_len,
                              TABLE_SIZE_MASK, SKIP_LENGTH, HASH2_ZERO);
    }
}

__host__ void initTable(bool *src, uint32_t *table, status *vec,
                             int states_num, int state_len, int uchar_len,
                             int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO)
{
    u_char *temp;
    gpuErrchk(cudaMalloc(&temp, uchar_len * states_num * sizeof(u_char)));

    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, states_num);
    const int block_num = int(std::ceil(states_num / float(thread_num)));
    
    initTable_kernel<<<block_num, thread_num>>>(src, table, temp, vec,
                                                states_num, state_len, uchar_len,
                                                TABLE_SIZE_MASK, SKIP_LENGTH, HASH2_ZERO);
    
    CUDASYNC();
    gpuErrchk(cudaFree(temp));
}

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void statesCpy_kernel(const bool *__restrict__ src, FourBytes *__restrict__ dst, const int *__restrict__ valid_idx,
                                 int valid_num, int state_len, int fourbytes_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / state_len;
    int j = tid % state_len;

    if (i < valid_num)
    {
        atomicOr(dst + (i * fourbytes_len) + (j / 32), 
                 (FourBytes(src[valid_idx[i] * state_len + j]) << (j % 32)));
    }
}