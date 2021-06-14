#ifndef __STATES_H__
#define __STATES_H__

#include <cstring>
#include <cmath>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <torch/extension.h>

#include "utilities.h"
#include "facedata.h"

#include "hash.h"

__host__ void fillTable(uint32_t *data, int len, uint32_t value);

__host__ void initTable(bool *src, uint32_t *table, status *vec,
                        int states_num, int state_len, int uchar_len,
                        int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO);

__global__ void statesCpy_kernel(const bool *__restrict__ src, FourBytes *__restrict__ dst, const int *__restrict__ valid_idx,
                                 int valid_num, int state_len, int fourbytes_len);

//

struct statesCpyFunc
{
    const status *query;
    __host__ __device__ statesCpyFunc(const status *query_) : query(query_) {}
    __host__ __device__ bool operator()(const int x)
    {
        return query[x] == SUCCESS;
    }
};

//

template <typename FloatType>
__global__ void pointsCpy_kernel(const FloatType *__restrict__ src, FACEDATA<FloatType> *__restrict__ dst, const int *__restrict__ valid_idx, int valid_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < valid_num)
    {
        dst[tid] = FACEDATA<FloatType>(src + valid_idx[tid] * 3);
    }
}

//

template <typename FloatType>
__global__ void statesDecompress_kernel(FloatType *__restrict__ dst, const FourBytes *__restrict__ src, int state_len, int fourbytes_len, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / state_len;
    int j = tid % state_len;

    if (i < num)
    {
        dst[tid] = FloatType((src[fourbytes_len * i + (j / 32)] >> (j % 32)) & FourBytes(1));
    }
}

template <typename FloatType>
__global__ void statesCompress_kernel(FourBytes *__restrict__ dst, const FloatType *__restrict__ src, int state_len, int fourbytes_len, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / state_len;
    int j = tid % state_len;

    if (i < num)
    {
        atomicOr(dst + (i * fourbytes_len) + (j / 32), FourBytes(src[tid]) << (j % 32));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
class STATES
{
public:
    FourBytes *buffer = nullptr; // compressed states
    FloatType *decoded_buffer = nullptr;
    FACEDATA<FloatType> *facesdata = nullptr;
    uint32_t *table = nullptr; // HashSet

    int uchar_len = 0;
    int fourbytes_len = 0;

    int states_num = 0;     // the num of states
    int state_len = 0;      // the size of one state
    int states_max_num = 0; // the capacity of buffer
    uint32_t HASH2_ZERO = 0;

    int BATCH_SIZE_MAX;
    const int BASE_NUM = _STATES_BASE_NUM;
    const int INCREASE = _STATES_INCREASE;
    const int TABLE_SIZE = _STATES_TABLE_SIZE;   // size of hash table, must be power of 2
    const int SKIP_LENGTH = _STATES_SKIP_LENGTH; // skip size of hash table

    __host__ void init(int state_len_, int BATCH_SIZE_MAX_);
    __host__ void reinit(const torch::Tensor &states_, const torch::Tensor &points_);

    __host__ thrust::device_vector<status> table_cpy(bool *src, uint32_t *dst);
    __host__ void states_cpy(bool *src, FourBytes *dst, thrust::device_vector<int> &valid_idx, int valid_num);
    __host__ void points_cpy(FloatType *points_ptr, FACEDATA<FloatType> *facesdata_ptr, thrust::device_vector<int> &valid_idx, int valid_num);
    __host__ void get_hash2_zero(uint32_t &value);

    __host__ void pop_back(FloatType *&state, FACEDATA<FloatType> *&face, int num);
    __host__ void push_back(FloatType *state, FACEDATA<FloatType> *facedata, int num);
    __host__ void cudaFreeMemory();
};

////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void STATES<FloatType>::init(int state_len_, int BATCH_SIZE_MAX_) // init memory
{
    if (buffer)
        gpuErrchk(cudaFree(buffer));
    if (decoded_buffer)
        gpuErrchk(cudaFree(decoded_buffer));
    if (facesdata)
        gpuErrchk(cudaFree(facesdata));
    if (table)
        gpuErrchk(cudaFree(table));

    state_len = state_len_;
    fourbytes_len = int(std::ceil(state_len / 32.0));
    uchar_len = int(std::ceil(state_len / 8.0));
    states_max_num = BASE_NUM;
    gpuErrchk(cudaMalloc(&buffer, states_max_num * fourbytes_len * sizeof(FourBytes)));
    gpuErrchk(cudaMalloc(&decoded_buffer, BATCH_SIZE_MAX_ * state_len * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&facesdata, states_max_num * sizeof(FACEDATA<FloatType>)));
    gpuErrchk(cudaMalloc(&table, TABLE_SIZE * sizeof(uint32_t)));

    get_hash2_zero(HASH2_ZERO);

    BATCH_SIZE_MAX = BATCH_SIZE_MAX_;
}

template <typename FloatType>
__host__ void STATES<FloatType>::reinit(const torch::Tensor &states_, const torch::Tensor &points_)
{
    if (int(states_.size(0)) > states_max_num)
    {
        int offset = 0;
        while ((1 << offset) < int(states_.size(0)))
            offset++;
        states_max_num = MAX2(BASE_NUM, int(1 << offset));
        if (buffer)
            gpuErrchk(cudaFree(buffer));
        if (facesdata)
            gpuErrchk(cudaFree(facesdata));
        gpuErrchk(cudaMalloc(&buffer, states_max_num * fourbytes_len * sizeof(FourBytes)));
        gpuErrchk(cudaMalloc(&facesdata, states_max_num * sizeof(FACEDATA<FloatType>)));
    }

    states_num = int(states_.size(0));

    // NOTE: it will remove duplicated states
    auto ret = table_cpy(states_.data_ptr<bool>(), table); // init hash table
    int valid_num = thrust::reduce(ret.begin(), ret.end(), 0, thrust::plus<status>());
    thrust::device_vector<int> idx(states_num);
    thrust::device_vector<int> valid_idx(valid_num);
    thrust::sequence(idx.begin(), idx.end());
    thrust::copy_if(idx.begin(), idx.end(), valid_idx.begin(), statesCpyFunc(thrust::raw_pointer_cast(ret.data())));
    states_cpy(states_.data_ptr<bool>(), buffer, valid_idx, valid_num);         // copy to state buffer
    points_cpy(points_.data_ptr<FloatType>(), facesdata, valid_idx, valid_num); // copy to face buffer
    states_num = valid_num;
}

template <typename FloatType>
__host__ thrust::device_vector<status> STATES<FloatType>::table_cpy(bool *src, uint32_t *dst) // they are on GPU
{
    fillTable(dst, TABLE_SIZE, HASH2_ZERO);
    thrust::device_vector<status> vec(states_num);
    initTable(src, dst, thrust::raw_pointer_cast(vec.data()),
              states_num, state_len, uchar_len,
              TABLE_SIZE - 1, SKIP_LENGTH, HASH2_ZERO);
    return vec;
}

template <typename FloatType>
__host__ void STATES<FloatType>::states_cpy(bool *src, FourBytes *dst, thrust::device_vector<int> &valid_idx, int valid_num) // they are on GPU
{
    const int total_num = state_len * valid_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaMemset(dst, 0, valid_num * fourbytes_len * sizeof(FourBytes));
    statesCpy_kernel<<<block_num, thread_num>>>(src, dst, thrust::raw_pointer_cast(valid_idx.data()), valid_num, state_len, fourbytes_len);
    CUDASYNC();
}

template <typename FloatType>
__host__ void STATES<FloatType>::points_cpy(FloatType *points_ptr, FACEDATA<FloatType> *facesdata_ptr, thrust::device_vector<int> &valid_idx, int valid_num) // they are on GPU
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, valid_num);
    const int block_num = int(std::ceil(valid_num / float(thread_num)));
    pointsCpy_kernel<<<block_num, thread_num>>>(points_ptr, facesdata_ptr, thrust::raw_pointer_cast(valid_idx.data()), valid_num);
    CUDASYNC();
}

template <typename FloatType>
__host__ void STATES<FloatType>::get_hash2_zero(uint32_t &value)
{
    u_char *host_uchar_state = new u_char[uchar_len];
    std::memset(host_uchar_state, 0, uchar_len * sizeof(u_char)); // it is impossible that all neurons are deactivated
    value = hash2(host_uchar_state, uchar_len);
    delete[] host_uchar_state;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void STATES<FloatType>::pop_back(FloatType *&state, FACEDATA<FloatType> *&face, int num = 1)
{
    if (num <= BATCH_SIZE_MAX)
    {
        states_num -= num;

        const int total_num = state_len * num;
        const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
        const int block_num = int(std::ceil(total_num / float(thread_num)));
        statesDecompress_kernel<<<block_num, thread_num>>>(decoded_buffer, buffer + states_num * fourbytes_len,
                                                           state_len, fourbytes_len, num);
        CUDASYNC();

        state = decoded_buffer;
        face = facesdata + states_num; // an offset pointer
    }
    else
    {
        state = nullptr;
        face = nullptr;
        std::cout << "Error: The num of pop-outs exceeds `BATCH_SIZE_MAX`! (In function `pop_back`)" << std::endl;
    }
}

template <typename FloatType>
__host__ void STATES<FloatType>::push_back(FloatType *state, FACEDATA<FloatType> *facedata, int num = 1)
{
    if (states_max_num - states_num < num)
    {
        FourBytes *temp_buffer;
        gpuErrchk(cudaMalloc(&temp_buffer, states_max_num * fourbytes_len * sizeof(FourBytes)));
        gpuErrchk(cudaMemcpy(temp_buffer, buffer, states_max_num * fourbytes_len * sizeof(FourBytes), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaFree(buffer));
        int inc = INCREASE;
        while (inc * states_max_num - states_num < num)
            inc *= INCREASE;
        gpuErrchk(cudaMalloc(&buffer, inc * states_max_num * fourbytes_len * sizeof(FourBytes)));
        gpuErrchk(cudaMemcpy(buffer, temp_buffer, states_max_num * fourbytes_len * sizeof(FourBytes), cudaMemcpyDeviceToDevice));
        cudaFree(temp_buffer);
        //
        FACEDATA<FloatType> *temp_facesdata;
        gpuErrchk(cudaMalloc(&temp_facesdata, states_max_num * sizeof(FACEDATA<FloatType>)));
        gpuErrchk(cudaMemcpy(temp_facesdata, facesdata, states_max_num * sizeof(FACEDATA<FloatType>), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaFree(facesdata));
        gpuErrchk(cudaMalloc(&facesdata, inc * states_max_num * sizeof(FACEDATA<FloatType>)));
        gpuErrchk(cudaMemcpy(facesdata, temp_facesdata, states_max_num * sizeof(FACEDATA<FloatType>), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaFree(temp_facesdata));
        //
        states_max_num *= inc;
    }

    const int total_num = state_len * num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaMemset(buffer + states_num * fourbytes_len, 0, num * fourbytes_len * sizeof(FourBytes));
    statesCompress_kernel<<<block_num, thread_num>>>(buffer + states_num * fourbytes_len, state,
                                                     state_len, fourbytes_len, num);
    CUDASYNC();

    gpuErrchk(cudaMemcpy(facesdata + states_num, facedata, num * sizeof(FACEDATA<FloatType>), cudaMemcpyDeviceToDevice));
    states_num += num;
}

template <typename FloatType>
__host__ void STATES<FloatType>::cudaFreeMemory()
{
    if (buffer)
        gpuErrchk(cudaFree(buffer));
    if (decoded_buffer)
        gpuErrchk(cudaFree(decoded_buffer));
    if (facesdata)
        gpuErrchk(cudaFree(facesdata));
    if (table)
        gpuErrchk(cudaFree(table));

    buffer = nullptr;
    decoded_buffer = nullptr;
    facesdata = nullptr;
    table = nullptr;
}

#endif