
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "utilities.h"


__host__ void cudaMemcpyStrided_D2D(FourBytes *dst, size_t stride_in_4bytes, int repeat_num, const FourBytes *src, size_t _4bytes_num);

__host__ void thrustSortBatched_fill(int *segments, int *key_vec, int one_batch_len, int batch_num);

__host__ void cudaInitCntX(int *cnt_x1, int *num, int *startIdx, int startIdx_stride, int batch_size);

__host__ void cudaInitCntX2(int *cnt_x2, int *num, int old_num, int *cnt_x1, int *x_bs_buffer, int batch_size, int x_max);

__host__ void cudaUpdateStartIdx(int *startIdx, int startIdx_stride, int *cnt_x2, int num);

__host__ void cudaInitCntX4(int *cnt_x2, int old_num, int *cnt_x1, int *num,int *bs_buffer1, int *bs_buffer2, int *bs_buffer3, int *bs_buffer4, int batch_size, int state_len);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    copy src to dst (both are on device)
        sources are from the same place: src
        destinations are to many places: dst (offset by stride_in_4bytes)
*/

__global__ void cudaMemcpyStrided_D2D_kernel(FourBytes*__restrict__ dst, int stride_in_4bytes, int repeat_num, const FourBytes* __restrict__ src, int _4bytes_num, int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / _4bytes_num;
    int j = tid - _4bytes_num * i;

    if(tid < total_num)
        dst[stride_in_4bytes * i + j] = src[j]; // copy to
}

__host__ void cudaMemcpyStrided_D2D(FourBytes* dst, size_t stride_in_4bytes, int repeat_num, const FourBytes* src, size_t _4bytes_num)
{   
    const int total_num = int(_4bytes_num) * repeat_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int( std::ceil(total_num / float(thread_num)) );
    cudaMemcpyStrided_D2D_kernel<<<block_num, thread_num>>>(dst, stride_in_4bytes, repeat_num, src, _4bytes_num, total_num);
    CUDASYNC();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    sort in batch
*/

__global__ void thrustSortBatched_fill_kernel(int* __restrict__ segments, int* __restrict__ key_vec, int one_batch_len, int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < total_num)
    {
        segments[tid] = tid / one_batch_len;
        key_vec[tid]  = tid % one_batch_len;
    }
}

__host__ void thrustSortBatched_fill(int* segments, int* key_vec, int one_batch_len, int batch_num)
{
    const int total_num = one_batch_len * batch_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int( std::ceil(total_num / float(thread_num)) );
    thrustSortBatched_fill_kernel <<<block_num, thread_num>>>(segments, key_vec, one_batch_len, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cudaUpdateStartIdx_kernel(int *__restrict__ startIdx, int startIdx_stride,
                                          const int *__restrict__ cnt_x2, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (num - 1)

    if (tid < num)
    {
        if (cnt_x2[5 * tid + 4] == 1) // okey
        {
            startIdx[cnt_x2[5 * tid + 0] * startIdx_stride] = cnt_x2[5 * tid + 2]; // i
        }
    }
}

__host__ void cudaUpdateStartIdx(int *startIdx, int startIdx_stride,
                                 int *cnt_x2, int num)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, num);
    const int block_num = int(std::ceil(num / float(thread_num)));
    cudaUpdateStartIdx_kernel<<<block_num, thread_num>>>(startIdx, startIdx_stride, cnt_x2, num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cudaInitCntX21_kernel(const int *__restrict__ cnt_x2, int old_num,
                                      int *__restrict__ x_bs_buffer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (old_num - 1)
    int state_idx = cnt_x2[5 * tid + 0];
    int x = cnt_x2[5 * tid + 1];

    if (tid < old_num)
    {
        atomicMax(x_bs_buffer + state_idx, x);
    }
}

__global__ void cudaInitCntX22_kernel(const int *__restrict__ cnt_x2, int old_num,
                                     int *__restrict__ x_bs_buffer, int x_max)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (old_num - 1)
    int state_idx = cnt_x2[5 * tid + 0];
    int x = cnt_x2[5 * tid + 1];

    if (tid < old_num && (cnt_x2[5 * tid + 4] == 1 || x > x_max))
        x_bs_buffer[state_idx] = -1;
}

__global__ void cudaInitCntX23_kernel(int *__restrict__ cnt_x1, int *__restrict__ num,
                                      const int *__restrict__ x_bs_buffer, int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (batch_size - 1)

    if (tid < batch_size)
    {
        int x = x_bs_buffer[tid];
        if (x != -1)
        {
            int old = atomicAdd(num, 1);
            cnt_x1[5 * old + 0] = tid;
            cnt_x1[5 * old + 1] = x + 1; // next one
        }
    }
}

__host__ void cudaInitCntX2(int *cnt_x2, int *num, int old_num,
                            int *cnt_x1,
                            int *x_bs_buffer, int batch_size, int x_max)
{
    thrust::device_ptr<int> dev_ptr(x_bs_buffer);
    thrust::fill(dev_ptr, dev_ptr + batch_size, -1);

    int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, old_num);
    int block_num = int(std::ceil(old_num / float(thread_num)));
    cudaInitCntX21_kernel<<<block_num, thread_num>>>(cnt_x2, old_num, x_bs_buffer);
    CUDASYNC();

    thread_num = MIN2(_MAX_THREADS_PER_BLOCK, old_num);
    block_num = int(std::ceil(old_num / float(thread_num)));
    cudaInitCntX22_kernel<<<block_num, thread_num>>>(cnt_x2, old_num, x_bs_buffer, x_max);
    CUDASYNC();

    thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    block_num = int(std::ceil(batch_size / float(thread_num)));
    cudaInitCntX23_kernel<<<block_num, thread_num>>>(cnt_x1, num, x_bs_buffer, batch_size);
    CUDASYNC();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void cudaInitCntX_kernel(int *__restrict__ cnt_x1, int *__restrict__ num,
                                    const int *__restrict__ startIdx, int startIdx_stride,
                                    int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch_size)
    {
        if (startIdx[tid * startIdx_stride] == -1)
        {
            int old = atomicAdd(num, 1);
            cnt_x1[old * 5 + 0] = tid; // state_idx
            cnt_x1[old * 5 + 1] = 0;   // x
        }
    }
}

__host__ void cudaInitCntX(int *cnt_x1, int *num,
                           int *startIdx, int startIdx_stride,
                           int batch_size)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    const int block_num = int(std::ceil(batch_size / float(thread_num)));
    cudaInitCntX_kernel<<<block_num, thread_num>>>(cnt_x1, num, startIdx, startIdx_stride, batch_size);
    CUDASYNC();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void cudaInitCntX41_kernel(const int *__restrict__ cnt_x2, int old_num, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                                      int *__restrict__ bs_buffer3, int *__restrict__ bs_buffer4)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (old_num - 1)
    
    if (tid < old_num)
    {
        int state_idx = cnt_x2[5 * tid + 0];
        int gt_startIdx = cnt_x2[5 * tid + 1];
        int wrap_i = cnt_x2[5 * tid + 2];

        bs_buffer3[state_idx] = gt_startIdx; // gt_startIdx is the same
        
        atomicMax(bs_buffer4 + state_idx, wrap_i); // find max
    }   
}

__global__ void cudaInitCntX42_kernel(const int *__restrict__ bs_buffer1, const int *__restrict__ bs_buffer2, int *__restrict__ bs_buffer3, int *__restrict__ bs_buffer4, 
                                      int batch_size, int state_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if(tid < batch_size && bs_buffer3[tid] != -1)
    {
        if(bs_buffer1[tid] >=0 && bs_buffer2[tid] >=0) // find a vertex
        {
            bs_buffer3[tid] = bs_buffer1[tid];
            bs_buffer4[tid] = bs_buffer2[tid];
        }
        else if (bs_buffer1[tid]==-1 && bs_buffer2[tid]==-1)
        {
            bs_buffer4[tid]++;
            if(bs_buffer4[tid] >= state_len) // we cannot find solution after looping over all the inequs
            {
                // we cannot find solution
                bs_buffer3[tid] = -1;
                bs_buffer4[tid] = -1;
#ifdef _ENABLE_KERNEL_PRINTF
                printf("we cannot find solution: state_idx = %d\n", tid);
#endif
            }
        }
        else if(bs_buffer1[tid]==-2 && bs_buffer2[tid]==-2) // close
        {
            bs_buffer3[tid] = -1;
            bs_buffer4[tid] = -1;
        }
    }
        
}

__global__ void cudaInitCntX43_kernel(int *__restrict__ cnt_x1, int *__restrict__ num, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                                      const int *__restrict__ bs_buffer3, const int *__restrict__ bs_buffer4,
                                      int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (batch_size - 1)

    if (tid < batch_size)
    {
        int gt_startIdx = bs_buffer3[tid];
        int wrap_i = bs_buffer4[tid];
        if (gt_startIdx != -1 && wrap_i != -1)
        {
            int old = atomicAdd(num, 1);
            int old5 = 5 * old;
            cnt_x1[old5 + 0] = tid;
            cnt_x1[old5 + 1] = gt_startIdx;
            cnt_x1[old5 + 2] = wrap_i;
        }
    }
}

__host__ void cudaInitCntX4(int *cnt_x2, int old_num, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                            int *cnt_x1, int *num,
                            int *bs_buffer1, int *bs_buffer2, int *bs_buffer3, int *bs_buffer4, 
                            int batch_size, int state_len)
{
    thrust::device_ptr<int> bs_buffer3_ptr(bs_buffer3);
    thrust::device_ptr<int> bs_buffer4_ptr(bs_buffer4);
    thrust::fill(bs_buffer3_ptr, bs_buffer3_ptr + batch_size, -1);
    thrust::fill(bs_buffer4_ptr, bs_buffer4_ptr + batch_size, -1);

    int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, old_num);
    int block_num = int( std::ceil(old_num / float(thread_num)));
    cudaInitCntX41_kernel<<<block_num, thread_num>>>(cnt_x2, old_num, bs_buffer3, bs_buffer4);
    CUDASYNC();

    thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    block_num = int( std::ceil(batch_size / float(thread_num)));
    cudaInitCntX42_kernel<<<block_num, thread_num>>>(bs_buffer1, bs_buffer2, bs_buffer3, bs_buffer4, batch_size, state_len);
    CUDASYNC();

    thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    block_num = int( std::ceil(batch_size / float(thread_num)));
    cudaInitCntX43_kernel<<<block_num, thread_num>>>(cnt_x1, num, bs_buffer3, bs_buffer4, batch_size);
    CUDASYNC();
}