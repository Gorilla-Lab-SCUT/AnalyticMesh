
#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <cmath>

#include "utilities.h"
#include "vecmat.h"
#include "facedata.h"
#include "hash.h"

////

__host__ void cudaMemcpyStrided_D2D(FourBytes *dst, size_t stride_in_4bytes, int repeat_num, const FourBytes *src, size_t _4bytes_num);

__host__ void thrustSortBatched_fill(int *segments, int *key_vec, int one_batch_len, int batch_num);

__host__ void cudaInitCntX(int *cnt_x1, int *num, int *startIdx, int startIdx_stride, int batch_size);

__host__ void cudaInitCntX2(int *cnt_x2, int *num, int old_num, int *cnt_x1, int *x_bs_buffer, int batch_size, int x_max);

__host__ void cudaUpdateStartIdx(int *startIdx, int startIdx_stride, int *cnt_x2, int num);

__host__ void cudaInitCntX4(int *cnt_x2, int old_num, int *cnt_x1, int *num, int *bs_buffer1, int *bs_buffer2, int *bs_buffer3, int *bs_buffer4, int batch_size, int state_len);

////

__device__ __forceinline__ float r_sqrt(float x) { return rsqrtf(x); }
__device__ __forceinline__ double r_sqrt(double x) { return rsqrt(x); }

////

__device__ __forceinline__ float fp_abs(float x) { return fabsf(x); }
__device__ __forceinline__ double fp_abs(double x) { return fabs(x); }

////

__device__ double atomicInc(double *__restrict__ address)
{
    unsigned long long int *address_ = (unsigned long long int *)address;
    unsigned long long int old = *address_, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_, assumed, __double_as_longlong(1 + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ float atomicInc(float *__restrict__ address)
{
    unsigned int *address_ = (unsigned int *)address;
    unsigned int old = *address_, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_, assumed, __float_as_uint(1 + __uint_as_float(assumed)));
    } while (assumed != old);

    return __uint_as_float(old);
}

__device__ int atomicInc(int *__restrict__ address)
{
    unsigned int *address_ = (unsigned int *)address;
    unsigned int old = *address_, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_, assumed, 1 + assumed);
    } while (assumed != old);

    return old;
}

////

inline cublasStatus_t cublasGeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc)
{
    return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline cublasStatus_t cublasGeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc)
{
    return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

////

inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

////

inline cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasGemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

////

inline cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount)
{
    return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t cublasGemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount)
{
    return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    mask some rows of matrices
    NOTE: `stride` is not in byte
*/

template <int COL, typename FloatType>
__global__ void cudaMaskRowsStrided_kernel(const FloatType *__restrict__ mask, int mask_stride,
                                           const FloatType *__restrict__ src, int src_stride, // multiples of `COL`
                                           FloatType *__restrict__ dst, int dst_stride,       // multiples of `COL`
                                           int elem_num,
                                           int repeat_num,
                                           int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / elem_num;
    int j = tid - i * elem_num;
    int tid_COL = j / COL;

    if (tid < total_num)
        dst[i * dst_stride + j] = src[i * src_stride + j] * mask[i * mask_stride + tid_COL];
}

template <int COL, typename FloatType>
__host__ void cudaMaskRowsStrided(FloatType *mask, int mask_stride,
                                  FloatType *src, int src_stride, // multiples of `COL`
                                  FloatType *dst, int dst_stride, // multiples of `COL`
                                  int elem_num,
                                  int repeat_num)
{
    const int total_num = repeat_num * elem_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaMaskRowsStrided_kernel<COL, FloatType><<<block_num, thread_num>>>(mask, mask_stride, src, src_stride, dst, dst_stride, elem_num, repeat_num, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    Matrices multiplication in batch
    NOTE: `stride` is not in byte
*/

template <typename FloatType>
__host__ void cublasMatMulBatched(cublasHandle_t handle,
                                  const MAT<FloatType> &A,
                                  FloatType *B, int height_b, int width_b, int stride_b,
                                  FloatType *C, int stride_c,
                                  int num,
                                  FloatType beta_ = 0.0,
                                  int stride_a = 0) // C = A * B
{
    assert(A.width == height_b);

    const FloatType alpha = 1.0;
    const FloatType beta = beta_;

    cublasErrchk(cublasGemmStridedBatched(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          width_b, A.height, height_b,
                                          &alpha,
                                          B, width_b, stride_b,
                                          A.data, A.width, stride_a,
                                          &beta,
                                          C, width_b, stride_c,
                                          num));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    Matrices vectors multiplication with adding biases in batch
    NOTE: `stride` is not in byte
*/

template <typename FloatType>
__host__ void cublasMatVecMulAddBatched(cublasHandle_t handle,
                                        const MAT<FloatType> &W,
                                        FloatType *x, int len_x, int stride_x,
                                        const VEC<FloatType> &b,
                                        FloatType *y, int len_y, int stride_y,
                                        int num) // y = W * x + b
{
    assert(W.width == len_x);
    assert(W.height == len_y);
    if (b.data)
        assert(b.len == len_y);

    const FloatType alpha = 1.0;
    const FloatType beta = 1.0; // plus biases

    if (b.data)
    {
        cudaMemcpyStrided_D2D((FourBytes *)y, stride_y * sizeof(FloatType) / 4, num,
                              (FourBytes *)(b.data), b.len * sizeof(FloatType) / 4);
    }

    cublasErrchk(cublasGemm(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            len_y, num, len_x,
                            &alpha,
                            W.data, len_x,
                            x, stride_x,
                            &beta,
                            y, stride_y));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    sign the constraints
    NOTE: `stride` is not in byte
*/

// template <int COL, typename FloatType>
// __global__ void cudaSignConstraints_kernel(const FloatType *__restrict__ mask, int mask_stride,
//                                            FloatType *__restrict__ data, int data_stride, // multiples of `COL`, in-place
//                                            int num, int total_num)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int i = tid / data_stride;
//     int j = tid - i * data_stride;
//     int tid_COL = j / COL;

//     if (tid < total_num)
//         data[tid] = data[tid] * (1.0 - 2.0 * mask[i * mask_stride + tid_COL]); // negative
// }

// template <int COL, typename FloatType>
// __host__ void cudaSignConstraints(FloatType *mask, int mask_stride,
//                                   FloatType *data, int data_stride, // multiples of `COL`, in-place
//                                   int num)
// {
//     const int total_num = data_stride * num;
//     const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
//     const int block_num = int(std::ceil(total_num / float(thread_num)));
//     cudaSignConstraints_kernel<COL, FloatType><<<block_num, thread_num>>>(mask, mask_stride, data, data_stride, num, total_num);
//     CUDASYNC();
// }

template <int COL, typename FloatType>
__global__ void cudaSignConstraints_kernel(const FloatType *__restrict__ mask, int mask_stride,
                                           FloatType *__restrict__ data, int data_stride, // multiples of `COL`, in-place
                                           int num, int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / (COL * mask_stride);
    int j = tid - i * (COL * mask_stride);
    int tid_COL = j / COL;

    if (tid < total_num)
        data[i * data_stride + j] *= (1.0 - 2.0 * mask[i * mask_stride + tid_COL]); // negative
}

template <int COL, typename FloatType>
__host__ void cudaSignConstraints(FloatType *mask, int mask_stride,
                                  FloatType *data, int data_stride, // multiples of `COL`, in-place
                                  int num)
{
    const int total_num = COL * mask_stride * num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaSignConstraints_kernel<COL, FloatType><<<block_num, thread_num>>>(mask, mask_stride, data, data_stride, num, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    mask matrix and add to dst (for identity skip connection)
*/

template <int COL, typename FloatType>
__global__ void cudaMaskRowsAddStrided_kernel(const FloatType *__restrict__ mask, int mask_stride,
                                              const FloatType *__restrict__ src, int src_dst_stride, // multiples of `COL`
                                              FloatType *__restrict__ dst,
                                              int elem_num,
                                              int repeat_num,
                                              int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / elem_num;
    int j = tid - i * elem_num;
    int tid_COL = j / COL;
    int src_dst_idx = i * src_dst_stride + j;

    if (tid < total_num)
        dst[src_dst_idx] = dst[src_dst_idx] + src[src_dst_idx] * mask[i * mask_stride + tid_COL];
}

template <int COL, typename FloatType>
__host__ void cudaMaskRowsAddStrided(FloatType *mask, int mask_stride,
                                     FloatType *src, int src_dst_stride, // multiples of `COL`
                                     FloatType *dst,
                                     int elem_num,
                                     int repeat_num)
{
    const int total_num = repeat_num * elem_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaMaskRowsAddStrided_kernel<COL, FloatType><<<block_num, thread_num>>>(mask, mask_stride, src, src_dst_stride, dst, elem_num, repeat_num, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    Matrices addition in batch (dst <- dst + src1)
*/

template <typename FloatType>
__global__ void cudaMatAddBatched_kernel(const FloatType *__restrict__ src1, int stride_src1,
                                         FloatType *__restrict__ dst, int stride_dst,
                                         int elem_num,
                                         int repeat_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / elem_num;
    int j = tid - i * elem_num;

    if (tid < elem_num * repeat_num)
        dst[i * stride_dst + j] = dst[i * stride_dst + j] + src1[i * stride_src1 + j];
}

template <typename FloatType>
__host__ void cudaMatAddBatched(FloatType *src1, int stride_src1,
                                FloatType *dst, int stride_dst,
                                int elem_num,
                                int repeat_num)
{
    const int total_num = repeat_num * elem_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaMatAddBatched_kernel<FloatType><<<block_num, thread_num>>>(src1, stride_src1, dst, stride_dst, elem_num, repeat_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void cudaAddIdentityBatched_kernel(FloatType *__restrict__ data,
                                              int ld,
                                              int ld_num,
                                              int stride,
                                              int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / ld_num;
    int j = tid - i * ld_num;
    int idx = i * stride + j * (ld + 1);

    if (tid < total_num)
        data[idx] = data[idx] + FloatType(1.0);
}

template <typename FloatType>
__host__ void cudaAddIdentityBatched(FloatType *data,
                                     int ld,
                                     int ld_num,
                                     int stride,
                                     int repeat_num)
{
    const int total_num = ld_num * repeat_num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaAddIdentityBatched_kernel<FloatType><<<block_num, thread_num>>>(data, ld, ld_num, stride, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    divided dst by norm calculated from row of src (used to calc distance in 3D cell)
*/

template <typename FloatType>
__global__ void cudaRow3NormDiv_kernel(const FloatType *__restrict__ src, FloatType *__restrict__ dst, int len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid3 = tid * 3;

    if (tid < len)
        dst[tid] = dst[tid] * r_sqrt(src[tid3] * src[tid3] +
                                     src[tid3 + 1] * src[tid3 + 1] +
                                     src[tid3 + 2] * src[tid3 + 2]);
}

template <typename FloatType>
__host__ void cudaRow3NormDiv(FloatType *src, FloatType *dst, int len)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, len);
    const int block_num = int(std::ceil(len / float(thread_num)));
    cudaRow3NormDiv_kernel<FloatType><<<block_num, thread_num>>>(src, dst, len);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* 
    another way to calc distance (distance on w_equ plane)
 */

template <typename FloatType>
__global__ void cudaWrapDistance_kernel(const FloatType *__restrict__ w_inequ, const FloatType *__restrict__ w_equ,
                                        FloatType *__restrict__ dst,
                                        int state_len, int batch_size, int total_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid3 = tid * 3;
    int idxequ = tid3 / state_len;

    if (tid < total_num)
    {
        FloatType n1n2 = w_inequ[tid3 + 0] * w_equ[idxequ + 0] +
                         w_inequ[tid3 + 1] * w_equ[idxequ + 1] +
                         w_inequ[tid3 + 2] * w_equ[idxequ + 2];
        dst[tid] = dst[tid] * r_sqrt(w_inequ[tid3 + 0] * w_inequ[tid3 + 0] +
                                     w_inequ[tid3 + 1] * w_inequ[tid3 + 1] +
                                     w_inequ[tid3 + 2] * w_inequ[tid3 + 2] -
                                     n1n2 * n1n2 /
                                         (w_equ[idxequ + 0] * w_equ[idxequ + 0] +
                                          w_equ[idxequ + 1] * w_equ[idxequ + 1] +
                                          w_equ[idxequ + 2] * w_equ[idxequ + 2]));
    }
}

template <typename FloatType>
__host__ void cudaWrapDistance(FloatType *w_inequ, FloatType *w_equ,
                               FloatType *dst,
                               int state_len, int batch_size)
{
    const int total_num = state_len * batch_size;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaWrapDistance_kernel<FloatType><<<block_num, thread_num>>>(w_inequ, w_equ, dst, state_len, batch_size, total_num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
__global__ void cudaCheckAnyEqual_kernel(const DataType *__restrict__ data, int stride, int data_num,
                                         DataType target_data_value,
                                         bool *__restrict__ flag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < data_num)
    {
        if (data[tid * stride] == target_data_value)
        {
            *flag = true;
        }
    }
}

template <typename DataType>
__host__ bool cudaCheckAnyEqual(DataType *data, int stride, int data_num,
                                DataType target_data_value) // return true if any of them is equal to `target_data_value`
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, data_num);
    const int block_num = int(std::ceil(data_num / float(thread_num)));
    thrust::device_vector<bool> flag(1);
    flag[0] = false;
    cudaCheckAnyEqual_kernel<DataType><<<block_num, thread_num>>>(data, stride, data_num, target_data_value, thrust::raw_pointer_cast(flag.data()));
    CUDASYNC();
    return flag[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
    solve 3x3 (A * x + b == 0)
 */

#define DET_3x3(a0, a1, a2, a3, a4, a5, a6, a7, a8) (a0 * (a4 * a8 - a5 * a7) - a1 * (a3 * a8 - a5 * a6) + a2 * (a3 * a7 - a4 * a6))

#ifdef _FLOAT32_USE_64PRECISION_WHEN_NEED

__device__ __forceinline__ void solve_3x3(const double *__restrict__ A, const double *__restrict__ b, double *__restrict__ x)
{
    double d0 = DET_3x3(A[0], A[1], A[2],
                        A[3], A[4], A[5],
                        A[6], A[7], A[8]);
    double d1 = DET_3x3((-b[0]), A[1], A[2],
                        (-b[1]), A[4], A[5],
                        (-b[2]), A[7], A[8]);
    double d2 = DET_3x3(A[0], (-b[0]), A[2],
                        A[3], (-b[1]), A[5],
                        A[6], (-b[2]), A[8]);
    double d3 = DET_3x3(A[0], A[1], (-b[0]),
                        A[3], A[4], (-b[1]),
                        A[6], A[7], (-b[2]));
    x[0] = d1 / d0;
    x[1] = d2 / d0;
    x[2] = d3 / d0;
}

__device__ __forceinline__ void solve_3x3(const float *__restrict__ A, const float *__restrict__ b, float *__restrict__ x)
{
    double d0 = DET_3x3(double(A[0]), double(A[1]), double(A[2]),
                        double(A[3]), double(A[4]), double(A[5]),
                        double(A[6]), double(A[7]), double(A[8]));
    double d1 = DET_3x3(double(-b[0]), double(A[1]), double(A[2]),
                        double(-b[1]), double(A[4]), double(A[5]),
                        double(-b[2]), double(A[7]), double(A[8]));
    double d2 = DET_3x3(double(A[0]), double(-b[0]), double(A[2]),
                        double(A[3]), double(-b[1]), double(A[5]),
                        double(A[6]), double(-b[2]), double(A[8]));
    double d3 = DET_3x3(double(A[0]), double(A[1]), double(-b[0]),
                        double(A[3]), double(A[4]), double(-b[1]),
                        double(A[6]), double(A[7]), double(-b[2]));
    x[0] = float(d1 / d0);
    x[1] = float(d2 / d0);
    x[2] = float(d3 / d0);
}

#else

template <typename FloatType>
__device__ __forceinline__ void solve_3x3(const FloatType *__restrict__ A, const FloatType *__restrict__ b, FloatType *__restrict__ x)
{
    FloatType d0 = DET_3x3(A[0], A[1], A[2],
                           A[3], A[4], A[5],
                           A[6], A[7], A[8]);
    FloatType d1 = DET_3x3((-b[0]), A[1], A[2],
                           (-b[1]), A[4], A[5],
                           (-b[2]), A[7], A[8]);
    FloatType d2 = DET_3x3(A[0], (-b[0]), A[2],
                           A[3], (-b[1]), A[5],
                           A[6], (-b[2]), A[8]);
    FloatType d3 = DET_3x3(A[0], A[1], (-b[0]),
                           A[3], A[4], (-b[1]),
                           A[6], A[7], (-b[2]));
    x[0] = d1 / d0;
    x[1] = d2 / d0;
    x[2] = d3 / d0;
}

#endif

template <typename FloatType>
__global__ void solve_3x3_kernel(const FloatType *__restrict__ As, const FloatType *__restrict__ bs, FloatType *__restrict__ xs, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num)
    {
        solve_3x3(As + 9 * tid,
                  bs + 3 * tid,
                  xs + 3 * tid);
    }
}

template <typename FloatType, bool UseCublas = false>
__host__ void solve_3x3_Batched(FloatType *As, FloatType *bs, FloatType *xs, int num)
{
    if (UseCublas)
    {
        // TODO
#warning `UseCublas` branch of function `solve_3x3_Batched` is not implemented.
    }
    else
    {
        const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, num);
        const int block_num = int(std::ceil(num / float(thread_num)));
        solve_3x3_kernel<FloatType><<<block_num, thread_num>>>(As, bs, xs, num);
        CUDASYNC();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

// (x>=0)  --- mapping to --->  (i, j) and (i > j)
__host__ __device__ __forceinline__ void get_i_j(int x, int ij_max, int &i, int &j)
{
    int k = x / 2;
#ifdef __CUDA_ARCH__
    int a = int(ceilf(sqrtf(2 * k + 2.25f) - 1.5f));
#else
    int a = int(std::ceil(std::sqrt(2 * k + 2.25f) - 1.5f));
#endif
    int r = x - a * (a + 1);
    int A = 2 * a + r / (a + 1);
    int B = r % (a + 1);
    i = (A + 1) / 2 + B + 1;
    j = A + 1 - i;
    i = (i < ij_max ? i : (ij_max - 1));
    j = (j < ij_max ? j : (ij_max - 1));
}

#ifdef _REMOVE_ZERO_CONS
template <typename FloatType>
__device__ __forceinline__ bool isZeroCons(const FloatType *__restrict__ this_w, const FloatType *__restrict__ this_b)
{
    return fp_abs(this_w[0]) <= FP_ZERO_CONS<FloatType> &&
           fp_abs(this_w[1]) <= FP_ZERO_CONS<FloatType> &&
           fp_abs(this_w[2]) <= FP_ZERO_CONS<FloatType> &&
           fp_abs(this_b[0]) <= FP_ZERO_CONS<FloatType>;
}
#endif

template <typename FloatType>
__global__ void fillAs_kernel(const int *__restrict__ cnt_x1, int num,            // state_idx, x, gt_i, gt_j, ok (inplace)
                              const int *__restrict__ dist_idx,                   // size == (batch_size, state_len)
                              FloatType *__restrict__ As,                         // size == (batch_size, 9) == (batch_size, 3, 3)
                              FloatType *__restrict__ bs,                         // size == (batch_size, 3)
                              int *__restrict__ cnt_x2, int *__restrict__ As_num, // one element
                              const FloatType *__restrict__ w_inequ,
                              const FloatType *__restrict__ b_inequ,
                              const FloatType *__restrict__ w_equ,
                              const FloatType *__restrict__ b_equ,
                              int state_len,
                              int x_max,
                              int s_max)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (s_max - 1)
    int state_idx, x;

    if (tid < s_max)
    {
        int mod = tid % num;
        state_idx = cnt_x1[5 * mod + 0];
        x = cnt_x1[5 * mod + 1] + (tid / num);

        if (x <= x_max)
        {
            int state_idx_state_len = state_idx * state_len;
            int state_idx3 = 3 * state_idx;

            int i, j;
            get_i_j(x, state_len, i, j); // decode

            int real_i = dist_idx[state_idx_state_len + i];
            int real_j = dist_idx[state_idx_state_len + j];
            int row1 = 3 * (state_idx_state_len + real_i);
            int row2 = 3 * (state_idx_state_len + real_j);

#ifdef _REMOVE_ZERO_CONS
            if (isZeroCons(w_inequ + row1, b_inequ + row1 / 3) &&
                isZeroCons(w_inequ + row2, b_inequ + row2 / 3) &&
                isZeroCons(w_equ + state_idx3, b_equ + state_idx))
            {
                return;
            }
#endif

            int old = atomicAdd(As_num, 1);
            int old9 = 9 * old;
            int old3 = 3 * old;
            int old5 = 5 * old;
            As[old9 + 0] = w_inequ[row1 + 0];
            As[old9 + 1] = w_inequ[row1 + 1];
            As[old9 + 2] = w_inequ[row1 + 2];
            As[old9 + 3] = w_inequ[row2 + 0];
            As[old9 + 4] = w_inequ[row2 + 1];
            As[old9 + 5] = w_inequ[row2 + 2];
            As[old9 + 6] = w_equ[state_idx3 + 0];
            As[old9 + 7] = w_equ[state_idx3 + 1];
            As[old9 + 8] = w_equ[state_idx3 + 2];
            bs[old3 + 0] = b_inequ[row1 / 3];
            bs[old3 + 1] = b_inequ[row2 / 3];
            bs[old3 + 2] = b_equ[state_idx];
            cnt_x2[old5 + 0] = state_idx;
            cnt_x2[old5 + 1] = x;
            cnt_x2[old5 + 2] = real_i; // the real i
            cnt_x2[old5 + 3] = real_j; // the real j
            cnt_x2[old5 + 4] = 1;      // default is ok (set to 0 for not okey)
        }
    }
}

template <typename FloatType>
__host__ void fillAs(int *cnt_x1, int num,     // state_idx, x, gt_i, gt_j, ok
                     int *dist_idx,            // size == (batch_size, state_len)
                     FloatType *As,            // size == (batch_size, 9) == (batch_size, 3, 3)
                     FloatType *bs,            // size == (batch_size, 3)
                     int *cnt_x2, int *As_num, // one element
                     FloatType *w_inequ,
                     FloatType *b_inequ,
                     FloatType *w_equ,
                     FloatType *b_equ,
                     int state_len,
                     int x_max,
                     int s_max)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, s_max);
    const int block_num = int(std::ceil(s_max / float(thread_num)));
    fillAs_kernel<FloatType><<<block_num, thread_num>>>(cnt_x1, num, dist_idx, As, bs, cnt_x2, As_num, w_inequ, b_inequ, w_equ, b_equ, state_len, x_max, s_max);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void cudaCheckCondition_kernel(const FloatType *__restrict__ xs,
                                          int *__restrict__ cnt_x2, // state_idx, x, gt_i, gt_j, ok
                                          const FloatType *__restrict__ w_inequ,
                                          const FloatType *__restrict__ b_inequ,
                                          int state_len, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (num * state_len - 1)
    int num_idx = tid / state_len;                   // 0 ~ (num - 1)
    int which_state_idx = tid % state_len;           // 0 ~ (state_len - 1)

    if (num_idx < num)
    {
        int state_idx = cnt_x2[5 * num_idx + 0];
        int i = cnt_x2[5 * num_idx + 2];
        int j = cnt_x2[5 * num_idx + 3];

        int offset = 3 * (state_idx * state_len + which_state_idx);
        int num_idx3 = 3 * num_idx;
        FloatType ans = w_inequ[offset + 0] * xs[num_idx3 + 0] +
                        w_inequ[offset + 1] * xs[num_idx3 + 1] +
                        w_inequ[offset + 2] * xs[num_idx3 + 2] +
                        b_inequ[offset / 3];

#ifdef _REMOVE_ZERO_CONS
        if (!isfinite(ans))
        {
            cnt_x2[5 * num_idx + 4] = 0;
        }
#endif

#ifdef _CONSTRAINTS_USE_GEOMETRIC_DISTANCE
        FloatType r_scale = r_sqrt(w_inequ[offset + 0] * w_inequ[offset + 0] +
                                   w_inequ[offset + 1] * w_inequ[offset + 1] +
                                   w_inequ[offset + 2] * w_inequ[offset + 2]);

        if (ans * r_scale > FP_EPS<FloatType> && which_state_idx != i && which_state_idx != j)
        {
            cnt_x2[5 * num_idx + 4] = 0; // set to zero for not okey
        }

#else
        if (ans > FP_EPS<FloatType> && which_state_idx != i && which_state_idx != j)
        {
            cnt_x2[5 * num_idx + 4] = 0; // set to zero for not okey
        }
#endif
    }
}

/*
    To determine whether or not the inequality constraint is satisfied
 */

template <typename FloatType>
__host__ void cudaCheckCondition(FloatType *xs,
                                 int *cnt_x2, // (state_idx, x, gt_i, gt_j, ok)
                                 FloatType *w_inequ,
                                 FloatType *b_inequ,
                                 int state_len, int num)
{
    const int total_num = state_len * num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaCheckCondition_kernel<FloatType><<<block_num, thread_num>>>(xs, cnt_x2, w_inequ, b_inequ, state_len, num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void calcStartIdx(int *startIdx, int startIdx_stride, // modify in place (output)
                           int *dist_idx,                      // size == (batch_size, state_len)
                           FloatType *As,                      // size == (s_max, 9) == (batch_size, 3, 3)
                           FloatType *bs,                      // size == (s_max, 3)
                           FloatType *xs,                      // size == (s_max, 3)
                           int *cnt_x,                         // (state_idx, x, gt_i, gt_j, ok)
                           int s_max,                          // the capacity of As / bs / xs / cnt_x
                           FloatType *w_inequ,
                           FloatType *b_inequ,
                           FloatType *w_equ,
                           FloatType *b_equ,
                           int state_len,
                           int batch_size)
{
    if (cudaCheckAnyEqual(startIdx, startIdx_stride, batch_size, -1) == false)
        return;

    // 0 <= x <= x_max
    // int x_max = (state_len - 1) * (state_len - 2);
    int x_max = MIN2((state_len - 1) * (state_len - 2), 812);

    thrust::device_vector<int> As_num(1);
    int *As_num_ptr = thrust::raw_pointer_cast(As_num.data());
    As_num[0] = 0;

    cudaInitCntX(cnt_x + 0, As_num_ptr, startIdx, startIdx_stride, batch_size);

    while (true)
    {
        // fill equations
        int num = As_num[0]; // num of cnt_x
        As_num[0] = 0;

        fillAs(cnt_x + 0, num, dist_idx, As, bs, cnt_x + 5 * batch_size, As_num_ptr,
               w_inequ, b_inequ, w_equ, b_equ, state_len, x_max, s_max);
        num = As_num[0]; // num of As (may be in 0 ~ (s_max - 1))

        // solve equations (stored in xs)
        solve_3x3_Batched(As, bs, xs, num);

        // check (ok==0 for not okey)
        cudaCheckCondition(xs, cnt_x + 5 * batch_size, w_inequ, b_inequ, state_len, num);

        // update to startIdx (for output)
        cudaUpdateStartIdx(startIdx, startIdx_stride, cnt_x + 5 * batch_size, num);

        // reinit
        As_num[0] = 0;
        cudaInitCntX2(cnt_x + 5 * batch_size, As_num_ptr, num, cnt_x + 0, cnt_x + 5 * batch_size + 5 * s_max, batch_size, x_max);

        if (As_num[0] == 0)
            break;
    }
    /*
        NOTE: if returned startIdx is still -1, then it means we can't find startIdx, and this value should be discarded.
     */
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void cudaInitCntX3_kernel(int *__restrict__ cnt_x1, int *__restrict__ num, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                                     int *__restrict__ tabij,
                                     FloatType *__restrict__ vertices_buffer, int vert_max,
                                     const int *__restrict__ startIdx, int startIdx_stride,
                                     int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch_size)
    {
        tabij[(2 + vert_max) * tid] = 0;               // clear tabij
        vertices_buffer[(1 + vert_max * 3) * tid] = 0; // clear vert
        int start_idx = startIdx[tid * startIdx_stride];
        if (start_idx != -1)
        {
            int old = atomicAdd(num, 1);
            cnt_x1[old * 5 + 0] = tid;       // state_idx 0~1023
            cnt_x1[old * 5 + 1] = start_idx; // gt_startIdx
            cnt_x1[old * 5 + 2] = 0;         // wrap_i

            tabij[(2 + vert_max) * tid] = 1;
            tabij[(2 + vert_max) * tid + 1] = start_idx;
        }
    }
}

template <typename FloatType>
__host__ void cudaInitCntX3(int *cnt_x1, int *num,
                            int *tabij,
                            FloatType *vertices_buffer, int vert_max,
                            int *startIdx, int startIdx_stride,
                            int batch_size)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    const int block_num = int(std::ceil(batch_size / float(thread_num)));
    cudaInitCntX3_kernel<<<block_num, thread_num>>>(cnt_x1, num, tabij, vertices_buffer, vert_max, startIdx, startIdx_stride, batch_size);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool existInArray(const int *__restrict__ array, int query, int arrayLen)
{
    bool flag = false;
    int tmp;
    for (int i = 0; i < arrayLen; ++i)
    {
        // to positive
        tmp = array[i];
        tmp = (tmp >= 0) ? (tmp) : (-tmp - 1);

        //
        flag = (flag || (tmp == query));
    }
    return flag;
}

template <typename FloatType>
__global__ void fillAs2_kernel(const int *__restrict__ cnt_x1, int num,            // s_max x (state_idx, gt_startIdx, wrap_i, real_i, ok)   (inplace)
                               const int *__restrict__ dist_idx,                   // size == (batch_size, state_len)
                               FloatType *__restrict__ As,                         // size == (batch_size, 9) == (batch_size, 3, 3)
                               FloatType *__restrict__ bs,                         // size == (batch_size, 3)
                               int *__restrict__ cnt_x2, int *__restrict__ As_num, // one element
                               const int *__restrict__ tabij, int vert_max,
                               const FloatType *__restrict__ w_inequ,
                               const FloatType *__restrict__ b_inequ,
                               const FloatType *__restrict__ w_equ,
                               const FloatType *__restrict__ b_equ,
                               int state_len,
                               int s_max,
                               int *__restrict__ idx_offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (s_max - 1)
    int state_idx, start, wrap_i, real_i;
    int state_idx_state_len;

    if (tid < s_max)
    {
        int mod = tid % num;
        state_idx = cnt_x1[5 * mod + 0]; // 0~1023
        start = cnt_x1[5 * mod + 1];
        state_idx_state_len = state_idx * state_len;
        const int *tab = tabij + (2 + vert_max) * state_idx;

        while (true)
        {
            wrap_i = cnt_x1[5 * mod + 2] + atomicAdd(idx_offset + state_idx, 1);
            real_i = dist_idx[state_idx_state_len + wrap_i];
            if (wrap_i >= state_len)
            {
                break;
            }

            if ((real_i == tab[1] && tab[0] >= 3) || // to close
                (!existInArray(tab + 1, real_i, tab[0])))
            {
                int state_idx3 = 3 * state_idx;

                int row1 = 3 * (state_idx_state_len + real_i);
                int row2 = 3 * (state_idx_state_len + start);

#ifdef _REMOVE_ZERO_CONS
                if (isZeroCons(w_inequ + row1, b_inequ + row1 / 3) &&
                    isZeroCons(w_inequ + row2, b_inequ + row2 / 3) &&
                    isZeroCons(w_equ + state_idx3, b_equ + state_idx))
                {
                    continue;
                }
#endif

                int old = atomicAdd(As_num, 1);
                int old9 = 9 * old;
                int old3 = 3 * old;
                int old5 = 5 * old;
                As[old9 + 0] = w_inequ[row1 + 0];
                As[old9 + 1] = w_inequ[row1 + 1];
                As[old9 + 2] = w_inequ[row1 + 2];
                As[old9 + 3] = w_inequ[row2 + 0];
                As[old9 + 4] = w_inequ[row2 + 1];
                As[old9 + 5] = w_inequ[row2 + 2];
                As[old9 + 6] = w_equ[state_idx3 + 0];
                As[old9 + 7] = w_equ[state_idx3 + 1];
                As[old9 + 8] = w_equ[state_idx3 + 2];
                bs[old3 + 0] = b_inequ[row1 / 3];
                bs[old3 + 1] = b_inequ[row2 / 3];
                bs[old3 + 2] = b_equ[state_idx];
                cnt_x2[old5 + 0] = state_idx;
                cnt_x2[old5 + 1] = start;  // start (real)
                cnt_x2[old5 + 2] = wrap_i; // wrap_i
                cnt_x2[old5 + 3] = real_i; // real i
                cnt_x2[old5 + 4] = 1;      // default is ok (set to 0 for not okey)

                break;
            }
        }
    }
}

template <typename FloatType>
__host__ void fillAs2(int *cnt_x1, int num,     // s_max x (state_idx, gt_startIdx, wrap_i, real_i, ok)   (inplace)
                      int *dist_idx,            // size == (batch_size, state_len)
                      FloatType *As,            // size == (batch_size, 9) == (batch_size, 3, 3)
                      FloatType *bs,            // size == (batch_size, 3)
                      int *cnt_x2, int *As_num, // one element
                      int *tabij, int vert_max,
                      FloatType *w_inequ,
                      FloatType *b_inequ,
                      FloatType *w_equ,
                      FloatType *b_equ,
                      int state_len,
                      int s_max,
                      int *bs_buffer1, int batch_size)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, s_max);
    const int block_num = int(std::ceil(s_max / float(thread_num)));
    gpuErrchk(cudaMemset(bs_buffer1, 0, batch_size * sizeof(int))); // offset value
    fillAs2_kernel<FloatType><<<block_num, thread_num>>>(cnt_x1, num, dist_idx, As, bs, cnt_x2, As_num, tabij, vert_max, w_inequ, b_inequ, w_equ, b_equ, state_len, s_max, bs_buffer1);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void cudaCheckCondition2_kernel(const FloatType *__restrict__ xs,
                                           int *__restrict__ cnt_x2, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                                           const FloatType *__restrict__ w_inequ,
                                           const FloatType *__restrict__ b_inequ,
                                           int state_len, int num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (num * state_len - 1)
    int num_idx = tid / state_len;                   // 0 ~ (num - 1)
    int which_state_idx = tid % state_len;           // 0 ~ (state_len - 1)

    if (num_idx < num)
    {
        int state_idx = cnt_x2[5 * num_idx + 0]; // 0~1023
        int i = cnt_x2[5 * num_idx + 1];
        int j = cnt_x2[5 * num_idx + 3]; // real

        int offset = 3 * (state_idx * state_len + which_state_idx);
        int num_idx3 = 3 * num_idx;

#ifdef _FLOAT32_USE_64PRECISION_WHEN_NEED
        FloatType ans = FloatType(double(w_inequ[offset + 0]) * double(xs[num_idx3 + 0]) +
                                  double(w_inequ[offset + 1]) * double(xs[num_idx3 + 1]) +
                                  double(w_inequ[offset + 2]) * double(xs[num_idx3 + 2]) +
                                  double(b_inequ[offset / 3]));
#else
        FloatType ans = w_inequ[offset + 0] * xs[num_idx3 + 0] +
                        w_inequ[offset + 1] * xs[num_idx3 + 1] +
                        w_inequ[offset + 2] * xs[num_idx3 + 2] +
                        b_inequ[offset / 3];
#endif

#ifdef _REMOVE_ZERO_CONS
        if (!isfinite(ans))
        {
            cnt_x2[5 * num_idx + 4] = 0;
        }
#endif

#ifdef _CONSTRAINTS_USE_GEOMETRIC_DISTANCE
        FloatType r_scale = r_sqrt(w_inequ[offset + 0] * w_inequ[offset + 0] +
                                   w_inequ[offset + 1] * w_inequ[offset + 1] +
                                   w_inequ[offset + 2] * w_inequ[offset + 2]);

        if (ans * r_scale > FP_EPS<FloatType> && which_state_idx != i && which_state_idx != j)
        {
            cnt_x2[5 * num_idx + 4] = 0; // set to zero for not okey
        }
#else
        if (ans > FP_EPS<FloatType> && which_state_idx != i && which_state_idx != j)
        {
            cnt_x2[5 * num_idx + 4] = 0; // set to zero for not okey
        }
#endif
    }
}

/*
    To determine whether or not the inequality constraint is satisfied
 */

template <typename FloatType>
__host__ void cudaCheckCondition2(FloatType *xs,
                                  int *cnt_x2, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                                  FloatType *w_inequ,
                                  FloatType *b_inequ,
                                  int state_len, int num)
{
    const int total_num = state_len * num;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    cudaCheckCondition2_kernel<FloatType><<<block_num, thread_num>>>(xs, cnt_x2, w_inequ, b_inequ, state_len, num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define DOT_PRODUCT(a, b) (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

template <typename FloatType>
__device__ bool isPivotValid_(const FloatType *__restrict__ neq, const FloatType *__restrict__ n1,
                              const FloatType *__restrict__ n2, const FloatType *__restrict__ n3) // project to 2D plane, and calc cosines angle
{
    FloatType A12 = DOT_PRODUCT(n1, n2);
    FloatType A23 = DOT_PRODUCT(n2, n3);
    FloatType A13 = DOT_PRODUCT(n1, n3);
    FloatType A22 = DOT_PRODUCT(n2, n2);
    FloatType C1 = DOT_PRODUCT(n1, neq);
    FloatType C2 = DOT_PRODUCT(n2, neq);
    FloatType C3 = DOT_PRODUCT(n3, neq);
    FloatType B = DOT_PRODUCT(neq, neq);
    return ((A12 * A23 - A22 * A13) - (A12 * C2 * C3 + A23 * C1 * C2 - A13 * C2 * C2 - A22 * C1 * C3) / B) > 0;
}

template <typename FloatType>
__device__ int isPivotValid(const FloatType *__restrict__ this_w_equ, const FloatType *__restrict__ this_w_inequ,
                            const int *__restrict__ tab, int new_i, int *__restrict__ the_last_i_idx)
{
    int tab_idx = tab[0];
    int cnt = 0;
    int the_ref_i_idx;

    // find `*the_last_i_idx` and `the_ref_i_idx`
    while (tab_idx >= 1 && cnt < 2)
    {
        if (tab[tab_idx] >= 0) // positive
        {
            if (cnt == 0)
            {
                *the_last_i_idx = tab_idx;
                cnt++;
            }
            else if (cnt == 1)
            {
                the_ref_i_idx = tab_idx;
                cnt++;
            }
        }
        tab_idx--;
    }
    if (cnt != 2)
        return (-1); // fail to compare

    // Judge if pivoting is successful by the angle of cosines
    return int(isPivotValid_(this_w_equ, this_w_inequ + 3 * tab[the_ref_i_idx], this_w_inequ + 3 * tab[*the_last_i_idx], this_w_inequ + 3 * new_i));
}

template <typename FloatType>
__device__ bool isSameVertices(const FloatType *__restrict__ v1, const FloatType *__restrict__ v2)
{
    return (fp_abs(v1[0] - v2[0]) + fp_abs(v1[1] - v2[1]) + fp_abs(v1[2] - v2[2]) < FP_SAME_EPS<FloatType>);
}

template <typename FloatType>
__global__ void cudaUpdateVert_kernel(FloatType *__restrict__ vertices_buffer, int vert_max,
                                      const FloatType *__restrict__ w_equ, const FloatType *__restrict__ w_inequ, int state_len,
                                      int *__restrict__ bs_buffer1, int *__restrict__ bs_buffer2, int *__restrict__ mutex, int batch_size,
                                      const FloatType *__restrict__ xs,
                                      int *__restrict__ tabij,                 // [N]-[3]-[4]-[0]-[2]-[1]-[3] ==> N is the number of total elems, the last one is equal to the first one ideally when finish
                                      const int *__restrict__ cnt_x2, int num) // (state_idx, gt_startIdx, wrap_i, real_i, ok)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ (num - 1)

    if (tid < num)
    {
        int state_idx = cnt_x2[5 * tid + 0]; // 0~1023
        int *tab = tabij + (vert_max + 2) * state_idx;
        FloatType *vert = vertices_buffer + (1 + 3 * vert_max) * state_idx;
        bool close = ((tab[0] >= 4 && tab[1] == tab[tab[0]]) || // close loop
                      tab[0] >= vert_max + 1);                  // no more than maximum
        int new_i = cnt_x2[5 * tid + 3];
        bool okey = (cnt_x2[5 * tid + 4] == 1);

        if (okey && (!close))
        {
            if (atomicCAS(mutex + state_idx, -1, tid) == -1) // lock for state_idx
            {
                int startIndex;
                int the_last_i_idx;

                int is_pivot_valid = isPivotValid(w_equ + 3 * state_idx, w_inequ + 3 * state_len * state_idx,
                                                  tab, new_i, &the_last_i_idx);
                if (is_pivot_valid == int(false))
                {
                    int tab_old = atomicInc(tab) + 1;
                    tab[tab_old] = (-new_i - 1); // append as negative
                    startIndex = the_last_i_idx; // positive
                }
                else
                {
                    if (int(vert[0]) > 0 &&
                        isSameVertices(vert + (3 * int(vert[0]) - 2), xs + 3 * tid))
                    {
                        // append as the positive index
                        int tab_old = atomicInc(tab) + 1;
                        tab[tab_old] = new_i; // real_i
                        startIndex = new_i;

                        tab[the_last_i_idx] = (-tab[the_last_i_idx] - 1);
                    }
                    else
                    {
                        // append vertex
                        int old = 3 * atomicInc(vert) + 1;
                        vert[old + 0] = xs[tid * 3 + 0]; // x
                        vert[old + 1] = xs[tid * 3 + 1]; // y
                        vert[old + 2] = xs[tid * 3 + 2]; // z

                        // append as the positive index
                        int tab_old = atomicInc(tab) + 1;
                        tab[tab_old] = new_i; // real_i
                        startIndex = new_i;
                    }
                }

                // close
                if ((tab[0] >= 4 && tab[1] == tab[tab[0]]) || tab[0] >= vert_max + 1)
                {
                    bs_buffer1[state_idx] = -2;
                    bs_buffer2[state_idx] = -2;
                }
                else // not close
                {
                    bs_buffer1[state_idx] = startIndex;
                    bs_buffer2[state_idx] = 0;
                }
            }
        }
    }
}

template <typename FloatType>
__host__ void cudaUpdateVert(FloatType *vertices_buffer, int vert_max,
                             FloatType *w_equ, FloatType *w_inequ, int state_len,
                             int *bs_buffer1, int *bs_buffer2, int *bs_buffer3, int batch_size,
                             FloatType *xs,
                             int *tabij,
                             int *cnt_x2, int num)
{
    thrust::device_ptr<int> bs_buffer1_ptr(bs_buffer1);
    thrust::device_ptr<int> bs_buffer2_ptr(bs_buffer2);
    thrust::device_ptr<int> bs_buffer3_ptr(bs_buffer3);
    thrust::fill(bs_buffer1_ptr, bs_buffer1_ptr + batch_size, -1);
    thrust::fill(bs_buffer2_ptr, bs_buffer2_ptr + batch_size, -1);
    thrust::fill(bs_buffer3_ptr, bs_buffer3_ptr + batch_size, -1);

    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, num);
    const int block_num = int(std::ceil(num / float(thread_num)));
    cudaUpdateVert_kernel<<<block_num, thread_num>>>(vertices_buffer, vert_max, w_equ, w_inequ, state_len, bs_buffer1, bs_buffer2, bs_buffer3, batch_size, xs, tabij, cnt_x2, num);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void calcPivoting(int *startIdx, int startIdx_stride,       // may contain -1 value, this should not be processed
                           FloatType *vertices_buffer, int vert_max, // vertices
                           int *dist_idx,                            // size == (batch_size, state_len)
                           FloatType *As,                            // size == (s_max, 9) == (batch_size, 3, 3)
                           FloatType *bs,                            // size == (s_max, 3)
                           FloatType *xs,                            // size == (s_max, 3)
                           int *cnt_x,                               // s_max x (state_idx, gt_startIdx, wrap_i, real_i, ok)    NOTE: wrap[wrap_i] = real_i
                           int s_max,                                // the capacity of As / bs / xs / cnt_x
                           int *tabij,                               // whether form a close loop
                           FloatType *w_inequ,
                           FloatType *b_inequ,
                           FloatType *w_equ,
                           FloatType *b_equ,
                           int state_len,
                           int batch_size)
{
    thrust::device_vector<int> As_num(1);
    int *As_num_ptr = thrust::raw_pointer_cast(As_num.data());
    As_num[0] = 0;

    // initialize tabij、cnt_x1、vertices_buffer
    cudaInitCntX3(cnt_x + 0, As_num_ptr, tabij, vertices_buffer, vert_max, startIdx, startIdx_stride, batch_size);

    while (true)
    {
        // fill equations
        int num = As_num[0]; // num of cnt_x
        As_num[0] = 0;

        // fill in cnt_x2
        fillAs2(cnt_x + 0, num, dist_idx, As, bs, cnt_x + 5 * batch_size, As_num_ptr,
                tabij, vert_max, w_inequ, b_inequ, w_equ, b_equ, state_len, s_max,
                cnt_x + 5 * batch_size + 5 * s_max + 0 * batch_size, batch_size);
        num = As_num[0]; // num of As

        // solve 3x3
        solve_3x3_Batched(As, bs, xs, num);

        // set cnt_x's okey flag
        cudaCheckCondition2(xs, cnt_x + 5 * batch_size, w_inequ, b_inequ, state_len, num);

        // update vertices_buffer、tab、bs_buffer1、bs_buffer2
        cudaUpdateVert(vertices_buffer, vert_max,
                       w_equ, w_inequ, state_len,
                       cnt_x + 5 * batch_size + 5 * s_max + 0 * batch_size,
                       cnt_x + 5 * batch_size + 5 * s_max + 1 * batch_size,
                       cnt_x + 5 * batch_size + 5 * s_max + 2 * batch_size,
                       batch_size, xs, tabij, cnt_x + 5 * batch_size, num);

        // reinit
        As_num[0] = 0;
        cudaInitCntX4(cnt_x + 5 * batch_size, num, // (state_idx, gt_startIdx, wrap_i, real_i, ok)
                      cnt_x + 0, As_num_ptr,
                      cnt_x + 5 * batch_size + 5 * s_max + 0 * batch_size,
                      cnt_x + 5 * batch_size + 5 * s_max + 1 * batch_size,
                      cnt_x + 5 * batch_size + 5 * s_max + 2 * batch_size,
                      cnt_x + 5 * batch_size + 5 * s_max + 3 * batch_size,
                      batch_size, state_len);
        if (As_num[0] == 0)
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void correctNormalDirection_kernel(FloatType *__restrict__ vertices_buffer, int vert_stride,
                                              const FloatType *__restrict__ w_equ,
                                              int batch_size,
                                              bool flip_insideout)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    FloatType *vert_ = vertices_buffer + tid * vert_stride;
    int num = int(vert_[0]);

    if (tid < batch_size && num >= 3)
    {
        FloatType ax = vert_[4] - vert_[1];
        FloatType ay = vert_[5] - vert_[2];
        FloatType az = vert_[6] - vert_[3];

        FloatType bx = vert_[7] - vert_[4];
        FloatType by = vert_[8] - vert_[5];
        FloatType bz = vert_[9] - vert_[6];

        FloatType cx = ay * bz - az * by;
        FloatType cy = az * bx - ax * bz;
        FloatType cz = ax * by - ay * bx;

        FloatType dir = cx * w_equ[3 * tid + 0] +
                        cy * w_equ[3 * tid + 1] +
                        cz * w_equ[3 * tid + 2];

        if ((dir < 0) ^ flip_insideout) // correct direction
        {
            FloatType x, y, z;
            int half_num = num / 2;
            for (int i = 0; i < half_num; ++i)
            {
                x = vert_[3 * i + 1];
                y = vert_[3 * i + 2];
                z = vert_[3 * i + 3];

                vert_[3 * i + 1] = vert_[3 * (num - i) - 2];
                vert_[3 * i + 2] = vert_[3 * (num - i) - 1];
                vert_[3 * i + 3] = vert_[3 * (num - i) - 0];

                vert_[3 * (num - i) - 2] = x;
                vert_[3 * (num - i) - 1] = y;
                vert_[3 * (num - i) - 0] = z;
            }
        }
    }
}

template <typename FloatType>
__host__ void correctNormalDirection(FloatType *vertices_buffer, int vert_max,
                                     FloatType *w_equ,
                                     int batch_size,
                                     bool flip_insideout)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, batch_size);
    const int block_num = int(std::ceil(batch_size / float(thread_num)));
    correctNormalDirection_kernel<<<block_num, thread_num>>>(vertices_buffer, 1 + 3 * vert_max, w_equ, batch_size, flip_insideout);
    CUDASYNC();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__global__ void inferNewStates_kernel(int *__restrict__ test_states_num, FloatType *__restrict__ temp_states, u_char *__restrict__ temp_uchar_states, // temp buffer to save the uchar (like temp_states)
                                      int *__restrict__ real_append_num, FloatType *__restrict__ write_states, FACEDATA<FloatType> *__restrict__ write_facedata,
                                      const FloatType *__restrict__ read_states, int state_len, int uchar_len,
                                      uint32_t *__restrict__ inplace_table, int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO,
                                      const int *__restrict__ tabij, const FloatType *__restrict__ vertices_buffer,
                                      int vert_max, int batch_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < batch_size * vert_max)
    {
        int batch_idx = tid % batch_size;
        int vert_idx = tid / batch_size;

        const int *tab = tabij + (2 + vert_max) * batch_idx + 1;
        const FloatType *vert = vertices_buffer + (1 + 3 * vert_max) * batch_idx + 1;

        int tabij_num = tab[-1];
        if (vert_idx < tabij_num - 1)
        {
            int switch_i = tab[vert_idx];
            if (switch_i >= 0 && switch_i < state_len) // must be positive && not extra constraint ????
            {
                int the_states_index = atomicAdd(test_states_num, 1); // used for counting

                int offset = state_len * the_states_index;

#ifdef _USE_CUDAMEMCPYASYNC // slower
                cudaMemcpyAsync(temp_states + offset,
                                read_states + state_len * batch_idx,
                                state_len * sizeof(FloatType),
                                cudaMemcpyDeviceToDevice);
                cudaDeviceSynchronize(); // must sync
#else
                // faster
                FloatType *temp_states__ = temp_states + offset;
                const FloatType *read_states__ = read_states + state_len * batch_idx;
                for (int i = 0; i < state_len; ++i)
                {
                    temp_states__[i] = read_states__[i];
                }
#endif

                temp_states[offset + switch_i] = FloatType(1) - temp_states[offset + switch_i];

                status okey = try_insert(temp_states + offset, state_len,
                                         inplace_table,
                                         temp_uchar_states + uchar_len * the_states_index, uchar_len,
                                         TABLE_SIZE_MASK, SKIP_LENGTH, HASH2_ZERO);

                if (okey == SUCCESS)
                {
                    int real_append_index = atomicAdd(real_append_num, 1);

                    // find the two vertices
                    int cnt_idx = -1;
                    int v1_idx;
                    int vert_idx_ = (vert_idx == 0) ? (tabij_num - 1) : vert_idx;
                    for (int i = 1; i < tabij_num; ++i) // except the first one
                    {
                        if (tab[i] >= 0)
                        {
                            cnt_idx++;
                        }
                        if (i == vert_idx_)
                        {
                            v1_idx = cnt_idx;
                        }
                    }
                    int v2_idx = (v1_idx + 1) % int(vert[-1]);

                    ///
                    write_facedata[real_append_index].start_idx = switch_i;
                    write_facedata[real_append_index].midpoint[0] = (vert[3 * v1_idx + 0] + vert[3 * v2_idx + 0]) / 2;
                    write_facedata[real_append_index].midpoint[1] = (vert[3 * v1_idx + 1] + vert[3 * v2_idx + 1]) / 2;
                    write_facedata[real_append_index].midpoint[2] = (vert[3 * v1_idx + 2] + vert[3 * v2_idx + 2]) / 2;

                    /// save the state
                    cudaMemcpyAsync(write_states + state_len * real_append_index,
                                    temp_states + offset,
                                    state_len * sizeof(FloatType),
                                    cudaMemcpyDeviceToDevice);
                }
            }
        }
    }
}

template <typename FloatType>
__host__ int inferNewStates(FloatType *temp_states, u_char *temp_uchar_states,
                            FloatType *write_states, FACEDATA<FloatType> *write_facedata,
                            FloatType *read_states, int state_len, int uchar_len,
                            uint32_t *inplace_table, int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO,
                            int *tabij, FloatType *vertices_buffer,
                            int vert_max, int batch_size)
{
    thrust::device_vector<int> num(2);
    num[0] = 0;
    num[1] = 0;
    int *test_states_num = thrust::raw_pointer_cast(num.data() + 0);
    int *real_append_num = thrust::raw_pointer_cast(num.data() + 1);

    const int total_num = batch_size * vert_max;
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, total_num);
    const int block_num = int(std::ceil(total_num / float(thread_num)));
    inferNewStates_kernel<<<block_num, thread_num>>>(test_states_num, temp_states, temp_uchar_states,
                                                     real_append_num, write_states, write_facedata,
                                                     read_states, state_len, uchar_len,
                                                     inplace_table, TABLE_SIZE_MASK, SKIP_LENGTH, HASH2_ZERO,
                                                     tabij, vertices_buffer,
                                                     vert_max, batch_size);

    CUDASYNC();

    return num[1]; // return `real_append_num`
}

#endif