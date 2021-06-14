#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <cstdio>

#include "macro.h"

#define MIN2(x, y) (((x) < (y)) ? (x) : (y))
#define MAX2(x, y) (((x) > (y)) ? (x) : (y))

using FourBytes = int; // `int` or `unsigned int`

template <typename FloatType>
constexpr FloatType FP_EPS = std::is_same<FloatType, float>::value ? _FP_EPS_FLOAT : _FP_EPS_DOUBLE;

template <typename FloatType>
constexpr FloatType FP_SAME_EPS = std::is_same<FloatType, float>::value ? _FP_SAME_EPS_FLOAT : _FP_SAME_EPS_DOUBLE;

template <typename FloatType>
constexpr int FP_POINTHASH_MUL = std::is_same<FloatType, float>::value ? _FP_POINTHASH_MUL_FLOAT : _FP_POINTHASH_MUL_DOUBLE;

#ifdef _REMOVE_ZERO_CONS
template <typename FloatType>
constexpr FloatType FP_ZERO_CONS = std::is_same<FloatType, float>::value ? _FP_ZERO_CONS_FLOAT : _FP_ZERO_CONS_DOUBLE;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////

namespace CUBLAS_ERR
{
static const char *cuBLASGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS: The operation completed successfully.";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED: The cuBLAS library was not initialized.";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed inside the cuBLAS library.";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE: An unsupported value or parameter was passed to the function (a negative vector size, for example).";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH: The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR: An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED: The GPU program failed to execute.";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR: An internal cuBLAS operation failed.";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED: The functionnality requested is not supported";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR: The functionnality requested requires some license and an error was detected when trying to check the current licensing.";
    }

    return "<unknown>";
}

inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuBLASassert: %s %s %d\n", cuBLASGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

}; // namespace CUBLAS_ERR

namespace CUDA_ERR
{
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
}; // namespace CUDA_ERR

#define gpuErrchk(ans)                                  \
    {                                                   \
        CUDA_ERR::gpuAssert((ans), __FILE__, __LINE__); \
    }

#define cublasErrchk(ans)                                    \
    {                                                        \
        CUBLAS_ERR::cublasAssert((ans), __FILE__, __LINE__); \
    }

////////////////////////////////////////////////////////////////////////////////////////////////

namespace DBPRINT
{
__host__ __device__ inline void dbPrint(const char *words, const char *file, const int line)
{
    if (words)
        printf("DEBUG_PRINT ==> file: %s | line: %d | message: %s\n", file, line, words);
    else
        printf("DEBUG_PRINT ==> file: %s | line: %d\n", file, line);
}
}; // namespace DBPRINT

#define DB_PRINT(words)                                \
    {                                                  \
        DBPRINT::dbPrint((words), __FILE__, __LINE__); \
    }

////////////////////////////////////////////////////////////////////////////////////////////////

class GPUTIMER
{
private:
    cudaEvent_t start_event, stop_event;
    cudaStream_t stream;

public:
    __host__ GPUTIMER(cudaStream_t stream_ = 0);
    __host__ ~GPUTIMER();
    __host__ void start();
    __host__ void stop();
    __host__ float get_time();
    __host__ std::string get_time_str(float elapsedTime);
    __host__ std::string get_time_str();
};

////////////////////////////////////////////////////////////////////////////////////////////////

void checkDeviceInfo();

#endif