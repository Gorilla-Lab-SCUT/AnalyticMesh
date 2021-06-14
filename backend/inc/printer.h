#ifndef __PRINTER_H__
#define __PRINTER_H__

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "facedata.h"

namespace PRINTER
{
template <typename T>
__host__ void h_print(T *ptr, int rows, int cols = -1, const char *ending = "\n");

__device__ void d_print(float *ptr, int rows, int cols = -1, const char *ending = "\n");
__device__ void d_print(double *ptr, int rows, int cols = -1, const char *ending = "\n");
__device__ void d_print(bool *ptr, int rows, int cols = -1, const char *ending = "\n");
__device__ void d_print(int *ptr, int rows, int cols = -1, const char *ending = "\n");
__device__ void d_print(u_char *ptr, int rows, int cols = -1, const char *ending = "\n");

template <typename DataType>
__host__ void dump(std::string file_name, DataType *ptr_, int height, int width, bool ptr_on_device = true, int precision = 15);

template <typename DataType>
__host__ void dump(DataType *ptr_, int height, int width, bool ptr_on_device = true, int precision = 6);

template <typename FloatType>
__host__ void dump(std::string file_name, FACEDATA<FloatType> *ptr_, int num, bool ptr_on_device = true, int precision = 15);
} // namespace PRINTER

////////////////////////////////////////////////////////////////

#define DEFINE_D_PRINT_FUNC(T, str, expr1, expr2)                                    \
    __device__ void PRINTER::d_print(T *ptr, int rows, int cols, const char *ending) \
    {                                                                                \
        if (cols > 0)                                                                \
        {                                                                            \
            printf("shape==(%d, %d) : \n", rows, cols);                              \
            for (int i = 0; i < rows; ++i)                                           \
            {                                                                        \
                for (int j = 0; j < cols; ++j)                                       \
                {                                                                    \
                    printf(str, expr1);                                              \
                }                                                                    \
                printf("\n");                                                        \
            }                                                                        \
            printf(ending);                                                          \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            printf("shape==(%d, ) : \n", rows);                                      \
            for (int i = 0; i < rows; ++i)                                           \
            {                                                                        \
                printf(str, expr2);                                                  \
            }                                                                        \
            printf("\n");                                                            \
            printf(ending);                                                          \
        }                                                                            \
    }

template <typename T>
__host__ void PRINTER::h_print(T *ptr, int rows, int cols, const char *ending)
{
    if (cols > 0) // 2D array
    {
        std::cout << "shape==(" << rows << ", " << cols << ") : " << std::endl;
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                if (std::is_same<T, u_char>::value) // uchar
                    std::cout << int(ptr[cols * i + j]) << '\t';
                else
                    std::cout << ptr[cols * i + j] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << ending;
    }
    else // 1D array (show in one row)
    {
        std::cout << "shape==(" << rows << ", ) : " << std::endl;
        for (int i = 0; i < rows; ++i)
        {
            if (std::is_same<T, u_char>::value) // uchar
                std::cout << int(ptr[i]) << '\t';
            else
                std::cout << ptr[i] << '\t';
        }
        std::cout << std::endl;
        std::cout << ending;
    }
}

DEFINE_D_PRINT_FUNC(float, "%f\t", ptr[cols * i + j], ptr[i])
DEFINE_D_PRINT_FUNC(double, "%f\t", ptr[cols * i + j], ptr[i])
DEFINE_D_PRINT_FUNC(bool, "%d  ", ptr[cols * i + j], ptr[i])
DEFINE_D_PRINT_FUNC(int, "%d\t", ptr[cols * i + j], ptr[i])
DEFINE_D_PRINT_FUNC(u_char, "%d\t", int(ptr[cols * i + j]), int(ptr[i]))

template <typename DataType>
__host__ void PRINTER::dump(std::string file_name, DataType *ptr_, int height, int width, bool ptr_on_device, int precision)
{
    DataType *ptr = nullptr;
    if (ptr_on_device)
    {
        ptr = new DataType[height * width];
        gpuErrchk(cudaMemcpy(ptr, ptr_, height * width * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    else
    {
        ptr = ptr_;
    }

    std::ofstream stm(file_name);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            stm << std::setprecision(precision) << ptr[i * width + j] << ' ';
        }
        stm << '\n';
    }
    stm.close();

    if (ptr_on_device)
        delete[] ptr;
}

template <typename DataType>
__host__ void PRINTER::dump(DataType *ptr_, int height, int width, bool ptr_on_device, int precision)
{
    DataType *ptr = nullptr;
    if (ptr_on_device)
    {
        ptr = new DataType[height * width];
        gpuErrchk(cudaMemcpy(ptr, ptr_, height * width * sizeof(DataType), cudaMemcpyDeviceToHost));
    }
    else
    {
        ptr = ptr_;
    }

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            std::cout << std::setprecision(precision) << ptr[i * width + j] << ' ';
        }
        std::cout << '\n';
    }

    if (ptr_on_device)
        delete[] ptr;
}

template <typename FloatType>
__host__ void PRINTER::dump(std::string file_name, FACEDATA<FloatType> *ptr_, int num, bool ptr_on_device, int precision)
{
    FACEDATA<FloatType> *ptr = nullptr;
    if (ptr_on_device)
    {
        ptr = new FACEDATA<FloatType>[num];
        gpuErrchk(cudaMemcpy(ptr, ptr_, num * sizeof(FACEDATA<FloatType>), cudaMemcpyDeviceToHost));
    }
    else
    {
        ptr = ptr_;
    }

    std::ofstream stm(file_name);
    for (int i = 0; i < num; ++i)
    {
        stm << std::setprecision(precision) << ptr[i].midpoint[0] << ' ' << ptr[i].midpoint[1] << ' ' << ptr[i].midpoint[2] << ' ' << ptr[i].start_idx << std::endl;
    }
    stm.close();

    if (ptr_on_device)
        delete[] ptr;
}

#endif