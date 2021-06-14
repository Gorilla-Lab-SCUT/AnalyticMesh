#ifndef __VECMAT_H__
#define __VECMAT_H__

#include <torch/extension.h>

template <typename FloatType>
class VEC
{
public:
    FloatType *data = nullptr;
    int len = 0;
    __host__ VEC() {}
    __host__ VEC(const torch::Tensor &bias)
    {
        data = bias.data_ptr<FloatType>();
        len = int(bias.size(0));
    }
};

////////////////////////////////////////////////////////////////////////

template <typename FloatType>
class MAT
{
public:
    FloatType *data = nullptr;
    int height = 0;
    int width = 0;
    __host__ MAT() {}
    __host__ MAT(const torch::Tensor &weight)
    {
        data = weight.data_ptr<FloatType>();
        height = int(weight.size(0));
        width = int(weight.size(1));
    }
    __host__ MAT(FloatType *data_, int height_, int width_) : data(data_), height(height_), width(width_) {}
    __host__ FloatType &operator()(const int i, const int j)
    {
        return data[i * width + j];
    }
};

#endif