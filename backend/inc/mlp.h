
#ifndef __MLP_H__
#define __MLP_H__

#include <vector>
#include <torch/extension.h>
#include "vecmat.h"

template <typename FloatType>
class MLP
{
public:
    int fc_layers_num = 0;
    int max_width = 0;
    MAT<FloatType> *weights = nullptr;
    VEC<FloatType> *biases = nullptr;
    MAT<FloatType> *transform = nullptr;
    MAT<int> arc_table;
    std::vector<int> offset_vec;

    __host__ MLP() {}

    __host__ void init(const torch::Tensor &arc_table_, const std::vector<int> &nodesnum);
    __host__ void reinit(const std::vector<torch::Tensor> &weights_,
                         const std::vector<torch::Tensor> &biases_,
                         const std::vector<torch::Tensor> &arc_tm_);
    __host__ void cudaFreeMemory();
};

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void MLP<FloatType>::init(const torch::Tensor &arc_table_, const std::vector<int> &nodesnum) // init architecture
{
    arc_table.data = arc_table_.data_ptr<int>();
    arc_table.height = int(arc_table_.size(0));
    arc_table.width = int(arc_table_.size(1));

    fc_layers_num = int(nodesnum.size() - 1);

    if (offset_vec.size())
        offset_vec.clear();
    int s = 0;
    offset_vec.push_back(0);
    for (int i = 1; i <= fc_layers_num; ++i)
    {
        s += nodesnum[i];
        offset_vec.push_back(s);
    }

    int transform_size = 0;
    for (int i = 0; i < arc_table.height; ++i)
    {
        for (int j = 0; j < arc_table(i, 0); ++j)
        {
            transform_size = MAX2(transform_size, arc_table(i, 2 + 2 * j));
        }
    }
    transform_size += 1; // not index

    if (weights)
        delete[] weights;
    if (biases)
        delete[] biases;
    if (transform)
        delete[] transform;
    weights = new MAT<FloatType>[fc_layers_num];
    biases = new VEC<FloatType>[fc_layers_num];
    transform = new MAT<FloatType>[transform_size];

    max_width = 0;
    for (auto w : nodesnum)
        max_width = MAX2(max_width, w);
}

template <typename FloatType>
__host__ void MLP<FloatType>::reinit(const std::vector<torch::Tensor> &weights_,
                                     const std::vector<torch::Tensor> &biases_,
                                     const std::vector<torch::Tensor> &arc_tm_) // init params
{
    for (int i = 0; i < fc_layers_num; ++i)
    {
        weights[i] = MAT<FloatType>(weights_[i]);
        biases[i] = VEC<FloatType>(biases_[i]);
    }

    for (size_t i = 0; i < arc_tm_.size(); ++i)
        transform[i] = MAT<FloatType>(arc_tm_[i]);
}

template <typename FloatType>
__host__ void MLP<FloatType>::cudaFreeMemory()
{
    if (weights)
        delete[] weights;
    if (biases)
        delete[] biases;
    if (transform)
        delete[] transform;
    weights = nullptr;
    biases = nullptr;
    transform = nullptr;
}

#endif