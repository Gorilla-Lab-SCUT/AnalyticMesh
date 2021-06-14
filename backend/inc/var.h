#ifndef __VAR_H__
#define __VAR_H__

#include <thrust/device_vector.h>

#include "utilities.h"
#include "states.h"
#include "mlp.h"
#include "polymesh.h"

template <typename FloatType>
class VAR
{
public:
    // const setting
    const int BATCH_SIZE_MAX = _VAR_BATCH_SIZE_MAX;
    const int SOLVE_SIZE_MAX = _VAR_SOLVE_SIZE_MAX;
    const int VERT_MAX = _VAR_VERT_MAX;

    // init somethings
    MLP<FloatType> mlp;
    STATES<FloatType> states;
    PolyMesh<FloatType, _VAR_VERT_MAX> polymesh;

    int num_extra_constraints;

    FloatType *w_inequ = nullptr;
    FloatType *b_inequ = nullptr;
    FloatType *w_equ = nullptr;
    FloatType *b_equ = nullptr;
    FloatType *w_temp = nullptr;
    FloatType *b_temp = nullptr;
    FloatType *distance = nullptr;
    FloatType *As = nullptr;
    FloatType *bs = nullptr;
    FloatType *xs = nullptr;
    int *cnt_x = nullptr;
    FloatType *vertices_buffer = nullptr;
    int *tabij = nullptr;

    FloatType *temp_states = nullptr;
    u_char *temp_uchar_states = nullptr;
    FloatType *write_states = nullptr;
    FACEDATA<FloatType> *write_facedata = nullptr;

    thrust::device_vector<int> segments;
    thrust::device_vector<int> key_vec;
    int *dist_idx;

    cublasHandle_t handle;

    void Init(int num_extra_constraints_);
    void Destroy();
};

template <typename FloatType>
void VAR<FloatType>::Init(int num_extra_constraints_)
{
    num_extra_constraints = num_extra_constraints_;

    w_inequ = nullptr;
    b_inequ = nullptr;
    w_equ = nullptr;
    b_equ = nullptr;
    w_temp = nullptr;
    b_temp = nullptr;
    distance = nullptr;
    As = nullptr;
    bs = nullptr;
    xs = nullptr;
    cnt_x = nullptr;
    vertices_buffer = nullptr;
    tabij = nullptr;
    temp_states = nullptr;
    temp_uchar_states = nullptr;
    write_states = nullptr;
    write_facedata = nullptr;
    gpuErrchk(cudaMalloc(&w_inequ, 3 * (states.state_len + num_extra_constraints) * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&b_inequ, 1 * (states.state_len + num_extra_constraints) * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&w_equ, 3 * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&b_equ, 1 * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&w_temp, 3 * mlp.max_width * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&b_temp, 1 * mlp.max_width * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&distance, 1 * (states.state_len + num_extra_constraints) * BATCH_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&As, 9 * SOLVE_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&bs, 3 * SOLVE_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&xs, 3 * SOLVE_SIZE_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&cnt_x, (5 * BATCH_SIZE_MAX + 5 * SOLVE_SIZE_MAX + 4 * BATCH_SIZE_MAX) * sizeof(int)));
    gpuErrchk(cudaMalloc(&vertices_buffer, ((1 + 3 * VERT_MAX) * BATCH_SIZE_MAX) * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&tabij, ((2 + VERT_MAX) * BATCH_SIZE_MAX) * sizeof(int)));
    gpuErrchk(cudaMalloc(&temp_states, (VERT_MAX * BATCH_SIZE_MAX * states.state_len) * sizeof(FloatType)));    // do not add `num_extra_constraints` ???
    gpuErrchk(cudaMalloc(&temp_uchar_states, (VERT_MAX * BATCH_SIZE_MAX * states.uchar_len) * sizeof(u_char))); // do not add `num_extra_constraints` ???
    gpuErrchk(cudaMalloc(&write_states, (VERT_MAX * BATCH_SIZE_MAX * states.state_len) * sizeof(FloatType)));   // do not add `num_extra_constraints` ???
    gpuErrchk(cudaMalloc(&write_facedata, (VERT_MAX * BATCH_SIZE_MAX) * sizeof(FACEDATA<FloatType>)));

    segments.resize((states.state_len + num_extra_constraints) * BATCH_SIZE_MAX);
    key_vec.resize((states.state_len + num_extra_constraints) * BATCH_SIZE_MAX);
    dist_idx = thrust::raw_pointer_cast(key_vec.data());

    cublasErrchk(cublasCreate(&handle));
}

template <typename FloatType>
void VAR<FloatType>::Destroy()
{
    if (w_inequ)
        gpuErrchk(cudaFree(w_inequ));
    if (b_inequ)
        gpuErrchk(cudaFree(b_inequ));
    if (w_equ)
        gpuErrchk(cudaFree(w_equ));
    if (b_equ)
        gpuErrchk(cudaFree(b_equ));
    if (w_temp)
        gpuErrchk(cudaFree(w_temp));
    if (b_temp)
        gpuErrchk(cudaFree(b_temp));
    if (distance)
        gpuErrchk(cudaFree(distance));
    if (As)
        gpuErrchk(cudaFree(As));
    if (bs)
        gpuErrchk(cudaFree(bs));
    if (xs)
        gpuErrchk(cudaFree(xs));
    if (cnt_x)
        gpuErrchk(cudaFree(cnt_x));
    if (vertices_buffer)
        gpuErrchk(cudaFree(vertices_buffer));
    if (tabij)
        gpuErrchk(cudaFree(tabij));
    if (temp_states)
        gpuErrchk(cudaFree(temp_states));
    if (temp_uchar_states)
        gpuErrchk(cudaFree(temp_uchar_states));
    if (write_states)
        gpuErrchk(cudaFree(write_states));
    if (write_facedata)
        gpuErrchk(cudaFree(write_facedata));

    w_inequ = nullptr;
    b_inequ = nullptr;
    w_equ = nullptr;
    b_equ = nullptr;
    w_temp = nullptr;
    b_temp = nullptr;
    distance = nullptr;
    As = nullptr;
    bs = nullptr;
    xs = nullptr;
    cnt_x = nullptr;
    vertices_buffer = nullptr;
    tabij = nullptr;
    temp_states = nullptr;
    temp_uchar_states = nullptr;
    write_states = nullptr;
    write_facedata = nullptr;

    cublasErrchk(cublasDestroy(handle));
}

#endif