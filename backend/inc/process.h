#ifndef __PROCESS_H__
#define __PROCESS_H__

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include "utilities.h"
#include "kernel.h"
#include "mlp.h"
#include "facedata.h"
#include "var.h"

#include "vecmat.h"

template <typename FloatType>
void fill_constraints(MLP<FloatType> &mlp, STATES<FloatType> &states,
                      int batch_size,
                      FloatType *w_inequ,
                      FloatType *b_inequ,
                      FloatType *w_equ,
                      FloatType *b_equ,
                      FloatType *w_temp,
                      FloatType *b_temp,
                      MAT<FloatType> &w_extra_constraints,
                      VEC<FloatType> &b_extra_constraints,
                      int num_extra_constraints,
                      FloatType *states_buffer,
                      cublasHandle_t handle)
{
    int all_constraints_len = states.state_len + num_extra_constraints;

    // copy weights[0] and biases[0] to w_inequ and b_inequ respectively
    cudaMemcpyStrided_D2D((FourBytes *)w_inequ, 3 * all_constraints_len * sizeof(FloatType) / 4, batch_size,
                          (FourBytes *)(mlp.weights[0].data), 3 * mlp.biases[0].len * sizeof(FloatType) / 4);
    cudaMemcpyStrided_D2D((FourBytes *)b_inequ, all_constraints_len * sizeof(FloatType) / 4, batch_size,
                          (FourBytes *)(mlp.biases[0].data), mlp.biases[0].len * sizeof(FloatType) / 4);

    int i = 1;
    while (i < mlp.fc_layers_num - 1)
    {
        int state_offset = mlp.offset_vec[i];
        int state_dec = mlp.biases[i - 1].len;

        // multiplied by mask (save to `temp`)
        cudaMaskRowsStrided<3>(states_buffer + (state_offset - state_dec), states.state_len,
                               w_inequ + 3 * (state_offset - state_dec), 3 * all_constraints_len,
                               w_temp, 3 * mlp.max_width,
                               3 * state_dec,
                               batch_size);
        cudaMaskRowsStrided<1>(states_buffer + (state_offset - state_dec), states.state_len,
                               b_inequ + (state_offset - state_dec), all_constraints_len,
                               b_temp, mlp.max_width,
                               state_dec,
                               batch_size);

        // multiplied by weights[i]
        cublasMatMulBatched(handle,
                            mlp.weights[i],
                            w_temp, state_dec, 3, 3 * mlp.max_width,
                            w_inequ + 3 * state_offset, 3 * all_constraints_len,
                            batch_size);
        cublasMatVecMulAddBatched(handle,
                                  mlp.weights[i],
                                  b_temp, state_dec, mlp.max_width,
                                  mlp.biases[i],
                                  b_inequ + state_offset, mlp.biases[i].len, all_constraints_len,
                                  batch_size);

        // add connection
        int connection_num = mlp.arc_table(i - 1, 0);
        if (connection_num != 0)
        {
            for (int ci = 0; ci < connection_num; ++ci)
            {
                int from = mlp.arc_table(i - 1, 1 + 2 * ci);
                MAT<FloatType> transform_matrix = mlp.transform[mlp.arc_table(i - 1, 2 + 2 * ci)];

                int os = mlp.offset_vec[from - 1];
                int dec = mlp.weights[from - 1].height;

                if (transform_matrix.height == 0 && transform_matrix.width == 0) // identity matrix
                {
                    if (from == 0)
                    {
                        // add 3x3 identity matrix
                        cudaAddIdentityBatched(w_inequ + 3 * state_offset,
                                               3, 3, 3 * all_constraints_len,
                                               batch_size);
                    }
                    else
                    {
                        cudaMaskRowsAddStrided<3>(states_buffer + os, states.state_len,
                                                  w_inequ + 3 * os, 3 * all_constraints_len,
                                                  w_inequ + 3 * state_offset,
                                                  3 * dec,
                                                  batch_size);
                        cudaMaskRowsAddStrided<1>(states_buffer + os, states.state_len,
                                                  b_inequ + os, all_constraints_len,
                                                  b_inequ + state_offset,
                                                  dec,
                                                  batch_size);
                    }
                }
                else
                {
                    if (from == 0)
                    {
                        cudaMatAddBatched(transform_matrix.data, 0,
                                          w_inequ + 3 * state_offset, 3 * all_constraints_len,
                                          transform_matrix.height * transform_matrix.width,
                                          batch_size);
                    }
                    else
                    {
                        // mask
                        cudaMaskRowsStrided<3>(states_buffer + os, states.state_len,
                                               w_inequ + 3 * os, 3 * all_constraints_len,
                                               w_temp, 3 * mlp.max_width,
                                               3 * dec,
                                               batch_size);
                        cudaMaskRowsStrided<1>(states_buffer + os, states.state_len,
                                               b_inequ + os, all_constraints_len,
                                               b_temp, mlp.max_width,
                                               dec,
                                               batch_size);

                        // multiplied by weight
                        cublasMatMulBatched(handle,
                                            transform_matrix,
                                            w_temp, dec, 3, 3 * mlp.max_width,
                                            w_inequ + 3 * state_offset, 3 * all_constraints_len,
                                            batch_size,
                                            FloatType(1.0)); // add
                        cublasMatVecMulAddBatched(handle,
                                                  transform_matrix,
                                                  b_temp, dec, mlp.max_width,
                                                  VEC<FloatType>(),
                                                  b_inequ + state_offset, mlp.biases[i].len, all_constraints_len,
                                                  batch_size); //
                    }
                }
            }
        }

        i++;
    }

    int state_offset = mlp.offset_vec[i];
    int state_dec = mlp.biases[i - 1].len;

    // multiplied by mask
    cudaMaskRowsStrided<3>(states_buffer + (state_offset - state_dec), states.state_len,
                           w_inequ + 3 * (state_offset - state_dec), 3 * all_constraints_len,
                           w_temp, 3 * mlp.max_width,
                           3 * state_dec,
                           batch_size);
    cudaMaskRowsStrided<1>(states_buffer + (state_offset - state_dec), states.state_len,
                           b_inequ + (state_offset - state_dec), all_constraints_len,
                           b_temp, mlp.max_width,
                           state_dec,
                           batch_size);

    // multiplied by weights[i]
    cublasMatMulBatched(handle,
                        mlp.weights[i],
                        w_temp, state_dec, 3, 3 * mlp.max_width,
                        w_equ, 3, // w_inequ + 3 * state_offset, 3 * states.state_len,
                        batch_size);

    cublasMatVecMulAddBatched(handle,
                              mlp.weights[i],
                              b_temp, state_dec, mlp.max_width,
                              mlp.biases[i],
                              b_equ, 1, 1, // b_inequ + state_offset, state_inc, states.state_len,
                              batch_size);

    // sign the constraints
    cudaSignConstraints<3>(states_buffer, states.state_len,
                           w_inequ, 3 * all_constraints_len,
                           batch_size);
    cudaSignConstraints<1>(states_buffer, states.state_len,
                           b_inequ, all_constraints_len,
                           batch_size);

    // copy extra constraints
    cudaMemcpyStrided_D2D((FourBytes *)(w_inequ + 3 * state_offset), 3 * all_constraints_len * sizeof(FloatType) / 4, batch_size,
                          (FourBytes *)(w_extra_constraints.data), 3 * num_extra_constraints * sizeof(FloatType) / 4);
    cudaMemcpyStrided_D2D((FourBytes *)(b_inequ + state_offset), all_constraints_len * sizeof(FloatType) / 4, batch_size,
                          (FourBytes *)(b_extra_constraints.data), num_extra_constraints * sizeof(FloatType) / 4);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
void offset_b_equ(FloatType *b_equ, FloatType iso, int batch_size)
{
    thrust::device_ptr<FloatType> b_equ_ptr(b_equ);
    thrust::for_each(b_equ_ptr, b_equ_ptr + batch_size, thrust::placeholders::_1 -= iso);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType, bool DistanceIn3D = true>
void fill_distance(STATES<FloatType> &states,
                   FloatType *w_inequ,
                   FloatType *b_inequ,
                   FloatType *w_equ,
                   int batch_size,
                   FACEDATA<FloatType> *faces_buffer,
                   FloatType *distance,
                   int num_extra_constraints,
                   cublasHandle_t handle)
{
    int all_constraints_len = states.state_len + num_extra_constraints;

    // W * x + b (should be negative)
    gpuErrchk(cudaMemcpy(distance, b_inequ, all_constraints_len * batch_size * sizeof(FloatType), cudaMemcpyDeviceToDevice));
    cublasMatMulBatched(handle,
                        MAT<FloatType>(w_inequ, all_constraints_len, 3),
                        faces_buffer[0].midpoint, 3, 1, sizeof(FACEDATA<FloatType>) / sizeof(FloatType),
                        distance, all_constraints_len,
                        batch_size,
                        FloatType(1.0),
                        all_constraints_len * 3);
    // calc distance
    if (DistanceIn3D)
    {
        cudaRow3NormDiv(w_inequ, distance, all_constraints_len * batch_size); // distance in 3d cell (seem to be better)
    }
    else
    {
        cudaWrapDistance(w_inequ, w_equ, distance, all_constraints_len, batch_size); // distance on w_equ plane
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void sortDistance(FloatType *value, // value (will change??)
                           thrust::device_vector<int> &segments,
                           thrust::device_vector<int> &key_vec,
                           int one_batch_len,
                           int batch_num)
{
    thrust::device_ptr<FloatType> value_ptr(value); // it will not copy array

    thrustSortBatched_fill(thrust::raw_pointer_cast(segments.data()),
                           thrust::raw_pointer_cast(key_vec.data()),
                           one_batch_len,
                           batch_num); // init

    // sort
    auto first = thrust::make_zip_iterator(thrust::make_tuple(segments.begin(), key_vec.begin()));
    thrust::stable_sort_by_key(value_ptr, value_ptr + one_batch_len * batch_num, first, thrust::greater<FloatType>());

    // NOTE: thrust of old version may fail to execute this command (and compile with warnings)
    thrust::stable_sort_by_key(segments.begin(), segments.begin() + one_batch_len * batch_num, key_vec.begin(), thrust::less<int>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void fillStartIdx(FACEDATA<FloatType> *faces_buffer, // modify in place
                           int *dist_idx,
                           FloatType *As,
                           FloatType *bs,
                           FloatType *xs,
                           int *cnt_x,
                           int solve_size,
                           FloatType *w_inequ,
                           FloatType *b_inequ,
                           FloatType *w_equ,
                           FloatType *b_equ,
                           int state_len,
                           int batch_size)
{
    calcStartIdx(&faces_buffer[0].start_idx, sizeof(FACEDATA<FloatType>) / sizeof(int),
                 dist_idx, As, bs, xs, cnt_x, solve_size, w_inequ, b_inequ, w_equ, b_equ, state_len, batch_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void vertPivoting(FACEDATA<FloatType> *faces_buffer,
                           FloatType *vertices_buffer, int vert_max,
                           int *dist_idx,
                           FloatType *As,
                           FloatType *bs,
                           FloatType *xs,
                           int *cnt_x,
                           int s_max,
                           int *tabij,
                           FloatType *w_inequ,
                           FloatType *b_inequ,
                           FloatType *w_equ,
                           FloatType *b_equ,
                           int state_len,
                           int batch_size)
{
    calcPivoting(&faces_buffer[0].start_idx, sizeof(FACEDATA<FloatType>) / sizeof(int), vertices_buffer, vert_max, dist_idx,
                 As, bs, xs, cnt_x, s_max, tabij, w_inequ, b_inequ, w_equ, b_equ, state_len, batch_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void corrNormDir(FloatType *vertices_buffer, int vert_max,
                          FloatType *w_equ,
                          int batch_size, bool flip_insideout)
{
    correctNormalDirection(vertices_buffer, vert_max, w_equ, batch_size, flip_insideout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ void inferAndAppend(FloatType *states_buffer, int batch_size, VAR<FloatType> &var)
{
    int real_append_num = inferNewStates(var.temp_states, var.temp_uchar_states,
                                         var.write_states, var.write_facedata,
                                         states_buffer, var.states.state_len, var.states.uchar_len,
                                         var.states.table, var.states.TABLE_SIZE - 1, var.states.SKIP_LENGTH, var.states.HASH2_ZERO,
                                         var.tabij, var.vertices_buffer,
                                         var.VERT_MAX, batch_size);
    var.states.push_back(var.write_states, var.write_facedata, real_append_num);
}

#endif