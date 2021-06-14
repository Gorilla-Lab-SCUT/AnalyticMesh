#ifndef __POLYMESH_H__
#define __POLYMESH_H__

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <string>

#include "utilities.h"

template <typename FloatType, int VERT_MAX>
class PolyMesh
{
public:
    const int INIT_NUM = _POLYMESH_INIT_NUM;
    const int INC_RATE = _POLYMESH_INC_RATE;
    const int STRIDE = (1 + 3 * VERT_MAX);
    const int HASH_LEN = _POLYMESH_HASH_LEN;
    const int HASH_EMPTY = -1;
    const uint32_t SKIP_LENGTH = _POLYMESH_SKIP_LENGTH;
    const int STATES_ASSUMED_MAX = _POLYMESH_STATES_ASSUMED_MAX;

    int _verts_buffer_num;
    int _verts_buffer_capacity;
    FloatType *_verts_buffer = nullptr; // raw data (gpu)   init size = INIT_NUM ,  can extend

    int *hash_array = nullptr;               // (index, hash2)   size = HASH_LEN*2 ,  cannot extend
    int *unique_verts_offset_list = nullptr; //    size = STATES_ASSUMED_MAX ,  cannot extend

    FloatType *vertices_d = nullptr; // device    size = STATES_ASSUMED_MAX*3,  cannot extend
    int *faces_d = nullptr;          // device    size = STATES_ASSUMED_MAX*(1+VERT_MAX),  cannot extend
    FloatType *vertices = nullptr;   // host     size = STATES_ASSUMED_MAX*3,  cannot extend
    int *faces = nullptr;            // host     size = STATES_ASSUMED_MAX*(1+VERT_MAX),  cannot extend

    int verts_num = 0;
    int faces_num = 0;

    void init();
    void reinit();
    void cudaFreeMemory();
    void append_verts(FloatType *device_verts, int add_num);
    int uniqueVertHash();
    void indexingVertices(int verts_num, FloatType scale, FloatType center_x, FloatType center_y, FloatType center_z);
    int indexingFaces();
    void combine_mesh(FloatType scale, FloatType center_x, FloatType center_y, FloatType center_z);
    template <typename ExportFloatType, bool IsPoly>
    void export_mesh(std::string file_path);
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::init()
{
    if (_verts_buffer)
        gpuErrchk(cudaFree(_verts_buffer));
    if (hash_array)
        gpuErrchk(cudaFree(hash_array));
    if (unique_verts_offset_list)
        gpuErrchk(cudaFree(unique_verts_offset_list));
    if (vertices_d)
        gpuErrchk(cudaFree(vertices_d));
    if (faces_d)
        gpuErrchk(cudaFree(faces_d));
    if (vertices)
        delete[] vertices;
    if (faces)
        delete[] faces;

    _verts_buffer_capacity = INIT_NUM;
    gpuErrchk(cudaMalloc(&_verts_buffer, STRIDE * _verts_buffer_capacity * sizeof(FloatType)));

    gpuErrchk(cudaMalloc(&hash_array, HASH_LEN * 2 * sizeof(int)));

    gpuErrchk(cudaMalloc(&unique_verts_offset_list, STATES_ASSUMED_MAX * sizeof(int)));

    gpuErrchk(cudaMalloc(&vertices_d, (3) * STATES_ASSUMED_MAX * sizeof(FloatType)));
    gpuErrchk(cudaMalloc(&faces_d, (1 + VERT_MAX) * STATES_ASSUMED_MAX * sizeof(int)));
    vertices = new FloatType[(3) * STATES_ASSUMED_MAX];
    faces = new int[(1 + VERT_MAX) * STATES_ASSUMED_MAX];
}

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::reinit()
{
    _verts_buffer_num = 0;

    thrust::device_ptr<int> hash_ptr(hash_array);
    thrust::fill(hash_ptr, hash_ptr + HASH_LEN * 2, HASH_EMPTY);
}

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::cudaFreeMemory()
{
    if (_verts_buffer)
        gpuErrchk(cudaFree(_verts_buffer));
    if (hash_array)
        gpuErrchk(cudaFree(hash_array));
    if (unique_verts_offset_list)
        gpuErrchk(cudaFree(unique_verts_offset_list));
    if (vertices_d)
        gpuErrchk(cudaFree(vertices_d));
    if (faces_d)
        gpuErrchk(cudaFree(faces_d));
    if (vertices)
        delete[] vertices;
    if (faces)
        delete[] faces;

    _verts_buffer = nullptr;
    hash_array = nullptr;
    unique_verts_offset_list = nullptr;
    vertices_d = nullptr;
    faces_d = nullptr;
    vertices = nullptr;
    faces = nullptr;
}

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::append_verts(FloatType *device_verts, int add_num) // note: append in a raw mode, which contains zero term
{
    if (_verts_buffer_capacity - _verts_buffer_num < add_num)
    {
        FloatType *_verts_buffer_temp = nullptr;
        gpuErrchk(cudaMalloc(&_verts_buffer_temp, STRIDE * _verts_buffer_capacity * sizeof(FloatType)));
        gpuErrchk(cudaMemcpy(_verts_buffer_temp, _verts_buffer, STRIDE * _verts_buffer_capacity * sizeof(FloatType), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaFree(_verts_buffer));
        int inc = INC_RATE;
        while (inc * _verts_buffer_capacity - _verts_buffer_num < add_num)
            inc *= INC_RATE;
        gpuErrchk(cudaMalloc(&_verts_buffer, STRIDE * _verts_buffer_capacity * inc * sizeof(FloatType)));
        gpuErrchk(cudaMemcpy(_verts_buffer, _verts_buffer_temp, STRIDE * _verts_buffer_capacity * sizeof(FloatType), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaFree(_verts_buffer_temp));
        _verts_buffer_capacity *= inc;
    }

    gpuErrchk(cudaMemcpy(_verts_buffer + STRIDE * _verts_buffer_num, device_verts, STRIDE * add_num * sizeof(FloatType), cudaMemcpyDeviceToDevice));
    _verts_buffer_num += add_num;
}

/////////////////////////////////////////////////////

// FNV-1a hash (32bit)
__device__ uint32_t PointHash1(const int *__restrict__ data_)
{
    const u_char *uchar_ = (u_char *)data_;
    uint32_t hash = 0x811C9DC5;
#pragma unroll
    for (int i = 0; i < 12; ++i)
    {
        hash = (hash ^ uchar_[i]) * 0x01000193;
    }

    return hash;
}

// FNV-1 hash (32bit)
__device__ uint32_t PointHash2(const int *__restrict__ data_)
{
    const u_char *uchar_ = (u_char *)data_;
    uint32_t hash = 0x811C9DC5;
#pragma unroll
    for (int i = 0; i < 12; ++i)
    {
        hash = (hash * 0x01000193) ^ (uchar_[i]);
    }

    return hash;
}

//////

template <typename FloatType>
__global__ void uniqueVertHash_kernel(int *__restrict__ verts_num, int *__restrict__ unique_verts_offset_list,
                                      FloatType *__restrict__ _verts_buffer, int stride,
                                      int *__restrict__ hash_array, int HASH_EMPTY, uint32_t HASH_MASK, uint32_t SKIP_LENGTH,
                                      int process_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < process_num)
    {
        FloatType *_verts = _verts_buffer + stride * tid;
        int num = int(_verts[0]);
        uint32_t idx;
        int value;
        int data_[3];
        _verts = _verts + 1;
        for (int i = 0; i < num; ++i)
        {
            data_[0] = int(_verts[3 * i + 0] * FP_POINTHASH_MUL<FloatType>);
            data_[1] = int(_verts[3 * i + 1] * FP_POINTHASH_MUL<FloatType>);
            data_[2] = int(_verts[3 * i + 2] * FP_POINTHASH_MUL<FloatType>);
            idx = PointHash1(data_) & HASH_MASK;
            value = int(PointHash2(data_));

            while (true)
            {
                int prev = atomicCAS(&hash_array[2 * idx + 1], HASH_EMPTY, value);

                if (prev == value)
                    break;
                if (prev == HASH_EMPTY)
                {
                    int index = atomicAdd(verts_num, 1); // 0, 1, 2, ...
                    hash_array[2 * idx + 0] = index;
                    unique_verts_offset_list[index] = stride * tid + 1 + 3 * i; // offset
                    break;
                }

                idx = (idx + SKIP_LENGTH) & HASH_MASK;
            }
        }
    }
}

template <typename FloatType, int VERT_MAX>
int PolyMesh<FloatType, VERT_MAX>::uniqueVertHash()
{
    thrust::device_vector<int> verts_num(1);
    verts_num[0] = 0;

    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, _verts_buffer_num);
    const int block_num = int(std::ceil(_verts_buffer_num / float(thread_num)));
    uniqueVertHash_kernel<<<block_num, thread_num>>>(thrust::raw_pointer_cast(verts_num.data()), unique_verts_offset_list,
                                                     _verts_buffer, STRIDE,
                                                     hash_array, HASH_EMPTY, uint32_t(HASH_LEN - 1), SKIP_LENGTH,
                                                     _verts_buffer_num);
    CUDASYNC();

    return verts_num[0]; // num of vertices
}

//////

template <typename FloatType>
__global__ void indexingVertices_kernel(int verts_num, const int *__restrict__ unique_verts_offset_list,
                                        const FloatType *__restrict__ _verts_buffer,
                                        FloatType *__restrict__ vertices_d,
                                        FloatType scale, FloatType center_x, FloatType center_y, FloatType center_z)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < verts_num)
    {
        int offset = unique_verts_offset_list[tid];
        vertices_d[3 * tid + 0] = _verts_buffer[offset + 0] * scale + center_x;
        vertices_d[3 * tid + 1] = _verts_buffer[offset + 1] * scale + center_y;
        vertices_d[3 * tid + 2] = _verts_buffer[offset + 2] * scale + center_z;
    }
}

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::indexingVertices(int verts_num, FloatType scale, FloatType center_x, FloatType center_y, FloatType center_z)
{
    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, verts_num);
    const int block_num = int(std::ceil(verts_num / float(thread_num)));
    indexingVertices_kernel<<<block_num, thread_num>>>(verts_num, unique_verts_offset_list, _verts_buffer, vertices_d,
                                                       scale, center_x, center_y, center_z);
    CUDASYNC();

    gpuErrchk(cudaMemcpy(vertices, vertices_d, 3 * verts_num * sizeof(FloatType), cudaMemcpyDeviceToHost));
}

//////

template <typename FloatType>
__global__ void indexingFaces_kernel(int *__restrict__ faces_num, int *__restrict__ faces_d, int faces_stride,
                                     FloatType *__restrict__ _verts_buffer, int stride,
                                     const int *__restrict__ hash_array, int HASH_EMPTY, uint32_t HASH_MASK, uint32_t SKIP_LENGTH,
                                     int process_num)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < process_num)
    {
        FloatType *_verts = _verts_buffer + stride * tid;
        int num = int(_verts[0]);
        uint32_t idx;
        int value;
        int v_i;
        int face_i;
        int data_[3];
        _verts = _verts + 1;
        if (num >= 3)
        {
            face_i = atomicAdd(faces_num, 1);
            faces_d[faces_stride * face_i] = num;
        }
        else
        {
            return;
        }
        for (int i = 0; i < num; ++i)
        {
            data_[0] = int(_verts[3 * i + 0] * FP_POINTHASH_MUL<FloatType>);
            data_[1] = int(_verts[3 * i + 1] * FP_POINTHASH_MUL<FloatType>);
            data_[2] = int(_verts[3 * i + 2] * FP_POINTHASH_MUL<FloatType>);
            idx = PointHash1(data_) & HASH_MASK;
            value = int(PointHash2(data_));

            while (true)
            {
                if (hash_array[2 * idx + 1] == value)
                {
                    v_i = hash_array[2 * idx + 0];
                    break;
                }
                idx = (idx + SKIP_LENGTH) & HASH_MASK;
            }

            faces_d[faces_stride * face_i + 1 + i] = v_i;
        }
    }
}

template <typename FloatType, int VERT_MAX>
int PolyMesh<FloatType, VERT_MAX>::indexingFaces()
{
    thrust::device_vector<int> faces_num(1);
    faces_num[0] = 0;

    const int thread_num = MIN2(_MAX_THREADS_PER_BLOCK, _verts_buffer_num);
    const int block_num = int(std::ceil(_verts_buffer_num / float(thread_num)));
    indexingFaces_kernel<<<block_num, thread_num>>>(thrust::raw_pointer_cast(faces_num.data()), faces_d, 1 + VERT_MAX,
                                                    _verts_buffer, STRIDE,
                                                    hash_array, HASH_EMPTY, uint32_t(HASH_LEN - 1), SKIP_LENGTH,
                                                    _verts_buffer_num);
    CUDASYNC();

    gpuErrchk(cudaMemcpy(faces, faces_d, (1 + VERT_MAX) * faces_num[0] * sizeof(int), cudaMemcpyDeviceToHost));

    return faces_num[0];
}

//////

template <typename FloatType, int VERT_MAX>
void PolyMesh<FloatType, VERT_MAX>::combine_mesh(FloatType scale, FloatType center_x, FloatType center_y, FloatType center_z)
{
#warning It may contain degenerated faces (such as `A-B-C-C-D-A`)
    verts_num = uniqueVertHash();
    indexingVertices(verts_num, scale, center_x, center_y, center_z); // NOTE: output vertex = vertex * scale + center
    faces_num = indexingFaces();
}

//////

template <typename FloatType, int VERT_MAX>
template <typename ExportFloatType, bool IsPoly>
void PolyMesh<FloatType, VERT_MAX>::export_mesh(std::string file_path)
{
    std::string ExportFloatTypeStr = std::is_same<ExportFloatType, float>::value ? "float" : "double";

    int i, j;

    int faces_stride = 1 + VERT_MAX;
    int trifaces_num = 0;
    if (!IsPoly)
    {
        for (i = 0; i < faces_num; ++i)
        {
            trifaces_num += faces[faces_stride * i] - 2;
        }
    }

    std::ofstream ply_file(file_path, std::ios::out | std::ios::binary);
    ply_file << "ply\n";
    ply_file << "format binary_little_endian 1.0\n";
    ply_file << "element vertex " << verts_num << std::endl;
    ply_file << "property " << ExportFloatTypeStr << " x\n";
    ply_file << "property " << ExportFloatTypeStr << " y\n";
    ply_file << "property " << ExportFloatTypeStr << " z\n";
    ply_file << "element face " << (IsPoly ? faces_num : trifaces_num) << std::endl;
    ply_file << "property list uchar int vertex_index\n";
    ply_file << "end_header\n";

    // save vertices
    ExportFloatType x, y, z;
    for (i = 0; i < verts_num; ++i)
    {
        x = ExportFloatType(vertices[3 * i + 0]);
        y = ExportFloatType(vertices[3 * i + 1]);
        z = ExportFloatType(vertices[3 * i + 2]);
        ply_file.write((char *)&(x), sizeof(ExportFloatType));
        ply_file.write((char *)&(y), sizeof(ExportFloatType));
        ply_file.write((char *)&(z), sizeof(ExportFloatType));
    }

    // save faces
    u_char v_num;
    if (IsPoly)
    {
        for (i = 0; i < faces_num; ++i)
        {
            v_num = (u_char)(faces[faces_stride * i]);
            ply_file.write((char *)&v_num, sizeof(u_char));
            for (j = 0; j < v_num; ++j)
            {
                ply_file.write((char *)&faces[faces_stride * i + 1 + j], sizeof(int));
            }
        }
    }
    else
    {
        int the_triface_num;
        v_num = u_char(3);
        for (i = 0; i < faces_num; ++i)
        {
            the_triface_num = faces[faces_stride * i] - 2;
            for (j = 0; j < the_triface_num; ++j)
            {
                ply_file.write((char *)&v_num, sizeof(u_char));
                ply_file.write((char *)&faces[faces_stride * i + 1 + 0], sizeof(int));
                ply_file.write((char *)&faces[faces_stride * i + 1 + j + 1], sizeof(int));
                ply_file.write((char *)&faces[faces_stride * i + 1 + j + 2], sizeof(int));
            }
        }
    }

    ply_file.close();
}

#endif