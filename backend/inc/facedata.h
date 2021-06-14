#ifndef __FACEDATA_H__
#define __FACEDATA_H__

template <typename FloatType>
class FACEDATA
{
public:
    FloatType midpoint[3];
    int start_idx; // -1 stands for init_points

    __host__ __device__ FACEDATA();
    __host__ __device__ FACEDATA(const FloatType *midpoint_);
    __host__ __device__ FACEDATA(const FloatType *midpoint_, int start_idx_);
    __host__ __device__ FACEDATA(const FACEDATA &facedata);
    __host__ __device__ FACEDATA &operator=(const FACEDATA &facedata);
};

/////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
__host__ __device__ FACEDATA<FloatType>::FACEDATA()
{
    midpoint[0] = 0;
    midpoint[1] = 0;
    midpoint[2] = 0;
    start_idx = -1;
}

template <typename FloatType>
__host__ __device__ FACEDATA<FloatType>::FACEDATA(const FloatType *midpoint_)
{
    midpoint[0] = midpoint_[0];
    midpoint[1] = midpoint_[1];
    midpoint[2] = midpoint_[2];
    start_idx = -1;
}

template <typename FloatType>
__host__ __device__ FACEDATA<FloatType>::FACEDATA(const FloatType *midpoint_, int start_idx_)
{
    midpoint[0] = midpoint_[0];
    midpoint[1] = midpoint_[1];
    midpoint[2] = midpoint_[2];
    start_idx = start_idx_;
}

template <typename FloatType>
__host__ __device__ FACEDATA<FloatType>::FACEDATA(const FACEDATA<FloatType> &facedata)
{
    midpoint[0] = facedata.midpoint[0];
    midpoint[1] = facedata.midpoint[1];
    midpoint[2] = facedata.midpoint[2];
    start_idx = facedata.start_idx;
}

template <typename FloatType>
__host__ __device__ FACEDATA<FloatType> &FACEDATA<FloatType>::operator=(const FACEDATA<FloatType> &facedata)
{
    if (this != &facedata)
    {
        midpoint[0] = facedata.midpoint[0];
        midpoint[1] = facedata.midpoint[1];
        midpoint[2] = facedata.midpoint[2];
        start_idx = facedata.start_idx;
    }
    return *this;
}

#endif