#ifndef __HASH_H__
#define __HASH_H__

using status = int;
#define FAIL (status(0))
#define SUCCESS (status(1))

//////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ uint32_t hash1(const u_char *__restrict__ data, int numBytes); // FNV-1a hash (32bit)
__host__ __device__ uint32_t hash2(const u_char *__restrict__ data, int numBytes); // FNV-1 hash (32bit)

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename BinaryType>
__host__ __device__ void encode(const BinaryType *__restrict__ src, u_char *__restrict__ dst, int state_len)
{
    u_char byte;
    int j = 0, k;
    for (int i = 0; i < state_len; ++i)
    {
        k = i % 8;
        if (k == 0)
            byte = 0;
        byte |= (u_char(src[i]) << k);
        if (k == 7 || i == state_len - 1)
            dst[j++] = byte;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////

template <typename BinaryType>
__host__ __device__ status try_insert(const BinaryType *__restrict__ state, int state_len,
                                      uint32_t *__restrict__ table,
                                      u_char *__restrict__ device_uchar_state, int uchar_len,
                                      int TABLE_SIZE_MASK, int SKIP_LENGTH, uint32_t HASH2_ZERO)
{
    encode(state, device_uchar_state, state_len);
    uint32_t idx = hash1(device_uchar_state, uchar_len) & TABLE_SIZE_MASK;
    uint32_t value = hash2(device_uchar_state, uchar_len);
#ifndef __CUDA_ARCH__ // CPU
    while (table[idx] != HASH2_ZERO && table[idx] != value)
    {
        idx = (idx + SKIP_LENGTH) & TABLE_SIZE_MASK; // linear probing
    }
    if (table[idx] == HASH2_ZERO)
    {
        table[idx] = value;
        return SUCCESS; // successfully
    }
    return FAIL; // already exist

#else // GPU
    uint32_t prev;
    while (true)
    {
        prev = atomicCAS(&table[idx], HASH2_ZERO, value);
        if (prev == HASH2_ZERO || prev == value)
        {
            return (prev == HASH2_ZERO ? SUCCESS : FAIL);
        }
        idx = (idx + SKIP_LENGTH) & TABLE_SIZE_MASK;
    }

#endif
}

#endif