#ifndef __MACRO_H__
#define __MACRO_H__


/* global */
#define _CONSTRAINTS_USE_GEOMETRIC_DISTANCE    // the inequality is judged by geometric distance
#define _ENABLE_KERNEL_PRINTF                  // use printf inside kernel function
#define _MAX_THREADS_PER_BLOCK (1024)          // maximal number of threads within one block
#define CUDASYNC()                             // gpuErrchk(cudaDeviceSynchronize())
// #define _REMOVE_ZERO_CONS                   // eliminate zero inequality constraints/vertices
// #define _FLOAT32_USE_64PRECISION_WHEN_NEED  // convert to float64
// #define _USE_CUDAMEMCPYASYNC                // use asynchronous copy in inferNewStates_kernel


/* EPS */
// tolerance error in judging inequality conditions
#define _FP_EPS_FLOAT (1e-12f) // 1e-12
#define _FP_EPS_DOUBLE (1e-20) // 1e-20
// tolerance error for determining whether it is the same vertex
#define _FP_SAME_EPS_FLOAT (3e-7f)  // 3e-7
#define _FP_SAME_EPS_DOUBLE (3e-14) // 3e-14
// point hash, the amplification factor for converting to int
#define _FP_POINTHASH_MUL_FLOAT (int(1e6))
#define _FP_POINTHASH_MUL_DOUBLE (int(1e9))
// the threshold for inequality constraints that are judged to be invalid
#ifdef _REMOVE_ZERO_CONS
#define _FP_ZERO_CONS_FLOAT (1e-20f)
#define _FP_ZERO_CONS_DOUBLE (1e-40)
#endif


/* states.h */
#define _STATES_BASE_NUM (262144)     // initial capacity of state, 262144==2^18
#define _STATES_INCREASE (2)          // multiply the capacity with this value
#define _STATES_TABLE_SIZE (33554432) // Hash table size, must be power of 2, 33554432==2^25==128MB
#define _STATES_SKIP_LENGTH (1543)    // skip step of hash table


/* var.h */
#define _VAR_BATCH_SIZE_MAX (1024) // solve how many states within one batch
#define _VAR_SOLVE_SIZE_MAX (1024) // solve how many equations within one batch
#define _VAR_VERT_MAX (20)         // an upper limit on the number of vertices


/* polymesh.h */
#define _POLYMESH_INIT_NUM (262144)            // initial num of polygonal faces, 262144==2^18
#define _POLYMESH_INC_RATE (2)                 // increase rate for faces
#define _POLYMESH_HASH_LEN (33554432)          // Hash table, used to combine the same vertices (but request a hash table twice as long)
#define _POLYMESH_SKIP_LENGTH (1543)           // skip step
#define _POLYMESH_STATES_ASSUMED_MAX (8388608) // the maximum number of states, 8388608==2^23


#endif
