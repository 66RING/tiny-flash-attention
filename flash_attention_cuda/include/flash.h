#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Data structures

struct Qkv_params {
    using index_t = uint32_t;
    // The QKV matrices.
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    /// Store data for backward
    void* __restrict__ l_ptr;
    // Output matrix
    void* __restrict__ o_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t q_head_stride;
    index_t q_seqlen_stride;
    index_t q_dim_stride;
    index_t k_batch_stride;
    index_t k_head_stride;
    index_t k_seqlen_stride;
    index_t k_dim_stride;
    index_t v_batch_stride;
    index_t v_head_stride;
    index_t v_seqlen_stride;
    index_t v_dim_stride;

    index_t bs;
    index_t head;
    index_t seqlen;
    index_t seqlen_rounded;
    index_t dim;
    index_t block_m;
    index_t block_n;
    bool is_causal;
};

//////////////////////////////

// TODO: 泛型的妙用
template<typename T, int Headdim> void run_flash_attn_fwd_(Qkv_params &params, cudaStream_t stream);

