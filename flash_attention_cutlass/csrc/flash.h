#pragma once

#include <cstdint>

// TODO: 特种约束字段, e.g. __restrict__ 的效果
struct Qkv_params {
    using index_t = uint32_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // // The stride between rows of the Q, K and V matrices.
    // index_t q_batch_stride;
    // index_t k_batch_stride;
    // index_t v_batch_stride;
    // // TODO: 
    // index_t q_row_stride;
    // index_t k_row_stride;
    // index_t v_row_stride;
    // index_t q_head_stride;
    // index_t k_head_stride;
    // index_t v_head_stride;

    bool is_bf16;
};


struct Flash_fwd_params : public Qkv_params {
  size_t bs;
  size_t head;
  size_t q_seqlen;
  size_t dim;

  size_t k_head;
  size_t k_seqlen;

  // TODO: review the impl of flash
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
  // different from nheads (query).
  size_t h_h_k_ratio; // precompute head / k_head,
  size_t flat_seqlen;
  size_t kv_head_stride;
  size_t qo_head_stride;


  size_t bs_stride;
  size_t head_stride;
  size_t seqlen_stride;
  size_t dim_stride;

  float softmax_scale;
  float softmax_scale_log2;
  void *__restrict__ out_ptr;
  void *__restrict__ softmax_lse_ptr;
  void *__restrict__ score_max;
  void *__restrict__ score_sum;

  bool is_causal;
};

