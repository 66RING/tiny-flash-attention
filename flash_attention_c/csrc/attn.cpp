#include "attn.h"
#include "utils.h"
#include <omp.h>
#include <iostream>

struct attn_fwd_params {
  size_t bs;
  size_t head_num;
  // TODO: GQA support
  size_t q_seqlen;
  size_t head_dim;
  size_t k_seqlen;
  size_t kv_head_num;

  size_t stride_q_bs;
  size_t stride_q_head_num;
  size_t stride_q_seqlen;
  size_t stride_q_head_dim;

  size_t stride_kv_bs;
  size_t stride_kv_head_num;
  size_t stride_kv_seqlen;
  size_t stride_kv_head_dim;

  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  void *o_ptr;

  bool is_causal;
  float softmax_scale;
};


template <typename Attn_traits> void run_naive_attn(attn_fwd_params &params, typename Attn_traits::elem_type* attn_score, size_t stride_score_l1) {
  /*
     q k v.shape  (bs, head_num, seqlen, head_dim)
     attn_score.shape = (seqlen, seqlen), compute one by one
  */
  using elem_type = typename Attn_traits::elem_type;

  for (int bid = 0; bid < params.bs; bid++) {
    for (int hid = 0; hid < params.head_num; hid++) {
      #pragma omp parallel for
      for (int i = 0; i < params.q_seqlen; i++) {
        elem_type* q = static_cast<elem_type*>(params.q_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;

        float maxval = -INFINITY;

        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }

        // qk dot product
        for (int j = 0; j < kv_len; j++) {
          elem_type* k = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + hid * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          elem_type val = 0.0f;
          for (int dim = 0; dim < params.head_dim; dim++) {
              val += q[dim] * k[dim];
          }

          val *= params.softmax_scale;
          if (val > maxval) {
              maxval = val;
          }
          // set score[i, j]
          attn_score[i * stride_score_l1 + j] = val;
        }

        // NOTE: softmax
        float score_sum = 0.0f;
        for (int j = 0; j < kv_len; j++) {
          auto exp = expf(attn_score[i * stride_score_l1 + j] - maxval);
          score_sum += exp;
          attn_score[i * stride_score_l1 + j] = exp;
        }
        for (int j = 0; j < kv_len; j++) {
          attn_score[i * stride_score_l1 + j] /= score_sum;
        }

        // NOTE: compute qk @ v
        // (seqlen, seqlen) @ (seqlen, head_dim)
        elem_type* out = static_cast<elem_type*>(params.o_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;
        // init accumulators
        for (int dim = 0; dim < params.head_dim; dim++) {
            out[dim] = 0.0f;
        }
        for (int j = 0; j < kv_len; j++) {
          elem_type* v = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + hid * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          for (int dim = 0; dim < params.head_dim; dim++) {
            out[dim] += attn_score[i * stride_score_l1 + j] * v[dim];
          }
        }
      }
    }
  }
}


template <typename Attn_traits> void run_flash_attn(attn_fwd_params &params) {
  /*
     q k v.shape  (bs, head_num, seqlen, head_dim)
     attn_score.shape = (seqlen, seqlen), compute one by one
  */
  using elem_type = typename Attn_traits::elem_type;

  #pragma omp parallel for collapse(3)
  for (int bid = 0; bid < params.bs; bid++) {
    for (int hid = 0; hid < params.head_num; hid++) {
      for (int i = 0; i < params.q_seqlen; i++) {
        elem_type* q = static_cast<elem_type*>(params.q_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;
        // init accumulators with zero allocate
        elem_type* out = static_cast<elem_type*>(params.o_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;
        // history max
        float maxval = -INFINITY;
        // div delay till the end (only div once)
        float score_sum = 0.0f;
        // qk dot product
        // NOTE: and online softmax
        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }
        for (int j = 0; j < kv_len; j++) {
          float local_maxval = -INFINITY;
          elem_type* k = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + hid * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          // TODO: val need should be higher precision
          elem_type val = 0.0f;

          // q @ k
          for (int dim = 0; dim < params.head_dim; dim++) {
              val += q[dim] * k[dim];
          }
          val *= params.softmax_scale;

          // local_maxval always the real max
          local_maxval = std::max(maxval, val);

          // TODO: skip scale if no update?
          // TODO: exp2f?
          auto exp = expf(val - local_maxval);
          auto scale = expf(maxval - local_maxval);

          // rescale score sum
          score_sum *= scale;
          score_sum += exp;

          // NOTE: online softmax rescale, update
          // and compute qk @ v: (seqlen, seqlen) @ (seqlen, head_dim)
          elem_type* v = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + hid * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          for (int dim = 0; dim < params.head_dim; dim++) {
            // rescale score
            out[dim] *= scale;
            out[dim] += exp * v[dim];
          }

          // update max
          maxval = local_maxval;
        }

        // TODO: online rescale or delay till the end?
        for (int dim = 0; dim < params.head_dim; dim++) {
          out[dim] /= score_sum;
        }
      }
    }
  }
}

void set_params_fprop(attn_fwd_params &params,
                      // device pointers
                      const torch::Tensor q,
                      const torch::Tensor k,
                      const torch::Tensor v,
                      torch::Tensor out,
                      bool is_causal,
                      float softmax_scale) {
  params.bs = q.size(0);
  params.head_num = q.size(1);
  params.kv_head_num = k.size(1);
  params.q_seqlen = q.size(2);
  params.k_seqlen = k.size(2);
  params.head_dim = q.size(3);

  params.stride_q_bs = q.stride(0);
  params.stride_q_head_num = q.stride(1);
  params.stride_q_seqlen = q.stride(2);
  params.stride_q_head_dim = q.stride(3);

  params.stride_kv_bs = k.stride(0);
  params.stride_kv_head_num = k.stride(1);
  params.stride_kv_seqlen = k.stride(2);
  params.stride_kv_head_dim = k.stride(3);

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();

  params.is_causal = is_causal;
  params.softmax_scale = softmax_scale;
}



torch::Tensor naive_attn(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, bool is_causal = false, float softmax_scale=1) {
  TORCH_CHECK(q.device().is_cpu(), "q must be on CPU");
  TORCH_CHECK(k.device().is_cpu(), "k must be on CPU");
  TORCH_CHECK(v.device().is_cpu(), "v must be on CPU");

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);
  int kv_seqlen = k.size(2);
  // dim
  int dim = q.size(3);

  auto attn_score = torch::empty({seqlen, kv_seqlen}, q.options());
  auto out = torch::zeros_like(q);

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale);

  // TODO: hard code float
  run_naive_attn<Naive_fwd_traits<float>>(params, (float*)attn_score.data_ptr(), attn_score.stride(0));

  return out;
}


torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, bool is_causal = false, float softmax_scale=1) {
  TORCH_CHECK(q.device().is_cpu(), "q must be on CPU");
  TORCH_CHECK(k.device().is_cpu(), "k must be on CPU");
  TORCH_CHECK(v.device().is_cpu(), "v must be on CPU");

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);
  // dim
  int dim = q.size(3);

  auto out = torch::zeros_like(q);

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale);

  // TODO: hard code float
  run_flash_attn<Naive_fwd_traits<float>>(params);

  return out;
}



