#include "attn.h"
#include "fast_attn.h"
#include "utils.h"
#include "static_switch.h"
#include <cstring>
#include <omp.h>
#include <vector>
#include <iostream>

struct attn_fwd_params {
  size_t bs;
  size_t head_num;
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

  // gqa group size
  size_t gqa_group_size;

  void *q_ptr;
  void *k_ptr;
  void *v_ptr;
  void *o_ptr;

  bool is_causal;
  float softmax_scale;

  int num_threads;

  // small buffer for fp16 cast to fp32
  void *q_buffer_ptr;
  void *k_buffer_ptr;
  void *v_buffer_ptr;
  void *o_buffer_ptr;
  bool is_half;
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
        auto head_group_id = hid / params.gqa_group_size;

        float maxval = -INFINITY;

        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }

        // qk dot product
        for (int j = 0; j < kv_len; j++) {
          elem_type* k = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          float val = 0.0f;
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
          elem_type* v = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
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
        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }
        for (int j = 0; j < kv_len; j++) {
          float local_maxval = -INFINITY;
          auto head_group_id = hid / params.gqa_group_size;
          elem_type* k = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          elem_type* v = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          float val = 0.0f;

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

template <typename Attn_traits> void run_fast_flash_attn(attn_fwd_params &params) {
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
        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }

        // TODO: hard code for now
        using vec_t = float32x4_t;

        for (int j = 0; j < kv_len; j++) {
          float local_maxval = -INFINITY;
          auto head_group_id = hid / params.gqa_group_size;
          elem_type* k = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          elem_type* v = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          // fp32 accumulator
          float val = 0.0f;

          // q @ k
          val = row_qk_dot<vec_t>(q, k, params.head_dim);
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
          row_score_v<vec_t, elem_type>(out, v, params.head_dim, exp, scale);

          // update max
          maxval = local_maxval;
        }

        // rescale out with score_sum
        row_out_rescale<vec_t>(out, params.head_dim, 1.0f / score_sum);
      }
    }
  }
}


template <typename Attn_traits> void run_fast_flash_attn_cast_buffer(attn_fwd_params &params) {
  /*
     q k v.shape  (bs, head_num, seqlen, head_dim)
     attn_score.shape = (seqlen, seqlen), compute one by one
  */
  using elem_type = typename Attn_traits::elem_type;

  #pragma omp parallel for collapse(3) num_threads(params.num_threads) if (params.bs * params.head_num * params.q_seqlen > 0)
  for (int bid = 0; bid < params.bs; bid++) {
    for (int hid = 0; hid < params.head_num; hid++) {
      for (int i = 0; i < params.q_seqlen; i++) {
        int thread_id = omp_get_thread_num();
        // write buffer for fp32 casting
        float* q_buffer = static_cast<float*>(params.q_buffer_ptr) + thread_id * params.head_dim;
        float* k_buffer = static_cast<float*>(params.k_buffer_ptr) + thread_id * params.head_dim;
        float* v_buffer = static_cast<float*>(params.v_buffer_ptr) + thread_id * params.head_dim;
        float* o_buffer = static_cast<float*>(params.o_buffer_ptr) + thread_id * params.head_dim;


        elem_type* q_ptr = static_cast<elem_type*>(params.q_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;
        // init accumulators with zero allocate
        elem_type* out_ptr = static_cast<elem_type*>(params.o_ptr) + bid * params.stride_q_bs + hid * params.stride_q_head_num + i * params.stride_q_seqlen;
        // history max
        float maxval = -INFINITY;
        // div delay till the end (only div once)
        float score_sum = 0.0f;
        int kv_len = params.k_seqlen;
        if (params.is_causal) {
          kv_len = i + 1 + (params.k_seqlen - params.q_seqlen);
        }

        float* q = reinterpret_cast<float*>(q_ptr);
        float* out = reinterpret_cast<float*>(out_ptr);
        // NOTE: casting if necessary
        // TODO: naive fp16 inst support
        if (params.is_half) {
          auto_cast(q_buffer, q_ptr, params.head_dim);
          // o_buffer is an accumulator (read write buffer), need to clear
          std::memset(o_buffer, 0, params.head_dim * sizeof(float));
          q = q_buffer;
          out = o_buffer;
        }

        // TODO: hard code for now
        using vec_t = float32x4_t;

        for (int j = 0; j < kv_len; j++) {
          float local_maxval = -INFINITY;
          auto head_group_id = hid / params.gqa_group_size;
          elem_type* k_ptr = static_cast<elem_type*>(params.k_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;
          elem_type* v_ptr = static_cast<elem_type*>(params.v_ptr) + bid * params.stride_kv_bs + head_group_id * params.stride_kv_head_num + j * params.stride_kv_seqlen;

          float *k = reinterpret_cast<float*>(k_ptr);
          float *v = reinterpret_cast<float*>(v_ptr);

          // NOTE: casting if necessary
          // TODO: naive fp16 inst support
          if (params.is_half) {
            auto_cast(k_buffer, k_ptr, params.head_dim);
            auto_cast(v_buffer, v_ptr, params.head_dim);
            k = k_buffer;
            v = v_buffer;
          }


          // fp32 accumulator
          float val = 0.0f;

          // q @ k
          val = row_qk_dot<vec_t>(q, k, params.head_dim);
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
          row_score_v<vec_t>(out, v, params.head_dim, exp, scale);

          // update max
          maxval = local_maxval;
        }

        // rescale out with score_sum
        row_out_rescale<vec_t>(out, params.head_dim, 1.0f / score_sum);


        // auto write back and cast
        auto_write_back(out_ptr, out, params.head_dim);
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
                      float softmax_scale,
                      int num_threads=16,
                      void* q_buffer_ptr=nullptr,
                      void* k_buffer_ptr=nullptr,
                      void* v_buffer_ptr=nullptr,
                      void* o_buffer_ptr=nullptr
                      ) {
  params.bs = q.size(0);
  params.head_num = q.size(1);
  params.kv_head_num = k.size(1);
  params.q_seqlen = q.size(2);
  params.k_seqlen = k.size(2);
  params.head_dim = q.size(3);

  params.gqa_group_size = q.size(1) / k.size(1);

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

  params.num_threads = num_threads;
  params.is_half = q.scalar_type() != at::ScalarType::Float;

  params.q_buffer_ptr = q_buffer_ptr;
  params.k_buffer_ptr = k_buffer_ptr;
  params.v_buffer_ptr = v_buffer_ptr;
  params.o_buffer_ptr = o_buffer_ptr;
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
  auto input_options = q.options();

  // NOTE: naive casting
  if (q.scalar_type() != at::ScalarType::Float) {
    q = q.to(at::ScalarType::Float);
    k = k.to(at::ScalarType::Float);
    v = v.to(at::ScalarType::Float);
    out = out.to(at::ScalarType::Float);
    attn_score = attn_score.to(at::ScalarType::Float);
  }

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale);

  // TODO: hard code float
  run_naive_attn<Naive_fwd_traits<float>>(params, (float*)attn_score.data_ptr(), attn_score.stride(0));

  return out.to(input_options);
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

  auto input_options = q.options();
  auto out = torch::zeros_like(q);

  // NOTE: naive casting
  if (q.scalar_type() != at::ScalarType::Float) {
    q = q.to(at::ScalarType::Float);
    k = k.to(at::ScalarType::Float);
    v = v.to(at::ScalarType::Float);
    out = out.to(at::ScalarType::Float);
  }

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale);

  // TODO: hard code float
  run_flash_attn<Naive_fwd_traits<float>>(params);

  return out.to(input_options);
}


torch::Tensor fast_flash_attn(torch::Tensor q, torch::Tensor k,
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

  auto input_options = q.options();
  auto out = torch::zeros_like(q);

  // NOTE: naive casting
  if (q.scalar_type() != at::ScalarType::Float) {
    q = q.to(at::ScalarType::Float);
    k = k.to(at::ScalarType::Float);
    v = v.to(at::ScalarType::Float);
    out = out.to(at::ScalarType::Float);
  }

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale);
  // TODO: hard code float
  run_fast_flash_attn<Naive_fwd_traits<float>>(params);

  return out.to(input_options);
}


torch::Tensor fast_flash_attn_cast_buffer(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, torch::Tensor data_buffer, bool is_causal = false, float softmax_scale=1) {
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

  auto input_options = q.options();
  auto out = torch::empty_like(q);

  TORCH_CHECK(data_buffer.size(0) == 4, "data_buffer must have 4 tensors to store q, k, v, o");
  TORCH_CHECK(data_buffer.device().is_cpu(), "v must be on CPU");
  TORCH_CHECK(data_buffer.size(2) == dim, "shape of data_buffer should be (4, num_threads, head_dim)");
  TORCH_CHECK(data_buffer.scalar_type() == at::ScalarType::Float, "data_buffer should be float type");

  int num_threads = data_buffer.size(1);
  auto q_buffer = data_buffer[0];
  auto k_buffer = data_buffer[1];
  auto v_buffer = data_buffer[2];
  auto o_buffer = data_buffer[3];

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale, num_threads, q_buffer.data_ptr(), k_buffer.data_ptr(), v_buffer.data_ptr(), o_buffer.data_ptr());

  FP_SWITCH(q.scalar_type(), "fast_flash_attn_cast_buffer", [&] {
      run_fast_flash_attn_cast_buffer<Naive_fwd_traits<scalar_t>>(params);
  });

  return out;
}


std::vector<torch::Tensor> fast_partial_attn_cast_buffer(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, torch::Tensor data_buffer, bool is_causal = false, float softmax_scale=1) {
  // TODO: float 16 is easy to be nan
  TORCH_CHECK(q.scalar_type() != at::ScalarType::Half, "fp16 is not supported yet, since nan");

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

  auto input_options = q.options();
  auto out = torch::empty_like(q);

  auto max = torch::empty({bs, head}, q.options());
  auto score_sum = torch::empty({bs, head}, q.options());

  TORCH_CHECK(data_buffer.size(0) == 4, "data_buffer must have 4 tensors to store q, k, v, o");
  TORCH_CHECK(data_buffer.device().is_cpu(), "v must be on CPU");
  TORCH_CHECK(data_buffer.size(2) == dim, "shape of data_buffer should be (4, num_threads, head_dim)");
  TORCH_CHECK(data_buffer.scalar_type() == at::ScalarType::Float, "data_buffer should be float type");

  int num_threads = data_buffer.size(1);
  auto q_buffer = data_buffer[0];
  auto k_buffer = data_buffer[1];
  auto v_buffer = data_buffer[2];
  auto o_buffer = data_buffer[3];

  attn_fwd_params params;
  set_params_fprop(params, q, k, v, out,
      is_causal, softmax_scale, num_threads, q_buffer.data_ptr(), k_buffer.data_ptr(), v_buffer.data_ptr(), o_buffer.data_ptr());

  FP_SWITCH(q.scalar_type(), "fast_flash_attn_cast_buffer", [&] {
      run_fast_flash_attn_cast_buffer<Naive_fwd_traits<scalar_t>>(params);
  });

  return {out};
}
