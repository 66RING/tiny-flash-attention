#pragma once

#include "flash.h"
#include <cstddef>
#include <cstdint> 
#include <torch/extension.h>

// NOTE:tensor malloc as device before we call
// e.g. data.to("cuda") in python
#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

void set_params_fprop(Qkv_params& params,
                      // sizes
                      const size_t bs, const size_t head, const size_t seqlen,
                      const size_t seqlen_rounded, const size_t dim,
                      const size_t block_m,
                      const size_t block_n,
                      // device pointers
                      const at::Tensor q, const at::Tensor k,
                      const at::Tensor v, at::Tensor out,
                      /* TODO: L ptr */
                      at::Tensor L, bool is_causal, float softmax_scale);


/// q, k, v with the shape of bs, head, seqlen, dim
torch::Tensor tiny_flash_attn_cuda(torch::Tensor& q, torch::Tensor& k,
                                   torch::Tensor& v, bool is_causal,
                                   float sm_scale);
