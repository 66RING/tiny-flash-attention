#pragma once

#include <cstddef>
#include <cstdint> 
#include <torch/extension.h>

torch::Tensor self_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_v1_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_v2_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v);

