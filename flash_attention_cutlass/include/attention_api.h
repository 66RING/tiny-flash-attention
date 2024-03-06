#pragma once

#include <cstddef>
#include <cstdint>
#include <torch/extension.h>
#include <vector>

#include "flash.h"

std::vector<torch::Tensor> flash_attention_v2_cutlass(torch::Tensor q, torch::Tensor k,
              torch::Tensor v, bool is_causal = false, float softmax_scale=1);

