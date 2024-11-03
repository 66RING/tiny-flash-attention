#pragma once

#include <iostream>

void hello_world() {
  std::cout << "Hello, World!" << std::endl;
}

// Naive attention impl
torch::Tensor naive_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false, float softmax_scale=1);

// Flash attention impl
torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false, float softmax_scale=1);

