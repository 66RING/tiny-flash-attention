#pragma once

#include <iostream>

void hello_world() {
  std::cout << "Hello, World!" << std::endl;
}

// Naive attention impl
torch::Tensor naive_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false, float softmax_scale=1);

// Flash attention impl
torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false, float softmax_scale=1);

// Fast Flash attention impl
torch::Tensor fast_flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool is_causal = false, float softmax_scale=1);

// Fast Flash attention impl with faster casting
// data_buffer.shape (4, num_thread, head_dim) for type casting of fp16/bf16
torch::Tensor fast_flash_attn_cast_buffer(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor data_buffer, bool is_causal = false, float softmax_scale=1);
