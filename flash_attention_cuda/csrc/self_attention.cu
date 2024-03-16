#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <torch/extension.h>
#include <torch/python.h>
#include "attention_api.cuh"

template <typename scalar_t>
__global__ void naive_nrow_gemm(scalar_t *A, scalar_t *B, scalar_t *C, scalar_t a, scalar_t b,
                                int64_t M, int64_t N, int64_t K, int64_t mBlock);

template <typename scalar_t>
__global__ void naive_pv(scalar_t *P, scalar_t *V, scalar_t *O, int M, int N, int mBlock);

template <typename scalar_t>
__global__ void row_softmax(scalar_t *input, scalar_t *output, int n);

// TODO: Add support for half
torch::Tensor self_attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  auto out = torch::zeros_like(q);
  // TODO: multihead
  // seqlen
  auto m = q.size(0);
  // dim
  auto n = q.size(1);

  int64_t mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  float sm_scale = 1.f / sqrtf(static_cast<float>(n));

  // Create intermediate date for the new shape of qk
  torch::TensorOptions options = q.options();
  std::vector<int64_t> shape = {m, m};
  torch::Tensor qk = torch::empty(shape, options);

  dim3 qk_block(m / mBlock, 1, 1);
  // NOTE: AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)
  // We need a way of determining at runtime what type a tensor is and then
  // selectively call functions with the corresponding correct type signature.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.type(), "QK", ([&] {
                               naive_nrow_gemm<scalar_t><<<1, qk_block>>>(
                                   q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                                   qk.data_ptr<scalar_t>(), sm_scale, 0.f, m, m, n, mBlock);
                             }));
  // Wait until kernel finish.
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());


  // QK[M, M]
  // TODO: too much thread may cause CUDA crash.
  dim3 sm_block(m, 1, 1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(qk.type(), "softmax(QK)", ([&] {
                              row_softmax<scalar_t><<<1, sm_block>>>(
                                  qk.data_ptr<scalar_t>(), qk.data_ptr<scalar_t>(), m);
                             }));
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());


  // QK[M, M] @ V[M, N]
  dim3 qkv_block(m / mBlock, 1, 1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(out.type(), "softmax(QK)V", ([&] {
                              naive_pv<scalar_t><<<1, qkv_block>>>(
                                  qk.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                                  out.data_ptr<scalar_t>(), m, n, mBlock);
                             }));
  // We can remove this sync and let user call torch.cuda.synchronize()
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return out;
}

// naive gemm implement with slice-k
// perform C = aA@B + bC
// A[M, K] x B[K, N] = C[M, N]
// each thread process mblock rows of A
// TODO: how to make data type more general
template <typename scalar_t>
__global__ void naive_nrow_gemm(scalar_t *A, scalar_t *B, scalar_t *C, scalar_t a, scalar_t b,
                                int64_t M, int64_t N, int64_t K, int64_t mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  // A[mBlock, K] x B[N, K].T = C[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      scalar_t sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      // C[M, N]
      // C = aA@B + bC
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

// perform QK[M, M] @ V[M, N]
template <typename scalar_t>
__global__ void naive_pv(scalar_t *P, scalar_t *V, scalar_t *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  int K = M;
  // P[mBlock, M] x V[M, N] = O[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      scalar_t sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      // C[M, N]
      O[i * N + j] = sum;
    }
  }
}

// each thread process one row of softmax
template <typename scalar_t>
__global__ void row_softmax(scalar_t *input, scalar_t *output, int n) {
  // assume id will not exceed row number of input
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  scalar_t row_max = -INFINITY;
  scalar_t sum = 0.f;

  // Find max
  for (int i = 0; i < n; i++) {
    row_max = max(input[idx * n + i], row_max);
  }

  // Compute numerator and denominator
  for (int i = 0; i < n; i++) {
    output[idx * n + i] = exp(input[idx * n + i] - row_max);
    sum += output[idx * n + i];
  }

  // Compute softmax
  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}
