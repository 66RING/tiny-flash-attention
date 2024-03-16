#include "attention_api.cuh"
#include <cassert>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/python.h>
#include "static_switch.h"


// BLOCK_M(Br or Brow), BLOCK_N(Bc or Bcol) can be determined at compile time
// just like offical implementation which use a template kernel to do that
// Dim is enumberated at runtime for all supported dim
template <typename Ty, int kBc = 4, int kBr = 4, int kDim = 4>
__global__ void flash_attention_v2_kernel(Ty *Q, Ty *K, Ty *V, Ty *O,
                                          int seqlen, int stride_head, Ty smScale) {
  // block size for K, V
  // group of row(seqlen)
  int groupSeq = (seqlen + kBc - 1) / kBc;
  // parallel process for V[Br, d]
  // group of column
  int groupTx = (kDim + kBc - 1) / kBc;
  int groupTy = (kDim + kBr - 1) / kBr;

  // load slice from global memory(HBM)
  __shared__ Ty sQ[kBr][kDim];
  __shared__ Ty sK[kBc][kDim];
  __shared__ Ty sV[kBc][kDim];
  // tmp o
  __shared__ Ty sO[kBr][kDim];
  __shared__ Ty sQK[kBr][kBc];
  // e^{x - max}
  __shared__ Ty sSafeE[kBr][kBc];
  // s stand for shared and local
  __shared__ Ty sDenom[kBr];
  __shared__ Ty sMax[kBr];

  // [0, Bc]
  int tx = threadIdx.x;
  // [0, Br]
  int ty = threadIdx.y;

  // each thread in the same blockIdx.x in the same (bs, head),
  // which shared memory and process a QKV
  int base_offset = blockIdx.x * stride_head;
  int row = ty + blockIdx.y * blockDim.y;

  // TODO: need a way to round up seqlen
  if (row >= seqlen) {
    return;
  }

  Q += base_offset;
  K += base_offset;
  V += base_offset;
  O += base_offset;

  // load q, o, max, denom from global memory to shared memory
  // Q[Br, dim]
  for (int i = 0; i < groupTx; i++) {
    sQ[ty][i * kBc + tx] = Q[row * kDim + i * kBc + tx];
    // NOTE:: accumulator zero init here
    sO[ty][i * kBc + tx] = 0;
  }

  sMax[ty] = -INFINITY;
  sDenom[ty] = 0;

  // load K, V block
  // Q[Br][dim] @ K[0..seqlen.step(Bc), dim]
  // compute partial sum of O[ty][dim] each iteration
  for (int j = 0; j < groupSeq; j++) {
    if ((j * kBc + tx) < seqlen) {
      // load k, v from global memory to shared memory
      // K[seqlen, dim], V[seqlen, dim]
      for (int i = 0; i < groupTy; i++) {
        // NOTE:
        // each thread.x copy a row of K to K.T
        // row0, t0:
        // row1, t1:
        // row2, t0:
        // row3, t2:
        sK[tx][i * kBr + ty] = K[j * kBc * kDim + tx * kDim + i * kBr + ty];
        sV[tx][i * kBr + ty] = V[j * kBc * kDim + tx * kDim + i * kBr + ty];
      }
    }

    // wait until g2s done
    __syncthreads();

    // compute qk
    Ty sum = 0.f;
    // result oriented: qk[y][x] from q[y] @ k[x]
    for (int i = 0; i < kDim; i++) {
      sum += sQ[ty][i] * sK[tx][i];
    }
    // sQK[Br, Bc]
    sQK[ty][tx] = sum * smScale;

    // wait until qk done
    __syncthreads();

    // compute local max of each row of qk
    Ty localMax = -INFINITY;
    for (int i = 0; i < kBc; i++) {
      localMax = max(localMax, sQK[ty][i]);
    }
    __syncthreads();
    // compute the max of each row
    Ty newMax = max(sMax[ty], localMax);

    // compute safe e(e^{x - max}) of each qk element
    sSafeE[ty][tx] = exp(sQK[ty][tx] - newMax);
    __syncthreads();

    // accumulate local denom of each row of qk with local max
    Ty localDenom = 0.f;
    for (int i = 0; i < kBc; i++) {
      localDenom += sSafeE[ty][i];
    }
    __syncthreads();

    // rescale history result
    Ty rescaleOld = exp(sMax[ty] - newMax);
    // rescale denom
    Ty newDenom = sDenom[ty] * rescaleOld + localDenom;

    // NOTE:
    // QK[Br, Bc] @ V[Bc, d] = O[Br, d]
    // tx in [0, Bc], ty in [0, Br]
    // slice-Bc and each O[ty, group.x] as accumulator
    for (int i = 0; i < groupTx; i++) {
      // NOTE: rescale old_o(numerator only for now) once: old_nume * rescale
      sO[ty][i * kBc + tx] = (sO[ty][i * kBc + tx] * rescaleOld);
      for (int k = 0; k < kBc; k++) {
        // NOTE:
        // accumulate numerator
        // new_nume = old_nume' + local_nume (Softmax(QK)@V)
        sO[ty][i * kBc + tx] += sSafeE[ty][k] * sV[k][i * kBc + tx];
      }
    }

    // update global max and denom
    sMax[ty] = newMax;
    sDenom[ty] = newDenom;
    __syncthreads();
  }

  // rescale O in the end
  for (int i = 0; i < groupTx; i++) {
    // copy sO[row, dim] to gO[row, dim]
    O[row * kDim + i * kBc + tx] = sO[ty][i * kBc + tx] / sDenom[ty];
  }
}

template <typename Ty, int kBc, int kBr, int kDim>
__global__ void flash_attention_v1_kernel(Ty *Q, Ty *K, Ty *V, Ty *O, Ty *gMax,
                                          Ty *gDenom, int seqlen, int stride_head, Ty smScale) {
  // block size for K, V
  // group of row(seqlen)
  int groupSeq = (seqlen + kBc - 1) / kBc;
  // parallel process for V[Br, d]
  // group of column
  int groupTx = (kDim + kBc - 1) / kBc;
  int groupTy = (kDim + kBr - 1) / kBr;

  // load slice from global memory(HBM)
  __shared__ Ty sQ[kBr][kDim];
  __shared__ Ty sK[kBc][kDim];
  __shared__ Ty sV[kBc][kDim];
  __shared__ Ty sO[kBr][kDim];
  __shared__ Ty sQK[kBr][kBc];

  __shared__ Ty sNewO[kBr][kDim];
  // e^{x - max}
  __shared__ Ty sSafeE[kBr][kBc];
  // s stand for shared and local
  __shared__ Ty sDenom[kBr];
  __shared__ Ty sMax[kBr];

  // [0, Bc]
  int tx = threadIdx.x;
  // [0, Br]
  int ty = threadIdx.y;

  int row = ty + blockIdx.y * blockDim.y;
  int base_offset = blockIdx.x;

  if (row >= seqlen) {
    return;
  }

  Q += base_offset * stride_head;
  K += base_offset * stride_head;
  V += base_offset * stride_head;
  O += base_offset * stride_head;
  gMax += base_offset * seqlen;
  gDenom += base_offset * seqlen;

  for (int j = 0; j < groupSeq; j++) {
    if ((j * kBc + tx) < seqlen) {
      // load k, v from global memory to shared memory
      // K[seqlen, dim], V[seqlen, dim]
      for (int i = 0; i < groupTy; i++) {
        // each thread.x copy a row of K to K.T
        // row0, t0:
        // row1, t1:
        // row2, t0:
        // row3, t2:
        sK[tx][i * kBr + ty] = K[j * kBc * kDim + tx * kDim + i * kBr + ty];
        sV[tx][i * kBr + ty] = V[j * kBc * kDim + tx * kDim + i * kBr + ty];
      }
    }

    if (row < seqlen) {
      // load q, o, max, denom from global memory to shared memory
      // Q[seqlen, dim]
      for (int i = 0; i < groupTx; i++) {
        sQ[ty][i * kBc + tx] = Q[row * kDim + i * kBc + tx];
        sO[ty][i * kBc + tx] = O[row * kDim + i * kBc + tx];
      }

      // NOTE: the drawback of flash attention 1 is here that it will load O,
      // max, denom from global memory to shared memory many time
      sMax[ty] = gMax[row];
      sDenom[ty] = gDenom[row];
    }

    // wait until g2s done
    __syncthreads();

    // compute qk
    Ty sum = 0.f;
    // result oriented: qk[y][x] from q[y] @ k[x]
    for (int i = 0; i < kDim; i++) {
      sum += sQ[ty][i] * sK[tx][i];
    }
    // sQK[Br, Bc]
    sQK[ty][tx] = sum * smScale;

    // wait until qk done
    __syncthreads();

    // compute local max of each row of qk
    Ty localMax = -INFINITY;
    for (int i = 0; i < kBc; i++) {
      localMax = max(localMax, sQK[ty][i]);
    }
    __syncthreads();

    // compute safe e(e^{x - max}) of each qk element
    sSafeE[ty][tx] = exp(sQK[ty][tx] - localMax);
    __syncthreads();

    // accumulate local denom of each row of qk with local max
    Ty localDenom = 0.f;
    for (int i = 0; i < kBc; i++) {
      localDenom += sSafeE[ty][i];
    }
    __syncthreads();

    // NOTE: this is a pure flash attention 1 implementation with many redundant
    // mul update global max of each row
    Ty newMax = max(sMax[ty], localMax);
    // rescale history result
    Ty rescaleOld = exp(sMax[ty] - newMax);
    // rescale result just computed above: sSafeE, localDenom
    Ty rescaleCur = exp(localMax - newMax);
    Ty newDenom = sDenom[ty] * rescaleOld + localDenom * rescaleCur;

    // clean each row of of sNewO
    for (int i = 0; i < groupTx; i++) {
      sNewO[ty][i * kBc + tx] = 0;
    }

    // NOTE:
    // QK[Br, Bc] @ V[Bc, d] = O[Br, d]
    // tx in [0, Bc], ty in [0, Br]
    // slice-Bc and each O[ty, group.x] as accumulator
    for (int k = 0; k < kBc; k++) {
      for (int i = 0; i < groupTx; i++) {
        // rescale numerator
        sNewO[ty][i * kBc + tx] +=
            sSafeE[ty][k] * rescaleCur * sV[k][i * kBc + tx];
      }
    }

    // NOTE: rescale output
    // old_nume = old_o * old_denom
    // new_o = (old_nume + new_nume) / new_denom
    for (int i = 0; i < groupTx; i++) {
      sNewO[ty][i * kBc + tx] = (/* new_nume */ sNewO[ty][i * kBc + tx] +
                                /* old_o */ sO[ty][i * kBc + tx] * rescaleOld *
                                    /* old_denom */ sDenom[ty]) /
                               newDenom;
    }

    __syncthreads();

    // update global o
    if (row < seqlen) {
      for (int i = 0; i < groupTx; i++) {
        // copy sO[row, dim] to gO[row, dim]
        O[row * kDim + i * kBc + tx] = sNewO[ty][i * kBc + tx];
      }
    }

    // update global max and denom
    gMax[row] = newMax;
    gDenom[row] = newDenom;
    __syncthreads();
  }
}

torch::Tensor flash_attention_v1_cuda(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);
  // dim
  int dim = q.size(3);
  float sm_scale = 1.f / sqrtf(static_cast<float>(dim));
  // offset 1 in head dim should skip seqlen * dim elements
  int stride_head = seqlen * dim;

  auto out = torch::zeros_like(q);

  // Create intermediate date for the new shape of max, denom
  torch::TensorOptions options = q.options();
  std::vector<int64_t> shape = {bs, head, seqlen};
  torch::Tensor gMax = torch::empty(shape, options);
  torch::fill(gMax, -INFINITY);
  torch::Tensor gDenom = torch::zeros(shape, options);

  const int Br = 2;
  const int Bc = 2;
  int Gc = bs * head;
  int Gr = (seqlen + Br - 1) / Br;

  // NOTE: each thread process a tile of Q
  dim3 grid = dim3(Gc, Gr);
  // NOTE: each block process a range row of Q
  dim3 block = dim3(Bc, Br);

  // NOTE: AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)
  // We need a way of determining at runtime what type a tensor is and then
  // selectively call functions with the corresponding correct type signature.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      q.type(), "flash_attn_v1", ([&] {
        FWD_HEADDIM_SWITCH(dim, [&]{
            flash_attention_v1_kernel<scalar_t, Bc, Br, kHeadDim>
                <<<grid, block>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                                  v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                                  gMax.data_ptr<scalar_t>(),
                                  gDenom.data_ptr<scalar_t>(), seqlen, stride_head, sm_scale);
        });
      }));

  // Wait until kernel finish.
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return out;
}

torch::Tensor flash_attention_v2_cuda(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  // batch size
  int bs = q.size(0);
  // head number
  int head = q.size(1);
  // seqlen
  int seqlen = q.size(2);
  // dim
  int dim = q.size(3);
  float sm_scale = 1.f / sqrtf(static_cast<float>(dim));
  // offset 1 in head dim should skip seqlen * dim elements
  int stride_head = seqlen * dim;

  auto out = torch::zeros_like(q);

  const int Br = 4;
  const int Bc = 4;
  // grid.x indicate the base offset
  int Gc = bs * head;
  // grid.y indicate the group of row
  int Gr = (seqlen + Br - 1) / Br;
  assert(dim % Bc == 0 && seqlen % Br == 0);

  // NOTE: each block process a range row of Q
  dim3 grid = dim3(Gc, Gr);
  // NOTE: each thread process a tile of Q
  dim3 block = dim3(Bc, Br);

  // NOTE: AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)
  // We need a way of determining at runtime what type a tensor is and then
  // selectively call functions with the corresponding correct type signature.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.type(), "flash_attn_v2", ([&] {
    FWD_HEADDIM_SWITCH(dim, [&]{
      flash_attention_v2_kernel<scalar_t, Bc, Br, kHeadDim><<<grid, block>>>(
          q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
          v.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), seqlen, stride_head, sm_scale);
    });
  }));

  // Wait until kernel finish.
  cudaDeviceSynchronize();
  CUDA_ERROR_CHECK(cudaGetLastError());

  return out;
}


