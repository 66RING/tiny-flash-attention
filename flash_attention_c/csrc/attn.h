#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <cmath>

template <typename T>
struct Naive_fwd_traits {
  using elem_type = T;
};
