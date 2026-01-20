#pragma once

#include <immintrin.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using float32x4_t = __m128;
using float32x8_t = __m256;
// #if defined(__AVX512F__)
using float32x16_t = __m512;

// using float16x8_t = __m128h;
// using float16x16_t = __m256h;
// using float16x32_t = __m512h;

// TODO:
using fp16_t = c10::Half;
using bf16_t = c10::BFloat16;


// bf16 casting
void cvt_bf16_to_fp32_row(float *dst, const bf16_t *src, size_t n);
void cvt_fp32_to_bf16_row(bf16_t *dst, const float *src, size_t n);

void cvt_fp16_to_fp32_row(float * dst, const fp16_t * src, size_t n);
void cvt_fp32_to_fp16_row(fp16_t * dst, const float * src, size_t n);


template <typename T>
void auto_cast(float *dst, T *src, size_t n);

template <typename T>
void auto_write_back(T *dst, float *src, size_t n);

template <> inline void auto_cast(float *dst, bf16_t *src, size_t n) {
  cvt_bf16_to_fp32_row(dst, src, n);
}

template <> inline void auto_cast(float *dst, fp16_t *src, size_t n) {
  cvt_fp16_to_fp32_row(dst, src, n);
}

template <> inline void auto_cast(float *dst, float *src, size_t n) {
  // empty
}

template<> inline void auto_write_back(bf16_t *dst, float *src, size_t n) {
  cvt_fp32_to_bf16_row(dst, src, n);
}

template<> inline void auto_write_back(fp16_t *dst, float *src, size_t n) {
  cvt_fp32_to_fp16_row(dst, src, n);
}

template<> inline void auto_write_back(float *dst, float *src, size_t n) {
  // empty
}

// debug tools
#ifndef NDEBUG // DEBUG
  #define ASSERT(cond, msg) assert(cond && msg)

  // e.g. data.to("cuda") in python
  #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
  #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
  #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
  #define CUDA_ERROR_CHECK(condition)                                            \
    do {                                                                         \
      cudaError_t error = condition;                                             \
      if (error != cudaSuccess) {                                                \
        printf("CUDA_CHECK error in line %d of file %s \
                : %s \n",                                                        \
               __LINE__, __FILE__, cudaGetErrorString(error));      \
        exit(EXIT_FAILURE);                                                      \
      }                                                                          \
    } while (0)


#else // not DEBUG
  #define ASSERT(cond, msg) {} while(0)

  #define CHECK_CUDA(x) do { } while (0)
  #define CHECK_CONTIGUOUS(x) do { } while (0)
  #define CHECK_INPUT(x) do { } while (0)
  #define CUDA_ERROR_CHECK(condition) do { condition; } while (0)


#endif



