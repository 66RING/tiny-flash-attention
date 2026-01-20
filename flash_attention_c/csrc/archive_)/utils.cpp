#include "utils.h"

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
float cvt_bf16_to_fp32_scalar(bf16_t h) {
  union {
    float f;
    uint32_t i;
  } u;
  // NOTE: (*(uint16_t*)&h) casting to "int" to do bit wise operation
  u.i = (uint32_t)(*(uint16_t*)&h) << 16;
  return u.f;
}


void cvt_bf16_to_fp32_row(float *dst, const bf16_t *src, size_t n) {
  size_t i = 0;
  // TODO: hard code mm256, __AVX512F__ support higher bits
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i,
                     _mm256_castsi256_ps(
                         _mm256_slli_epi32( // shift left logical 16bit
                             _mm256_cvtepu16_epi32(
                                 _mm_loadu_si128(
                                     (const __m128i *)(src + i))),
                             16)));
  }
  // leftover
  for (; i < n; i++) {
    dst[i] = cvt_bf16_to_fp32_scalar(src[i]);
  }
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
bf16_t cvt_fp32_to_bf16_scalar(float s) {
  bf16_t h;
  union {
    float f;
    uint32_t i;
  } u;
  u.f = s;
  if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
    (*(uint16_t*)&h) = (u.i >> 16) | 64; /* force to quiet */
    return h;
  }
  // NOTE: (*(uint16_t*)&h) casting to "int" to do bit wise operation
  (*(uint16_t*)&h) = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
  return h;
}

void cvt_fp32_to_bf16_row(bf16_t * dst, const float * src, size_t n) {
  int i = 0;
  // TODO: naive casting for now
  for (; i < n; i++) {
    dst[i] = cvt_fp32_to_bf16_scalar(src[i]);
  }
}

// TODO: naive fp32 <-> fp16
// Converts float32 to float16.
fp16_t cvt_fp32_to_fp16_scalar(float f) {
  // fp16_t res;
  // // use int to do bit wise copy
  // uint16_t tmp = f;
  // memcpy(&res, &tmp, sizeof(fp16_t));
  // return res;
  return (fp16_t)f;
}

// Converts float16 to float32.
float cvt_fp16_to_fp32_scalar(fp16_t h) {
  // // use int to do bit wise copy
  // uint16_t tmp;
  // memcpy(&tmp, &h, sizeof(fp16_t));
  // return (float)tmp;
  return (float)h;
}

void cvt_fp32_to_fp16_row(fp16_t * dst, const float * src, size_t n) {
  // TODO: SIMD
  int i = 0;
  for (; i < n; i++) {
      dst[i] = cvt_fp32_to_fp16_scalar(src[i]);
  }
}

void cvt_fp16_to_fp32_row(float * dst, const fp16_t * src, size_t n) {
  int i = 0;
  for (; i < n; i++) {
      dst[i] = cvt_fp16_to_fp32_scalar(src[i]);
  }
}

// TODO: aligned allocate -> ggml_aligned_malloc





