#pragma once

#include <assert.h>
#include "utils.h"

template <typename T, typename U> T load(const U *);
template <typename T> T zeros();
template <typename T, typename U> void store(U *p, T &a);
// TODO: rename to dot?
template <typename T> T fmadd(T &a, T &b, T &c);
template <typename T> T mul(T &a, T &b);
template <typename T> T sub(T &a, T &b);
template <typename T> T add(T &a, T &b);
// TODO: reduce half/bf16?
template <typename T> float reduce_sum(T &p);
template <typename T, typename U> T reg_init(const U &a);


#define REPEAT_ARG_4(fn, arg) fn(arg, arg, arg, arg)
#define REPEAT_ARG_8(fn, arg) fn(arg, arg, arg, arg, arg, arg, arg, arg)
#define REPEAT_ARG_16(fn, arg) fn(arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg, arg)

#define XMACROS_VEC_SIZE_TABLE(f) \
    f(float32x4_t, _mm, 4) \
    f(float32x8_t, _mm256, 8) \
    f(float32x16_t, _mm512, 16)

// TODO: rename for fp32

// generate reg_init()
#define _FUNCTION(T, bit, epr) template <> inline T reg_init(const float &p) { return REPEAT_ARG_##epr(bit##_set_ps, p); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate load()
#define _FUNCTION(T, bit, epr) template <> inline T load(const float *p) { return bit##_loadu_ps(p); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate store()
#define _FUNCTION(T, bit, epr) template <> inline void store(float *p, T& a) { bit##_storeu_ps(p, a); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate zeros()
#define _FUNCTION(T, bit, epr) template <> inline T zeros() { return bit##_setzero_ps(); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate fmadd()
#define _FUNCTION(T, bit, epr) template <> inline T fmadd(T &a, T &b, T &c) { return bit##_fmadd_ps(a, b, c); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate mul()
#define _FUNCTION(T, bit, epr) template <> inline T mul(T &a, T &b) { return bit##_mul_ps(a, b); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate sub()
#define _FUNCTION(T, bit, epr) template <> inline T sub(T &a, T &b) { return bit##_sub_ps(a, b); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION

// generate add()
#define _FUNCTION(T, bit, epr) template <> inline T add(T &a, T &b) { return bit##_add_ps(a, b); }
  XMACROS_VEC_SIZE_TABLE(_FUNCTION)
#undef _FUNCTION


// TODO: performance?
template <> inline float reduce_sum(float32x4_t &p) {
  const float32x4_t t0 = _mm_hadd_ps(p, p);
  return _mm_cvtss_f32(_mm_hadd_ps(t0, t0));
}

template <> inline float reduce_sum(float32x8_t &p) {
  const float32x4_t hi_p = _mm256_extractf128_ps(p, 1); // 提取高128位
  const float32x4_t lo_p = _mm256_castps256_ps128(p);   // 提取低128位
  const float32x4_t sum_p = _mm_add_ps(lo_p, hi_p);

  const float32x4_t t0 = _mm_hadd_ps(sum_p, sum_p);
  return _mm_cvtss_f32(_mm_hadd_ps(t0, t0));
}

template <> inline float reduce_sum(float32x16_t &p) {
  return _mm512_reduce_add_ps(p);
}

// qk = q @ k.T
// q.shape = (1, dim)
// k.shape = (1, dim)
// qk.shape = (1)
//
// for (int dim = 0; dim < params.head_dim; dim++) {
//     qk += q[dim] * k[dim];
// }
//
// return fp32 accumulator as attention score
template <typename vec_t, typename scalar_t>
inline float row_qk_dot(const scalar_t* q, const scalar_t* k, size_t dim) {
  // dot product: c = a @ b.T
  // TODO: array to do batching load or higher bits?
  vec_t a, b;
  vec_t c = zeros<vec_t>();
  constexpr int epr = sizeof(vec_t) / sizeof(scalar_t);
  ASSERT(dim % epr == 0, "dim must be multiple of epr");

  for (int i = 0; i < dim; i+=epr) {
    a = load<vec_t>(&q[i]);
    b = load<vec_t>(&k[i]);

    // TODO: inline or reuse
    c = fmadd<vec_t>(a, b, c);
  }

  // reduce c
  return reduce_sum<vec_t>(c);
}

// Perform rescale and accumulate v
//
// for (int dim = 0; dim < params.head_dim; dim++) {
//   out[dim] *= scale;
//   out[dim] += exp * v[dim];
// }
template <typename vec_t, typename scalar_t>
inline void row_score_v(scalar_t* o, const scalar_t* v, size_t dim, float exp, float scale) {
  vec_t scale_x = reg_init<vec_t>(scale);
  vec_t exp_x = reg_init<vec_t>(exp);

  vec_t o_x, v_x;

  constexpr int epr = sizeof(vec_t) / sizeof(scalar_t);
  ASSERT(dim % epr == 0, "dim must be multiple of epr");
  for (int i = 0; i < dim; i+=epr) {
    o_x = load<vec_t>(&o[i]);
    v_x = load<vec_t>(&v[i]);

    // rescale
    o_x = mul<vec_t>(o_x, scale_x);
    o_x = fmadd<vec_t>(exp_x, v_x, o_x);

    // update out
    store<vec_t>(&o[i], o_x);
  }
}

// Perform final rescale for out /= score_sum
//
// for (int dim = 0; dim < params.head_dim; dim++) {
//   out[dim] /= score_sum;
// }
template <typename vec_t, typename scalar_t>
inline void row_out_rescale(scalar_t* o, size_t dim, float inv_score_sum) {
  vec_t inv_score_sum_x = reg_init<vec_t>(inv_score_sum);

  vec_t o_x;
  constexpr int epr = sizeof(vec_t) / sizeof(scalar_t);
  ASSERT(dim % epr == 0, "dim must be multiple of epr");
  for (int i = 0; i < dim; i+=epr) {
    o_x = load<vec_t>(&o[i]);

    // rescale
    o_x = mul<vec_t>(o_x, inv_score_sum_x);

    // update out
    store<vec_t>(&o[i], o_x);
  }
}





