#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "kernel_traits.h"
#include "utils.h"

#define DEBUG

// data type to test
using FP = float;
using FPC = cute::half_t;
// Out type
#ifdef DEBUG
// 精度测试
using FPC_O = float;
#else
using FPC_O = cute::half_t;
#endif
// using FPC = double;
// BLOCK_M(Br, Brow), BLOCK_N(Bc, Bcol) can be determined at compile time
// just like offical implementation which use a template kernel to do that
// Block row size
// TODO: 测试这里多种shape
const int Bm = 64;
// Block column size
const int Bn = 64;
// TODO: 测试更大规模, 或者warps=2

// TODO: causal模式下, warp!=1情况有bug
// 使用kNThreads
const int Warps = 4;
const bool IS_CAUSAL = false;

const int BS = 2;
const int HEAD = 16;
const int SEQLEN = 128 * 3;
const int DIM = 64;
// const float softmax_scale = 1.f / sqrtf(static_cast<float>(SEQLEN));
const float softmax_scale = 1.f;

// debug only
int TX = 3;
int TY = 0;

// TODO: test trait
using Test_Traits = Flash_fwd_kernel_traits<DIM, Bm, Bn, Warps, FPC>;


// Shared Storage with Aligned addresses.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct SharedStorage {
  // TODO: Aligned的话smem的计算是否有问题
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};


#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                        \
             __LINE__, __FILE__, cudaGetErrorString(error));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr)                                                      \
  do {                                                                         \
    expr                                                                       \
  } while (0)
#else
#define DEBUG_BLOCK(...)                                                       \
  do {                                                                         \
  } while (0)
#endif

// TODO: 待功能差不多后再使用torch
void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t bs, const size_t head, const size_t seqlen,
                      const size_t dim,

                      const size_t bs_stride, const size_t head_stride,
                      const size_t seqlen_stride, const size_t dim_stride,

                      // // device pointers
                      // const torch::Tensor q,
                      // const torch::Tensor k,
                      // const torch::Tensor v,
                      // torch::Tensor out,

                      void *q, void *k, void *v, void *out,

                      float softmax_scale) {

  memset(&params, 0, sizeof(params));

  params.bs = bs;
  params.head = head;
  params.seqlen = seqlen;
  params.dim = dim;

  params.bs_stride = bs_stride;
  params.head_stride = head_stride;
  params.seqlen_stride = seqlen_stride;
  params.dim_stride = dim_stride;

  params.softmax_scale = softmax_scale;

  // TODO: get ptr
  params.q_ptr = q;
  params.k_ptr = k;
  params.v_ptr = v;
  params.out_ptr = out;
}

__global__ void naive_nrow_gemm(FP *A, FP *B, FP *C, FP a, FP b,
                                int M, int N, int K, int mBlock);
__global__ void causal_mask_qk(FP *qk, int m);
__global__ void row_softmax(FP *input, FP *output, int n);
__global__ void naive_pv(FP *P, FP *V, FP *O, int M, int N,
                         int mBlock);

template<typename T, typename U>
bool all_close(T *A, U *B, int total_size);

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, int kNWarps,typename Engine, typename Layout>
inline __device__ void mask_within_nblock(Tensor<Engine, Layout> &tensor, const int m_block, const int nbi) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");

    // NOTE:
    // 确定一个MMA内的index也是一个难点
    // (nrow=(2, MMA_M), ncol=(2, MMA_N))形如:
    //    T1.V0 T1.V1
    //    T1.V0 T1.V1
    // 根据mma_tile的示意图来确定col和row值

    // NOTE:
    // 计算thread的处理范围, mask掉超出范围的部分
    //
    // NOTE:
    // % 32表示32做组, 因为SM80_16x8x16_F32F16F16F32_TN _1_2_1中最大线程数id是32
    // (lane_id % 4) * 2表示在哪个"颜色"的col(thread)中, *2是为了靠右(即处理的哪个value2)
    // 因此col_idx_offset表示当前thread所处理的单个Atom中4列的哪列

    // lane_id表示一个MMA tile中的"线程组"
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = kBlockN * nbi + (lane_id % 4) * 2;

    const int nrow_group = threadIdx.x / 32;
    const int row_idx_offset = kBlockM * m_block + lane_id / 4 + nrow_group * 16 /* 2*8 */;
    // (2, nrow), 2*8 for each
    const int group_stride = kNWarps * 16;

    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        // SM80_16x8x16_F32F16F16F32_TN中的一组中, 一行4个线程处理8个value
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            // j用于计算value 1和value 2对应col
            // col_idx最终表示当前thread所处理的value的列号
            const int col_idx = col_idx_base + j;

            // mask掉scores中(QK后的结果)超出范围的部分
            // 列号和行号对比

            // Without the "make_coord" we get wrong results
            // for nrow(2, MMA_M)
            #pragma unroll
            for (int mi = 0; mi < size<0, 0>(tensor); ++mi) {

              #pragma unroll
              for (int mj = 0; mj < size<0, 1>(tensor); ++mj) {
                const int row_idx = row_idx_offset + mi * 8 + mj * group_stride;
                if (col_idx > row_idx) {
                  tensor(make_coord(mi, mj), make_coord(j, nj)) = -INFINITY;
                }
              }

            }

        }
    }
}


// NOTE: A矩阵已经在寄存器中的gemm封装
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    // NOTE: 符合M N K描述: A[M, K] @ B[N, K] = C[M, N]
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    // NOTE: retile 成拷贝需要的大小
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// copy from S to D with tiled_copy
// TODO: 需要支持causal模式的的跳过拷贝
template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // TODO: 原版处这里identity_MN是用来跳过大块的block的, predicate用于跳过block内的拷贝
        // TODO: 添加predicate逻辑, 用于跳过无用拷贝
        // if (get<0>(identity_MN(0, m, 0)) < max_MN)
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        }
    }
}


// Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
    // TD [2023-08-13]: Same error as above on Cutlass 3.2
    // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
    //                    get<0, 1>(l),
    //                    get<1, 1, 1>(l));
    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)),
                       get<1>(get<1>(get<1>(l))));
};


// TODO: not work
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// TODO:
// https://github.com/NVIDIA/cutlass/issues/802
// TODO: convert出来后数据是否在寄存器?
template <typename Fragment>
inline __device__ auto convert_type_f32_to_f16(Fragment const &acc_fp32) {
  Tensor acc_fp16 = make_tensor<cute::half_t>(shape(acc_fp32));
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp16x2 = recast<__half2>(acc_fp16);
    for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
  }
  return acc_fp16;
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}



// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// TODO: 搞清楚经过convert_layout_acc_rowcol后(nrow=(2, MMA_M), ncol=(2, MMA_N))的数学含义
// 形象的解释是把
//    T1.V0
//    T1.V1
//    T1.V0
//    T1.V1
// 变为
//    T1.V0 T1.V1
//    T1.V0 T1.V1
// 这样符合MMA tile的行列直觉
template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
    // "int_tuple.hpp(74): error: conversion to inaccessible base class"
    // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

// TODO: is first优化初次rescale
template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    // NOTE: scores来自acc_s: Q@K.T
    // acc_s用来存储QK和softmax的结果[seqlen, seqlen]
    // acc_o用来存储softmax(QK)结果的分子部分, 用于rescale
    // 流式计算不断用当前分块计算的结果scors来rescale

    if (Is_first) {
        // NOTE: 优化, 第一次softmax不需要rescale, 只需要记录分子, max, sum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // 记录上一次的max
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        // TODO: reduce的实现学习一下
        // NOTE: 计算新max到scores_max
        // reduce_max包含步:
        //  1. 求当前thread内max: 遍历
        //  2. reduce thread间的max: 使用shift技巧reduce
        reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            // NOTE: 辅助变量: 当前max
            float scores_max_cur = scores_max(mi);
            // NOTE: 计算旧score的rescale值
            // NOTE: 因为QK(影响max)计算时没有考虑softmax_scale, 所以这里要补上
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            // NOTE: rescale旧分母部分
            scores_sum(mi) *= scores_scale;
            // NOTE: 旧分子部分rescale
            // acc_o_rowcol.shape = (nrow, ncol)
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        // NOTE: 计算新分子部分: 对所有scores进行rescale
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        // NOTE: 累加新分母
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        // NOTE:利用新分子来累加新分母
        //  1. 线程内累加: 遍历
        //  2. 线程间累加: 使用shift技巧reduce
        reduce_sum(scores, scores_sum_cur);
        // NOTE: 新分母累加到旧分母
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); }
    }
};

} // namespace flash

template <typename Kernel_traits, bool Is_causal=false, typename Params>
__global__ void flash_attention_v2_cutlass_kernel(const Params params) {

  using namespace cute;

  // num_m_block: seqlen group
  const int m_block = blockIdx.x;

  // bs * head
  const int base_id = blockIdx.y;
  // The thread index.
  const int tidx = threadIdx.x;
  const int bs_head_offset = base_id * params.head_stride;

  // TODO: 传入泛型
  // NOTE: 小技巧
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  // using TiledMMA = typename Kernel_traits::MMA;
  using TiledMMA = typename Kernel_traits::TiledMma;
  using index_t = typename Kernel_traits::index_t;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutVt = typename Kernel_traits::SmemLayoutVtransposed;
  using SmemLayoutVtNoSwizzle = typename Kernel_traits::SmemLayoutVtransposedNoSwizzle;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  // Shared memory.
  extern __shared__ char smem_[];
  using SharedStorage = SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  // TODO: base offset for MHA
  // NOTE: convert C pointer to Tensor for convenience
  Tensor Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
      make_shape(params.seqlen, params.dim),
      make_stride(params.dim, Int<1>{}));
  Tensor K = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset),
      make_shape(params.seqlen, params.dim),
      make_stride(params.dim, Int<1>{}));
  Tensor V = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset),
      make_shape(params.seqlen, params.dim),
      make_stride(params.dim, Int<1>{}));

  // 加载Q, K, V分块
  // (kBlockM, kHeadDim, num_tile_n)
  Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // (kBlockN, kHeadDim, num_tile_n)
  // NOTE: loading流水线, 初次加载所需K, V
  Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));

  // 获取MMA抽象
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

  // NOTE: copy抽象
  // NOTE: QKV gmem -> smem拷贝的抽象
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  // NOTE: 定义gmem -> smem拷贝的src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);


  // NOTE: 定义smem -> reg拷贝的dst
  // partition_fragment与partition类似, 只是返回的是寄存器表示
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
  Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)

  //
  // Copy Atom retiling
  //

  // TODO: 理解这里的atom retiling

  // NOTE: 准备拷贝Q, K, V到smem的copy对象
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  // TODO: 拷贝时转置
  // NOTE: smem->reg拷贝Vt
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // NOTE: 命名规则, t表示to, s/g表示位置(smem, gmem)
  // 从smem加载时做retiling
  // tKgK表示gmem中的K, 用作gmem->smem的src
  // tKsK表示smem中的K, 用作gmem->smem的dst
  // tSsK表示smem中的K, 用作smem->reg的src

  // 流水线加载初始Q, K
  // 加载Q到smem
  flash::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // 加载K到smem
  flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  // 开始执行异步拷贝
  cute::cp_async_fence();

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // step1: slice-k compute QK block
  // Q[BLOCK_M, BLOCK_N] @ K[BLOCK_M, BLOCK_N].T = O[BLOCK_M, BLOCK_M]
  //
  // step2:
  // advance K, V

  // NOTE: K, V分块的数量: 处理的区间
  const int n_block_min = 0;
  // NOTE: 1. mask between N BLOCKs if is causal mode
  int seqlen_start = m_block * kBlockM;
  int seqlen_end = (m_block + 1) * kBlockM;
  int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.seqlen, kBlockN);

  // NOTE: 需要记录的max
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});
  // NOTE: 需要记录的denom
  Tensor scores_sum = make_fragment_like(scores_max);

  clear(rAccOut);

  for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
    auto rAccScore = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

    clear(rAccScore);

    // 等待Q, K的gmem -> smem拷贝完成, 即Q, K就绪
    // wait<0>表示等待还剩0个未完成
    flash::cp_async_wait<0>();
    __syncthreads();

    // gemm的同时异步加载V
    gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    // 异步加载V到smem
    flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // 发起异步拷贝
    cute::cp_async_fence();

    // O = Q@K.T
    // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );

    Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. mask within N BLOCKs
    if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
    }

    // NOTE: 等待V加载完成, 为下个K加载准备初始状态
    flash::cp_async_wait<0>();
    __syncthreads();

    // advance K
    if (nbi != n_block_max - 1) {
      gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      cute::cp_async_fence();
    }

    // 计算softmax
    // NOTE: rAccOut记录softmax后所有的分子
    nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
      flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

    // 实际执行QK @ V
    // (score AKA rAccScore): QK[M, N] @ V[N, dim]
    // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
    // TODO: convert_type目前写死
    Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
    // NOTE: Convert from layout C to layout A
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

    flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // NOTE: 最后统一除上分母部分
  // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
  Tensor acc_o_rowcol = make_tensor(rAccOut.data(), flash::convert_layout_acc_rowcol(rAccOut.layout()));
  // for row
  #pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    float scale = inv_sum;
    // for col
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { 
      acc_o_rowcol(mi, ni) *= scale; 
    }
  }

  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = flash::convert_type_f32_to_f16(rAccOut);
  // 复用sQ的smem做sO的拷出
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)

  // Partition sO to match the accumulator partitioning
  // TODO: review
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // NOTE: 先拷贝到smem
  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor O = make_tensor(
      // Use ElementAccum(f32) to debug
#ifdef DEBUG
      make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.out_ptr) + bs_head_offset),
#else
      make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset),
#endif
      make_shape(params.seqlen, params.dim),
      make_stride(params.dim, Int<1>{}));
  Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // 创建到smem -> gmem的拷贝
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();

  // NOTE:: 再拷贝到gmem

  // TODO: review, 这里两个copy的作用
  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  flash::copy(gmem_tiled_copy_O, tOrO, tOgO);
}

void flash_attention_v2_cuda(FPC *Q, FPC *K, FPC *V, FPC_O *O, int bs, int head, int seqlen, int dim) {
  using Kernel_traits = Test_Traits;
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutKV;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutKV;

  // Q smem size + KV smem size
  constexpr int kSmemSize = Kernel_traits::kSmemSize;

  int bs_stride = head * seqlen * dim;
  int head_stride = seqlen * dim;
  int seqlen_stride = dim;
  int dim_stride = 1;
  // int smem_size = kSmemSize;
  constexpr size_t smem_size = size_t(sizeof(SharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  Flash_fwd_params params;
  set_params_fprop(params, bs, head, seqlen, dim, bs_stride, head_stride,
                   seqlen_stride, dim_stride, Q, K, V, O, softmax_scale);

  const int num_m_block =
      (params.seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

  auto kernel = &flash_attention_v2_cutlass_kernel<Kernel_traits, IS_CAUSAL, Flash_fwd_params>;
  // NOTE: smem过大时需要设置
  if (smem_size >= 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  dim3 grid(num_m_block, params.bs * params.head, 1);
  dim3 block(size(Kernel_traits::kNThreads));

  kernel<<<grid, block, smem_size>>>(params);
  CUDA_CHECK(cudaGetLastError());

  cudaDeviceSynchronize();
}

void self_attention_run(FP *Q, FP *K, FP *V, FP *O, int m, int n, bool is_causal, float sm_scale = 1) {
  int mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  FP *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(FP) * m * m);

  dim3 qk_block(m / mBlock, 1, 1);
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
  cudaDeviceSynchronize();

  // causal mask
  if (is_causal == true) {
    // QK[M, M]
    dim3 grid(m / mBlock, 1, 1);
    dim3 block(mBlock, 1, 1);
    causal_mask_qk<<<grid, block>>>(sm_o, m);
  }

  {
    // TODO: test QK only
    FP *h_sm_o = new FP[m * m];
    cudaMemcpy(h_sm_o, sm_o, sizeof(FP) * m * m, cudaMemcpyDeviceToHost);
    Tensor Self = make_tensor(h_sm_o, make_shape(m, m), make_stride(m, 1));
    auto tile = make_tile(8, 8);
    print("self QK: \n");
    print("%d,%d:\n", TX, TY);
    print_tensor(local_tile(Self, tile, make_coord(TX, TY)));
    // print("0,1:\n");
    // print_tensor(local_tile(Self, tile, make_coord(0, 1)));
    // print("1,0:\n");
    // print_tensor(local_tile(Self, tile, make_coord(1, 0)));
    // print("1,1:\n");
    // print_tensor(local_tile(Self, tile, make_coord(1, 1)));
    free(h_sm_o);
  }

  // QK[M, M]
  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
  cudaDeviceSynchronize();
  {
    // TODO: test QK only
    FP *h_sm_o = new FP[m * m];
    cudaMemcpy(h_sm_o, sm_o, sizeof(FP) * m * m, cudaMemcpyDeviceToHost);
    Tensor Self = make_tensor(h_sm_o, make_shape(m, m), make_stride(m, 1));
    auto tile = make_tile(8, 8);
    auto coor = make_coord(TX, TY);
    Tensor tSelf = local_tile(Self, tile, coor);
    print("self softmax(QK): \n");
    print_tensor(tSelf);
    free(h_sm_o);
  }

  // QK[M, M] @ V[M, N]
  dim3 qkv_block(m / mBlock, 1, 1);
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  cudaDeviceSynchronize();

  {
    FP *h_sm_o = new FP[m * n];
    cudaMemcpy(h_sm_o, O, sizeof(FP) * m * n, cudaMemcpyDeviceToHost);
    Tensor Self = make_tensor(h_sm_o, make_shape(m, n), make_stride(n, 1));
    auto tile = make_tile(8, 8);
    auto coor = make_coord(TX, TY);
    print("self O: \n");
    print_tensor(local_tile(Self, tile, make_coord(TX, TY)));
    print("x,1:\n");
    print_tensor(local_tile(Self, tile, make_coord(TX, TY+1)));
    free(h_sm_o);
  }

  cudaFree(sm_o);
}

void self_attention_cuda(FP *Q, FP *K, FP *V, FP *O, int bs, int head, int seqlen, int dim, bool is_causal, float sm_scale = 1) {
  int stride = seqlen * dim;
  int total_size = bs * head * seqlen * dim;
  for (int i = 0; i < bs * head; i++) {
    self_attention_run(Q, K, V, O, seqlen, dim, is_causal, sm_scale);
    Q += stride;
    K += stride;
    V += stride;
    O += stride;
  }
}

// naive gemm implement with slice-k
// perform C = aA@B + bC
// A[M, K] x B[K, N] = C[M, N]
// each thread process mblock rows of A
__global__ void naive_nrow_gemm(FP *A, FP *B, FP *C, FP a, FP b,
                                int M, int N, int K, int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  // A[mBlock, K] x B[N, K].T = C[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      FP sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      // C[M, N]
      // C = aA@B + bC
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

__global__ void causal_mask_qk(FP *qk, int m) {
  // each thread process a row
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  int row = idx;
  for (int j = row + 1; j < m; j++) {
    qk[row * m + j] = -INFINITY;
  }
}

// perform QK[M, M] @ V[M, N]
__global__ void naive_pv(FP *P, FP *V, FP *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  // each thread process a range of rows
  idx *= mBlock;

  int K = M;
  // P[mBlock, M] x V[M, N] = O[mBlock, N]
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      FP sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      // C[M, N]
      O[i * N + j] = sum;
    }
  }
}

// each thread process one row of softmax
__global__ void row_softmax(FP *input, FP *output, int n) {
  // assume id will not exceed row number of input
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  FP max = -INFINITY;
  FP sum = 0.f;

  // Find max
  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max) {
      max = input[idx * n + i];
    }
  }

  // Compute numerator and denominator
  for (int i = 0; i < n; i++) {
    output[idx * n + i] = exp2(input[idx * n + i] - max);
    sum += output[idx * n + i];
  }

  // Compute softmax
  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}

void test_attention() {
  int bs = BS;
  int head = HEAD;
  // seqlen
  int m = SEQLEN;
  // dim
  int n = DIM;
  int total_size = bs * head * m * n;

  // Host pointer
  FP *h_K = new FP[total_size];
  FP *h_Q = new FP[total_size];
  FP *h_V = new FP[total_size];
  FP *h_O = new FP[total_size];

  FPC *h_K2 = new FPC[total_size];
  FPC *h_Q2 = new FPC[total_size];
  FPC *h_V2 = new FPC[total_size];
  FPC_O *h_O2 = new FPC_O[total_size];

  // 初始化 K, Q, V
  for (int i = 0; i < total_size; ++i) {
    h_K[i] = static_cast<FP>(rand()) / RAND_MAX;
    h_Q[i] = static_cast<FP>(rand()) / RAND_MAX;
    h_V[i] = static_cast<FP>(rand()) / RAND_MAX;
    // h_K[i] = static_cast<FP>(0.0001f * i);
    // h_Q[i] = static_cast<FP>(0.0001f * i);
    // h_V[i] = static_cast<FP>(0.0001f * i);

    h_Q2[i] = FPC(h_Q[i]);
    h_K2[i] = FPC(h_K[i]);
    h_V2[i] = FPC(h_V[i]);
  }

  FP *d_K, *d_Q, *d_V, *d_O;
  FPC *d_K2, *d_Q2, *d_V2;
  FPC_O *d_O2;
  // Malloc device memory
  cudaMalloc((void **)&d_K, sizeof(FP) * total_size);
  cudaMalloc((void **)&d_Q, sizeof(FP) * total_size);
  cudaMalloc((void **)&d_V, sizeof(FP) * total_size);
  cudaMalloc((void **)&d_O, sizeof(FP) * total_size);

  cudaMalloc((void **)&d_K2, sizeof(FPC) * total_size);
  cudaMalloc((void **)&d_Q2, sizeof(FPC) * total_size);
  cudaMalloc((void **)&d_V2, sizeof(FPC) * total_size);
  cudaMalloc((void **)&d_O2, sizeof(FPC_O) * total_size);

  // Copy data from host to device
  cudaMemcpy(d_K, h_K, sizeof(FP) * total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q, h_Q, sizeof(FP) * total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, sizeof(FP) * total_size, cudaMemcpyHostToDevice);

  cudaMemcpy(d_K2, h_K2, sizeof(FPC) * total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q2, h_Q2, sizeof(FPC) * total_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V2, h_V2, sizeof(FPC) * total_size, cudaMemcpyHostToDevice);


  // Run test
  for (int i = 0; i < 1; i++) {
    // Launch kernel
    bool is_causal = IS_CAUSAL;
    self_attention_cuda(d_Q, d_K, d_V, d_O, bs, head, m, n, is_causal, softmax_scale);

    CUDA_CHECK(cudaGetLastError());
  }
  cudaDeviceSynchronize();

  int epoch = 100;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // test flash attention 2
  for (int i = 0; i < epoch; i++) {
    flash_attention_v2_cuda(d_Q2, d_K2, d_V2, d_O2, bs, head, m, n);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Result back to host
  cudaMemcpy(h_O, d_O, sizeof(FP) * total_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_O2, d_O2, sizeof(FPC_O) * total_size, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  Tensor Cute = make_tensor(h_O2, make_shape(m, n), make_stride(n, 1));
  auto tile = make_tile(8, 8);
  print("cute: \n");
  print_tensor(local_tile(Cute, tile, make_coord(TX, TY)));
  print_tensor(local_tile(Cute, tile, make_coord(TX, TY + 1)));

  assert(all_close(h_O, h_O2, total_size) && "flash attention 1 != flash attention 2");



  cudaFree(d_K);
  cudaFree(d_Q);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaFree(d_K2);
  cudaFree(d_Q2);
  cudaFree(d_V2);
  cudaFree(d_O2);
  free(h_Q);
  free(h_K);
  free(h_V);
  free(h_O);
  free(h_Q2);
  free(h_K2);
  free(h_V2);
  free(h_O2);
}

template <typename T, typename U>
bool all_close(T *A, U *B, int total_size) {
  for (int i = 0; i < total_size; i++) {
    if (fabs(A[i] - B[i]) > 1e-2) {
      printf("A[%d] = %f, B[%d] = %f\n", i, A[i], i, (float)B[i]);
      return false;
    }
  }
  return true;
}

int main() {
  int epoch = 1;
  for (int i = 0; i < epoch; i++)
    test_attention();

  return 0;
}

