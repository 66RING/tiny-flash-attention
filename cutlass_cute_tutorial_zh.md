---
title: cutlass cute实现flash attention
author: 66RING
date: 2024-05-08
tags: 
- cuda
- cutlass
- machine learning system
- mlsys
mathjax: true
---

# 用cutlass cute实现flash attention

flash attention自顶向下(虽然我学cutlass是自底向上学的但是感觉快速上手应该自顶向下学)。因为有了cutlass cute用户就可以方便的实现一些功能了, 即一些cuda编程的范式:

- cuda程序范式: global mem -> share mem -> reg -> compute
    * block tiling: 
        + aka 复用smem, gmem -> smem的拷贝
    * thread tiling:
        + aka 复用reg, smem -> reg的拷贝
    * 合并访存, 向量访存:
        + aka 向量指令, LDSM, ldmatrix指令
    * warp divergent线程束分化
        + aka warp负载均衡, 同理流水线气泡问题
    * bank conflict冲突消解: swizzle
        + aka 利用内存的多路通道
    * double buffering
        + aka 加载和计算的流水线
    * ...

需要自底向上学的朋友推荐看[reed哥的系列教程](https://www.zhihu.com/people/reed-84-49)


## Acknowledge

- 直接抄的flash attention的代码，但是从0写一遍抄一遍的
- 排雷了不少坑
- 简化了大量fa的工程考虑，只保留核心代码
- 纯cuda,不考虑pybind版本可以看[standalone文件夹](https://github.com/66RING/tiny-flash-attention/tree/main/flash_attention_cutlass/standalone_src)
- 太久没填坑可以直接makefile开学

## flash attention速通

TODO: 简单描述一下flash attention的本质: flash attention three easy pieces

- online safe softmax
- 两个gemm的融合
- rescale的数学原理


## 自顶向下cute flash attention

在不考虑使用cutlass的情况下, 纯cuda应该怎么写高性能算子:

1. 多维block tiling: 
    - 把数据从global memory拷贝到shared memory
    - 复用smem中的数据, 减少访问gmem的此时
2. 多维thread tiling
    - 把数据从shared memory拷贝到global memory
    - 复用寄存器中的数据
3. 进一步优化
4. 使用向量指令异步加载
    - LDSM
    - ldmatrix
5. 合并访存
6. bank conflict冲突消解
7. 传算交叠流水线: 一边gmem -> smem拷贝一边做reg的gemm计算

而cutlass cute则把原本需要手写的thread协同工作的代码抽象封装好了, 如需要协同做拷贝时可以`make_tiled_copy`创建一个拷贝对象, 需要协同计算时可以用`TiledMMA<T>`创建mma(matrix multiply accumulate)对象来做计算。

**只需要看懂mma布局就知道thread间如何协同的**, 后面[基础设施](#基础设施)章节会介绍


### Terms 名词解释

- 命名习惯: `tQgQ`
    * 看到cute的变量名可能一头雾水, 所以有必要解释一下
    * 如`auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0))`, `t`(to)表示是给什么用的, 这里只是抽象了一层还是Q本身所以直接用tQ。`g`表示该变量的位置在global memory中
    * 如`tSrQ`, `tSrK`表示是给attention **S**core计算使用的, 寄存器(reg)中的Q, K
    * 如`tOrVt`表示是给最终output用的, 寄存器中的转置过了的V
- MNK矩阵乘法表述法
    * 两个矩阵相乘需要至少一个维度相同, K就表示这个相同的维度是多少
    * `A[M, K] @ B[N, K]`
- MMA(matrix multiply accumulate)
    * 简单的说就是用于表示thread tiling的规模, 即一个thread block中用多少个thread怎么计算, cute会抽象成一个个mma对象
- MMA描述法: 描述底层执行`D = AB + C`要使用的指令, 用户可以根据需要指定
    * 描述方法: DABC + MNK
    - DABC: 描述了寄存器类型, 如`SM75_16x8x8_F32F16F16F32_TN`中`F32F16F16F32`就是DABC描述。表示DABC寄存器分别是`F32`, `F16`, `F16`, `F32`
    - MNK: 描述了矩阵乘法的规模, 如`SM75_16x8x8_F32F16F16F32_TN`中`16x8x8`就表示`D[M, N] = A[M, K] * B[N, K] + C[M, N]`
- Tiled_MMA: 描述多个MMA_Atom如何协作来完成一个大任务
    * AtomLayoutMNK: Tile内在MNK方向上重复几次Atom, **通过多线程重复**
    * ValueLayoutMNK: Atom内在MNK方向上重复几次计算, **单线程内重复计算**
- BlockM
    * Q的分块计算的粒度
- BlockN
    * KV的分块计算的粒度


### 基础设施

- 查看MMA布局

使用这个[mma布局打印脚本](https://gist.github.com/66RING/2e188b73fdf703e9f9dfc7371814dd15)可以打印, 使用方法如下: 修改不同mma指令`SM80_16x8x16_F32F16F16F32_TN`来测试。

```cpp
  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }
```

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/mma.webp)

图片含义：T0, T1...表示thread，T0内V0, V1表示thread T0所负责的数据

- 打印tensor

直接使用cute提供的`print_tensor`, `print_layout`可以在命令行打印出tensor数据, 方便调试。e.g.

```cpp
// Convert a C pointer into cutlass Tensor
// with info like shape (M, K) and stride (K, 1)
const int M = 4;
const int K = 8;

Tensor A = make_tensor(c_host_ptr, make_shape(M, K), make_stride(K, 1));
cute::print_tensor(A);
cute::print_layout(A.layout());

/*
ptr[32b](0x7ffe79dcbbe0) o (4,8):(8,1):
    0    1    2    3    4    5    6    7
    8    9   10   11   12   13   14   15
   16   17   18   19   20   21   22   23
   24   25   26   27   28   29   30   31
(4,8):(8,1)
       0    1    2    3    4    5    6    7 
    +----+----+----+----+----+----+----+----+
 0  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
    +----+----+----+----+----+----+----+----+
 1  |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    +----+----+----+----+----+----+----+----+
 2  | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
    +----+----+----+----+----+----+----+----+
 3  | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
    +----+----+----+----+----+----+----+----+
*/
```

使用`local_tile`打印一个tile(一个tensor切片)

```cpp
cute::print_tensor(A);
auto A00 = local_tile(A, make_tile(2, 2), make_coord(0, 0));
auto A01 = local_tile(A, make_tile(2, 2), make_coord(0, 1));
auto A10 = local_tile(A, make_tile(2, 2), make_coord(1, 0));
cute::print_tensor(A00);
cute::print_tensor(A01);
cute::print_tensor(A10);

/*

cute::print_tensor(A);
ptr[32b](0x7ffc3fe94680) o (4,8):(1,4):
    0    4    8   12   16   20   24   28
    1    5    9   13   17   21   25   29
    2    6   10   14   18   22   26   30
    3    7   11   15   19   23   27   31

cute::print_tensor(A00);
ptr[32b](0x7ffc3fe94680) o (2,2):(1,4):
    0    4
    1    5

cute::print_tensor(A01);
ptr[32b](0x7ffc3fe946a0) o (2,2):(1,4):
    8   12
    9   13

cute::print_tensor(A10);
ptr[32b](0x7ffc3fe94688) o (2,2):(1,4):
    2    6
    3    7

*/

```


### attention计算的线程模型

单线程的attention计算belike: `q[seqlen, headdim] @ k[seqlen, headdim].T @ v[seqlen, headdim]`

而多线性的attention计算只需要从q的维度切分(想象成自回归场景下, 一次计算一个token的attention, 这里是并行的计算多个"单"query的attention)，每个thread负责BlockM个token的single head attention计算。即

如果输入的形状为`[bs, head, seqlen, headdim]`则总线程数为`bs x head x seqlen/BlockM`, 每个thread计算`[BlockM, headdim]`的query attention计算。在bs x head维度和seqlen维度都并行。

对应到每个独立的thread block上也是同理, 开辟`bs x head x seqlen/BlockM`个独立的线程块进行多个token的并行计算。

```cpp
dim3 grid(ceil_div(params.seqlen, BlockM), params.bs * params.head, 1);
```

TODO: 示意图

### 二维block tiling

flash attention 2的计算流程如下图所示, Q按inner loop顺序分别和K, V分开进行计算得到partial sum, 最后将partial sum累加得到和Q形状一样的输出。伪码描述为(先不用考虑online softmax和rescale的原理)

```python
flash_attention_2():
    # outter loop
    parallel do q[NUM_BLOCK_M]:
        # inner loop
        for i in range(NUM_BLOCK_N):
            qk = q @ k[i].T
            score = online_softmax(qk)
            out += score @ v[i]
        rescale(out)
```

你可能发现outter loop和inner loop和流传甚广的经典的flash attention那张三角形的图不一样。这是因为那张图的flash attention 1时期的实现。

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/flash_attention2.png)

利用cute的api可以快速制造q, k, v分块:

1. 用`make_tensor()`把裸指针封装成tensor方便后续操作
2. 使用`local_tile(tensor, tile, coord)`从tensor中取出一组/一个分块
3. 创建`Copy_Atom`拷贝对象实现global memory到shared memory的数据拷贝, 简单易用的多维block tiling

首先使用`make_tensor`API可以把传入的裸指针转换成更方便使用的Tensor。这里把完整`seqlen x dim`的QKV对象创建了出来，方便后面使用cute的API做`q_slice[i++]`之类的操作。不用担心`make_tensor`会产生额外的开销, 因为它不会。

```cpp
  // dim3 grid(ceil_div(params.seqlen, BlockM), params.bs * params.head, 1);

  const int m_block = blockIdx.x;
  const int bs_head_offset = blockIdx.y * params.seqlen * params.dim;

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
```

根据block id加载thread block对应的qkv分块。`local_tile(tensor, tile, coord)`可以把tensor抽象成由多个tile组成的数组(可以多多维), 然后使用coord去索引取出需要的部分。这里取出了当前thread block负责的Q分块，并取出第一个kv分块做后续"传算交叠流水线"的prefill.

因为这里Q的shape是`seqlen, kHeadDim`, 所以拆分成多个`[kBlockM, kHeadDim]`的块后可索引的coord为`[seqlen/kBlockM, kHeadDim/kHeadDim]`。取出`[m_block, _]`, 相当于python中的`[m_block, :]`这样的索引方式, 其中`m_block`索引维度的会被squeeze, 而`_`索引的维度会保留。所以最终的shape为`(kBlockM, kHeadDim, num_tile_n=1)`

```cpp
  // 加载Q, K, V分块
  // (kBlockM, kHeadDim, num_tile_n)
  Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // (kBlockN, kHeadDim, num_tile_n)
  // NOTE: loading流水线, 初次加载所需K, V
  Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
```

**将数据从global memory拷贝到shared memory来做多维的block tiling**: 定义从global memory到share memory拷贝的对象, 这样可以减少用户直接使用gpu指令。具体拷贝对象怎么构造后续再说, 简单的说就是使用一个config来配置用什么方法拷贝(异步的, 向量的)。

```cpp
  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

  // NOTE: 定义gmem -> smem拷贝的src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
```

其中, `gmem_thr_copy_QKV.partition_S()`创建拷贝的源地址对象, `gmem_thr_copy_QKV.partition_D()`创建拷贝的目标地址对象。因为gQ我们在创建分块时第二个维度用满了, 所以`make_coord(m_block, _)`提取出来也只有一个元素, 直接用`0`索引掉。

```
// tQgQ: tQ: 用于(t)表示/计算Q. gQ: 是global memory上的数据
// tQsQ: tQ: 用于(t)表示/计算Q. sQ: 是shared memory上的数据
```

然后使用API即可实现一个多维数据的拷贝。

```cpp
  // NOTE: gmem_tiled_copy_QKV为cute抽象出来的拷贝对象Copy_Atom, 表示用一组thread来做拷贝
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // 开始执行异步拷贝
  cute::cp_async_fence();
```

具体`gmem_thr_copy_QKV`拷贝对象的构造方法后面再说, 只需要传入一个异步拷贝的参数和规模布局即可用上向量指令做异步拷贝。

> 这是不是比手写gpu指令的block tiling各种拷贝简单多了:


### 二维thread tiling

本章节开始进入inner loop部分

```python
flash_attention_2():
    # outter loop
    parallel do q[NUM_BLOCK_M]:
        # inner loop
        for i in range(NUM_BLOCK_N):
            qk = q @ k[i].T
            score = online_softmax(qk)
            out += score @ v[i]
        rescale(out)
```

整体流程如下

1. pipeline prefill: load(q), load(k[0])
2. pipeline start
3. async_load(next(v)) && compute q @ k.T
4. softmax(qk)
5. async_load(next(k)) && compute qk @ v
6. pipeline finish
7. rescale

其中做gemm计算时都会从smem拷贝多维的数据到寄存器中做一个thread tiling。thread tiling可以复用已经拷贝到寄存器的数据，减少smem到reg拷贝的次数。如下图所示, 当gemm计算第0行时, BX0和A0X计算完成后, BX1可以直接利用已经在寄存器的A0X而不用再次做smem到reg的加载。

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/thread_tiling.png)

从gemm的角度出发看多维thread tiling的实现。使用`cute::copy`把smem中的数据`tCsA`拷贝到寄存器中`tCrA`后直接使用`cute::gemm`做多维thread tiling的gemm计算。具体thread tiling的布局通过可以通过[打印mma](#基础设施)查看。

```cpp
template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // NOTE: 构造smem -> reg拷贝的目的地址寄存器对象
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);

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
```

for循环前先做一次`cute::copy`是为了构造传算交叠(communication compute overlap)的流水线。即做smem->reg拷贝的同时做gemm。

回到cutlass flash attention的代码。使用cute提供的API构造gemm需要的寄存器对象。TODO: 具体`SmemCopyAtom`拷贝对象的构造方法后面再说, 只需要传入一个异步拷贝的参数和规模布局即可。

使用`partition_fragment_A`, `partition_fragment_B`, `partition_fragment_C`创建寄存器对象, 准备做thread tiling: 把数据从smem拷贝到reg, 并利用reg中的数据做矩阵乘法。

```cpp
  // NOTE: 定义smem -> reg拷贝的dst
  // partition_fragment与partition类似, 只是返回的是寄存器表示
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
  Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)
  // 创建输出的累加器accumulator output
  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // NOTE: 准备拷贝Q, K, V到smem的copy对象

  // 创建smem -> reg的拷贝对象
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  // 根据thread id找到当前thread负责的部分
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // 用partition_S创建smem -> reg的源地址对象
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  ...
```

inner loop部分代码如下。其中, 创建`auto rAccScore = partition_fragment_C()`来**融合两个gemm**: `score = q@k.T`的gemm和`out = score @ v`的gemm。

需要注意**融合两个gemm的坑点**, 因为要融合两个gemm, gemm-I的输出`score = q@k.T`要作为第二个gemm-II的输入`out = score @ v`, 所以**gemm-I的输出C layout需要和gemm-II的输入A layout一致**才能直接使用。通过打印mma指令发现`SM80_16x8x16_F32F16F16F32_TN`就符合这种要求。

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/mma.webp)

[ColfaxResearch的实现](https://github.com/ColfaxResearch/cutlass-kernels/blob/c796d779c9991213252e9f0a07e5516c8d829e3f/src/fmha/fmha_forward.cu#L114)似乎不用考虑这点, 用`rs_op_selector`和`ss_op_selector`两个API就把MMA配置好了。如果有人知道是怎么回事pls let me know.


```cpp
/*
flash_attention_2():
    # outter loop
    parallel do q[NUM_BLOCK_M]:
        # inner loop
        for i in range(NUM_BLOCK_N):
            qk = q @ k[i].T
            score = online_softmax(qk)
            out += score @ v[i]
        rescale(out)
*/
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
```

伪码和代码的对应情况如下:

```python
# inner loop
for nbi in range(NUM_BLOCK_N):
    # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
    qk = q @ k[nbi].T # flash::gemm_smem()
    score = online_softmax(qk) # softmax_rescale_o()
    # v[nbi]: gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    out += score @ v[nbi] # gemm_A_in_regs()
```

### 传算交叠流水线

- 异步拷贝

创建gmem到smem的拷贝对象时使用`SM80_CP_ASYNC_CACHEGLOBAL`指令来创建异步拷贝的Copy atom对象。

```cpp
using Gmem_copy_struct = std::conditional_t<
    Has_cp_async,
    SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    DefaultCopy
>;
using GmemTiledCopyQKV = decltype(
    make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                    GmemLayoutAtom{},
                    Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
```


- 流水线

伪码描述如下, 计算q@k时可以加载v, 计算qk@v时加载下一次迭代需要的k。目前只是用double buffering的方式预取1个kv. 如果每次预取多个kv还需要考虑smem大小对性能的影响。

```python
# inner loop
async_load(k[0]) # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
for nbi in range(NUM_BLOCK_N):
    # 加载v的同时计算q@k
    async_load(v[nbi]) # v[nbi]: gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    qk = q @ k[nbi].T # flash::gemm_smem()
    score = online_softmax(qk) # softmax_rescale_o()

    # 计算qk @ v的同时加载下一次迭代需要的k
    async_load(k[nbi]) # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
    out += score @ v[nbi] # gemm_A_in_regs()
```

在cutlass cute中使用也很简单, 构造好异步拷贝对象后发起异步拷贝即可。

```cpp
    // gemm的同时异步加载V
    gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    // 异步加载V到smem
    flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // 发起异步拷贝
    cute::cp_async_fence();

    // NOTE: 拷贝的同时执行gemm
    // O = Q@K.T
    // NOTE: 加载smem中的数据到reg再做gemm, **加载期间执行retile**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );
```


### 其他细节

- causal模式的提前返回
    * block间早退
    * **block内mask**: thread在mma中的定位
- 结果拷贝回global memory返回
    * 同样利用smem, 先从reg拷贝到smem再从smem拷贝到gmem
    * 这样可以用更大的位宽
- online safe softmax
- pybind和模板展开
    * 官方实现用了很多模板，本质就是1. 枚举所有可能的分块策略 2. 每个config写一个文件加速编译 3. 每个模板写个文件微调最佳config
    * python中接入cpp代码可以看这个[仓库](https://github.com/66RING/pytorch-cuda-binding-tutorial)

后面再展开补充，感兴趣的朋友可以先看源码注释。

### 其他优化

- bank conflict重复避免
    * swizzle
    * cutlass cute封装好了用swizzle解决bank conflict, 在创建拷贝对象时使用即可
- 转置优化
    * 拷贝时直接拷贝到转换后的目标地址, 从而不必开辟新的空间
    * 创建拷贝对象时, 配置布局时把dst的布局转置掉即可
- [高性能的reduce实现](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
    * 优化线程束分化问题(warp divergent)

TODO: 细节展开

### 稍微一点自底向上

> 深入的自底向上可以看[reed哥的系列教程](https://www.zhihu.com/people/reed-84-49)

TODO: 挑选几个重要的


### 主要坑点

- 两个gemm的融合的layout问题: gemm-I, gemm-II
    * 输入输出的布局比较讲究: gemm-I的输出C layout要和gemm-II的输入A layout一致












