
# Tái hiện Flash Attention bằng Cutlass Cute

Có thể học flash attention theo hướng từ trên xuống (top-down) (dù bản thân tôi học Cutlass theo hướng từ dưới lên, nhưng tôi thấy muốn nắm bắt nhanh thì nên học từ trên xuống). Nhờ có Cutlass Cute, người dùng có thể dễ dàng cài đặt một số chức năng, tức là một số mô thức (paradigm) của lập trình CUDA:

- Mô thức lập trình CUDA: global mem -> share mem -> reg -> compute
    * Block Tiling:
        + hay còn gọi là tái sử dụng shared memory (smem), sao chép từ global memory (gmem) sang shared memory (smem)
    * Thread Tiling:
        + hay còn gọi là tái sử dụng thanh ghi (reg) và shared memory (smem), sao chép từ shared memory (smem) vào thanh ghi (reg)
    * Gộp truy cập bộ nhớ, truy cập bộ nhớ dạng vector:
        + hay còn gọi là lệnh vector, LDSM (Load Shared Memory), lệnh ldmatrix
    * Warp Divergence (phân kỳ warp):
        + hay còn gọi là cân bằng tải giữa các warp, tương tự vấn đề "bong bóng" (bubble) trong pipeline
    * Giải quyết Bank Conflict: Swizzle
        + hay còn gọi là tận dụng tính chất đa kênh của bộ nhớ
    * Double Buffering (bộ đệm kép)
        + hay còn gọi là pipeline giữa nạp dữ liệu và tính toán
    * ...

Với những ai cần học theo hướng từ dưới lên, tôi khuyên đọc [tutorial chính thức của cute](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)

## Ghi nhận

- Dựa trên flash attention chính thức, nhưng được viết lại từ đầu, loại bỏ nhiều "mẹo" ẩn.
- Tôi đã đơn giản hóa rất nhiều cân nhắc về mặt kỹ thuật (engineering) của mã gốc, chỉ giữ lại phần cốt lõi.
- Thuần CUDA, không xét phiên bản Pybind: có thể xem trong [thư mục standalone](https://github.com/66RING/tiny-flash-attention/tree/main/flash_attention_cutlass/standalone_src).
- Đã lâu tôi chưa quay lại hoàn thiện, nên bạn có thể dùng thẳng Makefile để bắt đầu học.


## Ba mảnh ghép đơn giản của flash attention

TODO: Mô tả ngắn gọn bản chất của flash attention: flash attention three easy pieces

- Online safe softmax
- Hợp nhất (fusion) hai phép GEMM
- Nguyên lý toán học của rescaling


## Cute Flash Attention theo hướng từ trên xuống

Nếu viết các toán tử (operator) hiệu năng cao bằng CUDA thuần mà không dùng Cutlass, ta sẽ tiếp cận như sau:

1. Block tiling đa chiều:
    - Sao chép dữ liệu từ global memory sang shared memory.
    - Tái sử dụng dữ liệu trong shared memory để giảm số lần truy cập global memory.
2. Thread tiling đa chiều:
    - Sao chép dữ liệu từ shared memory vào thanh ghi.
    - Tái sử dụng dữ liệu trong thanh ghi.
3. Tối ưu thêm.
4. Dùng lệnh vector để nạp bất đồng bộ:
    - LDSM
    - ldmatrix
5. Gộp truy cập bộ nhớ.
6. Giải quyết bank conflict.
7. Pipeline chồng lấp truyền tải và tính toán: vừa sao chép dữ liệu từ global memory sang shared memory, vừa thực hiện tính GEMM trên thanh ghi.

Tuy nhiên, Cutlass Cute đã trừu tượng hóa và đóng gói sẵn phần mã vốn phải viết tay để các thread phối hợp với nhau. Ví dụ, khi cần phối hợp để sao chép, bạn có thể dùng `make_tiled_copy` để tạo một đối tượng sao chép; khi cần phối hợp để tính toán, bạn có thể dùng `TiledMMA<T>` để tạo đối tượng MMA (matrix multiply accumulate) để tính.

**Chỉ cần hiểu bố cục (layout) MMA là đủ để hiểu các thread phối hợp với nhau thế nào.** Phần [Công cụ](#công-cụ) bên dưới sẽ giới thiệu điều này.


### Giải thích thuật ngữ

- **Quy ước đặt tên**: tQgQ
    * Có thể bạn sẽ bối rối với tên biến của cute, nên cần giải thích một chút.
    * Ví dụ, `auto tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0))`, `t` (to) cho biết biến dùng cho mục đích gì; ở đây nó chỉ là một lớp trừu tượng bọc quanh chính Q, nên được đặt thẳng là `tQ`. `g` cho biết biến nằm trong global memory.
    * Ví dụ, `tSrQ`, `tSrK` cho biết chúng dùng để tính **S**core (điểm attention), là Q, K nằm trong thanh ghi (reg).
    * Ví dụ, `tOrVt` cho biết nó dùng cho output cuối cùng, là V đã chuyển vị nằm trong thanh ghi.
- Ký hiệu MNK cho phép nhân ma trận
    * Hai ma trận nhân với nhau cần có ít nhất một chiều giống nhau, K biểu thị chiều chung này.
    * `A[M, K] @ B[N, K]`
- MMA (Matrix Multiply Accumulate)
    * Nói đơn giản, nó dùng để biểu thị quy mô của thread tiling, tức là trong một thread block dùng bao nhiêu thread và chúng tính toán như thế nào. Cute trừu tượng hóa điều này thành từng đối tượng MMA.
- Mô tả MMA: mô tả lệnh được dùng để thực thi `D = AB + C` ở tầng thấp, người dùng có thể chỉ định theo nhu cầu.
    * Định dạng mô tả: DABC + MNK
    * DABC: mô tả kiểu thanh ghi, ví dụ trong `SM75_16x8x8_F32F16F16F32_TN`, `F32F16F16F32` là phần mô tả DABC. Nó cho biết các thanh ghi D, A, B, C lần lượt là F32, F16, F16, F32.
    * MNK: mô tả quy mô phép nhân ma trận, ví dụ trong `SM75_16x8x8_F32F16F16F32_TN`, `16x8x8` biểu thị `D[M, N] = A[M, K] * B[N, K] + C[M, N]`.
- Tiled_MMA: mô tả cách nhiều MMA_Atom phối hợp để hoàn thành một tác vụ lớn.
    * AtomLayoutMNK: số lần lặp lại Atom theo các hướng MNK bên trong một tile, **lặp lại bằng nhiều thread**.
    * ValueLayoutMNK: số lần lặp lại phép tính theo các hướng MNK bên trong một Atom, **lặp lại trong cùng một thread**.
- BlockM
    * Độ mịn (granularity) khi chia khối để tính Q.
- BlockN
    * Độ mịn khi chia khối để tính KV.


### Công cụ

- In bố cục MMA

Bạn có thể dùng [script in bố cục MMA](https://gist.github.com/66RING/2e188b73fdf703e9f9dfc7371814dd15) này để in bố cục. Cách dùng như sau: thay đổi các lệnh MMA khác nhau, ví dụ `SM80_16x8x16_F32F16F16F32_TN`, để thử nghiệm.

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

Ý nghĩa của hình: T0, T1... biểu thị các thread, V0, V1 bên trong T0 biểu thị dữ liệu mà thread T0 phụ trách.

- In tensor

Bạn có thể dùng trực tiếp `print_tensor` và `print_layout` do Cute cung cấp để in dữ liệu tensor ra dòng lệnh nhằm gỡ lỗi. Ví dụ:

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

Dùng `local_tile` để in một tile (một lát cắt của tensor).

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


### Mô hình thread của Flash Attention

Phép tính attention đơn luồng trông như sau: `q[seqlen, headdim] @ k[seqlen, headdim].T @ v[seqlen, headdim]`.

Còn phép tính attention đa luồng chỉ cần chia theo chiều của q (hãy hình dung trong kịch bản tự hồi quy (autoregressive), mỗi lần tính attention cho một token; ở đây là tính song song attention cho nhiều query "đơn lẻ"), mỗi thread chịu trách nhiệm tính single-head attention cho BlockM token. Tức là:

Nếu shape đầu vào là `[bs, head, seqlen, headdim]`, tổng số thread là `bs x head x seqlen/BlockM`, và mỗi thread tính attention cho query có kích thước `[BlockM, headdim]`. Việc này song song trên cả chiều bs x head lẫn chiều seqlen.

Tương tự với từng thread block độc lập: cấp phát `bs x head x seqlen/BlockM` thread block độc lập để tính song song cho nhiều token.

```cpp
dim3 grid(ceil_div(params.seqlen, BlockM), params.bs * params.head, 1);
```

TODO: hình minh họa


### Block Tiling hai chiều

Quy trình tính toán của Flash Attention 2 được minh họa trong sơ đồ dưới đây. Q được tính lần lượt với K và V theo thứ tự của vòng lặp trong để thu được các tổng riêng phần (partial sum). Cuối cùng, các tổng riêng phần được cộng dồn để ra output có cùng shape với Q. Mã giả (chưa xét nguyên lý của online softmax và rescale) như sau.

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

Bạn có thể nhận thấy vòng lặp ngoài và vòng lặp trong khác với sơ đồ tam giác kinh điển của Flash Attention đang được lan truyền rộng rãi. Đó là vì sơ đồ kia là của cách cài đặt Flash Attention 1.

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/flash_attention2.png)

Với API của Cute, ta có thể nhanh chóng tạo các khối cho q, k, v:

- Dùng `make_tensor()` để bọc con trỏ thô thành tensor, thuận tiện cho các thao tác sau này.
- Dùng `local_tile(tensor, tile, coord)` để lấy ra một nhóm/một khối từ tensor.
- Tạo đối tượng sao chép `Copy_Atom` để sao chép dữ liệu từ global memory sang shared memory, cung cấp cách làm block tiling đa chiều đơn giản, dễ dùng.

Trước tiên, API `make_tensor` được dùng để chuyển con trỏ thô được truyền vào thành một Tensor tiện dụng hơn. Ở đây ta tạo một đối tượng QKV hoàn chỉnh kích thước `seqlen x dim`, giúp dùng API của Cute cho các thao tác như `q_slice[i++]` thuận tiện hơn. Đừng lo `make_tensor` gây thêm chi phí, vì nó không tạo ra chi phí nào cả.

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

Nạp khối QKV tương ứng với thread block dựa theo block ID. `local_tile(tensor, tile, coord)` trừu tượng hóa tensor thành một mảng gồm nhiều tile (trên nhiều chiều), sau đó dùng coord để đánh chỉ số và lấy ra phần cần thiết. Ở đây ta lấy ra khối Q mà thread block hiện tại phụ trách, đồng thời lấy ra khối KV đầu tiên để "nạp trước" (prefill) cho pipeline "chồng lấp truyền tải và tính toán" phía sau.

Vì shape của Q ở đây là `seqlen`, `kHeadDim`, khi chia nó thành nhiều khối `[kBlockM, kHeadDim]`, ta có thể đánh chỉ số bằng `coord` dưới dạng `[seqlen/kBlockM, kHeadDim/kHeadDim]`. Lấy `[m_block, _]` tương đương với cách đánh chỉ số `[m_block, :]` trong Python. Khi đó, chiều được đánh chỉ số bởi `m_block` sẽ bị "ép" (squeeze) đi, còn chiều được đánh chỉ số bởi `_` sẽ được giữ lại. Vì vậy shape cuối cùng là `(kBlockM, kHeadDim, num_tile_n=1)`.

```cpp
  // load q, k, v block
  // (kBlockM, kHeadDim, num_tile_n)
  Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  // (kBlockN, kHeadDim, num_tile_n)
  // NOTE: compute commu overlap pipeline load first q, k
  Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
```

**Sao chép dữ liệu từ global memory sang shared memory để thực hiện block tiling đa chiều**: định nghĩa một đối tượng sao chép từ global memory sang shared memory, giúp người dùng ít phải dùng trực tiếp các lệnh GPU. Cách dựng đối tượng sao chép sẽ được bàn ở phần sau; nói đơn giản, nó được cấu hình qua một config để chỉ định phương thức sao chép (bất đồng bộ, vector hóa).

```cpp
  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
  Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtNoSwizzle{});

  // NOTE: define object of gmem -> smem copy, src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
```

Trong quá trình này, `gmem_thr_copy_QKV.partition_S()` tạo đối tượng địa chỉ nguồn (source) cho việc sao chép, còn `gmem_thr_copy_QKV.partition_D()` tạo đối tượng địa chỉ đích (destination). Vì khi tạo khối cho gQ ta đã dùng trọn chiều thứ hai, nên việc lấy ra bằng `make_coord(m_block, _)` chỉ cho ra một phần tử, do đó ta dùng thẳng `0` để đánh chỉ số.

```
// tQgQ: tQ: dùng để (t) tính Q. gQ: dữ liệu trong global memory
// tQsQ: tQ: dùng để (t) tính Q. sQ: dữ liệu trong shared memory
```

Sau đó, dùng API là có thể sao chép dữ liệu đa chiều.

```cpp
  // NOTE: gmem_tiled_copy_QKV là đối tượng sao chép Copy_Atom do cute trừu tượng hóa, biểu thị việc dùng một nhóm thread để sao chép
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // start async copy
  cute::cp_async_fence();
```

Cách dựng đối tượng sao chép `gmem_thr_copy_QKV` sẽ được bàn ở phần sau. Bạn chỉ cần truyền vào một tham số sao chép bất đồng bộ và layout quy mô là có thể dùng lệnh vector để sao chép bất đồng bộ.

> đơn giản hơn nhiều so với việc tự viết tay lệnh GPU cho block tiling và đủ kiểu sao chép


### Thread Tiling hai chiều

Bây giờ ta đi vào phần vòng lặp trong của mục này.

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

Quy trình tổng thể như sau:

1. pipeline prefill: load(q), load(k[0])
2. pipeline start
3. async_load(next(v)) && compute q @ k.T
4. softmax(qk)
5. async_load(next(k)) && compute qk @ v
6. pipeline finish
7. rescale

Trong lúc tính gemm, dữ liệu đa chiều được sao chép từ shared memory vào thanh ghi để thực hiện thread tiling. Thread tiling cho phép tái sử dụng dữ liệu đã được sao chép vào thanh ghi, giảm số lần sao chép từ shared memory vào thanh ghi. Như trong sơ đồ dưới đây, khi tính hàng đầu tiên của gemm, sau khi tính xong BX0 và A0X, BX1 có thể dùng ngay A0X đã nằm trong thanh ghi mà không phải nạp lại từ shared memory.

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/thread_tiling.png)

Nhìn cách cài đặt thread tiling đa chiều từ góc độ gemm: ta dùng `cute::copy` để sao chép dữ liệu tCsA từ shared memory vào thanh ghi tCrA, sau đó dùng thẳng `cute::gemm` để tính gemm với thread tiling đa chiều. Bố cục cụ thể của thread tiling có thể xem bằng cách in [mma](#công-cụ).

```cpp
template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    // NOTE: construct dst object of smem -> reg copy
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

Trước vòng lặp for, ta thực hiện một lần `cute::copy` để dựng pipeline chồng lấp truyền tải và tính toán. Nghĩa là vừa sao chép smem->reg vừa tính gemm.

Quay lại mã Flash Attention dùng Cutlass, ta dùng API do Cute cung cấp để dựng các đối tượng thanh ghi cần cho gemm. TODO: Cách dựng cụ thể đối tượng sao chép SmemCopyAtom sẽ được bàn sau, nhưng bạn chỉ cần truyền vào một tham số sao chép bất đồng bộ và layout quy mô.

Dùng `partition_fragment_A, partition_fragment_B, partition_fragment_C` để tạo các đối tượng thanh ghi, chuẩn bị cho thread tiling: sao chép dữ liệu từ shared memory vào thanh ghi, và nhân ma trận bằng dữ liệu trong thanh ghi.

```cpp
    // NOTE: construct mma object in register
  // partition_fragment can create a object in register
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ); // (MMA,MMA_M,MMA_K)
  Tensor tSrK  = thr_mma.partition_fragment_B(sK); // (MMA,MMA_N,MMA_K)
  Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle); // (MMA, MMA_K,MMA_N)
  // construct output accumulator
  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // construct copy object of smem -> reg
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  // select thread work by thread id 
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  // use partition_S to construct src object of Copy_Atom
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
  ...
```

Mã của vòng lặp trong như sau. Ở đây, `auto rAccScore = partition_fragment_C()` được tạo ra để **hợp nhất hai gemm**: gemm `score = q@k.T` và gemm `out = score @ v`.


Cần lưu ý **cái bẫy khi hợp nhất hai gemm**. Vì phải hợp nhất hai gemm, output của gemm-I, `score = q@k.T`, cần được dùng làm input của gemm-II, `out = score @ v`, **nên layout C của output gemm-I phải khớp với layout A của input gemm-II thì mới dùng trực tiếp được**. Qua việc in các lệnh MMA, ta thấy `SM80_16x8x16_F32F16F16F32_TN` thỏa mãn yêu cầu này.

![](https://raw.githubusercontent.com/66RING/66RING/master/.github/images/Notes/universe/ml/cutlass_flash_attention_top_down/mma.webp)

[Cách cài đặt của ColfaxResearch](https://github.com/ColfaxResearch/cutlass-kernels/blob/c796d779c9991213252e9f0a07e5516c8d829e3f/src/fmha/fmha_forward.cu#L114) dường như không cần xét đến điểm này, họ dùng hai API `rs_op_selector` và `ss_op_selector` để cấu hình MMA. Nếu ai biết cơ chế hoạt động của nó, xin hãy cho tôi biết.

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

    // Chờ việc sao chép Q, K từ gmem -> smem hoàn tất, tức Q, K đã sẵn sàng
    // wait<0> nghĩa là chờ đến khi còn 0 thao tác chưa hoàn tất
    flash::cp_async_wait<0>();
    __syncthreads();

    // Nạp V bất đồng bộ trong lúc tính gemm
    gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    // Nạp V bất đồng bộ vào smem
    flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // Khởi động sao chép bất đồng bộ
    cute::cp_async_fence();

    // O = Q@K.T
    // NOTE: Nạp dữ liệu từ smem vào reg rồi mới tính gemm, **thực hiện retile trong lúc nạp**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );

    Tensor scores = make_tensor(rAccScore.data(), flash::convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. mask within N BLOCKs
    if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      flash::mask_within_nblock<kBlockM, kBlockN, kNWarps>(scores, m_block, nbi);
    }

    // NOTE: Chờ V nạp xong, chuẩn bị trạng thái ban đầu cho lần nạp K tiếp theo
    flash::cp_async_wait<0>();
    __syncthreads();

    // advance K
    if (nbi != n_block_max - 1) {
      gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      flash::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      cute::cp_async_fence();
    }

    // Tính softmax
    // NOTE: rAccOut lưu toàn bộ tử số sau softmax
    nbi == 0 ? flash::softmax_rescale_o</*Is_first=*/true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
      flash::softmax_rescale_o</*Is_first=*/false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

    // Thực sự tính QK @ V
    // (score AKA rAccScore): QK[M, N] @ V[N, dim]
    // NOTE: DABC: F32F16F16F32, convert D type(F32) to A type(F16)
    // TODO: convert_type hiện đang bị viết cứng (hard-code)
    Tensor rP = flash::convert_type_f32_to_f16(rAccScore);
    // NOTE: Convert from layout C to layout A
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<TiledMMA>(scores.layout()));

    flash::gemm_A_in_regs(rAccOut, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }
```

Tương ứng giữa mã giả và mã thật như sau:

```python
# inner loop
for nbi in range(NUM_BLOCK_N):
    # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
    qk = q @ k[nbi].T # flash::gemm_smem()
    score = online_softmax(qk) # softmax_rescale_o()
    # v[nbi]: gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    out += score @ v[nbi] # gemm_A_in_regs()
```

### Pipeline chồng lấp truyền tải và tính toán

- Sao chép bất đồng bộ

Khi tạo đối tượng sao chép từ global memory sang shared memory, dùng lệnh `SM80_CP_ASYNC_CACHEGLOBAL` để tạo đối tượng Copy_Atom bất đồng bộ.

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


- Pipeline

Mã giả như sau: khi tính q@k thì nạp v, và khi tính qk@v thì nạp k cần cho vòng lặp kế tiếp. Hiện tại chỉ dùng double buffering để nạp trước (prefetch) 1 bộ kv. Nếu mỗi lần nạp trước nhiều bộ kv, cần cân nhắc ảnh hưởng của kích thước shared memory đến hiệu năng.

```python
# inner loop
async_load(k[0]) # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
for nbi in range(NUM_BLOCK_N):
    # Vừa nạp v vừa tính q@k
    async_load(v[nbi]) # v[nbi]: gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    qk = q @ k[nbi].T # flash::gemm_smem()
    score = online_softmax(qk) # softmax_rescale_o()

    # Vừa tính qk @ v vừa nạp k cần cho vòng lặp kế tiếp
    async_load(k[nbi]) # k[nbi]: gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
    out += score @ v[nbi] # gemm_A_in_regs()
```

Dùng cách này trong Cutlass Cute cũng rất đơn giản. Sau khi dựng xong đối tượng sao chép bất đồng bộ, chỉ cần khởi động việc sao chép bất đồng bộ.

```cpp
    // Nạp V bất đồng bộ trong lúc tính gemm
    gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi, _));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    // Nạp V bất đồng bộ vào smem
    flash::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    // Khởi động sao chép bất đồng bộ
    cute::cp_async_fence();

    // NOTE: Tính gemm trong lúc sao chép
    // O = Q@K.T
    // NOTE: Nạp dữ liệu từ smem vào reg rồi mới tính gemm, **thực hiện retile trong lúc nạp**
    flash::gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K
    );
```


### Các chi tiết khác

- Trả về sớm trong chế độ causal
    * Thoát sớm giữa các block
    * **Mask bên trong block**: xác định vị trí của thread trong MMA (Matrix Multiply and Accumulate)
- Sao chép kết quả trở lại global memory
    * Tận dụng shared memory (smem): sao chép từ thanh ghi sang smem trước, rồi từ smem sang global memory (gmem)
    * Cách này cho phép dùng độ rộng bit lớn hơn
- Online Safe Softmax
- Pybind và mở rộng template
    * Bản cài đặt chính thức dùng rất nhiều template, về bản chất là:
        1. Liệt kê mọi chiến lược chia khối có thể có
        2. Viết mỗi cấu hình thành một file riêng để tăng tốc biên dịch
        3. Viết mỗi template thành một file riêng để tinh chỉnh ra cấu hình tốt nhất
- Để tích hợp mã C++ vào Python, bạn có thể tham khảo [repository](https://github.com/66RING/pytorch-cuda-binding-tutorial) này

Các chi tiết khác sẽ được bổ sung sau. Bạn đọc quan tâm có thể xem trước phần chú thích trong mã nguồn.

### Các tối ưu khác

- Tránh bank conflict
    * Swizzling
    * Cutlass đã đóng gói sẵn swizzle để giải quyết bank conflict. Chỉ cần dùng khi tạo đối tượng sao chép.
- Tối ưu chuyển vị
    * Sao chép thẳng tới địa chỉ đích đã chuyển vị, tránh phải cấp phát vùng nhớ mới
    * Khi tạo đối tượng sao chép, cấu hình layout sao cho layout đích (dst) được chuyển vị
- [Cài đặt reduce hiệu năng cao](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
    * Tối ưu warp divergence (phân kỳ warp)
- 
TODO: Triển khai chi tiết


### Một chút theo hướng từ dưới lên

Để hiểu sâu theo hướng từ dưới lên, hãy tham khảo [tutorial chính thức của cute](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)

TODO: Chọn ra vài điểm quan trọng


### Mẹo quan trọng nhất

- Hợp nhất hai gemm: gemm-I, gemm-II
    - Layout của input và output rất quan trọng: layout C của output gemm-I phải khớp với layout A của input gemm-II
