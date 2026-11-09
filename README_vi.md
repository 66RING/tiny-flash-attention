# Tiny FlashAttention

WIP (đang hoàn thiện)

Một bản cài đặt [flash attention](https://github.com/Dao-AILab/flash-attention) tối giản bằng python, rust, cuda và c, phục vụ mục đích học tập.

- [phiên bản python](#flash-attention-2)
    * [x] [mã python thuần, cài đặt đơn giản](./flash_attention_py/tiny_flash_attn.py)
- [phiên bản triton](#triton-flash-attention-2)
    * [x] [mã triton](./flash_attention_py/tiny_flash_attn_triton.py)
- [phiên bản c]
    * [x] [mã c thuần, cài đặt đơn giản](./flash_attention_c/csrc/attn.cpp)
    * [x] [mã cuda đơn giản, chạy độc lập (standalone)](./flash_attention_cuda/standalone_src)
    * [x] [mã cuda đơn giản, có python binding](./flash_attention_cutlass/csrc/flash_attention.cu)
    * [x] [mã cuda dùng cutlass](./flash_attention_cutlass/csrc/flash_attention.cu)
- [phiên bản rust]

## Thực hành flash attention với cutlass cute

môi trường của tôi: cutlass v3.4, torch 1.14, cuda 12.4

- [tutorial tiếng Việt](./cutlass_cute_tutorial_vi.md)
- [tutorial tiếng Anh](./cutlass_cute_tutorial_en.md)
- [tutorial tiếng Trung](./cutlass_cute_tutorial_zh.md)


## Thuật toán

- attention
- softmax
    * $s(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$
    * Hàm mũ dễ bị tràn số, dẫn đến mất độ chính xác
- safe softmax
    * $s(x_i) = \frac{e^{x_i - max(x)}}{\sum_j{e^{x_j - max(x)}}} = \frac{e^{-max(x)} \times e^{x_i}}{e^{-max(x)} \times \sum_j{e^x_j}}$
    * Trừ đi giá trị lớn nhất ở phần số mũ
- online softmax
    * Vấn đề của softmax ở trên là: max ở tử số và sum ở mẫu số đều cần đọc toàn bộ vector để lấy giá trị max và sum, không thân thiện với bộ nhớ đệm (SRAM)
    * Thuật toán online softmax tính riêng tử số và mẫu số, cuối cùng mới gộp lại
        1. Tính max theo từng khối, đồng thời cập nhật lặp tổng (sum) ở mẫu số, thu được hệ số chuẩn hóa (normalization factor)
            - TODO
        2. scaling
- flash attention 1
    * tiling
    * SRAM
- flash attention 2

## Mẹo

- `softmax_lse`
    * lse là viết tắt của LogSumExp?
    * [lse chủ yếu giải quyết vấn đề tràn trên (overflow) hoặc tràn dưới (underflow) khi tính Softmax hoặc CrossEntropy](https://www.apispace.com/news/post/13827.html)
- `softmax_scale`
    * `q @ k.T / softmax_scale`
    * Sau khi thêm `softmax_scale`, mất mát độ chính xác không còn nghiêm trọng như trước (nhưng vẫn còn)

## Luồng xử lý

- Cách tính online của softmax: scale (cập nhật) + cộng dồn
- Cách tính online của s@v: scale (cập nhật) + cộng dồn
    1. cập nhật(O cũ) + O mới
    2. **Cách cập nhật: cập nhật max, cập nhật mẫu số**
        1. Cập nhật max: nhân cả tử số và mẫu số với $e^{max_old - max_new}$
        2. **Cập nhật mẫu số: nhân với mẫu số cũ trước, rồi chia cho mẫu số mới**

1. Nghĩ rõ việc chia khối để tính diễn ra như thế nào
2. Sau đó mới xét giá trị của từng khối đến từ đâu

Trường hợp không chia khối, giả sử shape của Q, K, V = (N, d)

Kết quả softmax nhân ma trận với V:

```
s = Q @ K.T = (N, d) @ (d, N) = (N, N)
attn = s @ V = (N, N) @ (N, d) = (N, d)
```

Chia khối kiểu naive: phần softmax nhân với phần tương ứng của V,

```
si = Qi @ Kj.T = (N/2, d) @ (d, N/2) = (N/2, N/2)
attni[N/2, :] = si @ Vj = (N/2, N/2) @ (N/2, d) = (N/2, d)
```


Chia khối kiểu online

TODO: hình minh họa

Vì vậy các output phải được cộng lại!


## Ý chính

- phép nhân ma trận
- shape
- thuật toán

## Thuật toán

- attention
- softmax
    * $s(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$
- safe softmax
- online softmax
    * thuật toán 1
    * cài đặt thuật toán 2
- flash attention 1
    * tiling
    * SRAM
- flash attention 2

### Online softmax 3 lượt (3 pass)

1. lượt 1: tính max theo từng khối
2. lượt 2: tính tổng (sum) ở mẫu số theo từng khối
3. lượt 3: thực hiện softmax(xi)

### Online softmax 2 lượt (2 pass)

1. lượt 1: vừa tính max theo từng khối, vừa cập nhật động tổng ở mẫu số
2. lượt 2: thực hiện softmax(xi)

$$d'_i = d'_{i-1}e^{m_{i-1} - m_{i}} + e^{x_i - m_{i}}$$

Số hạng $d'_{i-1}e^{m_{i-1} - m_{i}}$ giúp thay thế giá trị max đã lỗi thời

### Online attention 2 lượt (2 pass)

Phép nhân ma trận có tính kết hợp

### Online attention 1 lượt (1 pass)

### Tính OV theo khối

Với các O ở cùng vị trí

<img src="" alt="">

## Tổng kết

- Xử lý online sẽ gây **mất độ chính xác** (ít nhất là ở phiên bản tiny)

## flash attention 2

- Vấn đề của flash attention 1
    * li, mi, oi bị cập nhật quá thường xuyên
        + một mặt là có nhiều phép toán không phải nhân ma trận
            + chỉ cập nhật oi ở cuối cùng
        + mặt khác là ghi bộ nhớ quá thường xuyên
            + thứ tự của vòng lặp trong và vòng lặp ngoài

1. Giảm các phép toán không phải nhân ma trận (non-matmul)
2. Tính attn song song, kể cả khi chỉ có một head
3. Cố gắng tính toán nhiều hơn bên trong thread block, giảm giao tiếp giữa các nhóm

- luồng xử lý
    * So với flash attention 1
        + Các giá trị cục bộ (oi, mi, li) không cần cập nhật nhiều lần, một vòng lặp ngoài là xử lý xong một hàng

- mẹo
    * Cần đặc biệt chú ý shape của các khối trong flash attention 2

```python
# vòng lặp của flash attention 1
for j in range(k_block_num):
    kj = K_BLOCKS[j]
    vj = V_BLOCKS[j]

    for i in range(q_block_num):
        qi = Q_BLOCKS[i]

# vòng lặp của flash attention 2
for j in range(k_block_num):
    qi = Q_BLOCKS[i]

    for i in range(q_block_num):
        kj = K_BLOCKS[j]
        vj = V_BLOCKS[j]
```

## triton flash attention 2

[mã nguồn](./flash_attention_py/tiny_flash_attn_triton.py)

Dùng triton để cài đặt attention cho qkv có shape `bs, head, seqlen, dim`.

1. Xác định số thread block cần cho phép tính, tức là grid
    - Với flash attn 2, có thể song song hóa vòng lặp q bên ngoài, tức mỗi thread thực hiện attention giữa một phần của q với toàn bộ kv
    - Với việc chia khối Q (tức chia theo seqlen, chia theo token), nếu mỗi lần xử lý `BLOCK_M` token thì một lần tính attention đầy đủ cần `cdiv(seqlen, BLOCK_M)` thread; cdiv là phép chia làm tròn lên
    - Mỗi lần tính, kernel chỉ cần hai chiều cuối, tức (seqlen, dim), nên hai chiều đầu lớn bao nhiêu thì cần bấy nhiêu thread để xử lý. Vì vậy đặt `grid[1]` là `bs * head`
    - Do đó grid cuối cùng là `[cdiv(seqlen, BLOCKM), bs * head]`
2. Thiết kế kernel, thiết kế chương trình song song
    - Tính phần dữ liệu mà mỗi thread phụ trách
    - Tính offset cần thiết để truy cập `head+1` trong `(bs, head, seqlen, dim)`
        * Có thể dùng `Tensor.stride(dim)` để tính số phần tử cần bỏ qua khi truy cập phần tử kế tiếp của chiều dim
        * Dựa vào kích thước `bs*head` được ghi trong `grid[1]` và `q.stride(1)`, thread tìm ra phạm vi mình phụ trách
    - **Dùng API `tl.make_block_ptr()` để đọc qkv theo khối**, q chia khối theo `BLOCK_M`, kv chia khối theo `BLOCK_N`
        * Dùng tham số base để tìm đúng vị trí (bs, head)
        * Dùng tham số shape và order để định nghĩa bố cục bộ nhớ
            + Lấy khối q, v với `shape=(seqlen, dim)`, `order=(1, 0)`; `order=(1, 0)` nghĩa là chiều thứ hai nằm ở phía trong khi lưu trữ
            + Lấy khối k với `shape=(dim, seqlen)`, `order=(0, 1)`; `order=(0, 1)` nghĩa là chiều thứ hai nằm ở phía ngoài khi lưu trữ, tương đương với việc chuyển vị k
            + API sẽ dựng shape cần thiết theo thứ tự được chỉ định bởi order
        * Dùng `block_shape` để định nghĩa shape của toàn khối, còn tham số `shape` là kích thước của phần được đọc mỗi lần trong toàn khối
        * Dùng tham số `strides` để định nghĩa bước nhảy mỗi khi con trỏ khối q, k, v tăng lên
    - Q chia khối theo `BLOCK_M`, K và V chia khối theo `BLOCK_N`
3. Thuật toán flash attention 2
    - Vì CSE (common subexpression elimination - loại bỏ biểu thức con chung) và LICM (loop invariant code motion - di chuyển mã bất biến ra khỏi vòng lặp) không hỗ trợ `exp()` nên dùng `exp2()` thay thế, tức là `2^x`


## Tham khảo

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
