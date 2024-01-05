# Tiny FlashAttention

WIP

- [python version](#flash-attention-2)
    * [naive pure python code](./flash_attention_py/tiny_flash_attn.py)
- [triton version](#triton-flash-attention-2)
    * [triton code](./flash_attention_py/tiny_flash_attn_triton.py)
- [c version]
    * TODO: [naive pure c code]()
    * TODO: [naive cuda code]()
    * TODO: [cutlass cuda code]()
- [rust version]

A tiny [flash attention](https://github.com/Dao-AILab/flash-attention) implement in python, rust, cuda and c for learning purpose.

## tips

- `softmax_lse`
    * lse表示LogSumExp?
    * [lse主要解决计算Softmax或CrossEntropy2上溢(overflow)或下溢(underflow)的问题](https://www.apispace.com/news/post/13827.html)
- `softmax_scale`
    * `q @ k.T / softmax_scale`
    * 添加`softmax_scale`后精度损失没有那么严重了(但依然存在)

## flow

- softmax的online方法: scale(更新) + 累加
- s@v的online方法: scale(更新) + 累加
    1. 更新(旧O) + 新O
    2. **更新方法: 更新max, 更新分母**
        1. 更新max: 分子分母乘上$e^{max_old - max_new}$
        2. **更新分母: 先乘旧分母, 再除新分母**

1. 想清楚是怎么分块计算的
2. 再考虑块的值是怎么来的

不分块的情况, 设Q, K, V的shape=(N, d)

softmax结果和V矩阵乘:

```
s = Q @ K.T = (N, d) @ (d, N) = (N, N)
attn = s @ V = (N, N) @ (N, d) = (N, d)
```

分块native, softmax的部分和V的部分相乘, 

```
si = Qi @ Kj.T = (N/2, d) @ (d, N/2) = (N/2, N/2)
attni[N/2, :] = si @ Vj = (N/2, N/2) @ (N/2, d) = (N/2, d)
```


分块online

TODO: img

所以output是要相加的!


## Keynote

- the matmul
- the shape
- the algo

## The algo

- attention
- softmax
    * $s(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$
- safe softmax
- online softmax
    * algo1
    * impl algo2
- flash attention 1
    * tiling
    * SRAM
- flash attention 2

### 3 pass online softmax

1. pass1: 分块统计max
2. pass2: 分块求分母的sum
3. pass3: 执行softmax(xi)

### 2 pass online softmax

1. pass1: 分块统计max的同时动态更新分母的sum
2. pass2: 执行softmax(xi)

$$d'_i = d'_{i-1}e^{m_{i-1} - m_{i}} + e^{x_i - m_{i}}$$

$d'_{i-1}e^{m_{i-1} - m_{i}}$就能将过时的max给替换掉了

### 2 pass online attention

矩阵乘法满足结合律

### 1 pass online attention

### 分块OV的计算

对于相同位置的O

<img src="" alt="">

## sum

- online处理是会导致**精度损失**的(至少在tiny版本上)

## flash attention 2

- flash attention 1的问题
    * 频繁的li, mi, oi更新
        + 一方面是频繁的非矩阵乘法
            + oi最后更新
        + 一方面是频繁的写
            + 内外循环顺序

1. 减少非矩阵乘法(non-matmul)操作
2. 并行计算attn, 即使是单头
3. 考虑多在thread block内计算, 减少跨组通信

- flow
    * 与flash attention1对比
        + 局部值(oi, mi, li)就不用多次更新了, 一轮外部循环一行就能处理完成

- tips
    * flash attention 2中分块的形状要特别注意

```python
# flash attention 1 的循环
for j in range(k_block_num):
    kj = K_BLOCKS[j]
    vj = V_BLOCKS[j]

    for i in range(q_block_num):
        qi = Q_BLOCKS[i]

# flash attention 2 的循环
for j in range(k_block_num):
    qi = Q_BLOCKS[i]

    for i in range(q_block_num):
        kj = K_BLOCKS[j]
        vj = V_BLOCKS[j]
```

## triton flash attention 2

[source code](./flash_attention-py/tiny_flash_attn_triton.py)

用triton实现一个shape为`bs, head, seqlen, dim`的qkv的attention。

1. 考虑计算所需的thread blocks, 即grid
    - 对于flash attn 2, 可以将外层的q循环并行处理, 及每个thread执行的是一部分q和其他所有kv的attention
    - 对于Q的分块处理(即分seqlen, 即分token), 如果一次处理`BLOCK_M`个token, 那么一次完整的attention计算需要`cdiv(seqlen, BLOCK_M)`个thread, cdiv表示除法向上取整
    - 每次kernel计算只需要后两维度, 即(seqlen, dim), 那么前两个维度有多少就需要多少thread来处理。因此将`grid[1]`置为`bs * head`
    - 因此最终grid为`[cdiv(seqlen, BLOCKM), bs * head]`
2. kernel设计, 设计并行程序
    - 计算thread处理各自负责的数据
    - 计算`(bs, head, seqlen, dim)`访问`head+1`时需要的offset
        * 可以使用`Tensor.stride(dim)`计算访问dim这个维度的下一个元素时所需跳过的元素数
        * 根据`grid[1]`记录而`bs*head`的大小和`q.stride(1)`, thread找到自己负责的范围
    - **使用`tl.make_block_ptr()`API分块读取qkv**, q根据`BLOCK_M`分块, kv根据`BLOCK_N`分块
        * 使用base参数找到正确的(bs, head)位置
        * 使用shape和order参数定义内存布局
            + 取`shape=(seqlen, dim)`, `order=(1, 0)`的q, v块, `order=(1, 0)`表示第二个维度在存储中的内侧
            + 取`shape=(dim, seqlen)`, `order=(0, 1)`的k块, `order=(0, 1)`表示第二个维度在存储中的外侧, 相当于对k做转置
            + API会根据order指定的顺序去构造所需的shape
        * 使用`block_shape`定义整个块的shape, `shape`参数则是每次读取整个块中的一部分的大小
        * 使用`strides`参数定义每次q, k, v块指针递增时的步长
    - Q根据`BLOCK_M`分块, K和V根据`BLOCK_N`分块
3. flash attention 2算法
    - 因为CSE(common subexpression elimination), LICM(loop invariant code motion)不支持`exp()`所以使用`exp2()`代替, 即`2^x`














