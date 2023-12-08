# Tiny FlashAttention

- [python version]
- [rust version]
- [c version]

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






