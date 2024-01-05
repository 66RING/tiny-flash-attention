# Tiny FlashAttention

WIP

一个简易的[flash attention](https://github.com/Dao-AILab/flash-attention)实现。

## algo

- attention
- softmax
    * $s(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n}e^{x_j}}$
    * 指数容易溢出导致精度损失
- safe softmax
    * $s(x_i) = \frac{e^{x_i - max(x)}}{\sum_j{e^{x_j - max(x)}}} = \frac{e^{-max(x)} \times e^{x_i}}{e^{-max(x)} \times \sum_j{e^x_j}}$
    * 指数部分减去一个最大值
- online softmax
    * 上述softmax的问题在于, 分子处的max和分母的sum都需要读取整个向量以获取max和sum值, 缓存(SRAM)不够友好
    * online softmax的算法是分子分母分开算, 最后再整合
        1. 分块计算max, 并迭代出分母的sum, 得出normalization factor
            - TODO
        2. scaling
- flash attention 1
    * tiling
    * SRAM
- flash attention 2



## ref

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)




