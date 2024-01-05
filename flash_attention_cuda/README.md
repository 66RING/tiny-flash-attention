# tiny flash attention cuda version

## Term

- Log-Sum-Exp(LSE): max + log(sum)

## api note

- 如何在cu中分配内存, 如存储临时max, denom
- 如何做"批量"操作, 如复制a的0..3到b的0..3
- 一些关键字
    * `__shared__`
- 对比官方flash
    * `compute_attn_1rowblock`
- 高级库
    * `cutlass`

## TODO

- cuda里怎么做类型转换？
- cuda里怎么保证读的类型?

## roadmap

- [x]cuda binding
- []flash attn cuda
