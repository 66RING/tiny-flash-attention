# Tiny FlashAttention

WIP

A tiny [flash attention](https://github.com/Dao-AILab/flash-attention) implement in python, rust, cuda and c for learning purpose.

- [python version](#flash-attention-2)
    * [x] [naive pure python code](./flash_attention_py/tiny_flash_attn.py)
- [triton version](#triton-flash-attention-2)
    * [x] [triton code](./flash_attention_py/tiny_flash_attn_triton.py)
- [c version]
    * [x] [naive pure c code](./flash_attention_c/csrc/attn.cpp)
    * [x] [naive cuda code standalone](./flash_attention_cuda/standalone_src)
    * [x] [naive cuda code python binding](./flash_attention_cutlass/csrc/flash_attention.cu)
    * [x] [cutlass cuda code](./flash_attention_cutlass/csrc/flash_attention.cu)
- [rust version]

## cutlass cute flash attention in action

my env: cutlass v3.4, torch 1.14, cuda 12.4

- [en tutorial](./cutlass_cute_tutorial_en.md)
- [zh tutorial](./cutlass_cute_tutorial_zh.md)



