# flash attention implementin CUDA 

NOTE: specific pytorch version require to support the deserted API. Just use the standalone version or CUTLASS version.

## roadmap

- [x] naive self attention python
- [x] naive self attention cuda
- [x] naive self attention python API binding
    - TODO:
        * half support
        * make template data type more general
        * thread balance and too many thread may cause crash 
        * clean deprecated warning
- [x] flash attention 1 cuda
- [x] flash attention 2 cuda
- [x] flash attention 1/2 python binding
- [ ] split template and more general template(like dim and block size)
- [x] MHA support
- [ ] causal mode support
- [ ] flash attention cute
- [x] checkout `static_switch.h` in flash attention


## result

- You need **result-oriented programming** in CUDA
    * e.g. for `C[x, y]` should from thread (x, y)






