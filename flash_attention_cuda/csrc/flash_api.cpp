#include <torch/extension.h>
#include <torch/python.h>

#include "flash.h"
#include "flash_api.h"

void set_params_fprop(Qkv_params& params,
                      // sizes
                      const size_t bs, const size_t head, const size_t seqlen,
                      const size_t seqlen_rounded, const size_t dim,
                      const size_t block_m,
                      const size_t block_n,
                      // device pointers
                      const at::Tensor q, const at::Tensor k,
                      const at::Tensor v, at::Tensor out,
                      /* TODO: L ptr */
                      at::Tensor L, bool is_causal, float softmax_scale) {
    // Reset the parameters
    memset(&params, 0, sizeof(params));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("hello", &hello, "Prints hello world from c file");
    m.def("tiny_flash_attn_cuda", &tiny_flash_attn_cuda,
          "Flash attention 2 implement in cuda");
}
