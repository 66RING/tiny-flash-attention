#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("self_attention_cuda", &self_attention_cuda,
          "Naive Self attention implement in cuda");
    m.def("flash_attention_v1_cuda", &flash_attention_v1_cuda,
          "Flash attention v1 implement in cuda");
    m.def("flash_attention_v2_cuda", &flash_attention_v2_cuda,
          "Flash attention v2 implement in cuda");
}
