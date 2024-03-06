#include <torch/extension.h>
#include <torch/python.h>

#include "attention_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_v2_cutlass", &flash_attention_v2_cutlass,
          "Flash attention v2 implement in cutlass");
}
