#include <torch/extension.h>
#include "ops.h"

PYBIND11_MODULE(_kernels, m) {
	m.def("hello_world", &hello_world, "hello_world placeholder");
	m.def("naive_attn", &naive_attn, "Naive attention implementation on CPU");
	m.def("flash_attn", &flash_attn, "Flash attention implementation on CPU");
	m.def("fast_flash_attn", &fast_flash_attn, "Fast Flash attention implementation on CPU");

	m.def("fast_flash_attn_cast_buffer", &fast_flash_attn_cast_buffer, "test Fast Flash attention implementation on CPU");
}

