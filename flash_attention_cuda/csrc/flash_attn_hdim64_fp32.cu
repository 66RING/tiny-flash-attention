#include "flash.h"
#include <iostream>

// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool
// Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax>
// __global__ void flash_fwd_kernel(Flash_fwd_params params) {
//     static_assert(!(Is_causal && Is_local));  // If Is_local is true,
//     Is_causal should be false flash::compute_attn<Kernel_traits, Is_dropout,
//     Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax>(params);
// }

__global__ void flash_fwd_kernel(Qkv_params params) {}



// TODO: 简化版, 直接在这执行kernel
template <>
void run_flash_attn_fwd_<float, 128>(Qkv_params& params, cudaStream_t stream) {

    // TODO:
    auto grid = 1;
    auto threads = 1;
    auto smem_size = 1;
    auto kernel = &flash_fwd_kernel;
    kernel<<<grid, threads, smem_size, stream>>>(params);
    std::cout << "Call <float, 128>\n";
}
