#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/nn/functional.h>
#include <torch/python.h>
#include <vector>
// #include <cutlass/numeric_types.h>

#include "flash.h"
#include "flash_api.h"

void run_flash_attn_fwd(Qkv_params& params, cudaStream_t stream,
                        bool force_split_kernel = false) {
    // TODO:使用static switch来使用泛型
    // FP16_SWITCH(!params.is_bf16, [&] {
    //     FWD_HEADDIM_SWITCH(params.d, [&] {
    //         run_flash_attn_fwd_<elem_type, kHeadDim>(params, stream);
    //     });
    // });

    // using elem_type = cutlass::half_t;
    using elem_type = float;
    constexpr static int kHeadDim = 128;

    run_flash_attn_fwd_<elem_type, kHeadDim>(params, stream);
}

/// q, k, v with the shape of (bs, head, seqlen, dim)
torch::Tensor tiny_flash_attn_cuda(torch::Tensor& q, torch::Tensor& k,
                                   torch::Tensor& v, bool is_causal,
                                   float sm_scale) {
    // check in cuda check continous
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seqlen = sizes[2];
    const int dim = sizes[3];

    TORCH_CHECK(batch_size > 0, "batch size must be postive");

    // TODO: simplify
    at::Tensor q_padded, k_padded, v_padded;
    if (dim % 8 != 0) {
        q_padded = torch::nn::functional::pad(
            q, torch::nn::functional::PadFuncOptions({0, 8 - dim % 8}));
        k_padded = torch::nn::functional::pad(
            k, torch::nn::functional::PadFuncOptions({0, 8 - dim % 8}));
        v_padded = torch::nn::functional::pad(
            v, torch::nn::functional::PadFuncOptions({0, 8 - dim % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    // TODO:simplify
    out = torch::empty_like(q_padded);
    // TODO: unimplement L for backward
    at::Tensor L;

    // TODO: review
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int seqlen_rounded = round_multiple(seqlen, 128);

    // // TODO: review
    // // Otherwise the kernel will be launched from cuda:0 device
    // // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    Qkv_params params;

    constexpr size_t block_m = 64;
    constexpr size_t block_n = 32;

    set_params_fprop(params, batch_size, num_heads, seqlen, seqlen_rounded, dim,
                     block_m, block_n, q_padded, k_padded, v_padded, out, L,
                     is_causal, sm_scale);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_flash_attn_fwd(params, stream);

    std::cout << "tiny flash done\n";
    return q;
}
