import torch
import time
import math
from flash_attn import flash_attn_func

from build._kernels import naive_attn, flash_attn

@torch.inference_mode()
def ref_attn(q, k, v, causal=True, sm_scale=1):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    qlen = q.shape[-2]
    klen = k.shape[-2]
    if causal:
        gap = klen - qlen
        for i in range(qlen):
            p[:, :, i, i + gap + 1:] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p, v)
    return ref_out

def run_benchmark(epoch, warmup, func, *args, **kwargs):
    # warmup phase
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()
    time_s = time.time()
    for _ in range(epoch):
        _ = func(*args, **kwargs)
        torch.cuda.synchronize()
    time_e = time.time() - time_s
    return time_e


def main():
    torch.manual_seed(0)

    bs, head_num, seqlen, head_dim = 3, 32, 128, 128
    q = torch.rand(bs, head_num, seqlen, head_dim, dtype=torch.float32, device="cpu")
    k = torch.rand(bs, head_num, seqlen, head_dim, dtype=torch.float32, device="cpu")
    v = torch.rand(bs, head_num, seqlen, head_dim, dtype=torch.float32, device="cpu")
    is_causal = True
    softmax_scale = 1 / math.sqrt(head_dim)

    warmup = 10
    epoch = 10

    naive_out = naive_attn(q, k, v, is_causal, softmax_scale)
    fa_out = flash_attn(q, k, v, is_causal, softmax_scale)

    naive_time = run_benchmark(epoch, warmup, naive_attn, q, k, v, is_causal, softmax_scale)
    fa_time = run_benchmark(epoch, warmup, flash_attn, q, k, v, is_causal, softmax_scale)

    # warmup
    q = q.to("cuda")
    k = k.to("cuda")
    v = v.to("cuda")
    ref_time = run_benchmark(epoch, warmup, ref_attn, q, k, v, is_causal, softmax_scale)
    ref_out = ref_attn(q, k, v, is_causal, softmax_scale).cpu()


    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    fa_ref = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=softmax_scale).transpose(1, 2)
    fa_ref_time = run_benchmark(epoch, warmup, flash_attn_func, q, k, v, causal=is_causal, softmax_scale=softmax_scale)

    print(f"naive CPU time: {naive_time:.3f} s")
    print(f"flash CPU time: {fa_time:.3f} s")
    print(f"torch NAIVE time: {ref_time:.3f} s")

    print(f"naive_out: {naive_out} {naive_out.shape}")
    print("-----")
    print(f"fa_out: {fa_out} {fa_out.shape}")
    print("-----")
    print(f"fa_ref: {fa_ref} {fa_ref.shape}")
    print("-----")
    print(f"ref_out: {ref_out} {ref_out.shape}")

    assert torch.allclose(fa_out, ref_out, atol=1e-2)
    assert torch.allclose(naive_out, ref_out, atol=1e-2)





    pass

if __name__ == "__main__":
    main()
