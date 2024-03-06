import torch
import math
from attention_cutlass import flash_attention_v2_cutlass
import math
import time
# offical flash attention implement
from flash_attn import flash_attn_func as flash_attn_func_offical

'''
simple attention implement without multi head
'''

def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

def self_attention(q, k, v, causal=True, sm_scale=1):
    SEQLEN = q.shape[-2]
    M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
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
    # classic config
    # batch_size = 4
    # num head = [32, 16, 8]
    # seqlen = 4096
    # head dim = [64, 128, 256]

    # BS, HEAD, SEQLEN, DIM = 1, 2, 4 * 1024, 64
    BS, HEAD, SEQLEN, DIM = 2, 8, 2 * 1024, 64

    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)
    # q = (torch.arange(SEQLEN * DIM, device="cuda").reshape(BS, HEAD, SEQLEN, DIM) * 0.0001).half()
    # k = (torch.arange(SEQLEN * DIM, device="cuda").reshape(BS, HEAD, SEQLEN, DIM) * 0.0001).half()
    # v = (torch.arange(SEQLEN * DIM, device="cuda").reshape(BS, HEAD, SEQLEN, DIM) * 0.0001).half()

    warmup = 10
    epoch = 10

    debug_mode = False
    is_causal = True
    sm_scale = 1.0 / math.sqrt(SEQLEN);
    # sm_scale = 1.0

    base_time = run_benchmark(epoch, warmup, self_attention, q, k, v, causal=is_causal, sm_scale=sm_scale)
    print("baseline: \n", base_time * 1000 / epoch)
    flash2_time = run_benchmark(epoch, warmup, flash_attention_v2_cutlass, q, k, v, is_causal, sm_scale)
    print("flash2_cutlass_ref: \n", flash2_time * 1000 / epoch)

    fq = q.transpose(1, 2)
    fk = k.transpose(1, 2)
    fv = v.transpose(1, 2)

    official_ref_time = run_benchmark(epoch, warmup, flash_attn_func_offical, fq, fk, fv, causal=is_causal, softmax_scale=sm_scale)
    print("official_ref: \n", official_ref_time * 1000 / epoch)


    baseline = self_attention(q, k, v, causal=is_causal, sm_scale=sm_scale)
    flash2_cutlass_ref, _ = flash_attention_v2_cutlass(q, k, v, is_causal, sm_scale)
    official_result = flash_attn_func_offical(fq, fk, fv, causal=is_causal, softmax_scale=sm_scale)

    # print(baseline)
    # print(flash2_cutlass_ref)
    # print(official_result)

    assert torch.allclose(baseline, flash2_cutlass_ref, rtol=0, atol=1e-2)


if __name__ == "__main__":
    epoch = 1
    for _ in range(epoch):
        main()


