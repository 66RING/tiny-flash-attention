import torch
import math
from attention_cuda import self_attention_cuda, flash_attention_v1_cuda, flash_attention_v2_cuda
from flash_attn_triton import flash_attn_triton, ref_attn
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

def self_attention(q, k, v):
    # TODO: extract sm_scale
    q = q / math.sqrt(q.shape[-1])
    score = torch.matmul(q, k.transpose(-2, -1))
    s = torch.softmax(score, dim=-1)
    return s @ v


def main():
    BS, HEAD, SEQLEN, DIM = 1000, 1, 64, 64

    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)
    # q = torch.arange(16, dtype=torch.float32, device="cuda").reshape(1, 1, 4,4)
    # k = torch.arange(16, dtype=torch.float32, device="cuda").reshape(1, 1, 4,4)
    # v = torch.arange(16, dtype=torch.float32, device="cuda").reshape(1, 1, 4,4)
    warmup = 10
    epoch = 10
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

    for _ in range(warmup):
        # _ = self_attention_cuda(q, k, v)
        _ = flash_attention_v2_cuda(q, k, v)

    torch.cuda.synchronize()
    base_time = time.time()
    for _ in range(epoch):
        baseline = self_attention(q, k, v)
    torch.cuda.synchronize()
    base_time = time.time() - base_time

    # naive_time = time.time()
    # for _ in range(epoch):
    #     naive_cuda_ref = self_attention_cuda(q, k, v)
    #     torch.cuda.synchronize()
    # naive_time = time.time() - naive_time

    # flash1_time = time.time()
    # for _ in range(epoch):
    #     flash1_cuda_ref = flash_attention_v1_cuda(q, k, v)
    #     torch.cuda.synchronize()
    # flash1_time = time.time() - flash1_time

    torch.cuda.synchronize()
    flash2_time = time.time()
    for _ in range(epoch):
        flash2_cuda_ref = flash_attention_v2_cuda(q, k, v)
    torch.cuda.synchronize()
    flash2_time = time.time() - flash2_time

    flash_triton_time = time.time()
    for _ in range(epoch):
        flash_triton_ref = flash_attn_triton(q, k, v, causal=False, sm_scale=sm_scale).half()
    torch.cuda.synchronize()
    flash_triton_time = time.time() - flash_triton_time

    torch.cuda.synchronize()
    official_ref_time = time.time()
    for _ in range(epoch):
        official_result = flash_attn_func_offical(q, k, v, causal=False, softmax_scale=sm_scale)
    torch.cuda.synchronize()
    official_ref_time = time.time() - official_ref_time

    
    # print time in ms
    print("baseline: \n", base_time * 1000 / epoch)
    # print("naive_cuda_ref: \n", naive_time * 1000 / epoch)
    # print("flash1_cuda_ref: \n", flash1_time * 1000 / epoch)
    print("flash2_cuda_ref: \n", flash2_time * 1000 / epoch)
    print("flash_triton_ref: \n", flash_triton_time * 1000 / epoch)
    print("official_ref: \n", official_ref_time * 1000 / epoch)

    print("baseline: \n", baseline)
    # print("naive_cuda_ref: \n", naive_cuda_ref)
    # print("flash1_cuda_ref: \n", flash1_cuda_ref)
    print("flash2_cuda_ref: \n", flash2_cuda_ref)
    # print("flash_triton_ref: \n", flash_triton_ref)
    print("official_ref: \n", official_result)

    # assert torch.allclose(baseline, naive_cuda_ref, rtol=0, atol=1e-2)
    # assert torch.allclose(baseline, flash1_cuda_ref, rtol=0, atol=1e-2)
    assert torch.allclose(baseline, flash2_cuda_ref, rtol=0, atol=1e-2)
    # assert torch.allclose(baseline, official_result, rtol=0, atol=1e-2)

if __name__ == "__main__":
    epoch = 1
    for _ in range(epoch):
        main()

