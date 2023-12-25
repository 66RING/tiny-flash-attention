import torch
from tiny_flash_attn_2 import tiny_flash_attn_cuda

def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v


def tiny_flash_attn(q, k, v, causal=False, sm_scale=0.5):
    tiny_flash_attn_cuda(q, k, v, causal, sm_scale)

def main():
    BS, HEAD, SEQLEN, DIM = 1, 1, 128, 64
    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)
    tiny_flash_attn(q, k, v)

if __name__ == "__main__":
    main()
