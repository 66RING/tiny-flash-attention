import torch
import math
import tiny_flash_attn
from tiny_flash_attn_triton import flash_attn_triton, ref_attn

from flash_attn import flash_attn_func as flash_attn_func_cuda

class BaseAttention:
    def __init__(self):
        pass

    def attention(self, q, k, v):
        s = self.softmax(q @ k.T, dim=1)
        return s @ v

    def softmax(self, input, dim):
        raise "unimplement"

class NativeAttention(BaseAttention):
    def softmax(self, input, dim):
        return torch.softmax(input, dim)

class SafeAttention(BaseAttention):
    def softmax(self, input, dim):
        '''
        softmax with safe
        '''
        row_max = torch.max(input, dim=dim).values[:, None]
        # read++
        input_safe = input - row_max
        softmax_numerator = torch.exp(input_safe)
        # read++
        softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
        # read++
        return softmax_numerator / softmax_denominator

class OnlineSafeAttention(BaseAttention):
    '''
    The tiny flash attention implement
    '''
    def __init__(self):
        self.BLOCK_M = 4

    def attention(self, q, k, v, device='cuda'):
        return tiny_flash_attn.flash_attn(q, k, v, device, self.BLOCK_M)

    def attention_v1(self, q, k, v, device='cuda'):
        return tiny_flash_attn.flash_attn_v1(q, k, v, device, self.BLOCK_M)

    def attention_v2(self, q, k, v, device='cuda'):
        return tiny_flash_attn.flash_attn_v2(q, k, v, device, self.BLOCK_M)

    def attention_v2_multihead(self, q, k, v, device='cuda'):
        return tiny_flash_attn.flash_attn_v2_multihead(q, k, v, device, self.BLOCK_M)

def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16):
    q = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    return q, k, v

def main():
    BS, HEAD, SEQLEN, DIM = 1, 1, 128, 64

    q,k,v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float16)
    # softmax_scale = math.sqrt(q.shape[-1])
    # q /= softmax_scale

    native = NativeAttention()
    safe = SafeAttention()
    online = OnlineSafeAttention()

    native_result = native.attention(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    safe_result = safe.attention(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result1 = online.attention_v1(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result2 = online.attention_v2(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result2_multi = online.attention_v2_multihead(q, k, v)

    causal=False
    ref_result = ref_attn(q, k, v, causal=causal)
    triton_result = flash_attn_triton(q, k, v, causal=causal)
    official_result = flash_attn_func_cuda(q, k, v, causal=causal)

    # print(native_result)
    # print(safe_result)
    # print(online_result1)
    # print(online_result2)
    # print(online_result2_multi)
    print(ref_result)
    print(triton_result)
    print(official_result)

    # Assert attention output is same.
    # But it may have precision loss compared with native.
    assert torch.allclose(native_result, safe_result, rtol=0, atol=1e-2)
    assert torch.allclose(safe_result, online_result1.half(), rtol=0, atol=1e-2)
    assert torch.allclose(online_result2, online_result1, rtol=0, atol=1e-2)
    assert torch.allclose(online_result2_multi, online_result1, rtol=0, atol=1e-2)
    assert torch.allclose(triton_result, ref_result, rtol=0, atol=1e-2)

    # assert torch.allclose(official_result, triton_result, rtol=0, atol=1e-2)
    # assert torch.allclose(official_result, ref_result, rtol=0, atol=1e-2)

if __name__ == "__main__":
    main()

