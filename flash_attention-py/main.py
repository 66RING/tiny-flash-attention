import torch
import math
import tiny_flash_attn

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

    def attention(self, q, k, v, device='cpu'):
        return tiny_flash_attn.flash_attn(q, k, v, device, self.BLOCK_M)

    def attention_v1(self, q, k, v, device='cpu'):
        return tiny_flash_attn.flash_attn_v1(q, k, v, device, self.BLOCK_M)

    def attention_v2(self, q, k, v, device='cpu'):
        return tiny_flash_attn.flash_attn_v2(q, k, v, device, self.BLOCK_M)

    def attention_v2_multihead(self, q, k, v, device='cpu'):
        return tiny_flash_attn.flash_attn_v2_multihead(q, k, v, device, self.BLOCK_M)


def main():
    bs, head, seqlen, d = 1, 1, 160, 80
    q = torch.randn(bs, head, seqlen, d)
    # To avoid precision loss.
    # NOTE: softmax_scale should be implemented in the Attention class
    # But I just let native softmax to be itself.
    softmax_scale = math.sqrt(q.shape[-1])
    q /= softmax_scale
    k = torch.randn(bs, head, seqlen, d)
    v = torch.randn(bs, head, seqlen, d)

    native = NativeAttention()
    safe = SafeAttention()
    online = OnlineSafeAttention()

    native_result = native.attention(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    safe_result = safe.attention(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result1 = online.attention_v1(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result2 = online.attention_v2(torch.squeeze(q), torch.squeeze(k), torch.squeeze(v))
    online_result2_multi = online.attention_v2_multihead(q, k, v)
    online_result2_multi = torch.squeeze(online_result2_multi)

    print(native_result.to(device='cpu'))
    print(safe_result.to(device='cpu'))
    print(online_result1.to(device='cpu'))
    print(online_result2.to(device='cpu'))
    print(online_result2_multi.to(device='cpu'))

    # Assert attention output is same.
    # But it may have precision loss compared with native.
    assert torch.allclose(native_result.to(device='cpu'), safe_result.to(device='cpu'), rtol=1e-4, atol=1e-6)
    assert torch.allclose(safe_result.to(device='cpu'), online_result1.to(device='cpu'), rtol=1e-4, atol=1e-6)
    assert torch.allclose(online_result2.to(device='cpu'), online_result1.to(device='cpu'), rtol=1e-4, atol=1e-6)
    assert torch.allclose(online_result2_multi.to(device='cpu'), online_result1.to(device='cpu'), rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    main()

