import torch
import math

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
        assert q.shape == k.shape
        assert q.shape == v.shape

        q = q.to(device=device)
        k = k.to(device=device)
        v = v.to(device=device)
        # Create output buffer in HBM.
        output_buffer = torch.zeros(v.shape).to(device=device)
        # Create denominator buffer in HBM.
        l = torch.zeros(v.shape[:-1])[..., None].to(device=device)
        # Create max(x) buffer in HBM.
        m = torch.ones(v.shape[:-1])[..., None].to(device=device) * -torch.inf

        Q_BLOCKS = torch.split(q, self.BLOCK_M, dim=-2)
        K_BLOCKS = torch.split(k, self.BLOCK_M, dim=-2)
        V_BLOCKS = torch.split(v, self.BLOCK_M, dim=-2)
        O_BLOCKS = list(torch.split(output_buffer, self.BLOCK_M, dim=-2))
        L_BLOCKS = list(torch.split(l, self.BLOCK_M, dim=-2))
        M_BLOCKS = list(torch.split(m, self.BLOCK_M, dim=-2))

        k_block_num = k.shape[-2] // self.BLOCK_M
        for j in range(k_block_num):
            kj = K_BLOCKS[j]
            vj = V_BLOCKS[j]

            q_block_num = q.shape[-2] // self.BLOCK_M
            for i in range(q_block_num):
                qi = Q_BLOCKS[i]
                old_o = O_BLOCKS[i]
                old_d = L_BLOCKS[i]
                old_m = M_BLOCKS[i]

                # Compute qk.
                x_qkt = (qi @ kj.T)
                # Get local max of qk.
                local_m = torch.max(x_qkt, dim=1).values[:, None]

                # Compute new max.
                new_m = torch.maximum(old_m, local_m)
                # Compute numerator. e^{x - max(x)}.
                safe_e = torch.exp(x_qkt - new_m)
                # Compute new part of denominator.
                curr_d = torch.sum(safe_e, dim=1)[:, None]
                # Update denominator.
                new_d = old_d * torch.exp(old_m - new_m) + curr_d
                # Update old output and accumulate new output.
                new_o = old_o * torch.exp(old_m - new_m) * old_d / new_d + safe_e / new_d @ vj


                # # Compute numerator. e^{x - max(x)}
                # safe_e = torch.exp(x_qkt - local_m)
                # # Compute new part of denominator.
                # curr_d = torch.sum(safe_e, dim=1)[:, None]

                # # Update max.
                # new_m = torch.maximum(local_m, old_m)
                # # Update denominator.
                # new_d = old_d * torch.exp(old_m - new_m) + curr_d * torch.exp(local_m - new_m)
                # # Update old output and accumulate new output.
                # new_o = (old_d * torch.exp(old_m - new_m) * old_o / new_d) + (torch.exp(local_m - new_m) * safe_e / new_d @ vj)


                # Store new value.
                L_BLOCKS[i] = new_d
                M_BLOCKS[i] = new_m
                O_BLOCKS[i] = new_o

        output_buffer = torch.cat(O_BLOCKS, dim=-2)

        return output_buffer


def main():
    N, d = 1600, 40
    q = torch.randn(N, d)
    # To avoid precision loss.
    # softmax_scale should be implemented in the Attention class
    # But I just let native softmax to be itself.
    softmax_scale = math.sqrt(q.shape[-1])
    q /= softmax_scale
    k = torch.randn(N, d)
    v = torch.randn(N, d)

    native = NativeAttention()
    safe = SafeAttention()
    online = OnlineSafeAttention()

    native_result = native.attention(q, k, v)
    safe_result = safe.attention(q, k, v)
    online_result = online.attention(q, k, v, 'cuda')

    print(native_result.to(device='cpu'))
    print(safe_result.to(device='cpu'))
    print(online_result.to(device='cpu'))

    # Assert attention output is same.
    # But it may have precision loss compared with native.
    assert torch.allclose(native_result.to(device='cpu'), safe_result.to(device='cpu'), rtol=1e-4, atol=1e-6)
    assert torch.allclose(safe_result.to(device='cpu'), online_result.to(device='cpu'), rtol=1e-4, atol=1e-6)

if __name__ == "__main__":
    main()

