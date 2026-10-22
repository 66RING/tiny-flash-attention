import torch

from flash_attention_py.tiny_flash_attn import flash_attn


def test_flash_attn_returns_multihead_output():
    q = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]]
    )
    k = q.clone()
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]])

    expected = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
    actual = flash_attn(q, k, v, device="cpu", BLOCK_M=4)

    assert actual is not None
    torch.testing.assert_close(actual, expected)
