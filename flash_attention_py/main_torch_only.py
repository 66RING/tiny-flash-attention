import math

import torch
from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.utils.flop_counter import FlopCounterMode


def safe_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
    bs, seqlen, numhead, headdim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    qk = q @ k.transpose(2, 3)
    qk *= sm_scale

    if is_causal:
        mask = torch.tril(torch.ones(seqlen, seqlen, device=q.device))
        qk = qk.masked_fill(mask == 0, float("-inf"))

    # optional: use higher precision to do softmax
    qk = qk.float()

    #
    # safe softmax
    #
    row_max = qk.max(dim=-1, keepdim=True).values
    # safe score
    score = torch.exp(qk - row_max)
    score_sum = score.sum(dim=-1, keepdim=True)
    s = score / score_sum

    # #
    # # naive softmax
    # #
    # s = torch.softmax(qk, dim=-1)

    o = s.to(q.dtype) @ v
    o = o.transpose(1, 2)

    return o


def flash_attention_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
    """
    flash attention: Three Easy Pieces
        1. block tiling -> low intermediate data
        2. two gemm fused: gemm-I (q@k), gemm-II (s@v)
        3. online safe softmax: math equal fix up for block tiling

    algo(v2):
    parallel for {bsz, numhead, seqlen}
        for block_n in kvlen:
            gemm-I: s[1, block_n] = q[1, headdim] @ k[block_n, headdim].T
                1 for 1 of seqlen (typically block_m)
            online safe softmax
                safe softmax: exp(x - max)
                online safe softmax: exp(x - local_m) * rescale = global_sm
                    exp(x - local_m) * rescale = e^{x - local_m} * e^{local_m - new_m}
                                               = e^{x - local_m + local_m - new_m}
                                               = e^{x - new_m}
            gemm-II: o += s[1, block_n] @ v[block_n, headdim]
    """
    # NOTE: tiling size (terms):
    # q_tile = [block_m, headdim]
    # k_tile = [block_n, headdim]
    # v_tile = [block_n, headdim]
    # o_tile = q_tile = [block_m, headdim]
    block_m = 32
    block_n = 64

    bs, seqlen, numhead, headdim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    assert seqlen % block_m == 0 and seqlen % block_n == 0, "Simple for now."

    o = torch.empty_like(q)

    # parallel for in gpu
    for bid in range(bs):
        # parallel for in gpu
        for hid in range(numhead):
            ######################
            #   Global Memory
            ######################

            # NOTE:
            # FA1: overview
            # https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdMfo8veQRzPgt-PsLup9ttAZmMdufdV3N3Q&s

            # NOTE: share via global memory
            # need to gmem -> smem -> reg
            all_global_output = o.view(bs, numhead, seqlen // block_m, block_m, headdim)[bid, hid, :, :, :]
            all_global_sum = torch.zeros((seqlen // block_m, block_m, 1))
            all_global_max = torch.ones((seqlen // block_m, block_m, 1)) * -torch.inf

            # parallel for in gpu
            for j_tile, kv_start in enumerate(range(0, seqlen, block_n)):
                ######################
                #   Shared Memory
                ######################
                k_tile = k[bid, hid, kv_start : kv_start + block_n, :]
                v_tile = v[bid, hid, kv_start : kv_start + block_n, :]

                for i_tile, q_start in enumerate(range(0, seqlen, block_m)):
                    q_tile = q[bid, hid, q_start : q_start + block_m, :]

                    # since kv iter is outter loop. max, sum, out must shared via global memory
                    o_tile = all_global_output[i_tile]
                    global_sum = all_global_sum[i_tile]
                    global_max = all_global_max[i_tile]

                    qk = q_tile @ k_tile.T
                    qk = qk.float()

                    if is_causal:
                        row_indices = torch.arange(block_m, device=q.device)[:, None]
                        col_indices = torch.arange(block_n, device=q.device)[None, :]
                        absolute_pos_q = q_start + row_indices
                        absolute_pos_k = kv_start + col_indices
                        causal_mask = absolute_pos_k > absolute_pos_q
                        qk = qk.masked_fill(causal_mask == True, float("-inf"))

                    qk = qk * sm_scale
                    local_max = qk.max(dim=-1, keepdim=True).values

                    # rescale
                    new_max = torch.maximum(local_max, global_max)
                    rescale = torch.exp(global_max - new_max)

                    # latest score and sum with global max
                    local_score = torch.exp(qk - new_max)
                    local_sum = torch.sum(local_score, dim=-1, keepdim=True)

                    # NOTE:
                    # old_sum * rescale + new_sum
                    # where
                    #   old_sum = e^{x - old_m}
                    #   rescale = e^{old_m - new_m}
                    # so
                    #   old_sum * rescale = e ^ {x - old_m + old_m - new_m}
                    #                     = e ^ {x - new_m}
                    new_sum = global_sum * rescale + local_sum
                    # output = score / sum
                    #        = old_score / old_sum * old_sum * rescale = new_score

                    new_score = o_tile * rescale * global_sum + local_score.to(q.dtype) @ v_tile

                    global_sum.copy_(new_sum)
                    o_tile.copy_(new_score / new_sum)
                    global_max.copy_(new_max)

    o = o.transpose(1, 2)
    return o


def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
    """
    NOTE: what's different vs fa v1
        switch inner and outter loop
        v1:
            outter loop: iter over kv
            inner loop: iter over q and o
        v2:
            outter loop: iter over q and o
            inner loop: iter over kv
        so that
            1. less output tensor IO
            2. less output tensor rescale
            3. combine output tensor at the end(epilogue)

        you can simply checkout the 66ring/ans branch for a quick look.
    """
    # NOTE: tiling size (terms):
    # q_tile = [block_m, headdim]
    # k_tile = [block_n, headdim]
    # v_tile = [block_n, headdim]
    # o_tile = q_tile = [block_m, headdim]
    block_m = 32
    block_n = 64

    bs, seqlen, numhead, headdim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    assert seqlen % block_m == 0 and seqlen % block_n == 0, "Simple for now."

    o = torch.empty_like(q)

    # parallel for in gpu
    for bid in range(bs):
        # parallel for in gpu
        for hid in range(numhead):
            all_global_output = o.view(bs, numhead, seqlen // block_m, block_m, headdim)[bid, hid, :, :, :]
            ######################
            #   Global Memory
            ######################

            # parallel for in gpu
            for i_tile, q_start in enumerate(range(0, seqlen, block_m)):
                ######################
                #   Shared Memory
                ######################

                # >>>
                # >>> YOUR CORE HERE.
                # >>>
                q_tile = q[bid, hid, q_start : q_start + block_m, :]
                # a reference here, not materialize. no ldg usage.
                o_tile = all_global_output[i_tile]

                global_score = torch.zeros_like(o_tile)
                global_sum = torch.zeros((block_m, 1))
                global_max = torch.ones((block_m, 1)) * -torch.inf

                for j_tile, kv_start in enumerate(range(0, seqlen, block_n)):
                    # >>>
                    # >>> YOUR CORE HERE.
                    # >>>
                    k_tile = k[bid, hid, kv_start : kv_start + block_n, :]
                    v_tile = v[bid, hid, kv_start : kv_start + block_n, :]

                    qk = q_tile @ k_tile.T
                    qk = qk.float()

                    if is_causal:
                        row_indices = torch.arange(block_m, device=q.device)[:, None]
                        col_indices = torch.arange(block_n, device=q.device)[None, :]
                        absolute_pos_q = q_start + row_indices
                        absolute_pos_k = kv_start + col_indices
                        causal_mask = absolute_pos_k > absolute_pos_q
                        qk = qk.masked_fill(causal_mask == True, float("-inf"))

                    qk = qk * sm_scale
                    local_max = qk.max(dim=-1, keepdim=True).values

                    # rescale
                    new_max = torch.maximum(local_max, global_max)
                    rescale = torch.exp(global_max - new_max)

                    # latest score and sum with global max
                    local_score = torch.exp(qk - new_max)
                    local_sum = torch.sum(local_score, dim=-1, keepdim=True)

                    # NOTE:
                    # old_sum * rescale + new_sum
                    # where
                    #   old_sum = e^{x - old_m}
                    #   rescale = e^{old_m - new_m}
                    # so
                    #   old_sum * rescale = e ^ {x - old_m + old_m - new_m}
                    #                     = e ^ {x - new_m}
                    new_sum = global_sum * rescale + local_sum
                    # output = score / sum
                    #        = old_score / old_sum * old_sum * rescale = new_score
                    new_score = global_score * rescale + local_score.to(q.dtype) @ v_tile

                    # **shared memory io. no ldg involve**
                    global_sum = new_sum
                    global_max = new_max
                    global_score = new_score

                o_tile.copy_(global_score / global_sum)

    o = o.transpose(1, 2)
    return o


def get_tensors(BS, SEQLEN, HEAD, DIM):
    q = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)
    k = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)
    v = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)

    return q, k, v


@torch.no_grad()
def main():
    torch.manual_seed(13)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    BS, SEQLEN, HEAD, DIM = 3, 512, 8, 128
    q, k, v = get_tensors(BS, SEQLEN, HEAD, DIM)
    scale = 1 / math.sqrt(DIM)
    is_causal = True

    counter = FlopCounterMode(display=False)

    with counter:
        o = safe_self_attention(q, k, v, is_causal=is_causal, sm_scale=scale)
    print(f"torch self attention flops: {counter.get_total_flops()}")

    with counter:
        sdpa_o = sdpa(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal, scale=scale
        ).transpose(1, 2)
    print(f"sdpa flops: {counter.get_total_flops()}")

    fa_o = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=scale)
    fa_v1 = flash_attention_v1(q, k, v, is_causal=is_causal, sm_scale=scale)

    fa_v2 = flash_attention_v2(q, k, v, is_causal=is_causal, sm_scale=scale)

    torch.testing.assert_close(fa_o, o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(sdpa_o, o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(fa_v1, o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(fa_v2, o, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()

