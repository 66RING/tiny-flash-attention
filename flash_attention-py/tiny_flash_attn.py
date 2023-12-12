import torch
import math

def flash_attn_v1(q, k, v, device='cpu', BLOCK_M=4):
    '''
    The tiny flash attention implement
    '''
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

    Q_BLOCKS = torch.split(q, BLOCK_M, dim=-2)
    K_BLOCKS = torch.split(k, BLOCK_M, dim=-2)
    V_BLOCKS = torch.split(v, BLOCK_M, dim=-2)
    O_BLOCKS = list(torch.split(output_buffer, BLOCK_M, dim=-2))
    L_BLOCKS = list(torch.split(l, BLOCK_M, dim=-2))
    M_BLOCKS = list(torch.split(m, BLOCK_M, dim=-2))

    k_block_num = k.shape[-2] // BLOCK_M
    for j in range(k_block_num):
        kj = K_BLOCKS[j]
        vj = V_BLOCKS[j]

        q_block_num = q.shape[-2] // BLOCK_M
        for i in range(q_block_num):
            qi = Q_BLOCKS[i]
            old_o = O_BLOCKS[i]
            old_d = L_BLOCKS[i]
            old_m = M_BLOCKS[i]

            # Compute qk.
            x_qkt = (qi @ kj.T)
            # Get local max of qk.
            local_m = torch.max(x_qkt, dim=1).values[:, None]


            # # MatMul operator optimization version.
            # # Compute new max.
            # new_m = torch.maximum(old_m, local_m)
            # # Compute numerator. e^{x - max(x)}.
            # safe_e = torch.exp(x_qkt - new_m)
            # # Compute new part of denominator.
            # curr_d = torch.sum(safe_e, dim=1)[:, None]
            # # Update denominator.
            # new_d = old_d * torch.exp(old_m - new_m) + curr_d
            # # Update old output and accumulate new output.
            # new_o = old_o * torch.exp(old_m - new_m) * old_d / new_d + safe_e / new_d @ vj


            # Flash attention 1 with many redundant mul
            # Compute numerator. e^{x - max(x)}
            safe_e = torch.exp(x_qkt - local_m)
            # Compute new part of denominator.
            curr_d = torch.sum(safe_e, dim=1)[:, None]

            # Update max.
            new_m = torch.maximum(local_m, old_m)
            # Update denominator.
            new_d = old_d * torch.exp(old_m - new_m) + curr_d * torch.exp(local_m - new_m)
            # Update old output and accumulate new output.
            new_o = (old_d * torch.exp(old_m - new_m) * old_o / new_d) + (torch.exp(local_m - new_m) * safe_e / new_d @ vj)


            # Store new value.
            L_BLOCKS[i] = new_d
            M_BLOCKS[i] = new_m
            O_BLOCKS[i] = new_o

    output_buffer = torch.cat(O_BLOCKS, dim=-2)

    return output_buffer

def flash_attn_v2(q, k, v, device='cpu', BLOCK_M=4):
    '''
    The tiny flash attention implement
    '''
    assert q.shape == k.shape
    assert q.shape == v.shape

    q = q.to(device=device)
    k = k.to(device=device)
    v = v.to(device=device)
    # Create output buffer in HBM.
    output_buffer = torch.zeros(v.shape).to(device=device)

    Q_BLOCKS = torch.split(q, BLOCK_M, dim=-2)
    K_BLOCKS = torch.split(k, BLOCK_M, dim=-2)
    V_BLOCKS = torch.split(v, BLOCK_M, dim=-2)
    O_BLOCKS = list(torch.split(output_buffer, BLOCK_M, dim=-2))

    q_block_num = q.shape[-2] // BLOCK_M
    for j in range(q_block_num):
        qi = Q_BLOCKS[j]
        old_o = O_BLOCKS[j]
        # Create denominator buffer in HBM.
        old_d = torch.zeros((BLOCK_M, 1)).to(device=device)
        # Create max(x) buffer in HBM.
        old_m = torch.full((BLOCK_M, 1), -torch.inf).to(device=device)

        k_block_num = k.shape[-2] // BLOCK_M
        for i in range(k_block_num):
            kj = K_BLOCKS[i]
            vj = V_BLOCKS[i]

            # Compute qk.
            x_qkt = (qi @ kj.T)
            # Get local max of qk.
            local_m = torch.max(x_qkt, dim=1).values[:, None]

            # Compute new max.
            new_m = torch.maximum(old_m, local_m)
            # Compute numerator. i.e.: e^{x - max(x)}.
            safe_e = torch.exp(x_qkt - new_m)
            # Compute new part of denominator.
            curr_d = torch.sum(safe_e, dim=1)[:, None]
            # Update denominator.
            new_d = old_d * torch.exp(old_m - new_m) + curr_d
            # Update old output and accumulate new output.
            new_o = old_o * torch.exp(old_m - new_m) + safe_e  @ vj

            old_m = new_m
            old_d = new_d
            old_o = new_o

        O_BLOCKS[j] = old_o / old_d

    output_buffer = torch.cat(O_BLOCKS, dim=-2)

    return output_buffer


