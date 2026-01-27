# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import math
import torch
import triton
import triton.language as tl

# triton kernel
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=8),
#         # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=2),
#         # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=4),
#         # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=8),
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_S': 8, 'BLOCK_SIZE_W': 8}, num_warps=2),
        triton.Config({'BLOCK_SIZE_S': 8, 'BLOCK_SIZE_W': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE_S': 8, 'BLOCK_SIZE_W': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_W': 16}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=2),
        # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_W': 32}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=2),
        # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_W': 64}, num_warps=8),
    ],
    key=[]
)
@triton.jit
def afak_fwd_kernel(
    q_ptr, k_ptr, states_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, S:tl.constexpr, C: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Use multiple program IDs for better parallelization
    b_id = tl.program_id(axis=0)
    t_id = tl.program_id(axis=1)
    sw_block_id = tl.program_id(axis=2)
    num_s_blocks = triton.cdiv(S, BLOCK_SIZE_S)
    num_w_blocks = triton.cdiv(W, BLOCK_SIZE_W)
    SW = S + W

    # Compute base pointers
    q_base = q_ptr + b_id * T * C
    k_base = k_ptr + b_id * T * C
    states_base = states_ptr + b_id * T * S * C
    y_base = y_ptr + b_id * T * W

    # Fetch the query at [b_id, t_id, :]
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(B, T, C),
        strides=(T * C, C, 1),
        offsets=(b_id, t_id, 0),
        block_shape=(1, 1, C),
        order=(0, 1, 2),
    )
    q = tl.load(q_block_ptr) # (1, 1, C)

    if sw_block_id < num_s_blocks:
        s_first_id = sw_block_id * BLOCK_SIZE_S
        # Fetch the states at [b_id, t_id, s_first_id:s_first_id+BLOCK_SIZE_S, :]
        s_block_ptr = tl.make_block_ptr(
            base=states_ptr,
            shape=(B, T, S, C),
            strides=(T * S * C, S * C, C, 1),
            offsets=(b_id, t_id, s_first_id, 0),
            block_shape=(1, 1, BLOCK_SIZE_S, C),
            order=(0, 1, 2, 3),
        )
        s = tl.load(s_block_ptr) # (1, 1, BLOCK_SIZE_S, C)
        o = q[:, :, None, :] * s # (1, 1, BLOCK_SIZE_S, C)
        o = tl.sum(o, axis=-1) # (1, 1, BLOCK_SIZE_S)
        # Store the result
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, s_first_id),
            block_shape=(1, 1, BLOCK_SIZE_S),
            order=(0, 1, 2),
        )
        tl.store(y_block_ptr, o.to(y_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_S)
    else:
        w_first_id = (sw_block_id - num_s_blocks) * BLOCK_SIZE_W
        # Fetch the key at [b_id, t_id-W+1+(w_block_id*BLOCK_SIZE_W):t_id+(w_block_id*BLOCK_SIZE_W), :]
        # need to load the keys manually because make_block_ptr doesn't support masks
        tw_offs = tl.arange(0, BLOCK_SIZE_W)
        c_offs = tl.arange(0, C)
        k_block_ptr = k_base + (t_id - W + 1 + (w_first_id + tw_offs[:, None])) * C + c_offs[None, :]
        mask = w_first_id + tl.arange(0, BLOCK_SIZE_W)[:, None] > (W - t_id - 2)
        k = tl.load(k_block_ptr, mask=mask) # (BLOCK_SIZE_W, C)
        # Compute the dot product (but not with tl.dot because it has a minimum size of 16)
        y = q * k[None, :] # (1, BLOCK_SIZE_W, C)
        y = tl.sum(y, axis=-1) # (1, BLOCK_SIZE_W)
        # Store the result
        y_block_ptr = tl.make_block_ptr(
            base=y_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, S + w_first_id),
            block_shape=(1, 1, BLOCK_SIZE_W),
            order=(0, 1, 2),
        )
        tl.store(y_block_ptr, y[None, :].to(y_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_W)

# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE_C': bs_c,
#         }, num_warps=warps)
#         for bs_c in [16] #, 32, 64]
#         for warps in [2] # 4, 8]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_C': bs_c,
        }, num_warps=warps)
        for bs_c in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def afak_bwd_kernel(
    q_ptr, k_ptr, states_ptr, dy_ptr, dq_ptr, dk_ptr, ds_ptr,
    B: tl.constexpr, T: tl.constexpr, S: tl.constexpr, C: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Use multiple program IDs for better parallelization
    b_id = tl.program_id(axis=0)
    t_id = tl.program_id(axis=1)
    c_block_id = tl.program_id(axis=2)
    c_first_id = c_block_id * BLOCK_SIZE_C
    SW = S + W

    # Compute base pointers
    q_base = q_ptr + b_id * T * C
    k_base = k_ptr + b_id * T * C
    dy_base = dy_ptr + b_id * T * SW
    dq_base = dq_ptr + b_id * T * C
    dk_base = dk_ptr + b_id * T * C

    # First calculate the gradients for q
    # Fetch original keys at [b_id, t_id-W+1:t_id, c_first_id:c_first_id+BLOCK_SIZE_C]
    # using a block ptr also disallows the use of masks when loading, so let's just make a ptr manually
    tw_offs = tl.arange(0, W)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    k_block_ptr = k_base + (t_id - W + 1 + tw_offs[:, None]) * C + c_first_id + c_offs[None, :]
    mask = tl.arange(0, W)[:, None] > (W - t_id - 2)
    k = tl.load(k_block_ptr, mask=mask) # (W, BLOCK_SIZE_C)
    # Fetch output gradients at [b_id, t_id, S:W]
    dy_block_ptr = tl.make_block_ptr(
        base=dy_ptr,
        shape=(B, T, SW),
        strides=(T * SW, SW, 1),
        offsets=(b_id, t_id, S),
        block_shape=(1, 1, W),
        order=(0, 1, 2),
    )
    dy = tl.load(dy_block_ptr) # (1, 1, W)
    # Compute the gradients for q
    dqk = dy.permute(0, 2, 1) * k[None, :] # (1, W, BLOCK_SIZE_C)
    dqk = tl.sum(dqk, axis=1) # (1, BLOCK_SIZE_C)
    # Then we also have to add the gradients from the states
    # Fetch the states at [b_id, t_id, c_first_id:c_first_id+BLOCK_SIZE_C]
    s_block_ptr = tl.make_block_ptr(
        base=states_ptr,
        shape=(B, T, S, C),
        strides=(T * S * C, S * C, C, 1),
        offsets=(b_id, t_id, 0, c_first_id),
        block_shape=(1, 1, S, BLOCK_SIZE_C),
        order=(0, 1, 2, 3),
    )
    s = tl.load(s_block_ptr) # (1, 1, S, BLOCK_SIZE_C)
    # Fetch the output gradients at [b_id, t_id, :S]
    dy_block_ptr = tl.make_block_ptr(
        base=dy_ptr,
        shape=(B, T, SW),
        strides=(T * SW, SW, 1),
        offsets=(b_id, t_id, 0),
        block_shape=(1, 1, S),
        order=(0, 1, 2),
    )
    dy = tl.load(dy_block_ptr) # (1, 1, S)
    # Compute the gradients for q
    dqs = dy[:, :, :, None] * s # (1, 1, S, BLOCK_SIZE_C)
    dqs = tl.sum(dqs, axis=2) # (1, 1, BLOCK_SIZE_C)
    dq = dqk[None, :] + dqs # (1, 1, BLOCK_SIZE_C)
    # Store the result
    dq_block_ptr = tl.make_block_ptr(
        base=dq_ptr,
        shape=(B, T, C),
        strides=(T * C, C, 1),
        offsets=(b_id, t_id, c_first_id),
        block_shape=(1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2),
    )
    tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_C)

    # Calculate the gradients for states while we're at it
    # Fetch the query at [b_id, t_id, c_first_id:c_first_id+BLOCK_SIZE_C]
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(B, T, C),
        strides=(T * C, C, 1),
        offsets=(b_id, t_id, c_first_id),
        block_shape=(1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2),
    )
    q = tl.load(q_block_ptr) # (1, 1, BLOCK_SIZE_C)
    # Compute the gradients for states
    ds = dy[:, :, :, None] * q[:, :, None, :] # (1, 1, S, BLOCK_SIZE_C)
    # Store the result
    ds_block_ptr = tl.make_block_ptr(
        base=ds_ptr,
        shape=(B, T, S, C),
        strides=(T * S * C, S * C, C, 1),
        offsets=(b_id, t_id, 0, c_first_id),
        block_shape=(1, 1, S, BLOCK_SIZE_C),
        order=(0, 1, 2, 3),
    )
    tl.store(ds_block_ptr, ds.to(ds_block_ptr.dtype.element_ty)) # (1, 1, S, BLOCK_SIZE_C)

    # Then calculate the gradients for k
    # same thing here, let's just make the ptr manually
    tw_offs = tl.arange(0, W)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    q_block_ptr = q_base + (t_id + tw_offs[:, None]) * C + c_first_id + c_offs[None, :]
    mask = tl.arange(0, W)[:, None] < T - t_id
    q = tl.load(q_block_ptr, mask=mask) # (W, BLOCK_SIZE_C)
    # Fetch original gradients at [b_id, t_id, :]
    # This one is tricky bc we have to fetch a diagonal from dy
    # going from [b_id, t_id, W] to [b_id, t_id+W, 0]
    w_offs = tl.arange(0, W)
    diag_dy_base = dy_base + t_id * SW + S + tl.flip(w_offs, 0)
    dy_block_ptr = diag_dy_base + w_offs * SW
    mask = tl.arange(0, W) < T - t_id
    dy = tl.load(dy_block_ptr, mask=mask) # (W)
    # Compute the gradients for k
    dk = dy.reshape(W, 1) * q # (W, BLOCK_SIZE_C)
    dk = tl.sum(dk, axis=0) # (BLOCK_SIZE_C)
    # Store the result
    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr,
        shape=(B, T, C),
        strides=(T * C, C, 1),
        offsets=(b_id, t_id, c_first_id),
        block_shape=(1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2),
    )
    tl.store(dk_block_ptr, dk.reshape(1, 1, BLOCK_SIZE_C).to(dk_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_C)
    
class AttendFoldedAllKeysTriton(torch.autograd.Function):
    # @torch.compiler.disable
    @staticmethod
    def forward(ctx, q, k, states, W):
        B, T, C = q.shape
        B, T, S, C = states.shape
        q = q.contiguous()
        k = k.contiguous()
        states = states.contiguous()
        ctx.save_for_backward(q, k, states)
        ctx.W = W
        
        # Calculate grid dimensions
        grid = lambda meta: (B, T, triton.cdiv(S, meta['BLOCK_SIZE_S']) + triton.cdiv(W, meta['BLOCK_SIZE_W']))

        # Allocate output tensor
        y = torch.zeros((B, T, S+W), dtype=q.dtype, device=q.device).contiguous()
        
        # Launch kernel
        afak_fwd_kernel[grid](
            q, k, states, y,
            B, T, S, C, W,
        )
        
        return y
    
    # @torch.compiler.disable
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        q, k, states = ctx.saved_tensors
        B, T, S, C = states.shape
        W = ctx.W
        
        # Calculate grid dimensions
        grid = lambda meta: (B, T, triton.cdiv(C, meta['BLOCK_SIZE_C']))
        
        gq = torch.zeros_like(q).contiguous()
        gk = torch.zeros_like(k).contiguous()
        gs = torch.zeros_like(states).contiguous()

        # Launch kernel
        afak_bwd_kernel[grid](
            q, k, states, grad_output, gq, gk, gs,
            B, T, S, C, W
        )

        return gq, gk, gs, None

# triton kernel
# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE_C': bs_c,
#         }, num_warps=warps)
#         for bs_c in [16] #, 32, 64]
#         for warps in [2] # 4, 8]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_C': bs_c,
        }, num_warps=warps)
        for bs_c in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def afav_fwd_kernel(
    s_ptr, v_ptr, states_ptr, y_ptr,
    B: tl.constexpr, T: tl.constexpr, S: tl.constexpr, C: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Use multiple program IDs for better parallelization
    b_id = tl.program_id(axis=0)
    t_id = tl.program_id(axis=1)
    c_block_id = tl.program_id(axis=2)
    c_first_id = c_block_id * BLOCK_SIZE_C
    SW = S + W

    # Compute base pointers
    s_base = s_ptr + b_id * T * W
    v_base = v_ptr + b_id * T * C
    y_base = y_ptr + b_id * T * C

    # First we accumulate the values
    # Fetch the scores at [b_id, t_id, S:W]
    sv_block_ptr = tl.make_block_ptr(
        base=s_ptr,
        shape=(B, T, SW),
        strides=(T * SW, SW, 1),
        offsets=(b_id, t_id, S),
        block_shape=(1, 1, W),
        order=(0, 1, 2),
    )
    sv = tl.load(sv_block_ptr) # (1, 1, W)
    # Fetch the value at [b_id, t_id-W+1:t_id, c_first_id:c_first_id+BLOCK_SIZE_C]
    # need to load the keys manually because make_block_ptr doesn't support masks
    tw_offs = tl.arange(0, W)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    v_block_ptr = v_base + (t_id - W + 1 + tw_offs[:, None]) * C + c_first_id + c_offs[None, :]
    mask = tl.arange(0, W)[:, None] > (W - t_id - 2)
    v = tl.load(v_block_ptr, mask=mask) # (W, BLOCK_SIZE_C) but W can vary <W
    # Compute the dot product (but not with tl.dot because it has a minimum size of 16)
    # y = sv.permute(0, 2, 1) * v[None, :] # (1, W, BLOCK_SIZE_C)
    # y = tl.sum(y, axis=1, keep_dims=True) # (1, 1, BLOCK_SIZE_C)
    # turns out keep_dims kinda messes stuff up when later adding the accumulated states

    # Then we accumulate the states
    # Fetch the scores at [b_id, t_id, :S]
    ss_block_ptr = tl.make_block_ptr(
        base=s_ptr,
        shape=(B, T, SW),
        strides=(T * SW, SW, 1),
        offsets=(b_id, t_id, 0),
        block_shape=(1, 1, S),
        order=(0, 1, 2),
    )
    ss = tl.load(ss_block_ptr) # (1, 1, S)
    # Fetch the states at [b_id, t_id, c_first_id:c_first_id+BLOCK_SIZE_C]
    states_block_ptr = tl.make_block_ptr(
        base=states_ptr,
        shape=(B, T, S, C),
        strides=(T * S * C, S * C, C, 1),
        offsets=(b_id, t_id, 0, c_first_id),
        block_shape=(1, 1, S, BLOCK_SIZE_C),
        order=(0, 1, 2, 3),
    )
    states = tl.load(states_block_ptr) # (1, 1, S, BLOCK_SIZE_C)
    # Compute the dot product
    y = tl.sum(sv.permute(0, 2, 1) * v[None, :], axis=1) + tl.sum(ss[:, :, :, None] * states, axis=2).reshape(1, BLOCK_SIZE_C)

    # Store the result
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(B, T, C),
        strides=(T * C, C, 1),
        offsets=(b_id, t_id, c_first_id),
        block_shape=(1, 1, BLOCK_SIZE_C),
        order=(0, 1, 2),
    )
    tl.store(y_block_ptr, y[None, :].to(y_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_C)

# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE_S': bs_s,
#             'BLOCK_SIZE_W': bs_w
#         }, num_warps=warps)
#         for bs_s in [16] #, 32, 64]
#         for bs_w in [16] #, 32, 64]
#         for warps in [2] # 4, 8]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_S': bs_s,
            'BLOCK_SIZE_W': bs_w
        }, num_warps=warps)
        for bs_s in [8, 16] # 32, 64]
        for bs_w in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def afav_bwd_kernel(
    s_ptr, v_ptr, states_ptr, dy_ptr, ds_ptr, dv_ptr, dstates_ptr,
    B: tl.constexpr, T: tl.constexpr, S:tl.constexpr, C: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Use multiple program IDs for better parallelization
    b_id = tl.program_id(axis=0)
    t_id = tl.program_id(axis=1)
    sw_block_id = tl.program_id(axis=2)
    num_s_blocks = triton.cdiv(S, BLOCK_SIZE_S)
    num_w_blocks = triton.cdiv(W, BLOCK_SIZE_W)
    is_state = sw_block_id < num_s_blocks
    SW = S + W

    # Compute base pointers
    s_base = s_ptr + b_id * T * C
    v_base = v_ptr + b_id * T * C
    dy_base = dy_ptr + b_id * T * W
    ds_base = ds_ptr + b_id * T * C
    dv_base = dv_ptr + b_id * (T+W-1) * C + (W-1) * C # skip the first W-1 elements

    if not is_state:
        # Here we calculate the gradients for s [:, :, S:W] and for v
        w_first_id = (sw_block_id - num_s_blocks) * BLOCK_SIZE_W
        # First calculate the gradients for s
        # Fetch original output gradients at [b_id, t_id, :]
        dy_block_ptr = tl.make_block_ptr(
            base=dy_ptr,
            shape=(B, T, C),
            strides=(T * C, C, 1),
            offsets=(b_id, t_id, 0),
            block_shape=(1, 1, C),
            order=(0, 1, 2),
        )
        dy = tl.load(dy_block_ptr) # (1, 1, C)
        # Then calculate the gradients for v
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, S+w_first_id),
            block_shape=(1, 1, BLOCK_SIZE_W),
            order=(0, 1, 2),
        )
        s = tl.load(s_block_ptr) # (1, 1, BLOCK_SIZE_W)
        # Fetch original values at [b_id, t_id-W+1+(w_block_id*BLOCK_SIZE_W):t_id+(w_block_id*BLOCK_SIZE_W), :]
        # using a block ptr also disallows the use of masks when loading, so let's just make a ptr manually
        tw_offs = tl.arange(0, BLOCK_SIZE_W)
        c_offs = tl.arange(0, C)
        v_block_ptr = v_base + (t_id - W + 1 + (w_first_id + tw_offs[:, None])) * C + c_offs[None, :]
        mask = w_first_id + tl.arange(0, BLOCK_SIZE_W)[:, None] > (W - t_id - 2)
        v = tl.load(v_block_ptr, mask=mask) # (BLOCK_SIZE_W, C)

        # We already fetched output gradients dy at [b_id, t_id, :] w/ size (1, 1, C)
        # Compute the gradients for v
        dv = dy * s.reshape(1, BLOCK_SIZE_W, 1) # (1, BLOCK_SIZE_W, C)

        # Compute the gradients for q
        dsv = dy * v[None, :] # (1, BLOCK_SIZE_W, C)
        dsv = tl.sum(dsv, axis=-1) # (1, BLOCK_SIZE_W)

        # Store the result
        dsv_block_ptr = tl.make_block_ptr(
            base=ds_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, S+w_first_id),
            block_shape=(1, 1, BLOCK_SIZE_W),
            order=(0, 1, 2),
        )
        tl.store(dsv_block_ptr, dsv[None, :].to(dsv_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_W)

        # Store the result
        # need to make a ptr manually because make_block_ptr doesn't support masks
        tw_offs = tl.arange(0, BLOCK_SIZE_W)
        c_offs = tl.arange(0, C)
        dv_block_ptr = dv_base + (t_id - W + 1 + (w_first_id + tw_offs[:, None])) * C + c_offs[None, :]
        mask = w_first_id + tl.arange(0, BLOCK_SIZE_W)[:, None] > (W - t_id - 2)
        # now we have to atomically add the gradients to the original values
        tl.atomic_add(dv_block_ptr[None, :], dv)
    else:
        s_first_id = sw_block_id * BLOCK_SIZE_S
        # Here we calculate the gradients for s[:, :, :S] and for states
        # First calculate the gradients for s
        # Fetch states at [b_id, t_id, s_first_id:s_first_id+BLOCK_SIZE_S, :]
        states_block_ptr = tl.make_block_ptr(
            base=states_ptr,
            shape=(B, T, S, C),
            strides=(T * S * C, S * C, C, 1),
            offsets=(b_id, t_id, s_first_id, 0),
            block_shape=(1, 1, BLOCK_SIZE_S, C),
            order=(0, 1, 2, 3),
        )
        states = tl.load(states_block_ptr) # (1, 1, BLOCK_SIZE_S, C)
        # Fetch original output gradients at [b_id, t_id, :]
        dy_block_ptr = tl.make_block_ptr(
            base=dy_ptr,
            shape=(B, T, C),
            strides=(T * C, C, 1),
            offsets=(b_id, t_id, 0),
            block_shape=(1, 1, C),
            order=(0, 1, 2),
        )
        dy = tl.load(dy_block_ptr) # (1, 1, C)
        # Fetch the scores at [b_id, t_id, :S]
        ss_block_ptr = tl.make_block_ptr(
            base=s_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, s_first_id),
            block_shape=(1, 1, BLOCK_SIZE_S),
            order=(0, 1, 2),
        )
        ss = tl.load(ss_block_ptr) # (1, 1, BLOCK_SIZE_S)

        # Compute the gradients for s
        dss = dy[:, :, None, :] * states # (1, 1, BLOCK_SIZE_S, C)
        dss = tl.sum(dss, axis=-1) # (1, 1, BLOCK_SIZE_S)

        # Then calculate the gradients for states
        dstates = dy[:, :, None, :] * ss[:, :, :, None] # (1, 1, BLOCK_SIZE_S, C)

        # Store the result gradients of s at [b_id, t_id, :S]
        dss_block_ptr = tl.make_block_ptr(
            base=ds_ptr,
            shape=(B, T, SW),
            strides=(T * SW, SW, 1),
            offsets=(b_id, t_id, s_first_id),
            block_shape=(1, 1, BLOCK_SIZE_S),
            order=(0, 1, 2),
        )
        tl.store(dss_block_ptr, dss.to(dss_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_S)
        
        # Store the result gradients of states at [b_id, t_id, s_first_id:s_first_id+BLOCK_SIZE_S, :]
        dstates_block_ptr = tl.make_block_ptr(
            base=dstates_ptr,
            shape=(B, T, S, C),
            strides=(T * S * C, S * C, C, 1),
            offsets=(b_id, t_id, s_first_id, 0),
            block_shape=(1, 1, BLOCK_SIZE_S, C),
            order=(0, 1, 2, 3),
        )
        tl.store(dstates_block_ptr, dstates.to(dstates_block_ptr.dtype.element_ty)) # (1, 1, BLOCK_SIZE_S, C)

class AccumulateFoldedAllValuesTriton(torch.autograd.Function):
    # @torch.compiler.disable
    @staticmethod
    def forward(ctx, s, v, states, W):
        B, T, S, C = states.shape
        s = s.contiguous()
        v = v.contiguous()
        states = states.contiguous()
        ctx.save_for_backward(s, v, states)
        ctx.W = W
        
        # Calculate grid dimensions
        grid = lambda meta: (B, T, triton.cdiv(C, meta['BLOCK_SIZE_C']))

        # Allocate output tensor
        y = torch.zeros((B, T, C), dtype=v.dtype, device=v.device).contiguous()
        
        # Launch kernel
        afav_fwd_kernel[grid](
            s, v, states, y,
            B, T, S, C, W,
        )
        
        return y
    
    # @torch.compiler.disable
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        s, v, states = ctx.saved_tensors
        B, T, S, C = states.shape
        W = ctx.W
        
        # Calculate grid dimensions
        grid = lambda meta: (B, T, triton.cdiv(S, meta['BLOCK_SIZE_S']) + triton.cdiv(W, meta['BLOCK_SIZE_W']))
        
        gs = torch.zeros_like(s).contiguous()
        # for gv we want an additional W at the start of the time dimension bc we can't mask atomic add
        gv = torch.zeros((B, T+W-1, C), device=v.device).contiguous()
        gst = torch.zeros_like(states).contiguous()

        # Launch kernel
        afav_bwd_kernel[grid](
            s, v, states, grad_output, gs, gv, gst,
            B, T, S, C, W,
        )

        # No need for the additional W at the start of the time dimension for gv
        return gs, gv[:, W-1:].to(s.dtype), gst, None

# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE': bs,
#             'BLOCK_SIZE_S': bs_s,
#             'BLOCK_SIZE_C': bs_c
#         }, num_warps=warps)
#         for bs in [16] #, 32, 64]
#         for bs_s in [16] #, 32, 64]
#         for bs_c in [16] #, 32, 64]
#         for warps in [2] # 4, 8]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE': bs,
            'BLOCK_SIZE_S': bs_s,
            'BLOCK_SIZE_C': bs_c
        }, num_warps=warps)
        for bs in [16, 32, 64]
        for bs_s in [8, 16] #, 32, 64]
        for bs_c in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def cg2d_fwd_kernel(
    xg_ptr, gi_ptr, 
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr, T: tl.constexpr, nstages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # Add more constants for tiling
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Use multiple program IDs for better parallelization
    pid = tl.program_id(axis=0)
    # Compute batch, spatial, and channel indices
    num_s_blocks = tl.cdiv(S, BLOCK_SIZE_S)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    b = pid // (num_s_blocks * num_c_blocks)
    rem = pid % (num_s_blocks * num_c_blocks)
    s_block = rem // num_c_blocks
    c_block = rem % num_c_blocks

    # Compute actual indices
    s_offs = tl.arange(0, BLOCK_SIZE_S)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    s_mask = s_offs < (S - s_block * BLOCK_SIZE_S)
    c_mask = c_offs < (C - c_block * BLOCK_SIZE_C)
    s_offs = s_block * BLOCK_SIZE_S + s_offs
    c_offs = c_block * BLOCK_SIZE_C + c_offs

    # Compute base pointers
    xg_base = xg_ptr + b * T * S * C
    gi_base = gi_ptr + b * T * S

    # Precompute stages for better efficiency
    # nstages = tl.ceil(tl.log2(float(T))).to(tl.int32)
    offs = tl.arange(0, BLOCK_SIZE)
    
    for stage in tl.range(nstages): # CHANGE BACK TO tl.static_range() IN FINAL VERSION
        group_stride = 1 << stage
        # Process multiple elements per thread using BLOCK_SIZE
        for block_start in tl.range(0, T//2, BLOCK_SIZE):
            block_mask = offs < (T//2 - block_start)
            block_s_mask = block_mask[:, None] & s_mask[None, :]
            block_s_c_mask = block_mask[:, None, None] & s_mask[None, :, None] & c_mask[None, None, :]
            
            # Compute indices with vectorization
            initial_indices = group_stride + ((offs + block_start) // group_stride) * group_stride * 2
            t_targets = initial_indices + ((offs + block_start) % group_stride)
            t_adders = initial_indices - 1

            xg_targets_ptr = xg_base + t_targets[:, None, None] * S * C + s_offs[None, :, None] * C + c_offs[None, None, :] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            xg_adders_ptr = xg_base + t_adders[:, None, None] * S * C + s_offs[None, :, None] * C + c_offs[None, None, :] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            gi_targets_ptr = gi_base + t_targets[:, None] * S + s_offs[None, :] # (BLOCK_SIZE, BLOCK_SIZE_S)
            gi_adders_ptr = gi_base + t_adders[:, None] * S + s_offs[None, :] # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            xg_targets = tl.load(xg_targets_ptr, mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            xg_adders = tl.load(xg_adders_ptr, mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            gi_targets = tl.load(gi_targets_ptr, mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)
            gi_adders = tl.load(gi_adders_ptr, mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            # Compute and store results
            xg_targets += xg_adders * gi_targets[:, :, None] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            # Update gates
            gi_targets *= gi_adders # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            tl.store(xg_targets_ptr, xg_targets.to(xg_targets_ptr.dtype.element_ty), mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            tl.store(gi_targets_ptr, gi_targets.to(gi_targets_ptr.dtype.element_ty), mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)

# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE': bs,
#             'BLOCK_SIZE_S': bs_s,
#             'BLOCK_SIZE_C': bs_c
#         }, num_warps=warps)
#         for bs in [16] #, 32, 64]
#         for bs_s in [16] #, 32, 64]
#         for bs_c in [16] #, 32, 64]
#         for warps in [2] #, 32, 64]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE': bs,
            'BLOCK_SIZE_S': bs_s,
            'BLOCK_SIZE_C': bs_c
        }, num_warps=warps)
        for bs in [16, 32, 64]
        for bs_s in [8, 16] #, 32, 64]
        for bs_c in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def cg2d_gxg_bwd_kernel(
    gi_ptr, go_ptr,
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr, T: tl.constexpr, nstages: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Similar structure to forward kernel with reversed indices
    pid = tl.program_id(axis=0)
    num_s_blocks = tl.cdiv(S, BLOCK_SIZE_S)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    b = pid // (num_s_blocks * num_c_blocks)
    rem = pid % (num_s_blocks * num_c_blocks)
    s_block = rem // num_c_blocks
    c_block = rem % num_c_blocks

    s_offs = tl.arange(0, BLOCK_SIZE_S)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    s_mask = s_offs < (S - s_block * BLOCK_SIZE_S)
    c_mask = c_offs < (C - c_block * BLOCK_SIZE_C)
    s_offs = s_block * BLOCK_SIZE_S + s_offs
    c_offs = c_block * BLOCK_SIZE_C + c_offs

    gi_base = gi_ptr + b * T * S
    go_base = go_ptr + b * T * S * C

    # nstages = tl.ceil(tl.log2(float(T))).to(tl.int32)
    offs = tl.arange(0, BLOCK_SIZE)
    
    for stage in tl.range(nstages): # CHANGE BACK TO tl.static_range() IN FINAL VERSION
        group_stride = 1 << stage
        for block_start in tl.range(0, T//2, BLOCK_SIZE):
            block_mask = offs < (T//2 - block_start)
            block_s_mask = block_mask[:, None] & s_mask[None, :]
            block_s_c_mask = block_mask[:, None, None] & s_mask[None, :, None] & c_mask[None, None, :]
            
            initial_indices = T - 1 - group_stride - ((offs + block_start) // group_stride) * group_stride * 2
            t_targets = initial_indices - ((offs + block_start) % group_stride)
            t_adders = initial_indices + 1

            go_targets_ptr = go_base + t_targets[:, None, None] * S * C + s_offs[None, :, None] * C + c_offs[None, None, :] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            go_adders_ptr = go_base + t_adders[:, None, None] * S * C + s_offs[None, :, None] * C + c_offs[None, None, :] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            gi_targets_ptr = gi_base + t_targets[:, None] * S + s_offs[None, :] # (BLOCK_SIZE, BLOCK_SIZE_S) 
            gi_adders_ptr = gi_base + t_adders[:, None] * S + s_offs[None, :] # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            # Load with block masking
            go_targets = tl.load(go_targets_ptr, mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            go_adders = tl.load(go_adders_ptr, mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            gi_targets = tl.load(gi_targets_ptr, mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)
            gi_adders = tl.load(gi_adders_ptr, mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            # Compute and store results
            go_targets += go_adders * gi_targets[:, :, None] # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            gi_targets *= gi_adders # (BLOCK_SIZE, BLOCK_SIZE_S)
            
            tl.store(go_targets_ptr, go_targets.to(go_targets_ptr.dtype.element_ty), mask=block_s_c_mask) # (BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
            tl.store(gi_targets_ptr, gi_targets.to(gi_targets_ptr.dtype.element_ty), mask=block_s_mask) # (BLOCK_SIZE, BLOCK_SIZE_S)

# @triton.autotune(
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE': bs,
#             'BLOCK_SIZE_S': bs_s,
#             'BLOCK_SIZE_C': bs_c
#         }, num_warps=warps)
#         for bs in [16] #, 32, 64]
#         for bs_s in [16] #, 32, 64]
#         for bs_c in [16] #, 32, 64]
#         for warps in [2] #, 4, 8]
#     ],
#     key=[]
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE': bs,
            'BLOCK_SIZE_S': bs_s,
            'BLOCK_SIZE_C': bs_c
        }, num_warps=warps)
        for bs in [16, 32, 64]
        for bs_s in [8, 16] #, 32, 64]
        for bs_c in [16, 32, 64]
        for warps in [2, 4, 8]
    ],
    key=[]
)
@triton.jit
def cg2d_ggi_bwd_kernel(
    go_ptr, y_ptr, grad_gi_ptr,
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr, T: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    num_t_blocks = tl.cdiv(T, BLOCK_SIZE)
    num_s_blocks = tl.cdiv(S, BLOCK_SIZE_S)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    t_block = pid // (num_s_blocks * num_c_blocks)
    rem = pid % (num_s_blocks * num_c_blocks)
    s_block = rem // num_c_blocks
    c_block = rem % num_c_blocks

    t_offs = tl.arange(0, BLOCK_SIZE)
    s_offs = tl.arange(0, BLOCK_SIZE_S)
    c_offs = tl.arange(0, BLOCK_SIZE_C)
    t_mask = t_offs < (T - t_block * BLOCK_SIZE)
    s_mask = s_offs < (S - s_block * BLOCK_SIZE_S)
    c_mask = c_offs < (C - c_block * BLOCK_SIZE_C)
    t_offs = t_block * BLOCK_SIZE + t_offs
    s_offs = s_block * BLOCK_SIZE_S + s_offs
    c_offs = c_block * BLOCK_SIZE_C + c_offs

    # Compute grad_gi
    # torch:
    # grad_gi = grad_output * y
    # grad_gi = grad_gi.sum(-1)
    grad_gi_base = grad_gi_ptr + b * T * S
    t_first_id = t_block * BLOCK_SIZE
    s_first_id = s_block * BLOCK_SIZE_S
    c_first_id = c_block * BLOCK_SIZE_C
    # We can use make_block_ptr since the blocks we need are contiguous
    go_block_ptr = tl.make_block_ptr(
        base=go_ptr,
        shape=(B, T, S, C),
        strides=(T * S * C, S * C, C, 1),
        offsets=(b, t_first_id, s_first_id, c_first_id),
        block_shape=(1, BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C),
        order=(0, 1, 2, 3)
    )
    go_block = tl.load(go_block_ptr) # (1, BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(B, T, S, C),
        strides=(T * S * C, S * C, C, 1),
        offsets=(b, t_first_id, s_first_id, c_first_id), # y is already shifted to the right by 1
        block_shape=(1, BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C),
        order=(0, 1, 2, 3)
    )
    y_block = tl.load(y_block_ptr) # (1, BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)

    grad_gi = go_block * y_block  # (1, BLOCK_SIZE, BLOCK_SIZE_S, BLOCK_SIZE_C)
    grad_gi = tl.sum(grad_gi, axis=-1)  # (1, BLOCK_SIZE, BLOCK_SIZE_S)

    # Need to use atomic add for accumulation between S blocks, so we also need to use manual pointer bc it's what atomic add accepts
    grad_gi_block_ptr = grad_gi_base + t_offs[:, None] * S + s_offs[None, :]
    grad_gi_mask = t_mask[:, None] & s_mask[None, :]
    tl.atomic_add(grad_gi_block_ptr[None, :], grad_gi, mask=grad_gi_mask[None, :])

class CumulativeGating2DTriton(torch.autograd.Function):
    # @torch.compiler.disable
    @staticmethod
    def forward(ctx, xg, gi):
        xg = xg.contiguous()
        gi = gi.contiguous()
        orig_gi = gi.clone()
        B, T, S, C = xg.shape
        
        # Calculate grid dimensions
        grid = lambda meta: (B * triton.cdiv(S, meta['BLOCK_SIZE_S']) * triton.cdiv(C, meta['BLOCK_SIZE_C']),)

        # Launch kernel
        nstages = math.ceil(math.log2(T))
        cg2d_fwd_kernel[grid](
            xg, gi,
            B, S, C, T, nstages,
        )
        
        ctx.save_for_backward(xg, orig_gi)
        return xg
    
    # @torch.compiler.disable
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        y, gi = ctx.saved_tensors
        B, T, S, C = y.shape
        
        # Calculate grid dimensions
        grid = lambda meta: (B * triton.cdiv(S, meta['BLOCK_SIZE_S']) * triton.cdiv(C, meta['BLOCK_SIZE_C']),)

        gi = torch.cat((gi[:, 1:], torch.ones_like(gi[:, -1:])), dim=1).contiguous()
        grad_xg = grad_output.clone()
        y = torch.cat((torch.zeros_like(y[:, :1]), y[:, :-1]), dim=1).contiguous()
        grad_gi = torch.zeros((B, T, S), device=gi.device).contiguous() # torch.zeros_like(gi)

        # Launch kernel
        nstages = math.ceil(math.log2(T))
        cg2d_gxg_bwd_kernel[grid](
            gi, grad_xg,
            B, S, C, T, nstages,
        )

        # Launch kernel
        grid = lambda meta: (B, triton.cdiv(T, meta['BLOCK_SIZE']) * triton.cdiv(S, meta['BLOCK_SIZE_S']) * triton.cdiv(C, meta['BLOCK_SIZE_C']))
        cg2d_ggi_bwd_kernel[grid](
            grad_xg, y, grad_gi,
            B, S, C, T,
        )

        return grad_xg, grad_gi.to(gi.dtype)
    
# Parallel Semi-Compressed Attention
def parallel_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    window_size: int,
    num_heads: int,
    alibi: torch.Tensor,
    mask: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
    checkpoint_level: Optional[int] = 2,
    offsets: Optional[torch.LongTensor] = None,
    head_first: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, HQ, T, K]` if `head_first=True` else `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        s (torch.Tensor):
            slot representations of shape `[B, H, T, M]` if `head_first=True` else `[B, T, H, M]`.
        g (torch.Tensor):
            Forget gates of shape `[B, H, T, M]` applied to keys.
            If not provided, this function is equivalent to vanilla ABC.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `[N, H, K, M]` and `[N, H, M, V]`.
            Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `2`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
        offsets (Optional[torch.LongTensor]):
            Offsets of shape `[N+1]` defining the bos/eos positions of `N` variable-length sequences in the batch.
            For example,
            if `offsets` is `[0, 1, 3, 6, 10, 15]`, there are `N=5` sequences with lengths 1, 2, 3, 4 and 5 respectively.
            If provided, the inputs are concatenated and the batch size `B` is expected to be 1.
            Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (Tuple[torch.Tensor]):
            Final state tuple having tensors of shape `[N, H, K, M]` and `[N, H, M, V]` if `output_final_state=True`.
            `None` otherwise.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gsa import fused_recurrent_gsa
        # inputs with equal lengths
        >>> B, T, H, K, V, M = 4, 2048, 4, 512, 512, 64
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> s = torch.randn(B, T, H, M, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, M, device='cuda'))
        >>> h0 = (torch.randn(B, H, K, M, device='cuda'), torch.randn(B, H, M, V, device='cuda'))
        >>> o, (hk, hv) = chunk_gsa(q, k, v, s, g,
                                    initial_state=h0,
                                    output_final_state=True,
                                    head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `offsets` is required
        >>> q, k, v, s, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, s, g))
        # for a batch with 4 sequences, offsets with 5 start/end positions are expected
        >>> offsets = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, (hk_var, hv_var) = chunk_gsa(q, k, v, s, g,
                                                initial_state=h0,
                                                output_final_state=True,
                                                offsets=offsets,
                                                head_first=False)
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert hk.allclose(hk_var)
        >>> assert hv.allclose(hv_var)
    """
    if offsets is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `offsets`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state[0].shape[0] != len(offsets) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(offsets) - 1} rather than {initial_state[0].shape[0]}.")
    assert checkpoint_level in [0, 1, 2]
    
    if scale is None:
        scale = q.shape[-1] ** -0.5

    BH, T, S = g.shape
    # Do semi-compressed attention
    sg = torch.einsum('bts,btc->btsc', g, s)
    gi = 1 - g 
    states = CumulativeGating2DTriton.apply(sg, gi) # states (B*H, T, S, C) at all time steps
    scores = AttendFoldedAllKeysTriton.apply(q, k, states, window_size) * scale # scores (B*H, T, S+W)
    # bring back to (B, H, T, S+W) to apply alibi with shape (H, T, S+W)
    scores = scores.view(-1, num_heads, T, S + window_size) + alibi[:, :T]
    scores = scores.masked_fill(mask[:T] == 0, float('-inf'))
    scores = torch.softmax(scores, dim=-1).view(BH, T, S + window_size)
    o = AccumulateFoldedAllValuesTriton.apply(scores, v, states, window_size) # outputs (B*H, T, C)

    final_state = None # TODO: fix for inference
    return o, final_state
