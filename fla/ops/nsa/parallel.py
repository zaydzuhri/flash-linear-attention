# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.common.utils import (prepare_chunk_indices, prepare_lens,
                                  prepare_token_indices)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(S):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [G, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [G, BS]
        b_p = tl.exp(b_s - b_m[:, None])
        # [G]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [G, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


@triton.jit
def parallel_nsa_kernel_mask(
    block_indices,
    block_mask,
    T: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr
):
    i_b, i_t, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    b_m = b_i * BS <= i_t

    if b_i < NS:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


@triton.jit
def parallel_nsa_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    offsets,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ*K
    do += (bos + i_t) * HQ*V
    dq += (i_v * B * T + bos + i_t) * HQ*K
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    block_indices += (bos + i_t) * H*S + i_h * S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(S):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BV, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # [G, BS]
        b_s = tl.dot(b_q, b_k)
        b_p = tl.exp(b_s - b_lse[:, None])
        b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

        # [G, BV] @ [BV, BS] -> [G, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # [G, BS] @ [BS, BK] -> [G, BK]
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    offsets,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * B*T*H + bos * H + i_h) * K, (T, K), (H*K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H*M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            p_do = tl.make_block_ptr(do + (bos + i) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)

            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (NV, T, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        offsets=offsets,
        token_indices=token_indices,
        H=H,
        HQ=HQ,
        G=G,
        T=T,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    offsets: torch.LongTensor,
    block_size: int
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if offsets is not None:
        NS = triton.cdiv(prepare_lens(offsets).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(B, T, H*S)](
        block_indices=block_indices,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS
    )
    return block_mask


def parallel_nsa_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    delta = parallel_nsa_bwd_preprocess(o, do)

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (NV, T, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        offsets=offsets,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV
    )
    dq = dq.sum(0)

    if offsets is not None:
        chunk_indices = prepare_chunk_indices(offsets, BS)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    # [B, T, H, M]
    block_mask = parallel_nsa_block_mask(block_indices, offsets, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        offsets=offsets,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV
    )
    dk = dk.sum(0)
    return dq, dk, dv


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            offsets=offsets,
            token_indices=token_indices)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_size=ctx.block_size,
            scale=ctx.scale,
            offsets=ctx.offsets,
            token_indices=ctx.token_indices)
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]` if `head_first=False` else `[B, HQ, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=False` else `[B, T, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]` if `head_first=False` else `[B, HQ, T, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v, indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, indices))
    o = ParallelNSAFunction.apply(q, k, v, indices, block_size, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o
