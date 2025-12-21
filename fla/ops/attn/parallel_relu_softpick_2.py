# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from fla.ops.attn.parallel import parallel_attn_fwd, parallel_attn_bwd
from fla.ops.common.utils import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
import triton


@torch.compile
class ParallelReluSoftpick2AttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, scale, offsets):
        ctx.dtype = q.dtype

        chunk_size = min(128, max(16, triton.next_power_of_2(q.shape[1])))
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None

        o, lse = parallel_attn_fwd(
            q=q,
            k=k,
            v=v,
            scale=scale,
            chunk_size=chunk_size,
            offsets=offsets,
            indices=indices,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.chunk_size = chunk_size
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            scale=ctx.scale,
            chunk_size=ctx.chunk_size,
            offsets=ctx.offsets,
            indices=ctx.indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None


def parallel_relu_softpick_2_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> torch.Tensor:
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v = map(lambda x: rearrange(x, "b h t d -> b t h d"), (q, k, v))
    o = ParallelReluSoftpick2AttentionFunction.apply(q, k, v, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, "b t h d -> b h t d")
    return o
