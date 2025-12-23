# -*- coding: utf-8 -*-

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def parallel_gated_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_score: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if not head_first:
        q, k, v, gate_score = map(
            lambda x: rearrange(x, "b t h d -> b h t d"),
            (q, k, v, gate_score),
        )

    try:
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
    except TypeError:
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

    o = o * torch.sigmoid(gate_score)

    if not head_first:
        o = rearrange(o, "b h t d -> b t h d")

    return o
