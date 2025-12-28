# -*- coding: utf-8 -*-

from typing import Optional

import torch
from einops import rearrange


def softmax_plus_one(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_e = torch.exp(x - x_m)
    return x_e / (torch.exp(-x_m) + torch.sum(x_e, dim=dim, keepdim=True))


def naive_softmax_plus_one_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5
    if not head_first:
        q, k, v = map(lambda x: rearrange(x, "b t h d -> b h t d"), (q, k, v))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = torch.tril(torch.ones(k_len, k_len, device=q.device))
    wei = torch.matmul(q, k.transpose(2, 3))
    wei = wei * scale
    wei = wei.masked_fill(mask[k_len - q_len:k_len, :k_len] == 0, float("-inf"))
    wei = softmax_plus_one(wei.float(), dim=-1).to(q.dtype)
    o = torch.matmul(wei, v)
    if not head_first:
        o = rearrange(o, "b h t d -> b t h d")
    return o, wei


def reference_naive_softmax_plus_one_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    head_first: bool = False,
):
    return naive_softmax_plus_one_attn(q, k, v, scale=scale, head_first=head_first)
