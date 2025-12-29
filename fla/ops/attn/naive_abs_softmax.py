import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

def abs_softmax_1(x, dim: int = -1, eps: float = 1e-8):
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return a_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps)


def abs_softmax_2(x, dim: int = -1, eps: float = 1e-8):
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_e = torch.exp(x - x_m)
    a_x_e = torch.where(x.isfinite(), torch.abs(x_e), 0)
    return a_x_e / (torch.sum(a_x_e, dim=dim, keepdim=True) + eps)

def _naive_abs_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: int,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5
    if not head_first:
        q, k, v = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (q, k, v))

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    causal_mask = torch.tril(torch.ones(k_len, k_len, device=q.device, dtype=torch.bool))
    wei = torch.matmul(q, k.transpose(2, 3))
    wei = wei * scale
    wei = wei.masked_fill(causal_mask[k_len - q_len:k_len, :k_len] == 0, float('-inf'))
    if mode == 2:
        attn = abs_softmax_2(wei.float(), dim=-1).to(q.dtype)
    else:
        attn = abs_softmax_1(wei.float(), dim=-1).to(q.dtype)

    o = torch.matmul(attn, v)
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    return o, attn


def naive_abs_softmax_1_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    return _naive_abs_softmax_attn(q, k, v, mode=1, scale=scale, cu_seqlens=cu_seqlens, head_first=head_first)


def naive_abs_softmax_2_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    return _naive_abs_softmax_attn(q, k, v, mode=2, scale=scale, cu_seqlens=cu_seqlens, head_first=head_first)


def _reference_naive_abs_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: int,
    scale: Optional[float] = None,
    head_first: bool = False,
):
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5
    if not head_first:
        q, k, v = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (q, k, v))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = torch.tril(torch.ones(k_len, k_len, device=q.device, dtype=torch.bool))
    scores = torch.matmul(q, k.transpose(2, 3)) * scale
    scores = scores.masked_fill(mask[k_len - q_len:k_len, :k_len] == 0, float('-inf'))
    if mode == 2:
        attn = abs_softmax_2(scores.float(), dim=-1).to(q.dtype)
    else:
        attn = abs_softmax_1(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)
    if not head_first:
        out = rearrange(out, 'b h t d -> b t h d')
    return out, attn


def reference_naive_abs_softmax_1_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    head_first: bool = False,
):
    """Pure PyTorch reference for abs-softmax-1 attention (causal)."""
    return _reference_naive_abs_softmax_attn(q, k, v, mode=1, scale=scale, head_first=head_first)


def reference_naive_abs_softmax_2_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    head_first: bool = False,
):
    """Pure PyTorch reference for abs-softmax-2 attention (causal)."""
    return _reference_naive_abs_softmax_attn(q, k, v, mode=2, scale=scale, head_first=head_first)
