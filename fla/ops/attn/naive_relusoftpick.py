import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


def relu_softpick_1(x, dim: int = -1, eps: float = 1e-8):
    # relu-softpick-1: relu(exp(x)-1) / sum(relu(exp(x)-1))
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    r_x_e_1 = torch.where(x.isfinite(), r_x_e_1, 0)
    return r_x_e_1 / (torch.sum(r_x_e_1, dim=dim, keepdim=True) + eps)


def relu_softpick_2(x, dim: int = -1, eps: float = 1e-8):
    # relu-softpick-2: relu(exp(x)) / sum(relu(exp(x))) == softmax
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_e = torch.exp(x - x_m)
    r_x_e = F.relu(x_e)
    r_x_e = torch.where(x.isfinite(), r_x_e, 0)
    return r_x_e / (torch.sum(r_x_e, dim=dim, keepdim=True) + eps)


def _naive_relu_softpick_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: int,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
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
        attn = relu_softpick_2(wei.float(), dim=-1).to(q.dtype)
    else:
        attn = relu_softpick_1(wei.float(), dim=-1).to(q.dtype)
    o = torch.matmul(attn, v)
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    return o, attn


def naive_relu_softpick_1_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
):
    return _naive_relu_softpick_attn(q, k, v, mode=1, scale=scale, cu_seqlens=cu_seqlens, head_first=head_first)


def naive_relu_softpick_2_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
):
    return _naive_relu_softpick_attn(q, k, v, mode=2, scale=scale, cu_seqlens=cu_seqlens, head_first=head_first)


def _reference_naive_relu_softpick_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: int,
    scale: Optional[float] = None,
    head_first: bool = False
):
    """Pure PyTorch reference for relu-softpick attention (causal)."""
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
        attn = relu_softpick_2(scores.float(), dim=-1).to(q.dtype)
    else:
        attn = relu_softpick_1(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)
    if not head_first:
        out = rearrange(out, 'b h t d -> b t h d')
    return out, attn


def reference_naive_relu_softpick_1_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    head_first: bool = False
):
    return _reference_naive_relu_softpick_attn(q, k, v, mode=1, scale=scale, head_first=head_first)


def reference_naive_relu_softpick_2_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    head_first: bool = False
):
    return _reference_naive_relu_softpick_attn(q, k, v, mode=2, scale=scale, head_first=head_first)
