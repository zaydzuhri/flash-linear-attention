# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange


def naive_gated_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_score: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    """
    Elementwise gated attention using PyTorch SDPA.

    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of same shape format as q
        v: Value tensor of same shape format as q
        gate_score: Gate tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        scale: Scaling factor for attention scores (default: head_dim ** -0.5)
        cu_seqlens: Cumulative sequence lengths for packed sequences (unused in naive impl, kept for interface compatibility)
        head_first: If True, inputs are [batch, heads, seq, dim]; if False, [batch, seq, heads, dim]

    Returns:
        Tuple of (output, attention_weights) where output is the gated attention output
        and attention_weights are the attention weights before gating.

    Reference:
        Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
        https://arxiv.org/abs/2505.06708
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5

    # Convert to head_first format for computation
    if not head_first:
        q, k, v, gate_score = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (q, k, v, gate_score))

    q_len = q.shape[-2]
    k_len = k.shape[-2]

    # Create causal mask
    mask = torch.tril(torch.ones(k_len, k_len, device=q.device))
    wei = torch.matmul(q, k.transpose(2, 3))  # [batch, heads, q_len, k_len]
    wei = wei * scale
    wei = wei.masked_fill(mask[k_len - q_len:k_len, :k_len] == 0, float('-inf'))
    wei = torch.softmax(wei.float(), dim=-1).to(q.dtype)

    # Apply attention to values
    o = torch.matmul(wei, v)  # [batch, heads, q_len, head_dim]

    # Apply elementwise gating
    o = o * torch.sigmoid(gate_score)

    # Convert back to original format if needed
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')

    return o, wei
