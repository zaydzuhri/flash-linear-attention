# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.attn.gpt_oss_flex_attention_sink import HAS_FLEX_ATTENTION, flex_attention_with_sink

if TYPE_CHECKING:
    from fla.models.utils import Cache


class GptOssSinkAttention(nn.Module):
    """GPT-OSS attention sink variant using flex attention."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.0,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        attn_impl: str = "gpt_oss_flex_attention_sink",
        initializer_range: float = 0.006,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_key_value_groups = self.num_kv_groups
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.attn_impl = attn_impl
        self.initializer_range = initializer_range
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.sinks, mean=0.0, std=self.initializer_range)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError("flex_attention is not available in this PyTorch build.")
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if cu_seqlens is not None:
            raise NotImplementedError("gpt_oss_sink attention does not support cu_seqlens/varlen inputs.")

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if self.attn_impl != "gpt_oss_flex_attention_sink":
            raise ValueError(f"Unknown attention implementation: {self.attn_impl}")

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        o = flex_attention_with_sink(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            scale=self.scaling,
            sliding_window=self.window_size,
            compile=True,
            has_static_cache=True,
        )

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        attentions = None
        if output_attentions:
            attentions = None

        return o, attentions, past_key_values
