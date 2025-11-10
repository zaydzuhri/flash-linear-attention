# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops import parallel_attn, parallel_rectified_attn, parallel_softpick_attn, naive_attn, naive_rectified_attn, naive_softpick_attn

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        attn_impl: str = "flash_attn",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.attn_impl = attn_impl

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if "scaled" in self.attn_impl:
            self.s = nn.Parameter(torch.empty(self.num_heads, 1))
            self.register_buffer("logn", torch.log(torch.arange(2, self.max_position_embeddings*4+2, dtype=self.s.dtype)[:, None, None]))

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def reset_parameters(self):
        if "scaled" in self.attn_impl:
            nn.init.constant_(self.s, 0.3)
            self.logn.copy_(torch.log(torch.arange(2, self.max_position_embeddings*4+2, dtype=self.s.dtype)[:, None, None]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
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

        # if flash_attn_func is None:
        #     raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        if "scaled" in self.attn_impl:
            k_len = k.shape[1]
            q = q * self.s.to(q.dtype) * self.logn[k_len-q_len:k_len].to(q.dtype)

        # Contains at least one padding token in the sequence
        if self.attn_impl == "flash_attn":
            if attention_mask is not None:
                q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
                o = pad_input(o, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                ).unsqueeze(0)
            else:
                o = flash_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
        elif self.attn_impl == "parallel_attn":
            o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_scaled_attn":
            o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_rectified_attn":
            o = parallel_rectified_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_softpick_attn":
            o = parallel_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_scaled_softpick_attn":
            o = parallel_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_attn":
            o, attentions = naive_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_scaled_attn":
            o, attentions = naive_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_rectified_attn":
            o, attentions = naive_rectified_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_softpick_attn":
            o, attentions = naive_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_scaled_softpick_attn":
            o, attentions = naive_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        else:
            raise ValueError(f"Unknown attention implementation: {self.attn_impl}")

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions or "parallel" in self.attn_impl or "flash" in self.attn_impl:
            attentions = None

        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

class StochasticSoftpickAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        attn_impl: str = "flash_attn",
        stochastic_p: float = 0.5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.attn_impl = attn_impl
        self.stochastic_value = stochastic_p
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if "scaled" in self.attn_impl:
            self.s = nn.Parameter(torch.empty(self.num_heads, 1))
            self.register_buffer("logn", torch.log(torch.arange(2, self.max_position_embeddings*4+2, dtype=self.s.dtype)[:, None, None]))

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def reset_parameters(self):
        if "scaled" in self.attn_impl:
            nn.init.constant_(self.s, 0.3)
            self.logn.copy_(torch.log(torch.arange(2, self.max_position_embeddings*4+2, dtype=self.s.dtype)[:, None, None]))


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
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

        # if flash_attn_func is None:
        #     raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

        if "scaled" in self.attn_impl:
            k_len = k.shape[1]
            q = q * self.s.to(q.dtype) * self.logn[k_len-q_len:k_len].to(q.dtype)

        # Contains at least one padding token in the sequence

        p = torch.rand(1, device=q.device)
        stochastic_p = torch.tensor(self.stochastic_value, dtype=torch.float32, device=q.device)
        cond = torch.where(p < stochastic_p, torch.tensor(1, dtype=torch.bool, device=q.device), torch.tensor(0, dtype=torch.bool, device=q.device))
        if self.attn_impl == "flash_attn":
            if attention_mask is not None:
                q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q, k, v, attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
                o = pad_input(o, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o = flash_attn_varlen_func(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                ).unsqueeze(0)
            else:
                o = flash_attn_func(
                    q, k, v,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
        
        elif self.attn_impl == "parallel_attn":
            if cond:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_scaled_attn":
            if cond:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_rectified_attn":
            if cond:
                o = parallel_rectified_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_softpick_attn":
            if cond:
                o = parallel_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "parallel_scaled_softpick_attn":
            if cond:
                o = parallel_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_attn":
            if cond:
                o, attentions = naive_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_scaled_attn":
            if cond:
                o, attentions = naive_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_rectified_attn":
            if cond:
                o, attentions = naive_rectified_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_softpick_attn":
            if cond:
                o, attentions = naive_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        elif self.attn_impl == "naive_scaled_softpick_attn":
            if cond:
                o, attentions = naive_softpick_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
            else:
                o = parallel_attn(q, k, v, scale=self.head_dim**-0.5, cu_seqlens=cu_seqlens)
        else:
            raise ValueError(f"Unknown attention implementation: {self.attn_impl}")

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions or "parallel" in self.attn_impl or "flash" in self.attn_impl:
            attentions = None

        return o, attentions, past_key_values

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)
