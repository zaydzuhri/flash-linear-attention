# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm
from fla.modules.activations import swish, sigmoid
from fla.modules.layernorm import rms_norm_linear
from fla.ops.scan import parallel_scan, naive_recurrent_scan

if TYPE_CHECKING:
    from fla.models.utils import Cache

def build_alibi_tensor_scan(head_num, seq_len, window_len, state_size):
    slopes = torch.tensor([2 ** (-8.0 * i / head_num) for i in range(head_num)])
    alibi = torch.zeros((head_num, seq_len, window_len))
    for i in range(seq_len):
        for j in range(window_len):
            if i < window_len:
                alibi[:, i, j] = slopes * (j - window_len + 1) if i > (window_len - j - 2) else 0
            else:
                alibi[:, i, j] = alibi[:, window_len-1, j]
    # Now concat a zeros tensor of size (head_num, seq_len, state_size) to the left of the above square tensor
    alibi = torch.cat((torch.zeros(head_num, seq_len, state_size), alibi), dim=2)
    return alibi # shape: (head_num, seq_len, state_size + window_size) or (H, T, S + W)

def scores_mask(T, W, S):
    # create lower right triangle mask (W, W)
    mask = torch.tril(torch.ones(W, W)).flip(1)
    # concat ones with size (T-W, W) in 0th dim
    mask = torch.cat((mask, torch.ones(T-W, W)), dim=0)
    # concat ones with size (T, S) in 1st dim
    mask = torch.cat((torch.ones(T, S), mask), dim=1)
    return mask # shape: (T, S + W)

class SemiCompressedAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'parallel',
        hidden_size: int = 1024,
        window_size: int = 512,
        state_size: int = 64,
        gate_act: str = 'softmax',
        max_position_embeddings: Optional[int] = 2048,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_first: bool = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: Optional[int] = None,
        scale: Optional[float] = 1.,
        **kwargs
    ) -> SemiCompressedAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.state_size = state_size
        self.gate_act = gate_act
        self.max_position_embeddings = max_position_embeddings
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.scale = scale

        self.norm_first = norm_first

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        self.s_proj = nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.state_size, bias=False)

        self.norm = RMSNorm(self.hidden_size, elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.apply(self._initialize_weights)

        self.register_buffer('alibi', build_alibi_tensor_scan(self.num_heads, self.max_position_embeddings, self.window_size, self.state_size))
        self.register_buffer('mask', scores_mask(self.max_position_embeddings, self.window_size, self.state_size))

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'naive' if past_key_values is not None else self.mode

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        s = self.s_proj(hidden_states)
        g = self.g_proj(hidden_states)
        
        if self.gate_act == 'softmax':
            g = F.softmax(g, dim=-1)
        elif self.gate_act == 'sigmoid':
            g = sigmoid(g)
        else:
            raise NotImplementedError(f"Gate activation `{self.gate_act}` is not supported.")

        # KV cache is updated before going into SCAN
        if past_key_values is not None:
            k, v = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=q.shape[2],
                # We actually don't want to crop to window for the initial prompt, only for subsequent autoregressive tokens
                cache_kwargs=dict(window_size=self.window_size) if q.shape[-2] == 1 else dict()
            )['attn_state']

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'parallel':
            # Split heads (but merge with batch dimension because kernels receive (B T C) shape)
            q = rearrange(q, 'b t (h c) -> (b h) t c', h=self.num_heads)
            k = rearrange(k, 'b t (h c) -> (b h) t c', h=self.num_kv_heads)
            v = rearrange(v, 'b t (h c) -> (b h) t c', h=self.num_kv_heads)
            s = rearrange(s, 'b t (h c) -> (b h) t c', h=self.num_kv_heads)
            g = rearrange(g, 'b t (h s) -> (b h) t s', h=self.num_kv_heads)
            o, recurrent_state = parallel_scan(
                q=q,
                k=k,
                v=v,
                s=s,
                g=g,
                window_size=self.window_size,
                num_heads=self.num_heads,
                alibi=self.alibi.to(q.device),
                mask=self.mask.to(q.device),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                head_first=False
            )
            o = rearrange(o, '(b h) t c -> b t (h c)', h=self.num_heads)
        elif mode == 'naive':
            # TODO: Implement naive recurrent SCAN for inference
            q = rearrange(q, 'b t (h c) -> b h t c', h=self.num_heads)
            k = rearrange(k, 'b t (h c) -> b h t c', h=self.num_kv_heads)
            v = rearrange(v, 'b t (h c) -> b h t c', h=self.num_kv_heads)
            s = rearrange(s, 'b t (h c) -> b h t c', h=self.num_kv_heads)
            g = rearrange(g, 'b t (h s) -> b h t s', h=self.num_kv_heads)
            o, recurrent_state = naive_recurrent_scan(
                q=q,
                k=k,
                v=v,
                s=s,
                g=g,
                window_size=self.window_size,
                alibi=self.alibi.to(q.device),
                mask=self.mask.to(q.device),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                head_first=False
            )
            o = rearrange(o, 'b h t c -> b t (h c)', h=self.num_heads)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update the recurrent state after SCAN
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx
            )

        o = rms_norm_linear(swish(o), self.norm.weight, self.norm.bias, self.o_proj.weight, self.o_proj.bias)
        return o, None, past_key_values
