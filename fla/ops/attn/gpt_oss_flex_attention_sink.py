# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This file contains an adapted flex-attention sink kernel derived from Unsloth.
# Attribution required by the request: https://github.com/unslothai/unsloth

import functools
import math
import torch
import torch.nn.functional as F

FLEX_ATTENTION_KV_INCREMENT = 512


def _torch_compile(fn):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, fullgraph=False, dynamic=True)
        except TypeError:
            return torch.compile(fn, fullgraph=False)
    return fn


try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    from torch.nn.attention.flex_attention import AuxRequest, _score_mod_signature, _mask_mod_signature
    HAS_FLEX_ATTENTION = True
except Exception:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    _flex_attention = None
    _create_block_mask = None


if HAS_FLEX_ATTENTION:
    try:
        import torch._dynamo as _dynamo
    except Exception:
        pass
    vram_of_gpu = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        vram_of_gpu = min(
            torch.cuda.memory.mem_get_info(i)[-1] / 1024 / 1024 / 1024
            for i in range(torch.cuda.device_count())
        )
    kernel_options = None
    if vram_of_gpu is not None and vram_of_gpu <= 16:
        kernel_options = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
        }
    elif vram_of_gpu is not None and vram_of_gpu <= 24:
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        }
    if kernel_options is not None:
        _flex_attention = functools.partial(_flex_attention, kernel_options=kernel_options)

    uncompiled_flex_attention = _flex_attention
    flex_attention = _torch_compile(_flex_attention)
    _compiled_create_block_mask = _torch_compile(_create_block_mask)

    @functools.lru_cache
    def create_block_mask_cached(mask_mod, M, N, device="cuda"):
        return _create_block_mask(mask_mod, None, None, M, N, device=device)

    @functools.lru_cache
    def create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        return _create_block_mask(mask_mod, bsz, head, M, N, device=device)

    def compiled_create_block_mask_cached(mask_mod, M, N, device="cuda"):
        return _compiled_create_block_mask(mask_mod, None, None, M, N, device=device)

    def compiled_create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        return _compiled_create_block_mask(mask_mod, bsz, head, M, N, device=device)

    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
        return q_idx >= kv_idx

    def generate_causal_mask_with_padding(padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def _mask(batch_idx, head_idx, q_idx, kv_idx):
            q_start = q_idx >= padding_start_idx[batch_idx]
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return q_start & k_start & (q_idx >= kv_idx)

        _mask.__name__ = _mask.__doc__ = "causal_mask_with_left_padding"
        return _mask

    def generate_decoding_causal_mask_with_padding(padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def _mask(batch_idx, head_idx, q_idx, kv_idx):
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return k_start & (q_idx >= kv_idx)

        _mask.__name__ = _mask.__doc__ = "decoding_causal_mask_with_left_padding"
        return _mask

    @functools.lru_cache
    def generate_sliding_window_mask(window_size: int):
        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            windowed = q_idx - kv_idx < window_size
            return causal & windowed

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}"
        return sliding_window

    def generate_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            windowed = q_idx - kv_idx < window_size
            q_padded = q_idx >= padding_start_idx[batch_idx]
            k_padded = kv_idx >= padding_start_idx[batch_idx]
            return q_padded & k_padded & causal & windowed

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_with_left_padding_{window_size}"
        return sliding_window

    def generate_decoding_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        return generate_sliding_window_mask(window_size)

    def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: torch.tensor):
        def _score_mod(score, b, h, q, kv):
            return score_mod(score, b, h, q + _offset, kv)

        return _score_mod

    def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: torch.tensor):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + _offset, kv)

        return _mask_mod

    class FlexAttentionCache:
        __slots__ = (
            "offset",
            "offset_tensor",
            "mask_mod_with_offset",
            "block_mask",
            "mask_mod",
            "max_length",
            "block_size",
            "sliding_window",
            "block_mask_slice",
        )

        def __init__(self, key, mask_mod, sliding_window):
            bsz, heads_kv, qlen_kv, dim = key.shape
            if sliding_window is None:
                div, mod = divmod(qlen_kv, FLEX_ATTENTION_KV_INCREMENT)
                n = FLEX_ATTENTION_KV_INCREMENT * div + (FLEX_ATTENTION_KV_INCREMENT if mod != 0 else 0)
                self.offset = qlen_kv - 2
                if self.offset <= -2:
                    self.offset = -1
                self.sliding_window = None
            else:
                n = sliding_window
                self.offset = min(sliding_window, qlen_kv) - 2
                if self.offset <= -2:
                    self.offset = -1
                self.sliding_window = sliding_window - 1
            self.offset_tensor = torch.tensor(self.offset, device=key.device, dtype=torch.int32)
            self.block_mask = compiled_create_block_mask(mask_mod, bsz, heads_kv, n, n, device=key.device)
            self.mask_mod = mask_mod
            self.max_length = n
            self.block_size = self.block_mask.BLOCK_SIZE[0]
            self.mask_mod_with_offset = get_mask_mod_w_offset(self.mask_mod, self.offset_tensor)
            self.block_mask_slice = None

        def __call__(self, key):
            bsz, heads_kv, qlen_kv, dim = key.shape
            if (self.sliding_window is None) or (self.offset < self.sliding_window):
                self.offset += 1
                self.offset_tensor.add_(1)
            elif self.sliding_window is not None:
                return self.block_mask_slice
            if self.offset >= self.max_length:
                self.max_length += FLEX_ATTENTION_KV_INCREMENT
                self.block_mask = compiled_create_block_mask(
                    self.mask_mod, bsz, heads_kv, self.max_length, self.max_length, device=key.device
                )
                self.block_size = self.block_mask.BLOCK_SIZE[0]
            block_offset = self.offset // self.block_size
            block_mask_slice = self.block_mask[:, :, block_offset]
            block_mask_slice.mask_mod = self.mask_mod_with_offset
            block_mask_slice.seq_lengths = (1, qlen_kv)
            self.block_mask_slice = block_mask_slice
            return block_mask_slice

    def causal_mask_with_sink(batch, head, q_idx, kv_idx):
        causal = (q_idx + 1) >= kv_idx
        sink_first_column = kv_idx == 0
        return causal | sink_first_column

    @functools.lru_cache
    def generate_sliding_window_with_sink(window_size: int):
        def sliding_window(batch, head, q_idx, kv_idx):
            causal = (q_idx + 1) >= kv_idx
            windowed = (q_idx + 1) - kv_idx < window_size
            sink_first_column = kv_idx == 0
            return (causal & windowed) | sink_first_column

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}_sink"
        return sliding_window

    @functools.lru_cache
    def generate_sink_score_mod(sink_weights: torch.Tensor):
        def sink_score_mod(score, batch, head, q_idx, kv_idx):
            return torch.where(
                kv_idx == 0,
                sink_weights[head].to(score.dtype) + 0.0,
                score,
            )

        return sink_score_mod

    def old_flex_attention_with_sink(
        self_attn,
        query,
        key,
        value,
        attention_mask=None,
        scale=None,
        sliding_window=None,
        compile=True,
    ):
        if not self_attn.training:
            raise NotImplementedError("flex attention sink only supports training in this mode")
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        sink_weights = self_attn.sinks
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

        bsz, heads_q, qlen_q, dim = query.shape
        _, heads_kv, qlen_kv, _ = key.shape

        key_padded = torch.cat([key.new_zeros(bsz, heads_kv, 1, dim), key], dim=2)
        value_padded = torch.cat([value.new_zeros(bsz, heads_kv, 1, dim), value], dim=2)

        sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
        mask_mod = (
            generate_sliding_window_with_sink(sliding_window)
            if type(sliding_window) is int and sliding_window != 0
            else causal_mask_with_sink
        )
        score_mod = generate_sink_score_mod(sink_weights)
        block_mask = compiled_create_block_mask(mask_mod, qlen_q, qlen_kv + 1, device=key.device)
        attn_output = (flex_attention if compile else uncompiled_flex_attention)(
            query,
            key_padded,
            value_padded,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=enable_gqa,
            scale=scale,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def is_flex_attention_decoding(self_attn, query):
        if query.dim() == 4:
            bsz, heads_q, qlen_q, dim = query.shape
        else:
            bsz, qlen_q, dim = query.shape
        is_training = self_attn.training
        has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
        if is_training or (not is_training and (not has_flex_cache or qlen_q != 1)):
            return False
        return True

    def flex_attention_with_sink(
        self_attn,
        query,
        key,
        value,
        attention_mask=None,
        scale=None,
        sliding_window=None,
        compile=True,
        has_static_cache=True,
    ):
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        sink_weights = self_attn.sinks
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

        bsz, heads_q, qlen_q, dim = query.shape
        _, heads_kv, qlen_kv, _ = key.shape

        sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
        is_training = self_attn.training
        mask_mod = None
        block_mask = None
        has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
        if attention_mask is not None and has_static_cache:
            if is_training or (not is_training and (not has_flex_cache or qlen_q != 1)):
                if is_training:
                    if has_flex_cache:
                        del self_attn._flex_attention_cache
                else:
                    assert attention_mask is not None
                    assert attention_mask.dim() == 2, f"attention_mask has dim = {attention_mask.dim()}"
                    padding_start_idx = attention_mask.argmax(1).to(query.device)
                    do_padding = (
                        torch.arange(max(qlen_q, qlen_kv), device=query.device)
                        .repeat((bsz, 1))
                        .lt(padding_start_idx.unsqueeze(0).T)
                    )
                    query.transpose(2, 1)[do_padding[:, :qlen_q]] = 1
                    key.transpose(2, 1)[do_padding[:, :qlen_kv]] = -torch.inf
                    value.transpose(2, 1)[do_padding[:, :qlen_kv]] = 0
                    mask_mod = prefill_mask_mod = (
                        generate_sliding_window_mask_with_padding(sliding_window, padding_start_idx)
                        if type(sliding_window) is int and sliding_window != 0
                        else generate_causal_mask_with_padding(padding_start_idx)
                    )
                    decoding_mask_mod = (
                        generate_decoding_sliding_window_mask_with_padding(sliding_window, padding_start_idx)
                        if type(sliding_window) is int and sliding_window != 0
                        else generate_decoding_causal_mask_with_padding(padding_start_idx)
                    )
                    self_attn._flex_attention_cache = FlexAttentionCache(key, decoding_mask_mod, sliding_window)
            else:
                block_mask = self_attn._flex_attention_cache(key)
        if mask_mod is None:
            mask_mod = (
                generate_sliding_window_mask(sliding_window)
                if type(sliding_window) is int and sliding_window != 0
                else causal_mask
            )
        if block_mask is None:
            block_mask = compiled_create_block_mask(mask_mod, bsz, heads_q, qlen_q, qlen_kv, device=key.device)

        if compile:
            out = flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=None,
                enable_gqa=enable_gqa,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
        else:
            out = uncompiled_flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=None,
                enable_gqa=enable_gqa,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
        attn_output, aux = out
        logsumexp = aux.lse

        sink_scale = torch.sigmoid(logsumexp - sink_weights.unsqueeze(1))
        attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def flex_attention_with_sink_decoding(
        self_attn,
        query,
        key,
        value,
        scale=None,
    ):
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale
        block_mask = self_attn._flex_attention_cache(key)
        out = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=None,
            enable_gqa=enable_gqa,
            scale=scale,
            return_aux=AuxRequest(lse=True),
        )
        attn_output, aux = out
        return attn_output, aux.lse

    def flex_attention_add_sinks(
        self_attn,
        attn_output,
        logsumexp,
    ):
        logsumexp -= self_attn.sinks.unsqueeze(1)
        sink_scale = torch.sigmoid(logsumexp, out=logsumexp)
        attn_output *= sink_scale.unsqueeze(-1).to(attn_output.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def flash_attention_left_padded(
        self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        is_causal=True,
        window_size_left=None,
        dropout_p=0.0,
        scale=None,
    ):
        assert attention_mask.dtype in (torch.int32, torch.int64, torch.bool)
        device = query_states.device

        bsz, qlen = attention_mask.shape
        n_heads = self_attn.config.num_attention_heads
        n_kv_heads = getattr(self_attn.config, "num_key_value_heads", n_heads)
        head_dim = self_attn.head_dim

        bsz, heads_q, qlen_q, dim = query_states.shape
        _, heads_kv, qlen_kv, _ = key_states.shape

        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        seqlens = attention_mask.to(dtype=torch.int32, device=device).sum(dim=1)
        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item())

        flat_mask = attention_mask.reshape(-1).to(device=device)
        keep = flat_mask.nonzero(as_tuple=False).squeeze(-1)

        q_flat = q.reshape(bsz * qlen_q, n_heads, head_dim)
        k_flat = k.reshape(bsz * qlen_kv, n_kv_heads, head_dim)
        v_flat = v.reshape(bsz * qlen_kv, n_kv_heads, head_dim)

        q_unpad = q_flat.index_select(0, keep).contiguous()
        k_unpad = k_flat.index_select(0, keep).contiguous()
        v_unpad = v_flat.index_select(0, keep).contiguous()

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        kwargs = dict(scale=scale)
        if window_size_left is not None:
            kwargs["window_size_left"] = int(window_size_left)
            kwargs["window_size_right"] = 0

        attn_output, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query=q_unpad,
            key=k_unpad,
            value=v_unpad,
            cum_seq_q=cu_seqlens,
            cum_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
            return_debug_mask=False,
            **kwargs,
        )
        sink_scale = torch.sigmoid(logsumexp - self_attn.sinks.unsqueeze(1))
        attn_output = attn_output * sink_scale.unsqueeze(-1).transpose(0, 1).to(attn_output.dtype)

        out_flat = q_flat.new_zeros((bsz * qlen_q, n_heads, head_dim))
        out_flat[keep] = attn_output
        attn_output = out_flat.view(bsz, qlen_q, n_heads, head_dim)

        attn_output = attn_output.contiguous()
        return attn_output
else:
    def flex_attention_with_sink(*args, **kwargs):
        raise RuntimeError("flex_attention is not available in this PyTorch build")
