# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from einops import repeat


def naive_recurrent_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    window_size: int,
    alibi: torch.Tensor,
    mask: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False,
    head_first: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, T, C, W, S = *q.shape, window_size, g.shape[-1]
    Tk = k.shape[2]

    if scale is None:
        scale = C ** -0.5

    sg = torch.einsum("bhts, bhtc -> bhtsc", g, s) # (B, H, T, S, C)
    gi = 1 - g # (B, H, T, S)
    prev_state = initial_state if initial_state is not None else torch.zeros((B, H, S, C), device=q.device, dtype=q.dtype)
    outs = []

    for t in range(T): # this will only loop more than once in the first prefill pass
        prev_state = torch.einsum("bhs, bhsc -> bhsc", gi[:, :, t], prev_state) # (B, H, S, C)
        state = prev_state + sg[:, :, t] # (B, H, S, C)

        if T == Tk: # first prefill pass
            k_window = k[:, :, max(0, t - W):t] # (B, H, W, C)
            v_window = v[:, :, max(0, t - W):t]
        else: # subsequent passes
            k_window = k
            v_window = v
        Tw = k_window.shape[-2]
        # if the window crop is less than W, pad with zeros on the left
        if Tw < W:
            k_window = torch.cat((torch.zeros((B, H, W - Tw, C), device=k.device, dtype=k.dtype), k_window), dim=2)
            v_window = torch.cat((torch.zeros((B, H, W - Tw, C), device=v.device, dtype=v.dtype), v_window), dim=2)
        all_keys = torch.cat((state, k_window), dim=2) # (B, H, S, C) + (B, H, W, C) -> (B, H, S+W, C)
        all_values = torch.cat((state, v_window), dim=2) # (B, H, S, C) + (B, H, W, C) -> (B, H, S+W, C)
        scores = torch.einsum("bhc, bhxc -> bhx", q[:, :, 0], all_keys) * scale # (B, H, C) @ (B, H, S+W, C) -> (B, H, S+W)
        scores += alibi[:, Tw] # (B, H, S+W)
        scores = scores.masked_fill(mask[Tw] == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhx, bhxc -> bhc", scores, all_values)
        outs.append(out)

        prev_state = state
    final_state = prev_state
    outs = torch.stack(outs, dim=2)

    return outs, final_state
