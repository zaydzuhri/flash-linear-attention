# -*- coding: utf-8 -*-

from typing import Optional

import torch

from .parallel import parallel_attn


def parallel_gated_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_score: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> torch.Tensor:
    o = parallel_attn(
        q,
        k,
        v,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )
    return o * torch.sigmoid(gate_score)
