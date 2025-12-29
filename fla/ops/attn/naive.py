import torch
from typing import Optional
from einops import rearrange

def naive_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    if not head_first:
        q, k, v = map(lambda x: rearrange(x, 'b t h d -> b h t d'), (q, k, v))
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = torch.tril(torch.ones(k_len, k_len, device=q.device))
    wei = torch.matmul(q, k.transpose(2, 3)) # shape: (batch_size, num_heads, q_len, k_len)
    wei = wei * scale
    wei = wei.masked_fill(mask[k_len-q_len:k_len, :k_len] == 0, float('-inf'))
    wei = torch.softmax(wei.float(), dim=-1).to(q.dtype)
    o = torch.matmul(wei, v) # shape: (batch_size, num_heads, q_len, head_dim)
    if not head_first:
        o = rearrange(o, 'b h t d -> b t h d')
    return o, wei