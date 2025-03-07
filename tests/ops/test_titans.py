# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from ops.titans.fused_chunk import fused_chunk_titans_linear
from ops.titans.naive import chunk_titans_linear_ref


def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


def initialize_chunked_param(B, H, T, BT, dtype=torch.float32):
    # Calculate number of complete chunks and remaining elements
    num_complete_chunks = T // BT
    remainder = T % BT

    # Initialize for complete chunks
    if num_complete_chunks > 0:
        theta_chunks = torch.rand(B, H, num_complete_chunks, 1, dtype=dtype)
        theta_main = theta_chunks.repeat_interleave(
            BT, dim=2
        )  # Shape: (B, H, num_complete_chunks*BT, 1)
    else:
        theta_main = torch.empty(B, H, 0, 1, dtype=dtype)

    # Handle remaining elements if any
    if remainder > 0:
        theta_remainder = torch.rand(B, H, 1, 1, dtype=dtype)
        theta_remainder = theta_remainder.repeat_interleave(
            remainder, dim=2
        )  # Shape: (B, H, remainder, 1)

        # Concatenate main chunks with remainder
        theta = torch.cat([theta_main, theta_remainder], dim=2)
    else:
        theta = theta_main

    return theta


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("head_first", [True, False])
def test_naive_chunk_fwd(
    B: int, T: int, H: int, D: int, dtype: torch.dtype, scale: float, head_first: bool
):
    BT = 64
    # set seed
    torch.manual_seed(1)
    # we don't use such initialization in the original code
    # theta = initialize_chunked_param(B, H, T, BT, dtype)
    # alpha = initialize_chunked_param(B, H, T, BT, dtype)
    # eta = initialize_chunked_param(B, H, T, BT, dtype)
    theta = torch.rand(B, H, T, 1, dtype=dtype)
    alpha = torch.rand(B, H, T, 1, dtype=dtype)
    eta = torch.rand(B, H, T, 1, dtype=dtype)

    # titans normalize queries and keys using ℓ2-normalization
    q = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, dtype=dtype)
    w = torch.randn(H, D, dtype=dtype)
    b = torch.randn(H, D, dtype=dtype)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    if not head_first:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        theta = theta.permute(0, 2, 1, 3)
        alpha = alpha.permute(0, 2, 1, 3)
        eta = eta.permute(0, 2, 1, 3)
    q, k, v, w, b, theta, alpha, eta = map(
        lambda x: x.cuda().requires_grad_(False), (q, k, v, w, b, theta, alpha, eta)
    )
    # in titans paper, h0 is not learnable
    h0 = h0.cuda()

    ref_naive, ref_ht_naive = chunk_titans_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        chunk_size=BT,
        initial_state=h0.clone(),
        head_first=head_first,
        use_chunk=False,
    )
    ref, ref_ht = chunk_titans_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        chunk_size=BT,
        initial_state=h0.clone(),
        head_first=head_first,
        use_chunk=True,
    )

    assert_close(" o", ref, ref_naive, 0.006)
    assert_close("ht", ref_ht, ref_ht_naive, 0.005)


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [64])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [32])
@pytest.mark.parametrize("scale", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("head_first", [True, False])
def test_fused_chunk_fwd(
    B: int, T: int, H: int, D: int, dtype: torch.dtype, scale: float, head_first: bool
):
    BT = 1
    # set seed
    torch.manual_seed(1)
    # we don't use such initialization in the original code
    # theta = initialize_chunked_param(B, H, T, BT, dtype)
    # alpha = initialize_chunked_param(B, H, T, BT, dtype)
    # eta = initialize_chunked_param(B, H, T, BT, dtype)
    theta = torch.rand(B, H, T, 1, dtype=dtype)
    alpha = torch.rand(B, H, T, 1, dtype=dtype)
    eta = torch.rand(B, H, T, 1, dtype=dtype)

    if head_first:
        # titans normalize queries and keys using ℓ2-normalization
        q = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        k = F.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn(B, H, T, D, dtype=dtype)
        w = torch.randn(H, D, dtype=dtype)
        b = torch.randn(H, D, dtype=dtype)
        h0 = torch.randn(B, H, D, D, dtype=dtype)
    else:
        q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(
            dtype
        )
        v = torch.randn(B, T, H, D, dtype=dtype)
        w = torch.randn(H, D, dtype=dtype)
        b = torch.randn(H, D, dtype=dtype)
        h0 = torch.randn(B, H, D, D, dtype=dtype)
        # we need to reshape here because head_first is True
        theta = theta.permute(0, 2, 1, 3)
        alpha = alpha.permute(0, 2, 1, 3)
        eta = eta.permute(0, 2, 1, 3)
    q, k, v, w, b, theta, alpha, eta = map(
        lambda x: x.cuda().requires_grad_(False), (q, k, v, w, b, theta, alpha, eta)
    )
    # in titans paper, h0 is not learnable
    h0 = h0.cuda()

    ref_naive, ref_ht_naive = fused_chunk_titans_linear(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        chunk_size=BT,
        initial_state=h0.clone(),
        head_first=head_first,
    )
    ref, ref_ht = chunk_titans_linear_ref(
        q.clone(),
        k.clone(),
        v.clone(),
        w.clone(),
        b.clone(),
        theta.clone(),
        alpha.clone(),
        eta.clone(),
        output_final_state=True,
        chunk_size=BT,
        initial_state=h0.clone(),
        head_first=head_first,
        use_chunk=True,
    )

    # assert_close(" o", ref, ref_naive, 0.006)
    assert_close("ht", ref_ht, ref_ht_naive, 0.005)
