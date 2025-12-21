# -*- coding: utf-8 -*-

import torch
import triton

from fla.ops.attn.naive_relusoftpick import (
    naive_relu_softpick_attn,
    reference_naive_relu_softpick_attn,
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[64] + [128 * 2 ** i for i in range(0, 6)],
        line_arg="provider",
        line_vals=[
            "triton_fwd",
            "triton_fwdbwd",
            "ref_fwd",
            "ref_fwdbwd",
        ],
        line_names=[
            "triton_fwd",
            "triton_fwdbwd",
            "reference_fwd",
            "reference_fwdbwd",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "dotted"),
            ("red", "-"),
            ("red", "dotted"),
        ],
        ylabel="Execution Time (ms)",
        plot_name="relu_softpick_performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device

    dtype = torch.bfloat16
    requires_grad = True if "bwd" in provider else False
    B, H, D = 4, 8, 64

    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton_fwd":
        return triton.testing.do_bench(
            lambda: naive_relu_softpick_attn(q, k, v, head_first=False)[0],
            quantiles=quantiles,
        )
    if provider == "ref_fwd":
        return triton.testing.do_bench(
            lambda: reference_naive_relu_softpick_attn(q, k, v, head_first=False)[0],
            quantiles=quantiles,
        )
    if provider == "triton_fwdbwd":
        return triton.testing.do_bench(
            lambda: naive_relu_softpick_attn(q, k, v, head_first=False)[0].backward(do),
            quantiles=quantiles,
        )
    if provider == "ref_fwdbwd":
        return triton.testing.do_bench(
            lambda: reference_naive_relu_softpick_attn(q, k, v, head_first=False)[0].backward(do),
            quantiles=quantiles,
        )
    raise ValueError(f"Unknown provider {provider}")


def _peak_memory_bytes(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def print_memory_table():
    from fla.utils import device

    if device != "cuda":
        print("Skipping VRAM benchmark: CUDA device not available.")
        return

    def _fmt_bytes(num_bytes: int) -> str:
        """Human-friendly byte formatter."""
        units = ["B", "KB", "MB", "GB", "TB"]
        val = float(num_bytes)
        for unit in units:
            if val < 1024 or unit == units[-1]:
                return f"{val:.2f} {unit}"
            val /= 1024

    providers = [
        ("triton_fwd", lambda q, k, v, do: naive_relu_softpick_attn(q, k, v, head_first=False)[0]),
        ("ref_fwd", lambda q, k, v, do: reference_naive_relu_softpick_attn(q, k, v, head_first=False)[0]),
        ("triton_fwdbwd", lambda q, k, v, do: naive_relu_softpick_attn(q, k, v, head_first=False)[0].backward(do)),
        ("ref_fwdbwd", lambda q, k, v, do: reference_naive_relu_softpick_attn(q, k, v, head_first=False)[0].backward(do)),
    ]
    x_vals = [64] + [128 * 2 ** i for i in range(0, 6)]
    dtype = torch.bfloat16
    B, H, D = 4, 8, 64

    col_w = 16
    print("Peak VRAM (per run):")
    header = f"{'T':>6} " + " ".join(name.rjust(col_w) for name, _ in providers)
    print(header)
    for T in x_vals:
        q = torch.randn(B, T, H, D, device=device, requires_grad=True, dtype=dtype)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.ones_like(q, dtype=dtype)
        row = [f"{T:>6}"]
        for _, runner in providers:
            mem = _peak_memory_bytes(lambda: runner(q, k, v, do))
            row.append(_fmt_bytes(mem).rjust(col_w))
        print(" ".join(row))


if __name__ == "__main__":
    benchmark.run(print_data=True, save_path=".")
    print_memory_table()
