# -*- coding: utf-8 -*-

import torch

torch._dynamo.config.recompile_limit = 128 # Set to a higher limit

import triton

from fla.ops.attn.gpt_oss_flex_attention_sink import HAS_FLEX_ATTENTION, flex_attention_with_sink


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in (torch.float16, torch.bfloat16):
        return 2e-2, 2e-2
    return 1e-3, 1e-3


class _DummyAttn:
    def __init__(self, sinks: torch.Tensor, scale: float):
        self.sinks = sinks
        self.num_key_value_groups = 1
        self.scaling = scale
        self.sliding_window = None
        self.training = True


def _naive_attention_with_sink(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attn_logits = torch.matmul(query, key.transpose(2, 3)) * scale
    q_len = query.shape[-2]
    k_len = key.shape[-2]
    causal_mask = torch.tril(
        torch.ones((q_len, k_len), device=attn_logits.device, dtype=torch.bool),
        diagonal=k_len - q_len,
    )
    attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))
    sink_logits = sinks.view(1, -1, 1, 1).expand(attn_logits.shape[0], -1, q_len, 1)
    combined_logits = torch.cat([attn_logits, sink_logits], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined_logits, dim=-1)
    probs = probs[..., :-1]
    out = torch.matmul(probs.to(value.dtype), value)
    return out.transpose(1, 2).contiguous()


def check_accuracy():
    torch.manual_seed(0)
    device = torch.device("cuda")
    b, t, h, d = 2, 64, 4, 32
    scale = d ** -0.5
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        q = torch.randn(b, h, t, d, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        sinks = torch.randn(h, device=device, dtype=dtype)
        attn = _DummyAttn(sinks=sinks, scale=scale)
        with torch.no_grad():
            out_kernel = flex_attention_with_sink(attn, q, k, v, attention_mask=None, scale=scale, compile=True)
            out_ref = _naive_attention_with_sink(q, k, v, sinks, scale=scale)
        rtol, atol = _tolerances(dtype)
        torch.testing.assert_close(out_kernel, out_ref, rtol=rtol, atol=atol)
    print("[accuracy] OK")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[64] + [128 * 2 ** i for i in range(0, 6)],
        line_arg="provider",
        line_vals=[
            "kernel_fwd",
            "kernel_fwdbwd",
            "naive_fwd",
            "naive_fwdbwd",
        ],
        line_names=[
            "kernel_fwd",
            "kernel_fwdbwd",
            "naive_fwd",
            "naive_fwdbwd",
        ],
        styles=[
            ("blue", "-"),
            ("blue", "dotted"),
            ("red", "-"),
            ("red", "dotted"),
        ],
        ylabel="Execution Time (ms)",
        plot_name="gpt_oss_flex_attention_sink",
        args={},
    )
)
def benchmark(T, provider):
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError("flex_attention not available in this build.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    requires_grad = True if "bwd" in provider else False
    b, h, d = 4, 8, 64
    scale = d ** -0.5

    q = torch.randn(b, h, T, d, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    sinks = torch.randn(h, device=device, requires_grad=requires_grad, dtype=dtype)
    attn = _DummyAttn(sinks=sinks, scale=scale)
    do = torch.ones((b, T, h, d), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "kernel_fwd":
        return triton.testing.do_bench(
            lambda: flex_attention_with_sink(attn, q, k, v, attention_mask=None, scale=scale, compile=True),
            quantiles=quantiles,
        )
    if provider == "naive_fwd":
        return triton.testing.do_bench(
            lambda: _naive_attention_with_sink(q, k, v, sinks, scale=scale),
            quantiles=quantiles,
        )
    if provider == "kernel_fwdbwd":
        return triton.testing.do_bench(
            lambda: flex_attention_with_sink(attn, q, k, v, attention_mask=None, scale=scale, compile=True).backward(do),
            quantiles=quantiles,
        )
    if provider == "naive_fwdbwd":
        return triton.testing.do_bench(
            lambda: _naive_attention_with_sink(q, k, v, sinks, scale=scale).backward(do),
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
    if not torch.cuda.is_available():
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

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    b, h, d = 4, 8, 64
    scale = d ** -0.5
    providers = [
        ("kernel_fwd", lambda q, k, v, do: flex_attention_with_sink(_DummyAttn(sinks, scale), q, k, v, attention_mask=None, scale=scale, compile=True)),
        ("naive_fwd", lambda q, k, v, do: _naive_attention_with_sink(q, k, v, sinks, scale)),
        ("kernel_fwdbwd", lambda q, k, v, do: flex_attention_with_sink(_DummyAttn(sinks, scale), q, k, v, attention_mask=None, scale=scale, compile=True).backward(do)),
        ("naive_fwdbwd", lambda q, k, v, do: _naive_attention_with_sink(q, k, v, sinks, scale).backward(do)),
    ]

    x_vals = [64] + [128 * 2 ** i for i in range(0, 6)]
    col_w = 16
    print("Peak VRAM (per run):")
    header = f"{'T':>6} " + " ".join(name.rjust(col_w) for name, _ in providers)
    print(header)
    for T in x_vals:
        q = torch.randn(b, h, T, d, device=device, requires_grad=True, dtype=dtype)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        sinks = torch.randn(h, device=device, requires_grad=True, dtype=dtype)
        do = torch.ones((b, T, h, d), device=device, dtype=dtype)
        row = [f"{T:>6}"]
        for _, runner in providers:
            mem = _peak_memory_bytes(lambda: runner(q, k, v, do))
            row.append(_fmt_bytes(mem).rjust(col_w))
        print(" ".join(row))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError("flex_attention is not available in this PyTorch build.")
    check_accuracy()
    benchmark.run(print_data=True, save_path=".")
    print_memory_table()
