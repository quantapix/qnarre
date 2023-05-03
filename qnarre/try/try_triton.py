# %%

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x1_ptr, x2_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x1 = tl.load(x1_ptr + offsets, mask=mask)
    x2 = tl.load(x2_ptr + offsets, mask=mask)
    y = x1 + x2
    tl.store(y_ptr + offsets, y, mask=mask)


# %%


def add(x1: torch.Tensor, x2: torch.Tensor):
    y = torch.empty_like(x1)
    assert x1.is_cuda and x2.is_cuda and y.is_cuda
    n = y.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x1, x2, y, n, BLOCK=1024)
    return y


# %%

torch.manual_seed(0)
size = 98432
x1 = torch.rand(size, device="cuda")
x2 = torch.rand(size, device="cuda")
y1 = x1 + x2
y2 = add(x1, x2)
print(y1)
print(y2)
print(f"The maximum difference between torch and triton is " f"{torch.max(torch.abs(y1 - y2))}")

# %%


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x1 = torch.rand(size, device="cuda", dtype=torch.float32)
    x2 = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x1 + x2, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x1, x2), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%

benchmark.run(print_data=True, show_plots=True)

# %%
