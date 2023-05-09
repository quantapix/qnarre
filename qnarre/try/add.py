# %%

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x1_ptr, x2_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x1 = tl.load(x1_ptr + offs, mask=mask)
    x2 = tl.load(x2_ptr + offs, mask=mask)
    y = x1 + x2
    tl.store(y_ptr + offs, y, mask=mask)


# %%


def add(x1: torch.Tensor, x2: torch.Tensor):
    y = torch.empty_like(x1)
    assert x1.is_cuda and x2.is_cuda and y.is_cuda
    n = y.numel()
    grid = lambda x: (triton.cdiv(n, x["BLOCK"]),)
    add_kernel[grid](x1, x2, y, n, BLOCK=1024)
    return y


# %%

torch.manual_seed(0)
size = 98432
x1 = torch.rand(size, device="cuda")
x2 = torch.rand(size, device="cuda")
y_torch = x1 + x2
y_triton = add(x1, x2)
print(f"torch={y_torch}")
print(f"triton={y_triton}")
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(y_torch - y_triton))}"
)

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
    qs = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min, max = triton.testing.do_bench(lambda: x1 + x2, quantiles=qs)
    if provider == "triton":
        ms, min, max = triton.testing.do_bench(lambda: add(x1, x2), quantiles=qs)
    y = lambda ms: 12 * size / ms * 1e-6
    return y(ms), y(max), y(min)


# %%

benchmark.run(print_data=True, show_plots=True)

# %%
