# %%

import torch
import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    x2 = x - x_max[:, None]
    # read  MN elements ; write MN elements
    nr = torch.exp(x2)
    # read  MN elements ; write M  elements
    dr = nr.sum(dim=1)
    # read MN + M elements ; write MN elements
    y = nr / dr[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return y


# %%


@triton.jit
def softmax_kernel(y_ptr, x_ptr, x_stride, y_stride, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    x = x_ptr + pid * x_stride + offsets
    x = tl.load(x, mask=offsets < n_cols, other=-float("inf"))
    nr = tl.exp(x - tl.max(x, axis=0))
    dr = tl.sum(nr, axis=0)
    y = y_ptr + pid * y_stride + offsets
    tl.store(y, nr / dr, mask=offsets < n_cols)


# %%


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    n_warps = 4
    if BLOCK >= 2048:
        n_warps = 8
    if BLOCK >= 4096:
        n_warps = 16
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=n_warps,
        BLOCK=BLOCK,
    )
    return y


# %%

torch.manual_seed(0)
x = torch.randn(1823, 781, device="cuda")
y_torch = torch.softmax(x, axis=1)
y_triton = softmax(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# %%


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch-native", "torch-jit"],
        line_names=["Triton", "Torch (native)", "Torch (jit)"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=-1), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(show_plots=True, print_data=True)

# %%
