# %%

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    x1_ptr, x2_ptr, y_ptr,
    M, N, K,
    stride_m, stride_k1,
    stride_k2, stride_n,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
    ACT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m = tl.cdiv(M, BLOCK_M)
    n = tl.cdiv(N, BLOCK_N)
    g = GROUP * n
    first = (pid // g) * GROUP
    size = min(m - first, GROUP)
    pid_m = first + (pid % size)
    pid_n = (pid % g) // size
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    x1s = x1_ptr + (offs_m[:, None] * stride_m + offs_k[None, :] * stride_k1)
    x2s = x2_ptr + (offs_k[:, None] * stride_k2 + offs_n[None, :] * stride_n)
    y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x1 = tl.load(x1s, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        x2 = tl.load(x2s, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        y += tl.dot(x1, x2)
        x1s += BLOCK_K * stride_k1
        x2s += BLOCK_K * stride_k2
    if ACT == "leaky_relu":
        y = leaky_relu(y)
    y = y.to(tl.float16)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ys = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
    tl.store(ys, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def leaky_relu(x):
    y = x + 1
    return tl.where(y >= 0, y, 0.01 * y)


# %%

def matmul(x1, x2, act=""):
    assert x1.shape[1] == x2.shape[0]
    assert x1.is_contiguous()
    assert x2.is_contiguous()
    M, K = x1.shape
    K, N = x2.shape
    y = torch.empty((M, N), device=x1.device, dtype=x1.dtype)
    grid = lambda x: (triton.cdiv(M, x['BLOCK_M']) * triton.cdiv(N, x['BLOCK_N']),)
    matmul_kernel[grid](
        x1, x2, y,
        M, N, K,
        x1.stride(0), x1.stride(1),
        x2.stride(0), x2.stride(1),
        y.stride(0), y.stride(1),
        ACT=act
    )
    return y


# %%

torch.manual_seed(0)
x1 = torch.randn((512, 512), device='cuda', dtype=torch.float16)
x2 = torch.randn((512, 512), device='cuda', dtype=torch.float16)
y_torch = torch.matmul(x1, x2)
y_triton = matmul(x1, x2)
print(f"torch={y_torch}")
print(f"triton={y_triton}")
if torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# %%

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['cublas', 'triton'],
        line_names=["cuBLAS", "Triton"],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    x1 = torch.randn((M, K), device='cuda', dtype=torch.float16)
    x2 = torch.randn((K, N), device='cuda', dtype=torch.float16)
    qs = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min, max = triton.testing.do_bench(lambda: torch.matmul(x1, x2), quantiles=qs)
    if provider == 'triton':
        ms, min, max = triton.testing.do_bench(lambda: matmul(x1, x2), quantiles=qs)
    y = lambda x: 2 * M * N * K * 1e-12 / (x * 1e-3)
    return y(ms), y(max), y(min)

# %%

benchmark.run(show_plots=True, print_data=True)

# %%
