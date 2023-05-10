import pytest
import torch
import triton
import triton.language as tl

from flash_attention import flash_attn_func


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    M,
    Y,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start = tl.program_id(0)
    off = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    _, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    _, _, s_vk, _ = V.stride()
    _, s_yh, s_ym, s_yn = Y.stride()
    q = tl.load(Q + off * s_qh + offs_m[:, None] * s_qm + offs_d[None, :] * s_qk)
    ks = K + off * s_qh + offs_n[None, :] * s_kn + offs_d[:, None] * s_kk
    vs = V + off * s_qh + offs_n[:, None] * s_qm + offs_d[None, :] * s_qk
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    y = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for i in range(0, (start + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(ks + i * s_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (i + offs_n[None, :]), qk, float("-inf"))
        m2 = tl.maximum(tl.max(qk, 1), m)
        l *= tl.exp(m - m2)
        p = tl.exp(qk - m2[:, None])
        l2 = tl.sum(p, 1) + l
        l3 = 1.0 / l2
        p *= l3[:, None]
        y *= (l * l3)[:, None]
        v = tl.load(vs + i * s_vk)
        p = p.to(Q.dtype.element_ty)
        y += tl.dot(p, v)
        l = l2
        m = m2
    tl.store(L + off * N_CTX + offs_m, l)
    tl.store(M + off * N_CTX + offs_m, m)
    tl.store(Y + off * s_yh + offs_m[:, None] * s_ym + offs_d[None, :] * s_yn, y)


@triton.jit
def _bwd_prep(
    Y,
    DY,
    L,
    NewDY,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    y = tl.load(Y + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    dy = tl.load(DY + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    dy = dy / denom[:, None]
    delta = tl.sum(y * dy, axis=1)
    tl.store(NewDY + off_m[:, None] * D_HEAD + off_n[None, :], dy)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Y,
    DY,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off = tl.program_id(0)
    off_z = off // H
    off_h = off % H
    s_qz, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    Q += off_z * s_qz + off_h * s_qh
    K += off_z * s_qz + off_h * s_qh
    V += off_z * s_qz + off_h * s_qh
    DQ += off_z * s_qz + off_h * s_qh
    DK += off_z * s_qz + off_h * s_qh
    DV += off_z * s_qz + off_h * s_qh
    DY += off_z * s_qz + off_h * s_qh
    for i in range(0, num_block):
        lo = i * BLOCK_M
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = i * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        qs = Q + (offs_qm[:, None] * s_qm + offs_k[None, :] * s_qk)
        ks = K + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk)
        vs = V + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk)
        dqs = DQ + (offs_qm[:, None] * s_qm + offs_k[None, :] * s_qk)
        dys = DY + (offs_qm[:, None] * s_qm + offs_k[None, :] * s_qk)
        ds = D + off * N_CTX
        ms = M + off * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(ks)
        v = tl.load(vs)
        for j in range(lo, num_block * BLOCK_M, BLOCK_M):
            j += offs_m
            q = tl.load(qs)
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(j[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(ms + j)
            p = tl.exp(qk * sm_scale - m[:, None])
            dy = tl.load(dys)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), dy)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - tl.load(ds + j)[:, None]
            dp += tl.dot(dy, tl.trans(v))
            ds = p * dp * sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            dq = tl.load(dqs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dqs, dq)
            dqs += BLOCK_M * s_qm
            qs += BLOCK_M * s_qm
            dys += BLOCK_M * s_qm
        tl.store(DK + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk), dk)
        tl.store(DV + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk), dv)


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        y = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            m,
            y,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, y, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return y

    @staticmethod
    def backward(ctx, dy):
        BLOCK = 128
        q, k, v, o, l, m = ctx.saved_tensors
        dy = dy.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dy_scaled = torch.empty_like(dy)
        delta = torch.empty_like(l)
        _bwd_prep[(ctx.grid[0] * ctx.grid[1],)](
            o,
            dy,
            l,
            dy_scaled,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            dy_scaled,
            dq,
            dk,
            dv,
            l,
            m,
            delta,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv, None


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(4, 48, 1024, 64)])
def test_op(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.1, std=0.2)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.4, std=0.2)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.3, std=0.2)
        .requires_grad_()
    )
    sm_scale = 0.2
    dy = torch.randn_like(q)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    y_ref = torch.matmul(p, v)
    y_ref.backward(dy)
    dv_ref, v.grad = v.grad.clone(), None
    dk_ref, k.grad = k.grad.clone(), None
    dq_ref, q.grad = q.grad.clone(), None
    y_triton = attention(q, k, v, sm_scale)
    y_triton.backward(dy)
    dv_triton, v.grad = v.grad.clone(), None
    dk_triton, k.grad = k.grad.clone(), None
    dq_triton, q.grad = q.grad.clone(), None
    assert torch.allclose(y_ref, y_triton, atol=1e-2, rtol=0)
    assert torch.allclose(dv_ref, dv_triton, atol=1e-2, rtol=0)
    assert torch.allclose(dk_ref, dk_triton, atol=1e-2, rtol=0)
    assert torch.allclose(dq_ref, dq_triton, atol=1e-2, rtol=0)


BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 14)],
            line_arg="provider",
            line_vals=["triton", "flash"],
            line_names=["Triton", "Flash"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "D_HEAD": D_HEAD,
                "dtype": torch.float16,
                "mode": mode,
            },
        )
        for mode in ["fwd", "bwd"]
    ]
)
def benchmark(BATCH, H, N_CTX, D_HEAD, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        sm_scale = 1.3
        f = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            y = f()
            dy = torch.randn_like(y)
            f = lambda: y.backward(dy, retain_graph=True)
    else:
        assert provider == "flash"
        lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        qkv = torch.randn(
            (BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True
        )
        f = lambda: flash_attn_func(qkv, cu_seqlens, 0.0, N_CTX, causal=True)
        if mode == "bwd":
            y = f()
            dy = torch.randn_like(y)
            f = lambda: y.backward(dy, retain_graph=True)
    ms = triton.testing.do_bench(f, warmup=warmup, rep=rep)
    return ms


# only works on post-Ampere GPUs right now
benchmark.run(save_path=".", print_data=True)
