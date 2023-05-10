import flash_attn_cuda
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from einops import rearrange, repeat


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    # TMP,
    L,
    M,
    Y,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    start = tl.program_id(0)
    off = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_K)
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
    y = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    # ts = TMP + off * N_CTX + offs_m
    for i in range(0, (start + 1) * BLOCK_M, BLOCK_N):
        # i = tl.multiple_of(i, BLOCK_N)
        k = tl.load(ks + i * s_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)  # , tl.trans(k)) , trans_b=True)
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

        m2 = tl.max(qk, 1)
        p = tl.exp(qk - m2[:, None])
        m3 = tl.maximum(m, m2)
        alpha = tl.exp(m - m3)
        beta = tl.exp(m2 - m3)
        l2 = alpha * l + beta * tl.sum(p, 1)
        p_scale = beta / l2
        p = p * p_scale[:, None]
        y_scale = l / l2 * alpha
        tl.store(ts, y_scale)
        y_scale = tl.load(ts)  # BUG: have to store and immediately load
        y = y * y_scale[:, None]
        v = tl.load(vs + i * s_vk)
        p = p.to(v.dtype)
        y += tl.dot(p, v)
        l = l2
        m = m3

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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    o_zh = tl.program_id(0)
    o_z = o_zh // H
    o_h = o_zh % H
    s_qz, s_qh, s_qm, s_qk = Q.stride()
    _, _, s_kn, s_kk = K.stride()
    off = o_z * s_qz + o_h * s_qh
    offs_k = tl.arange(0, BLOCK_K)
    for i in range(0, num_block):
        i *= BLOCK_M
        offs_m = i + tl.arange(0, BLOCK_M)
        offs_n = i + tl.arange(0, BLOCK_M)
        qs = Q + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        ks = K + off + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk)
        vs = V + off + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk)
        dqs = DQ + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        dys = DY + off + (offs_m[:, None] * s_qm + offs_k[None, :] * s_qk)
        ds = D + o_zh * N_CTX
        ms = M + o_zh * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        k = tl.load(ks)
        v = tl.load(vs)
        for j in range(i, num_block * BLOCK_M, BLOCK_M):
            j += tl.arange(0, BLOCK_N)
            q = tl.load(qs)
            qk = tl.dot(q, tl.trans(k))  # , trans_b=True)
            qk = tl.where(j[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(ms + j)
            p = tl.exp(qk * sm_scale - m[:, None])
            dy = tl.load(dys)
            dv += tl.dot(
                tl.trans(p.to(Q.dtype.element_ty)), dy
            )  # p.to(dy.dtype), dy, trans_a=True)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - tl.load(ds + j)[:, None]
            dp += tl.dot(dy, tl.trans(v))  # , trans_b=True)
            ds = p * dp * sm_scale
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)  # ds.to(q.dtype), q, trans_a=True)
            dq = tl.load(dqs)  # , eviction_policy="evict_last")
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)  # ds.to(k.dtype), k)
            tl.store(dqs, dq)  # , eviction_policy="evict_last")
            qs += BLOCK_M * s_qm
            dqs += BLOCK_M * s_qm
            dys += BLOCK_M * s_qm
        tl.store(DK + off + (offs_n[:, None] * s_kn + offs_k[None, :] * s_kk), dk)
        tl.store(DV + off + (offs_n[:, None] * s_qm + offs_k[None, :] * s_qk), dv)


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        assert torch.cuda.get_device_capability()[0] > 7
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        y = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        tmp = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            # tmp,
            L,
            m,
            y,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_K=Lk,
            num_warps=num_warps,
            num_stages=2,  # =1,
        )

        ctx.save_for_backward(q, k, v, y, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK
        ctx.BLOCK_N = BLOCK
        ctx.BLOCK_K = Lk
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, y, l, m = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dy = dy.contiguous()
        dy_scaled = torch.empty_like(dy)
        delta = torch.empty_like(l)
        _bwd_prep[(ctx.grid[0] * ctx.grid[1],)](
            y,
            dy,
            l,
            dy_scaled,
            delta,
            BLOCK_M=ctx.BLOCK_M,
            D_HEAD=ctx.BLOCK_K,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            y,
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
            BLOCK_M=ctx.BLOCK_M,
            BLOCK_N=ctx.BLOCK_N,
            BLOCK_K=ctx.BLOCK_K,
            num_warps=8,
            num_stages=1,
        )
        return dq, dk, dv, None  # dq.to(q.dtype),


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


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, s = x.shape[0], x.shape[1:]
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=s.numel())
        ).reshape(-1, *s)

    @staticmethod
    def backward(ctx, x):
        (indices,) = ctx.saved_tensors
        assert x.ndim >= 2
        s = x.shape[1:]
        x = rearrange(x, "b ... -> b (...)")
        y = torch.zeros(
            [ctx.first_axis_dim, x.shape[1]],
            device=x.device,
            dtype=x.dtype,
        )
        y.scatter_(0, repeat(indices, "z -> z d", d=x.shape[1]), x)
        return y.reshape(ctx.first_axis_dim, *s), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert x.ndim >= 2
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        y[indices] = x
        # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
        return y

    @staticmethod
    def backward(ctx, x):
        (indices,) = ctx.saved_tensors
        y = x[indices]
        # y = torch.gather(x, 0, repeat(indices, 'z -> z d', d=x.shape[1]))
        return y, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, s = x.shape[0], x.shape[1:]
        second_dim = s.numel()
        y = x[indices]
        return y, x.detach()

    @staticmethod
    def backward(ctx, x, grad_residual):
        (indices,) = ctx.saved_tensors
        assert x.ndim >= 2
        s = x.shape[1:]
        assert grad_residual.shape[1:] == s
        y = grad_residual
        # y[indices] += x
        indices = indices.reshape(indices.shape[0], *((1,) * (x.ndim - 1)))
        indices = indices.expand_as(x)
        y.scatter_add_(0, indices, x)
        return y.reshape(ctx.first_axis_dim, *s), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(x, mask):
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(x, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(x, indices, batch, seqlen):
    dim = x.shape[-1]
    # y = torch.zeros((batch * seqlen), dim, device=x.device, dtype=x.dtype)
    # y[indices] = x
    y = index_put_first_axis(x, indices, batch * seqlen)
    return rearrange(y, "(b s) ... -> b s ...", b=batch)


def _get_block_size(device, head_dim, is_dropout):
    assert head_dim % 8 == 0 and head_dim <= 128
    return 256 if head_dim <= 64 else 128


def _flash_attn_forward(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    return_softmax,
    num_splits=0,
    generator=None,
):
    softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        return_softmax,
        num_splits,
        generator,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, rng_state, S_dmask


def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    rng_state=None,
    num_splits=0,
    generator=None,
):
    dout = dout.contiguous()
    _, _, _, softmax_d = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        num_splits,
        generator,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            torch.empty_like(qkv[:, 0]),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
            num_splits=1 if ctx.deterministic else 0,
        )
        return dqkv, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q,
            kv[:, 0],
            kv[:, 1],
            torch.empty_like(q),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, kv, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        _flash_attn_backward(
            dout,
            q,
            kv[:, 0],
            kv[:, 1],
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
            num_splits=1 if ctx.deterministic else 0,
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q,
            k,
            v,
            torch.empty_like(q),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
            num_splits=1 if ctx.deterministic else 0,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FlashAttnQKVPackedSplitFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen0,
        max_seqlen1,
        batch_size0,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        deterministic,
    ):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if dropout_p > 0:
            rng_state0 = torch.cuda.get_rng_state()
            generator1 = torch.Generator(device="cuda")
            rng_state1 = generator1.get_state()
        else:
            rng_state0, generator1, rng_state1 = None, None, None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out = torch.empty_like(qkv[:, 0])
        _, softmax_lse0, S_dmask0 = _flash_attn_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            cu_seqlens[: batch_size0 + 1],
            cu_seqlens[: batch_size0 + 1],
            max_seqlen0,
            max_seqlen0,
            dropout_p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _, softmax_lse1, S_dmask1 = _flash_attn_forward(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                out,
                cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:],
                max_seqlen1,
                max_seqlen1,
                dropout_p,
                softmax_scale,
                causal=causal,
                return_softmax=return_softmax,
                generator=generator1,
            )
        torch.cuda.current_stream().wait_stream(s)
        ctx.save_for_backward(
            qkv, out, softmax_lse0, softmax_lse1, cu_seqlens, rng_state0, rng_state1
        )
        ctx.dropout_p = dropout_p
        ctx.max_seqlen0 = max_seqlen0
        ctx.max_seqlen1 = max_seqlen1
        ctx.batch_size0 = batch_size0
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        if not return_softmax:
            return out
        else:
            max_seqlen_q = max(softmax_lse0.shape[2], softmax_lse1.shape[2])
            max_seqlen_k = max(S_dmask0.shape[3], S_dmask1.shape[3])
            softmax_lse = torch.cat(
                [
                    F.pad(softmax_lse0, (0, max_seqlen_q - softmax_lse0.shape[2])),
                    F.pad(softmax_lse1, (0, max_seqlen_q - softmax_lse1.shape[2])),
                ],
                dim=0,
            )
            return out, softmax_lse, S_dmask0, S_dmask1

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse0, softmax_lse1, cu_seqlens, rng_state0, rng_state1 = ctx.saved_tensors
        batch_size0 = ctx.batch_size0
        if rng_state0 is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state0)
        if rng_state1 is not None:
            generator1 = torch.Generator(device="cuda")
            generator1.set_state(rng_state1)
        else:
            generator1 = None
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout,
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            out,
            softmax_lse0,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens[: batch_size0 + 1],
            cu_seqlens[: batch_size0 + 1],
            ctx.max_seqlen0,
            ctx.max_seqlen0,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            num_splits=1 if ctx.deterministic else 0,
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _flash_attn_backward(
                dout,
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                out,
                softmax_lse1,
                dqkv[:, 0],
                dqkv[:, 1],
                dqkv[:, 2],
                cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:],
                ctx.max_seqlen1,
                ctx.max_seqlen1,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                generator=generator1,
                num_splits=1 if ctx.deterministic else 0,
            )
        torch.cuda.current_stream().wait_stream(s)
        if rng_state0 is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None, None, None, None


def flash_attn_unpadded_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    deterministic=False,
):
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
        deterministic,
    )


def flash_attn_unpadded_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    deterministic=False,
):
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
        deterministic,
    )


def flash_attn_unpadded_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    deterministic=False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
        deterministic,
    )


def flash_attn_unpadded_qkvpacked_split_func(
    qkv,
    cu_seqlens,
    max_seqlen0,
    max_seqlen1,
    batch_size0,
    dropout_p,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    deterministic=False,
):
    return FlashAttnQKVPackedSplitFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen0,
        max_seqlen1,
        batch_size0,
        dropout_p,
        softmax_scale,
        causal,
        return_attn_probs,
        deterministic,
    )


def flash_attn_func(
    qkv, cu_seqlens, dropout_p, max_s, softmax_scale=None, causal=False, return_attn_probs=False
):
    return flash_attn_unpadded_qkvpacked_func(
        qkv, cu_seqlens, max_s, dropout_p, softmax_scale, causal, return_attn_probs
    )


class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        qkv,
        key_padding_mask=None,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False,
    ):
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device
                )
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(
                        rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
                    ),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


class FlashMHA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)")), attn_weights
