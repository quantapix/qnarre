import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    TMP, L, M,
    Y,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_yz, stride_yh, stride_ym, stride_yn,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_d = tl.arange(0, BLOCK_DMODEL)
    start = tl.program_id(0)
    offs_m = start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    off = tl.program_id(1)
    off_q = off * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_y = off * stride_yh + offs_m[:, None] * stride_ym + offs_d[None, :] * stride_yn
    q = tl.load(Q + off_q)
    ks = K + off_k
    vs = V + off_v
    ts = TMP + off * N_CTX + offs_m
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    y = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for i in range(0, (start + 1) * BLOCK_M, BLOCK_N):
        i = tl.multiple_of(i, BLOCK_N)
        k = tl.load(ks + i * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        qk += tl.where(offs_m[:, None] >= (i + offs_n[None, :]), 0, float("-inf"))
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
        v = tl.load(vs + i * stride_vk)
        p = p.to(v.dtype)
        y += tl.dot(p, v)
        l = l2
        m = m3
    tl.store(L + off * N_CTX + offs_m, l)
    tl.store(M + off * N_CTX + offs_m, m)
    tl.store(Y + off_y, y)


@triton.jit
def _bwd_preprocess(
    Y, DY, L,
    NewDY, Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
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
    Q, K, V, sm_scale, Y, DY,
    DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off = tl.program_id(0)
    off_z = off // H
    off_h = off % H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    DY += off_z * stride_qz + off_h * stride_qh
    for i in range(0, num_block):
        lo = i * BLOCK_M
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = i * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        qs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        ks = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        vs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dqs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dys = DY + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        ds = D + off * N_CTX
        ms = M + off * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(ks)
        v = tl.load(vs)
        for j in range(lo, num_block * BLOCK_M, BLOCK_M):
            j += offs_m
            q = tl.load(qs)
            qk = tl.dot(q, k, trans_b=True)
            qk = tl.where(j[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(ms + j)
            p = tl.exp(qk * sm_scale - m[:, None])
            dy = tl.load(dys)
            dv += tl.dot(p.to(dy.dtype), dy, trans_a=True)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - tl.load(ds + j)[:, None]
            dp += tl.dot(dy, v, trans_b=True)
            ds = p * dp * sm_scale
            dk += tl.dot(ds.to(q.dtype), q, trans_a=True)
            dq = tl.load(dqs, eviction_policy="evict_last")
            dq += tl.dot(ds.to(k.dtype), k)
            tl.store(dqs, dq, eviction_policy="evict_last")
            dqs += BLOCK_M * stride_qm
            qs += BLOCK_M * stride_qm
            dys += BLOCK_M * stride_qm
        tl.store(DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk), dv)
        tl.store(DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk), dk)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        BLOCK = 128
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        tmp = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            tmp, L, m,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk, num_warps=num_warps,
            num_stages=1,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.BLOCK = BLOCK
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do, l,
            do_scaled, delta,
            BLOCK_M=ctx.BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )

        num_warps = 8
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do_scaled,
            dq, dk, dv,
            l, m,
            delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0],
            BLOCK_M=ctx.BLOCK, BLOCK_N=ctx.BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=num_warps,
            num_stages=1,
        )
        return dq.to(q.dtype), dk, dv, None


attention = _attention.apply
