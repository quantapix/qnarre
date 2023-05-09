import flash_attn_cuda
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, 'b ... -> b (...)'), 0,
                            repeat(indices, 'z -> z d', d=second_dim)).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = torch.zeros([ctx.first_axis_dim, grad_output.shape[1]],
                                  device=grad_output.device, dtype=grad_output.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, 'z -> z d', d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
                             dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        indices, = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices), indices,
            cu_seqlens, max_seqlen_in_batch)


def pad_input(hidden_states, indices, batch, seqlen):
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)

def _get_block_size(device, head_dim, is_dropout):
    assert head_dim % 8 == 0 and head_dim <= 128
    return 256 if head_dim <= 64 else 128


def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
                        generator=None):
    softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, False, causal, return_softmax, num_splits, generator
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, rng_state, S_dmask


def _flash_attn_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                         rng_state=None, num_splits=0, generator=None):
    dout = dout.contiguous()  # CUDA code assumes that dout is contiguous
    _, _, _, softmax_d = flash_attn_cuda.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal,
        num_splits, generator, rng_state)
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return dq, dk, dv, softmax_d


class FlashAttnQKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
                return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], torch.empty_like(qkv[:, 0]), cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen, dropout_p, softmax_scale, causal=causal,
            return_softmax=return_softmax
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
            dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse,
            dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, cu_seqlens,
            ctx.max_seqlen, ctx.max_seqlen, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,
        )
        return dqkv, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q, kv[:, 0], kv[:, 1], torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            max_seqlen_k, dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
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
            dout, q, kv[:, 0], kv[:, 1], out, softmax_lse,
            dq, dkv[:, 0], dkv[:, 1], cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,
        )
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
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
            dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, ctx.causal,
            rng_state=rng_state, num_splits=1 if ctx.deterministic else 0,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FlashAttnQKVPackedSplitFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        # Save rng_state because the backward pass will regenerate the dropout mask
        if dropout_p > 0:
            rng_state0 = torch.cuda.get_rng_state()
            generator1 = torch.Generator(device='cuda')
            rng_state1 = generator1.get_state()
        else:
            rng_state0, generator1, rng_state1 = None, None, None
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out = torch.empty_like(qkv[:, 0])
        _, softmax_lse0, S_dmask0 = _flash_attn_forward(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], out, cu_seqlens[:batch_size0 + 1],
            cu_seqlens[:batch_size0 + 1], max_seqlen0, max_seqlen0, dropout_p, softmax_scale,
            causal=causal, return_softmax=return_softmax
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _, softmax_lse1, S_dmask1 = _flash_attn_forward(
                qkv[:, 0], qkv[:, 1], qkv[:, 2], out, cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:], max_seqlen1, max_seqlen1, dropout_p, softmax_scale,
                causal=causal, return_softmax=return_softmax, generator=generator1
            )
        torch.cuda.current_stream().wait_stream(s)
        ctx.save_for_backward(qkv, out, softmax_lse0, softmax_lse1, cu_seqlens,
                              rng_state0, rng_state1)
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
            softmax_lse = torch.cat([F.pad(softmax_lse0, (0, max_seqlen_q - softmax_lse0.shape[2])),
                                     F.pad(softmax_lse1, (0, max_seqlen_q - softmax_lse1.shape[2]))],
                                    dim=0)
            return out, softmax_lse, S_dmask0, S_dmask1

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse0, softmax_lse1, cu_seqlens, rng_state0, rng_state1 = ctx.saved_tensors
        batch_size0 = ctx.batch_size0
        if rng_state0 is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state0)
        if rng_state1 is not None:
            generator1 = torch.Generator(device='cuda')
            generator1.set_state(rng_state1)
        else:
            generator1 = None
        dqkv = torch.empty_like(qkv)
        _flash_attn_backward(
            dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse0,
            dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens[:batch_size0 + 1],
            cu_seqlens[:batch_size0 + 1], ctx.max_seqlen0, ctx.max_seqlen0, ctx.dropout_p,
            ctx.softmax_scale, ctx.causal, num_splits=1 if ctx.deterministic else 0,
        )
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _flash_attn_backward(
                dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse1,
                dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens[batch_size0:],
                cu_seqlens[batch_size0:], ctx.max_seqlen1, ctx.max_seqlen1, ctx.dropout_p,
                ctx.softmax_scale, ctx.causal, generator=generator1,
                num_splits=1 if ctx.deterministic else 0,
            )
        torch.cuda.current_stream().wait_stream(s)
        if rng_state0 is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        return dqkv, None, None, None, None, None, None, None, None, None


def flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None,
                                       causal=False, return_attn_probs=False, deterministic=False):
    return FlashAttnQKVPackedFunc.apply(qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale,
                                        causal, return_attn_probs, deterministic)


def flash_attn_unpadded_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      dropout_p, softmax_scale=None, causal=False,
                                      return_attn_probs=False, deterministic=False):
    return FlashAttnKVPackedFunc.apply(q, kv, cu_seqlens_q, cu_seqlens_k,
                                       max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                                       return_attn_probs, deterministic)


def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             dropout_p, softmax_scale=None, causal=False, return_attn_probs=False,
                             deterministic=False):
    return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               dropout_p, softmax_scale, causal, return_attn_probs, deterministic)


def flash_attn_unpadded_qkvpacked_split_func(
        qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0, dropout_p, softmax_scale=None,
        causal=False, return_attn_probs=False, deterministic=False):
    return FlashAttnQKVPackedSplitFunc.apply(qkv, cu_seqlens, max_seqlen0, max_seqlen1, batch_size0,
                                             dropout_p, softmax_scale, causal, return_attn_probs,
                                             deterministic)


def flash_attn_func(qkv, cu_seqlens, dropout_p, max_s, softmax_scale=None, causal=False,
                     return_attn_probs=False):
    return flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_s, dropout_p, softmax_scale,
                                              causal, return_attn_probs)

class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                        device=qkv.device)
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context, attn_weights = self.inner_attn(qkv, key_padding_mask=key_padding_mask,
                                                need_weights=need_weights, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
