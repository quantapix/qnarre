import tensorflow as tf


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp,
                    d_model,
                    d_inner,
                    dropout,
                    kernel_initializer,
                    scope='ff',
                    is_training=True):
    output = inp
    with tf.variable_scope(scope):
        output = tf.layers.dense(inp,
                                 d_inner,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_1')
        output = tf.layers.dropout(output,
                                   dropout,
                                   training=is_training,
                                   name='drop_1')
        output = tf.layers.dense(output,
                                 d_model,
                                 kernel_initializer=kernel_initializer,
                                 name='layer_2')
        output = tf.layers.dropout(output,
                                   dropout,
                                   training=is_training,
                                   name='drop_2')
        output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def rel_multihead_attn(w,
                       r,
                       r_w_bias,
                       r_r_bias,
                       attn_mask,
                       mems,
                       d_model,
                       n_head,
                       d_head,
                       dropout,
                       dropatt,
                       is_training,
                       kernel_initializer,
                       scope='rel_attn'):
    scale = 1 / (d_head**0.5)
    with tf.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w],
                        0) if mems is not None and mems.shape.ndims > 1 else w
        w_heads = tf.layers.dense(cat,
                                  3 * n_head * d_head,
                                  use_bias=False,
                                  kernel_initializer=kernel_initializer,
                                  name='qkv')
        r_head_k = tf.layers.dense(r,
                                   n_head * d_head,
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer,
                                   name='r')

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec,
                              [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec,
                                   d_model,
                                   use_bias=False,
                                   kernel_initializer=kernel_initializer,
                                   name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def embedding_lookup(lookup_table, x, use_tpu=True):
    if use_tpu:
        n_token = tf.shape(lookup_table)[0]
        one_hot_idx = tf.one_hot(x, n_token)
        if one_hot_idx.shape.ndims == 2:
            return tf.einsum('nd,in->id', lookup_table, one_hot_idx)
        else:
            return tf.einsum('nd,ibn->ibd', lookup_table, one_hot_idx)
    else:
        return tf.nn.embedding_lookup(lookup_table, x)


def mask_adaptive_embedding_lookup(x,
                                   n_token,
                                   d_embed,
                                   d_proj,
                                   cutoffs,
                                   initializer,
                                   proj_initializer,
                                   div_val=1,
                                   proj_same_dim=True,
                                   scope='adaptive_embed',
                                   **kwargs):
    emb_scale = d_proj**0.5
    with tf.variable_scope(scope):
        if div_val == 1:
            lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                           initializer=initializer)
            y = embedding_lookup(lookup_table, x, use_tpu=False)
            if d_proj != d_embed:
                proj_W = tf.get_variable('proj_W', [d_embed, d_proj],
                                         initializer=proj_initializer)
                y = tf.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = tf.shape(x)
            y = tf.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (x >= l_idx) & (x < r_idx)
                    cur_x = tf.boolean_mask(x, mask) - l_idx
                    cur_d_embed = d_embed // (div_val**i)
                    lookup_table = tf.get_variable(
                        'lookup_table', [r_idx - l_idx, cur_d_embed],
                        initializer=initializer)
                    cur_y = embedding_lookup(lookup_table,
                                             cur_x,
                                             use_tpu=False)
                    if d_proj == cur_d_embed and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = tf.get_variable('proj_W',
                                                 [cur_d_embed, d_proj],
                                                 initializer=proj_initializer)
                        cur_y = tf.einsum('id,de->ie', cur_y, proj_W)
                    mask_idx = tf.to_int64(tf.where(mask))
                    y += tf.scatter_nd(mask_idx, cur_y,
                                       tf.to_int64(tf.shape(y)))
                    tables.append(lookup_table)
                    projs.append(proj_W)
            ret_params = [tables, projs]

    y *= emb_scale
    return y, ret_params


def mul_adaptive_embedding_lookup(x,
                                  n_token,
                                  d_embed,
                                  d_proj,
                                  cutoffs,
                                  initializer,
                                  proj_initializer,
                                  div_val=1,
                                  perms=None,
                                  proj_same_dim=True,
                                  scope='adaptive_embed'):
    """
  perms: If None, first compute W = W1 x W2 (projection for each bin),
      and then compute X x W (embedding lookup). If not None,
      use bin-based embedding lookup with max_bin_size defined by
      the shape of perms.
  """
    emb_scale = d_proj**0.5
    with tf.variable_scope(scope):
        if div_val == 1:
            lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                           initializer=initializer)
            y = embedding_lookup(lookup_table, x)
            if d_proj != d_embed:
                proj_W = tf.get_variable('proj_W', [d_embed, d_proj],
                                         initializer=proj_initializer)
                y = tf.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = tf.shape(x)
            if perms is None:
                cat_lookup = []
            else:
                cat_lookup = tf.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    cur_d_embed = d_embed // (div_val**i)
                    lookup_table = tf.get_variable(
                        'lookup_table', [r_idx - l_idx, cur_d_embed],
                        initializer=initializer)
                    if cur_d_embed == d_proj and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = tf.get_variable('proj_W',
                                                 [cur_d_embed, d_proj],
                                                 initializer=proj_initializer)
                    if perms is None:
                        cat_lookup.append(
                            tf.einsum('ie,ed->id', lookup_table, proj_W))
                    else:
                        # speed up the computation of the first bin
                        # also save some meory
                        if i == 0:
                            cur_y = embedding_lookup(lookup_table,
                                                     tf.minimum(x, r_idx - 1))
                            if proj_W is not None:
                                cur_y = tf.einsum('ibe,ed->ibd', cur_y, proj_W)
                            cur_y *= perms[i][:, :, None]
                            cat_lookup += cur_y
                        else:
                            cur_x = tf.einsum('ib,ibk->k',
                                              tf.to_float(x - l_idx), perms[i])
                            cur_x = tf.to_int32(cur_x)
                            cur_y = embedding_lookup(lookup_table, cur_x)
                            if proj_W is not None:
                                cur_y = tf.einsum('ke,ed->kd', cur_y, proj_W)
                            cat_lookup += tf.einsum('kd,ibk->ibd', cur_y,
                                                    perms[i])
                    tables.append(lookup_table)
                    projs.append(proj_W)
            if perms is None:
                cat_lookup = tf.concat(cat_lookup, 0)
                y = embedding_lookup(cat_lookup, x)
            else:
                y = cat_lookup
            ret_params = [tables, projs]

    y *= emb_scale
    return y, ret_params


def mask_adaptive_logsoftmax(hidden,
                             target,
                             n_token,
                             d_embed,
                             d_proj,
                             cutoffs,
                             params,
                             tie_projs,
                             initializer=None,
                             proj_initializer=None,
                             div_val=1,
                             scope='adaptive_softmax',
                             proj_same_dim=True,
                             return_mean=True,
                             **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    def _gather_logprob(logprob, target):
        lp_size = tf.shape(logprob)
        r = tf.range(lp_size[0])
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                        initializer=tf.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                 logits=output)
        else:
            cutoff_ends = [0] + cutoffs + [n_token]
            nll = tf.zeros_like(target, dtype=tf.float32)
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (target >= l_idx) & (target < r_idx)
                    mask_idx = tf.where(mask)
                    cur_target = tf.boolean_mask(target, mask) - l_idx
                    cur_d_embed = d_embed // (div_val**i)

                    if div_val == 1:
                        cur_W = params_W[l_idx:r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                            initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or
                                not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable(
                                'proj', [cur_d_embed, d_proj],
                                initializer=proj_initializer)
                    if i == 0:
                        cluster_W = tf.get_variable(
                            'cluster_W', [len(cutoffs), d_embed],
                            initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable(
                            'cluster_b', [len(cutoffs)],
                            initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)
                        head_logprob = tf.nn.log_softmax(head_logit)
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_logprob = _gather_logprob(cur_head_logprob,
                                                      cur_target)
                    else:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_hidden = tf.boolean_mask(hidden, mask)
                        tail_logit = tf.squeeze(
                            _logit(cur_hidden[None], cur_W, cur_b, cur_proj),
                            0)
                        tail_logprob = tf.nn.log_softmax(tail_logit)
                        cur_logprob = (
                            cur_head_logprob[:, cutoff_ends[1] + i - 1] +
                            _gather_logprob(tail_logprob, cur_target))
                    nll += tf.scatter_nd(mask_idx, -cur_logprob,
                                         tf.to_int64(tf.shape(nll)))
    if return_mean:
        nll = tf.reduce_mean(nll)
    return nll


def mul_adaptive_logsoftmax(hidden,
                            target,
                            n_token,
                            d_embed,
                            d_proj,
                            cutoffs,
                            params,
                            tie_projs,
                            initializer=None,
                            proj_initializer=None,
                            div_val=1,
                            perms=None,
                            proj_same_dim=True,
                            scope='adaptive_softmax',
                            **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if x.shape.ndims == 3:
            if proj is not None:
                y = tf.einsum('ibd,ed->ibe', y, proj)
            return tf.einsum('ibd,nd->ibn', y, W) + b
        else:
            if proj is not None:
                y = tf.einsum('id,ed->ie', y, proj)
            return tf.einsum('id,nd->in', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = tf.get_variable('bias', [n_token],
                                        initializer=tf.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                 logits=output)
            nll = tf.reduce_mean(nll)
        else:
            total_loss, total_cnt = 0, 0
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]

                    cur_d_embed = d_embed // (div_val**i)

                    if div_val == 1:
                        cur_W = params_W[l_idx:r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = tf.get_variable('b', [r_idx - l_idx],
                                            initializer=tf.zeros_initializer())
                    if tie_projs[i]:
                        if div_val == 1:
                            cur_proj = params_projs
                        else:
                            cur_proj = params_projs[i]
                    else:
                        if (div_val == 1 or
                                not proj_same_dim) and d_proj == cur_d_embed:
                            cur_proj = None
                        else:
                            cur_proj = tf.get_variable(
                                'proj', [cur_d_embed, d_proj],
                                initializer=proj_initializer)

                    if i == 0:
                        cluster_W = tf.get_variable(
                            'cluster_W', [len(cutoffs), d_embed],
                            initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable(
                            'cluster_b', [len(cutoffs)],
                            initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)

                        head_target = kwargs.get("head_target")
                        head_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=head_target, logits=head_logit)

                        masked_loss = head_nll * perms[i]
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(perms[i])

                        # head_logprob = tf.nn.log_softmax(head_logit)

                        # final_logprob = head_logprob * perms[i][:, :, None]
                        # final_target = tf.one_hot(target, tf.shape(head_logprob)[2])
                        # total_loss -= tf.einsum('ibn,ibn->', final_logprob, final_target)
                        # total_cnt += tf.reduce_sum(perms[i])
                    else:
                        cur_head_nll = tf.einsum('ib,ibk->k', head_nll,
                                                 perms[i])

                        cur_hidden = tf.einsum('ibd,ibk->kd', hidden, perms[i])
                        tail_logit = _logit(cur_hidden, cur_W, cur_b, cur_proj)

                        tail_target = tf.einsum('ib,ibk->k',
                                                tf.to_float(target - l_idx),
                                                perms[i])
                        tail_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.to_int32(tail_target), logits=tail_logit)

                        sum_nll = cur_head_nll + tail_nll
                        mask = tf.reduce_sum(perms[i], [0, 1])

                        masked_loss = sum_nll * mask
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(mask)

            nll = total_loss / total_cnt

    return nll


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

    return tf.stop_gradient(new_mem)


def transformer(dec_inp,
                target,
                mems,
                n_token,
                n_layer,
                d_model,
                d_embed,
                n_head,
                d_head,
                d_inner,
                dropout,
                dropatt,
                initializer,
                is_training,
                proj_initializer=None,
                mem_len=None,
                cutoffs=[],
                div_val=1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                use_tpu=True,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True,
                scope='transformer'):
    """
  cutoffs: a list of python int. Cutoffs for adaptive softmax.
  tie_projs: a list of python bools. Whether to tie the projections.
  use_tpu: if True, use one_hot in embedding lookup and bin-based implementation
        of adaptive softmax.
  perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
        Only used in the adaptive setting.
  """
    new_mems = []
    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                       initializer=initializer)
        else:
            r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                       initializer=initializer)
            r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                       initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        if proj_initializer is None:
            proj_initializer = initializer
        lookup_fn = (mul_adaptive_embedding_lookup
                     if use_tpu else mask_adaptive_embedding_lookup)
        embeddings, shared_params = lookup_fn(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            cutoffs=cutoffs,
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=div_val,
            perms=input_perms,
            proj_same_dim=proj_same_dim)

        attn_mask = _create_mask(qlen, mlen, same_length)

        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000**(tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.layers.dropout(embeddings, dropout, training=is_training)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)
                output = positionwise_FF(inp=output,
                                         d_model=d_model,
                                         d_inner=d_inner,
                                         dropout=dropout,
                                         kernel_initializer=initializer,
                                         is_training=is_training)

        output = tf.layers.dropout(output, dropout, training=is_training)

        logsoftmax_fn = (mul_adaptive_logsoftmax
                         if use_tpu else mask_adaptive_logsoftmax)
        loss = logsoftmax_fn(hidden=output,
                             target=target,
                             n_token=n_token,
                             d_embed=d_embed,
                             d_proj=d_model,
                             cutoffs=cutoffs,
                             params=shared_params,
                             tie_projs=tie_projs,
                             initializer=initializer,
                             proj_initializer=proj_initializer,
                             div_val=div_val,
                             perms=target_perms,
                             head_target=head_target,
                             proj_same_dim=proj_same_dim)
        return loss, new_mems


import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000**(torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        # https://nvidia.github.io/apex/layernorm.html
        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self,
                 n_head,
                 d_model,
                 d_head,
                 dropout,
                 dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None],
                                        -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None],
                                        -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0),
                                              attn_vec.size(1),
                                              self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self,
                 n_head,
                 d_model,
                 d_head,
                 dropout,
                 dropatt=0,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device,
                                   dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model,
                               self.n_head * self.d_head,
                               bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            #print(mems.dtype,w.dtype)
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head,
                                 self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head,
                                 self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head,
                                 self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head,
                                 self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn',
                          (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn',
                          (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None],
                    -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None],
                    -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0),
                                              attn_vec.size(1),
                                              self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn',
                          (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn',
                          (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None],
                                        -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None],
                                        -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0),
                                              attn_vec.size(1),
                                              self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


# default attention
class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        if d_inner > 0:
            self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout,
                                          **kwargs)
        self.pos_ff = PositionwiseFF(d_model,
                                     d_inner,
                                     dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        try:
            output = self.pos_ff(output)
        except:
            pass

        return output


# class RelLearnableDecoderLayer(nn.Module):
#     def __init__(self, n_head, d_model, d_head, d_inner, dropout,
#                  **kwargs):
#         super(RelLearnableDecoderLayer, self).__init__()
#
#         self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
#                                          **kwargs)
#         self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
#                                      pre_lnorm=kwargs.get('pre_lnorm'))
#
#     def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
#
#         output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
#                                attn_mask=dec_attn_mask,
#                                mems=mems)
#         output = self.pos_ff(output)
#
#         return output


# type 2 attention
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        if d_inner > 0:
            self.pos_ff = PositionwiseFF(d_model,
                                         d_inner,
                                         dropout,
                                         pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self,
                dec_inp,
                r,
                r_w_bias,
                r_r_bias,
                dec_attn_mask=None,
                mems=None):

        output = self.dec_attn(dec_inp,
                               r,
                               r_w_bias,
                               r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        try:
            output = self.pos_ff(output)
        except:
            pass

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self,
                 n_token,
                 d_embed,
                 d_proj,
                 cutoffs,
                 div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype,
                                   device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(self,
                 n_token,
                 n_layer,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout,
                 dropatt,
                 tie_weight=True,
                 d_embed=None,
                 div_val=1,
                 tie_projs=[False],
                 pre_lnorm=False,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 cutoffs=[],
                 adapt_inp=False,
                 same_length=False,
                 attn_type=0,
                 clamp_len=-1,
                 sample_softmax=-1,
                 fp32_embedding=False,
                 fp32_layernorm=False):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token,
                                          d_embed,
                                          d_model,
                                          cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(n_head,
                                                    d_model,
                                                    d_head,
                                                    d_inner,
                                                    dropout,
                                                    tgt_len=tgt_len,
                                                    ext_len=ext_len,
                                                    mem_len=mem_len,
                                                    dropatt=dropatt,
                                                    pre_lnorm=pre_lnorm))
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(n_head,
                                             d_model,
                                             d_head,
                                             d_inner,
                                             dropout,
                                             tgt_len=tgt_len,
                                             ext_len=ext_len,
                                             mem_len=mem_len,
                                             dropatt=dropatt,
                                             pre_lnorm=pre_lnorm))
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(n_head,
                                 d_model,
                                 d_head,
                                 d_inner,
                                 dropout,
                                 dropatt=dropatt,
                                 pre_lnorm=pre_lnorm))

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token,
                                                    d_embed,
                                                    d_model,
                                                    cutoffs,
                                                    div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[
                        i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head,
                                                      self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head,
                                                      self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head,
                             self.d_head))
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head,
                             self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        #print("word_emb dtype: ", self.word_emb.emb_layers[0].weight.dtype)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (
                torch.triu(all_ones, 1 + mlen) +
                torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None]  # -1
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen),
                                       diagonal=1 + mlen).byte()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen - 1,
                                   -1,
                                   -1.0,
                                   device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                #print("layer ", i, ": ", mems_i.dtype, layer)
                core_out = layer(core_out,
                                 pos_emb,
                                 self.r_w_bias,
                                 self.r_r_bias,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out,
                                 r_emb,
                                 self.r_w_bias[i],
                                 r_bias,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(klen - 1,
                                   -1,
                                   -1.0,
                                   device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out,
                                 dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)
        #print("core_out: ", core_out[0].dtype)
        #print("new_mems: ", new_mems[0].dtype)
        return core_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, mems=mems)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb, self.out_layer.bias, target,
                                  pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)),
                             target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data,
                                         B,
                                         tgt_len,
                                         device=device,
                                         ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token,
                                     args.n_layer,
                                     args.n_head,
                                     args.d_model,
                                     args.d_head,
                                     args.d_inner,
                                     args.dropout,
                                     dropatt=args.dropout,
                                     tie_weight=True,
                                     d_embed=d_embed,
                                     div_val=div_val,
                                     tie_projs=tie_projs,
                                     pre_lnorm=True,
                                     tgt_len=tgt_len,
                                     ext_len=ext_len,
                                     mem_len=mem_len,
                                     cutoffs=cutoffs,
                                     attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]
