# Copyright 2019 Quantapix Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = Q.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = Q.concat([Q.sin(sinusoid_inp), Q.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return Q.tile(pos_emb[:, None, :], [1, bsz, 1])
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
    with Q.variable_scope(scope):
        output = Q.layers.dense(inp,
                                d_inner,
                                activation=Q.nn.relu,
                                kernel_initializer=kernel_initializer,
                                name='layer_1')
        output = Q.layers.dropout(output,
                                  dropout,
                                  training=is_training,
                                  name='drop_1')
        output = Q.layers.dense(output,
                                d_model,
                                kernel_initializer=kernel_initializer,
                                name='layer_2')
        output = Q.layers.dropout(output,
                                  dropout,
                                  training=is_training,
                                  name='drop_2')
        output = Q.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    return output


def rel_shift(x):
    x_size = Q.shape(x)

    x = Q.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = Q.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = Q.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = Q.reshape(x, x_size)

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
    with Q.variable_scope(scope):
        qlen = Q.shape(w)[0]
        rlen = Q.shape(r)[0]
        bsz = Q.shape(w)[1]

        cat = Q.concat([mems, w],
                       0) if mems is not None and mems.shape.ndims > 1 else w
        w_heads = Q.layers.dense(cat,
                                 3 * n_head * d_head,
                                 use_bias=False,
                                 kernel_initializer=kernel_initializer,
                                 name='qkv')
        r_head_k = Q.layers.dense(r,
                                  n_head * d_head,
                                  use_bias=False,
                                  kernel_initializer=kernel_initializer,
                                  name='r')

        w_head_q, w_head_k, w_head_v = Q.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = Q.shape(w_head_k)[0]

        w_head_q = Q.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = Q.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = Q.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = Q.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = Q.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = Q.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = Q.nn.softmax(attn_score, 1)
        attn_prob = Q.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = Q.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = Q.shape(attn_vec)
        attn_vec = Q.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = Q.layers.dense(attn_vec,
                                  d_model,
                                  use_bias=False,
                                  kernel_initializer=kernel_initializer,
                                  name='o')
        attn_out = Q.layers.dropout(attn_out, dropout, training=is_training)

        output = Q.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
    return output


def embedding_lookup(lookup_table, x, use_tpu=True):
    if use_tpu:
        n_token = Q.shape(lookup_table)[0]
        one_hot_idx = Q.one_hot(x, n_token)
        if one_hot_idx.shape.ndims == 2:
            return Q.einsum('nd,in->id', lookup_table, one_hot_idx)
        else:
            return Q.einsum('nd,ibn->ibd', lookup_table, one_hot_idx)
    else:
        return Q.nn.embedding_lookup(lookup_table, x)


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
    with Q.variable_scope(scope):
        if div_val == 1:
            lookup_table = Q.get_variable('lookup_table', [n_token, d_embed],
                                          initializer=initializer)
            y = embedding_lookup(lookup_table, x, use_tpu=False)
            if d_proj != d_embed:
                proj_W = Q.get_variable('proj_W', [d_embed, d_proj],
                                        initializer=proj_initializer)
                y = Q.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = Q.shape(x)
            y = Q.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with Q.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (x >= l_idx) & (x < r_idx)
                    cur_x = Q.boolean_mask(x, mask) - l_idx
                    cur_d_embed = d_embed // (div_val**i)
                    lookup_table = Q.get_variable('lookup_table',
                                                  [r_idx - l_idx, cur_d_embed],
                                                  initializer=initializer)
                    cur_y = embedding_lookup(lookup_table,
                                             cur_x,
                                             use_tpu=False)
                    if d_proj == cur_d_embed and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = Q.get_variable('proj_W',
                                                [cur_d_embed, d_proj],
                                                initializer=proj_initializer)
                        cur_y = Q.einsum('id,de->ie', cur_y, proj_W)
                    mask_idx = Q.to_int64(Q.where(mask))
                    y += Q.scatter_nd(mask_idx, cur_y, Q.to_int64(Q.shape(y)))
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
    with Q.variable_scope(scope):
        if div_val == 1:
            lookup_table = Q.get_variable('lookup_table', [n_token, d_embed],
                                          initializer=initializer)
            y = embedding_lookup(lookup_table, x)
            if d_proj != d_embed:
                proj_W = Q.get_variable('proj_W', [d_embed, d_proj],
                                        initializer=proj_initializer)
                y = Q.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]
        else:
            tables, projs = [], []
            cutoff_ends = [0] + cutoffs + [n_token]
            x_size = Q.shape(x)
            if perms is None:
                cat_lookup = []
            else:
                cat_lookup = Q.zeros([x_size[0], x_size[1], d_proj])
            for i in range(len(cutoff_ends) - 1):
                with Q.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    cur_d_embed = d_embed // (div_val**i)
                    lookup_table = Q.get_variable('lookup_table',
                                                  [r_idx - l_idx, cur_d_embed],
                                                  initializer=initializer)
                    if cur_d_embed == d_proj and not proj_same_dim:
                        proj_W = None
                    else:
                        proj_W = Q.get_variable('proj_W',
                                                [cur_d_embed, d_proj],
                                                initializer=proj_initializer)
                    if perms is None:
                        cat_lookup.append(
                            Q.einsum('ie,ed->id', lookup_table, proj_W))
                    else:
                        # speed up the computation of the first bin
                        # also save some meory
                        if i == 0:
                            cur_y = embedding_lookup(lookup_table,
                                                     Q.minimum(x, r_idx - 1))
                            if proj_W is not None:
                                cur_y = Q.einsum('ibe,ed->ibd', cur_y, proj_W)
                            cur_y *= perms[i][:, :, None]
                            cat_lookup += cur_y
                        else:
                            cur_x = Q.einsum('ib,ibk->k',
                                             Q.to_float(x - l_idx), perms[i])
                            cur_x = Q.to_int32(cur_x)
                            cur_y = embedding_lookup(lookup_table, cur_x)
                            if proj_W is not None:
                                cur_y = Q.einsum('ke,ed->kd', cur_y, proj_W)
                            cat_lookup += Q.einsum('kd,ibk->ibd', cur_y,
                                                   perms[i])
                    tables.append(lookup_table)
                    projs.append(proj_W)
            if perms is None:
                cat_lookup = Q.concat(cat_lookup, 0)
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
            y = Q.einsum('ibd,ed->ibe', y, proj)
        return Q.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    def _gather_logprob(logprob, target):
        lp_size = Q.shape(logprob)
        r = Q.range(lp_size[0])
        idx = Q.stack([r, target], 1)
        return Q.gather_nd(logprob, idx)

    with Q.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = Q.get_variable('bias', [n_token],
                                       initializer=Q.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = Q.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                logits=output)
        else:
            cutoff_ends = [0] + cutoffs + [n_token]
            nll = Q.zeros_like(target, dtype=Q.float32)
            for i in range(len(cutoff_ends) - 1):
                with Q.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (target >= l_idx) & (target < r_idx)
                    mask_idx = Q.where(mask)
                    cur_target = Q.boolean_mask(target, mask) - l_idx
                    cur_d_embed = d_embed // (div_val**i)

                    if div_val == 1:
                        cur_W = params_W[l_idx:r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = Q.get_variable('b', [r_idx - l_idx],
                                           initializer=Q.zeros_initializer())
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
                            cur_proj = Q.get_variable(
                                'proj', [cur_d_embed, d_proj],
                                initializer=proj_initializer)
                    if i == 0:
                        cluster_W = Q.get_variable(
                            'cluster_W', [len(cutoffs), d_embed],
                            initializer=Q.zeros_initializer())
                        cluster_b = Q.get_variable(
                            'cluster_b', [len(cutoffs)],
                            initializer=Q.zeros_initializer())
                        cur_W = Q.concat([cur_W, cluster_W], 0)
                        cur_b = Q.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)
                        head_logprob = Q.nn.log_softmax(head_logit)
                        cur_head_logprob = Q.boolean_mask(head_logprob, mask)
                        cur_logprob = _gather_logprob(cur_head_logprob,
                                                      cur_target)
                    else:
                        cur_head_logprob = Q.boolean_mask(head_logprob, mask)
                        cur_hidden = Q.boolean_mask(hidden, mask)
                        tail_logit = Q.squeeze(
                            _logit(cur_hidden[None], cur_W, cur_b, cur_proj),
                            0)
                        tail_logprob = Q.nn.log_softmax(tail_logit)
                        cur_logprob = (
                            cur_head_logprob[:, cutoff_ends[1] + i - 1] +
                            _gather_logprob(tail_logprob, cur_target))
                    nll += Q.scatter_nd(mask_idx, -cur_logprob,
                                        Q.to_int64(Q.shape(nll)))
    if return_mean:
        nll = Q.reduce_mean(nll)
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
                y = Q.einsum('ibd,ed->ibe', y, proj)
            return Q.einsum('ibd,nd->ibn', y, W) + b
        else:
            if proj is not None:
                y = Q.einsum('id,ed->ie', y, proj)
            return Q.einsum('id,nd->in', y, W) + b

    params_W, params_projs = params[0], params[1]

    with Q.variable_scope(scope):
        if len(cutoffs) == 0:
            softmax_b = Q.get_variable('bias', [n_token],
                                       initializer=Q.zeros_initializer())
            output = _logit(hidden, params_W, softmax_b, params_projs)
            nll = Q.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                logits=output)
            nll = Q.reduce_mean(nll)
        else:
            total_loss, total_cnt = 0, 0
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with Q.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]

                    cur_d_embed = d_embed // (div_val**i)

                    if div_val == 1:
                        cur_W = params_W[l_idx:r_idx]
                    else:
                        cur_W = params_W[i]
                    cur_b = Q.get_variable('b', [r_idx - l_idx],
                                           initializer=Q.zeros_initializer())
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
                            cur_proj = Q.get_variable(
                                'proj', [cur_d_embed, d_proj],
                                initializer=proj_initializer)

                    if i == 0:
                        cluster_W = Q.get_variable(
                            'cluster_W', [len(cutoffs), d_embed],
                            initializer=Q.zeros_initializer())
                        cluster_b = Q.get_variable(
                            'cluster_b', [len(cutoffs)],
                            initializer=Q.zeros_initializer())
                        cur_W = Q.concat([cur_W, cluster_W], 0)
                        cur_b = Q.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(hidden, cur_W, cur_b, cur_proj)

                        head_target = kwargs.get("head_target")
                        head_nll = Q.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=head_target, logits=head_logit)

                        masked_loss = head_nll * perms[i]
                        total_loss += Q.reduce_sum(masked_loss)
                        total_cnt += Q.reduce_sum(perms[i])

                        # head_logprob = Q.nn.log_softmax(head_logit)

                        # final_logprob = head_logprob * perms[i][:, :, None]
                        # final_target = Q.one_hot(target, Q.shape(head_logprob)[2])
                        # total_loss -= Q.einsum('ibn,ibn->', final_logprob, final_target)
                        # total_cnt += Q.reduce_sum(perms[i])
                    else:
                        cur_head_nll = Q.einsum('ib,ibk->k', head_nll,
                                                perms[i])

                        cur_hidden = Q.einsum('ibd,ibk->kd', hidden, perms[i])
                        tail_logit = _logit(cur_hidden, cur_W, cur_b, cur_proj)

                        tail_target = Q.einsum('ib,ibk->k',
                                               Q.to_float(target - l_idx),
                                               perms[i])
                        tail_nll = Q.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=Q.to_int32(tail_target), logits=tail_logit)

                        sum_nll = cur_head_nll + tail_nll
                        mask = Q.reduce_sum(perms[i], [0, 1])

                        masked_loss = sum_nll * mask
                        total_loss += Q.reduce_sum(masked_loss)
                        total_cnt += Q.reduce_sum(mask)

            nll = total_loss / total_cnt

    return nll


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = Q.ones([qlen, qlen])
    mask_u = Q.matrix_band_part(attn_mask, 0, -1)
    mask_dia = Q.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = Q.zeros([qlen, mlen])
    ret = Q.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = Q.matrix_band_part(attn_mask, -1, 0)
        ret = Q.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = Q.concat([prev_mem, curr_out], 0)[-mem_len:]

    return Q.stop_gradient(new_mem)


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
    with Q.variable_scope(scope):
        if untie_r:
            r_w_bias = Q.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                      initializer=initializer)
            r_r_bias = Q.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                      initializer=initializer)
        else:
            r_w_bias = Q.get_variable('r_w_bias', [n_head, d_head],
                                      initializer=initializer)
            r_r_bias = Q.get_variable('r_r_bias', [n_head, d_head],
                                      initializer=initializer)

        qlen = Q.shape(dec_inp)[0]
        mlen = Q.shape(mems[0])[0] if mems is not None else 0
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

        pos_seq = Q.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = Q.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000**(Q.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = Q.layers.dropout(embeddings, dropout, training=is_training)
        pos_emb = Q.layers.dropout(pos_emb, dropout, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with Q.variable_scope('layer_{}'.format(i)):
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

        output = Q.layers.dropout(output, dropout, training=is_training)

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


class RelMultiHeadAttn(nn.Module):
    def __init__(self,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

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


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model,
                               self.n_head * self.d_head,
                               bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
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
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)
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


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D
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
        # scale


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
