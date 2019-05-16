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

import numpy as np

from qnarre.neura import tf
from qnarre.neura.layers import base


class Layer(base.Layer):
    def add_weight(self, *pa, **kw):
        return super().add_weight(*pa, dtype=tf.floatx(), **kw)


class TokDeduce(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'PAD',
                'brackets',
                'dim_embed',
                'dim_hidden',
                'emb_one_hot',
                'num_toks',
            ))

    def __init__(self, ps, table_ws, adapt_ws=None, **kw):
        super().__init__(ps, **kw)
        self._compute_output_and_mask_jointly = True
        self.adapt_ws = adapt_ws or []
        self.table_ws = table_ws
        self.table_bs = []

    def build(self, input_shape):
        cfg = self.cfg
        h = cfg.dim_hidden
        d = cfg.dim_embed or h
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            t = d // (len(bs)**i)
            self.table_bs.append(self.add_bias(f'table_b{i}', (e - b,)))
            if len(self.adapt_ws) == i:
                a = self.add_weight(f'adapt_w{i}', (t, h)) if t != h else None
                self.adapt_ws.append(a)
            b = e
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        s = tf.TensorShape((self.cfg.dim_hidden, ))
        return input_shape.concatenate(s)

    @tf.function
    def call(self, inputs):
        cfg = self.cfg
        ctx, x = inputs, tf.not_equal(inputs, cfg.PAD)
        y = tf.zeros_like(x, dtype=tf.floatx())
        bs = (cfg.brackets or []) + [cfg.num_toks]
        b = 0
        for i, e in enumerate(bs):
            m = (x >= (b or 1)) & (x < e)
            t = tf.boolean_mask(x, m) - b
            if i == 0:
                h = tf.log_softmax(self.logit(ctx, i))
                mh = tf.boolean_mask(h, m)
                u = _gather_logprob(mh, t)
            else:
                mh = tf.boolean_mask(h, m)
                mc = tf.boolean_mask(ctx, m)
                u = self.lookup(, i)
            y = tf.tensor_scatter_nd_add(y, tf.where(m), -u)
        y *= tf.int_shape(y)[-1]**0.5
        y._keras_mask = mask
        return y

    def logit(self, x, i):
        y = x
        a = self.adapt_ws[i]
        if a is not None:
            y = tf.einsum('ih,eh->ie', y, a)
        t = self.table_ws[i]
        b = self.table_bs[i]
        y = tf.einsum('ie,ne->in', y, t) + b
        return y

def mask_adaptive_logsoftmax(ctx,
                             tgt,
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

    params_W, params_projs = params[0], params[1]

    def _gather_logprob(logprob, tgt):
        lp_size = tf.shape(logprob)
        r = tf.range(lp_size[0])
        idx = tf.stack([r, tgt], 1)
        return tf.gather_nd(logprob, idx)

    with tf.variable_scope(scope):
        if len(cutoffs) == 0:
            output = _logit(ctx, params_W, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt,
                                                                logits=output)
        else:
            cutoff_ends = [0] + cutoffs + [n_token]
            for i in range(len(cutoff_ends) - 1):
                with tf.variable_scope('cutoff_{}'.format(i)):
                    l_idx, r_idx = cutoff_ends[i], cutoff_ends[i + 1]
                    mask = (tgt >= l_idx) & (tgt < r_idx)
                    mask_idx = tf.where(mask)
                    cur_tgt = tf.boolean_mask(tgt, mask) - l_idx
                    cur_d_embed = d_embed // (div_val**i)

                    if i == 0:
                        cluster_W = tf.get_variable(
                            'cluster_W', [len(cutoffs), d_embed],
                            initializer=tf.zeros_initializer())
                        cluster_b = tf.get_variable(
                            'cluster_b', [len(cutoffs)],
                            initializer=tf.zeros_initializer())
                        cur_W = tf.concat([cur_W, cluster_W], 0)
                        cur_b = tf.concat([cur_b, cluster_b], 0)

                        head_logit = _logit(ctx, cur_W, cur_b, cur_proj)
                        head_logprob = tf.nn.log_softmax(head_logit)
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_logprob = _gather_logprob(cur_head_logprob,
                                                      cur_tgt)
                    else:
                        cur_head_logprob = tf.boolean_mask(head_logprob, mask)
                        cur_ctx = tf.boolean_mask(ctx, mask)
                        tail_logit = tf.squeeze(
                            _logit(cur_ctx[None], cur_W, cur_b, cur_proj),
                            0)
                        tail_logprob = tf.nn.log_softmax(tail_logit)
                        cur_logprob = (
                            cur_head_logprob[:, cutoff_ends[1] + i - 1] +
                            _gather_logprob(tail_logprob, cur_tgt))
                    nll += tf.scatter_nd(mask_idx, -cur_logprob,
                                        tf.to_int64(tf.shape(nll)))
    if return_mean:
        nll = tf.reduce_mean(nll)
    return nll


def mul_adaptive_logsoftmax(ctx,
                            tgt,
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
            output = _logit(ctx, params_W, softmax_b, params_projs)
            nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt,
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

                        head_logit = _logit(ctx, cur_W, cur_b, cur_proj)

                        head_tgt = kwargs.get("head_tgt")
                        head_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=head_tgt, logits=head_logit)

                        masked_loss = head_nll * perms[i]
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(perms[i])

                        # head_logprob = tf.nn.log_softmax(head_logit)

                        # final_logprob = head_logprob * perms[i][:, :, None]
                        # final_tgt = tf.one_hot(tgt, tf.shape(head_logprob)[2])
                        # total_loss -= tf.einsum('ibn,ibn->', final_logprob, final_tgt)
                        # total_cnt += tf.reduce_sum(perms[i])
                    else:
                        cur_head_nll = tf.einsum('ib,ibk->k', head_nll,
                                                perms[i])

                        cur_ctx = tf.einsum('ibd,ibk->kd', ctx, perms[i])
                        tail_logit = _logit(cur_ctx, cur_W, cur_b, cur_proj)

                        tail_tgt = tf.einsum('ib,ibk->k',
                                               tf.to_float(tgt - l_idx),
                                               perms[i])
                        tail_nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.to_int32(tail_tgt), logits=tail_logit)

                        sum_nll = cur_head_nll + tail_nll
                        mask = tf.reduce_sum(perms[i], [0, 1])

                        masked_loss = sum_nll * mask
                        total_loss += tf.reduce_sum(masked_loss)
                        total_cnt += tf.reduce_sum(mask)

            nll = total_loss / total_cnt

    return nll
