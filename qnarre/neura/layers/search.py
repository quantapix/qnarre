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
# https://arxiv.org/pdf/1904.09751.pdf

from qnarre.neura import tf
from qnarre.neura import utils
from tensorflow.python.util import nest

from qnarre.neura.layers.base import Layer


class Beam(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'END',
                'beam_alpha',
                'beam_size',
                'num_toks',
            ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        self.to_logp = lambda *a, **kw: owner.to_logp(*a, **kw)

    def build(self, input_shape):
        cfg = self.cfg
        tgt = input_shape[0]
        assert tgt[0] == cfg.batch_size
        y = tf.constant([[0.] + [-float('inf')] * (cfg.beam_size - 1)])
        self._logp = tf.tile(y, [cfg.batch_size, 1])
        sh = (cfg.batch_size, cfg.beam_size)
        self._score = tf.ones(shape=sh) * utils.big_neg
        self._flag = tf.zeros(dtype='bool', shape=sh)
        return super().build(input_shape)

    @tf.function
    def call(self, inputs):
        cfg = self.cfg
        x, ctx = inputs
        x = x[:, None, ]
        self.tgt = self.out = tf.tile(x, [1, cfg.beam_size, 1])
        self.logp = self._logp
        self.score = self._score
        self.flag = self._flag
        i = 1
        while self.not_done(i):
            logp, idx = self.top_logp(ctx, i)
            tgt = self.append_tgt(idx, i)
            self.tgt, self.logp = self.top_tgt(tgt, logp)
            self.out, self.score, self.flag = self.top_out(tgt, logp, i)
            i += 1
        out = tf.where(tf.reduce_any(self.flag, axis=1), self.out, self.tgt)
        score = tf.where(tf.reduce_any(self.flag, axis=1), self.score,
                         self.logp)
        return out, score

    def not_done(self, i):
        y = self.score * tf.cast(self.flag, tf.floatx())
        y = tf.reduce_min(y, axis=1)
        fs = tf.reduce_any(self.flags, axis=1)
        old = y + (1. - tf.cast(fs, tf.floatx())) * utils.big_neg
        n = tf.int_shape(self.tgt)[-1]
        new = self.logp[:, 0] / self.penalty(n)
        done = tf.reduce_all(tf.greater(old, new))
        return tf.logical_and(tf.less(i, n), tf.logical_not(done))

    def top_logp(self, ctx, bias, i):
        cfg = self.cfg
        y = tf.zeros((
            cfg.batch_size,
            cfg.beam_size,
            cfg.num_toks,
        ))
        y += tf.expand_dims(self.logp, axis=2)
        b = tf.range(cfg.batch_size)
        ii = tf.constant([i] * cfg.batch_size)
        for j in range(cfg.beam_size):
            jj = tf.constant([j] * cfg.batch_size)
            sel = tf.stack([b, jj, ii])
            yj = self.to_logp(self.tgt[:, j, :], ctx, bias, i)[1]
            y = tf.tensor_scatter_nd_add(y, sel, yj)
        y = tf.reshape(y, (-1, cfg.beam_size * cfg.num_toks))
        logp, idx = tf.top_k(y, k=2 * cfg.beam_size)
        return logp, idx

    def append_tok(self, idx, i, **kw):
        cfg = self.cfg
        k = 2 * cfg.beam_size
        b = tf.range(cfg.batch_size * k) // k
        b = tf.reshape(b, (cfg.batch_size, k))
        beam = idx // cfg.num_toks
        sel = tf.stack([b, beam], axis=2)
        y = tf.gather_nd(self.tgt, sel)
        ii = tf.constant([i] * cfg.batch_size * k)
        ii = tf.reshape(ii, (cfg.batch_size, k))
        sel = tf.stack([b, beam, ii], axis=2)
        u = tf.expand_dims(idx % cfg.num_toks, axis=2)
        tgt = tf.tensor_scatter_nd_update(y, sel, u)
        return tgt

    def top_tgt(self, x, lp):
        cfg = self.cfg
        fs = tf.equal(x[:, :, -1], cfg.END)
        lp += tf.cast(fs, tf.floatx()) * utils.big_neg
        return self.top_beams([x, lp], lp)

    def top_out(self, x, lp, i):
        cfg = self.cfg
        score = lp / self.penalty(i + 1)
        flag = tf.equal(x[:, :, -1], cfg.END)
        score += (1. - tf.cast(flag, tf.floatx())) * utils.big_neg
        return self.top_beams([x, score, flag], score)

    def gather_beams(self, xs, beams, k):
        cfg = self.cfg
        b = tf.range(cfg.batch_size * k) // k
        b = tf.reshape(b, (cfg.batch_size, k))
        sel = tf.stack([b, beams], axis=2)
        return nest.map_structure(lambda x: tf.gather_nd(x, sel), xs)

    def top_beams(self, xs, vs):
        k = self.cfg.beam_size
        _, beams = tf.top_k(vs, k=k)
        return self.gather_beams(xs, beams, k)

    def penalty(self, n):
        n = tf.cast(n, tf.floatx())
        y = tf.pow(((5. + n) / 6.), self.cfg.beam_alpha)
        return y
