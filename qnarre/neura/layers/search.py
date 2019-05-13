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

from qnarre.neura import tf

from tensorflow.python.util import nest


class Beam(tf.Layer):
    def __init__(self, PS, to_logp, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.to_logp = to_logp

    def build(self, input_shape):
        PS = self.PS
        tgt = input_shape[0]
        assert tgt[0] == PS.batch_size
        y = tf.constant([[0.] + [-float('inf')] * (PS.beam_size - 1)])
        self._logp = tf.tile(y, [PS.batch_size, 1])
        sh = (PS.batch_size, PS.beam_size)
        self._score = tf.ones(shape=sh) * PS.big_neg
        self._flag = tf.zeros(dtype='bool', shape=sh)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        PS = self.PS
        tgt, ctx, bias = inputs
        tgt = tf.expand_dims(tgt, axis=1)
        self.tgt = self.out = tf.tile(tgt, [1, PS.beam_size, 1])
        self.logp = self._logp
        self.score = self._score
        self.flag = self._flag
        i = 1
        while self.not_done(i):
            logp, idx = self.top_logp(ctx, bias, i, **kw)
            tgt = self.append_tgt(idx, i, **kw)
            self.tgt, self.logp = self.top_tgt(tgt, logp)
            self.out, self.score, self.flag = self.top_out(tgt, logp, i)
            i += 1
        out = tf.where(tf.reduce_any(self.flag, axis=1), self.out, self.tgt)
        score = tf.where(tf.reduce_any(self.flag, axis=1), self.score,
                         self.logp)
        return out, score

    def not_done(self, i):
        PS = self.PS
        y = self.score * tf.cast(self.flag, tf.floatx())
        y = tf.reduce_min(y, axis=1)
        fs = tf.reduce_any(self.flags, axis=1)
        old = y + (1. - tf.cast(fs, tf.floatx())) * PS.big_neg
        n = tf.int_shape(self.tgt)[-1]
        new = self.logp[:, 0] / self.penalty(n)
        done = tf.reduce_all(tf.greater(old, new))
        return tf.logical_and(tf.less(i, n), tf.logical_not(done))

    def top_logp(self, ctx, bias, i, **kw):
        PS = self.PS
        y = tf.zeros((
            PS.batch_size,
            PS.beam_size,
            PS.vocab_size,
        ))
        y += tf.expand_dims(self.logp, axis=2)
        b = tf.range(PS.batch_size)
        ii = tf.constant([i] * PS.batch_size)
        for j in range(PS.beam_size):
            jj = tf.constant([j] * PS.batch_size)
            sel = tf.stack([b, jj, ii])
            yj = self.to_logp(self.tgt[:, j, :], ctx, bias, i, **kw)[1]
            y = tf.tensor_scatter_nd_add(y, sel, yj)
        y = tf.reshape(y, (-1, PS.beam_size * PS.vocab_size))
        logp, idx = tf.top_k(y, k=2 * PS.beam_size)
        return logp, idx

    def append_tok(self, idx, i, **kw):
        PS = self.PS
        k = 2 * PS.beam_size
        b = tf.range(PS.batch_size * k) // k
        b = tf.reshape(b, (PS.batch_size, k))
        beam = idx // PS.vocab_size
        sel = tf.stack([b, beam], axis=2)
        y = tf.gather_nd(self.tgt, sel)
        ii = tf.constant([i] * PS.batch_size * k)
        ii = tf.reshape(ii, (PS.batch_size, k))
        sel = tf.stack([b, beam, ii], axis=2)
        new = tf.expand_dims(idx % PS.vocab_size, axis=2)
        tgt = tf.tensor_scatter_nd_update(y, sel, new)
        return tgt

    def top_tgt(self, tgt, logp):
        PS = self.PS
        fs = tf.equal(tgt[:, :, -1], PS.END)
        logp += tf.cast(fs, tf.floatx()) * self.PS.big_neg
        return self.top_beams([tgt, logp], logp)

    def top_out(self, tgt, logp, i):
        PS = self.PS
        score = logp / self.penalty(i + 1)
        flag = tf.equal(tgt[:, :, -1], PS.END)
        score += (1. - tf.cast(flag, tf.floatx())) * PS.big_neg
        return self.top_beams([tgt, score, flag], score)

    def gather_beams(self, xs, beams, k):
        PS = self.PS
        b = tf.range(PS.batch_size * k) // k
        b = tf.reshape(b, (PS.batch_size, k))
        sel = tf.stack([b, beams], axis=2)
        return nest.map_structure(lambda x: tf.gather_nd(x, sel), xs)

    def top_beams(self, xs, vs):
        k = self.PS.beam_size
        _, beams = tf.top_k(vs, k=k)
        return self.gather_beams(xs, beams, k)

    def penalty(self, n):
        n = tf.cast(n, tf.floatx())
        y = tf.pow(((5. + n) / 6.), self.PS.beam_alpha)
        return y
