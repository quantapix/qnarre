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

import qnarre.neura as Q

from tensorflow.python.util import nest


class Beam(Q.Layer):
    def __init__(self, PS, to_logits, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.to_logits = to_logits

    def build(self, input_shape):
        PS = self.PS
        tgt = input_shape[0]
        sh = tgt[:1] + (PS.beam_size, ) + tgt[1:]
        kw = dict(shape=sh, dtype='int32', trainable=False)
        init = Q.constant_initializer(PS.PAD)
        self.active = self.add_variable(initializer=init, **kw)
        init = Q.constant_initializer(PS.UNK)
        self.settled = self.add_variable(initializer=init, **kw)
        kw.update(shape=sh[:-1], dtype=Q.floatx())
        init = np.array([0., PS.big_neg]).reshape([1, 2])
        init = Q.constant_initializer(init)
        self.logps = self.add_variable(initializer=init, **kw)
        init = Q.constant_initializer(PS.big_neg)
        self.scores = self.add_variable(initializer=init, **kw)
        kw.update(dtype='bool')
        self.flags = self.add_variable(initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        PS = self.PS
        tgt = inputs[0]
        tgt = Q.expand_dims(tgt, axis=1)
        tgt = Q.tile(tgt, [1, PS.beam_size, 1])
        i = 1
        while self.not_done(i):
            y, lps = self.expand(*inputs, i, **kw)
            self.active, self.logps = self.new_active(y, lps)
            self.settled, self.scores, self.flags = self.new_settled(i, y, lps)
            i += 1
        done = Q.where(Q.reduce_any(self.flags, 1), self.done, self.alive)
        scores = Q.where(Q.reduce_any(self.flags, 1), self.scores, self.logps)
        return done, scores

    def not_done(self, i):
        y = self.scores * Q.cast(self.flags, Q.floatx())
        old = Q.reduce_min(y, axis=1)
        fs = Q.reduce_any(self.flags, axis=1)
        old += (1. - Q.cast(fs, Q.floatx())) * self.PS.big_neg
        n = Q.int_shape(self.active)[-1]
        new = self.logps[:, 0] / self.penalty(n)
        done = Q.reduce_all(Q.greater(old, new))
        return Q.logical_and(Q.less(i, n), Q.logical_not(done))

    def expand(self, tgt, ctx, bias, i, **kw):
        PS = self.PS
        sh = Q.int_shape(self.active)
        lps = Q.zeros(sh[:2] + (PS.vocab_size,))
        b = Q.range(sh[0])
        ii = Q.constant(i, shape=sh[:1])
        for j in range(sh[1]):
            jj = Q.constant(j, shape=sh[:1])
            lp = self.to_logits(self.active[:, j, :], ctx, bias, i, **kw)[1]
            lps = Q.tensor_scatter_nd_update(lps, Q.stack([b, jj, ii]), lp)
        lps = Q.reshape(lps, (-1, sh[1] * PS.vocab_size))
        k = 2 * PS.beam_size
        lps, idx = Q.top_k(lps, k=k)
        y = self.gather_beams([self.active], idx // PS.vocab_size, k)
        y2 = Q.expand_dims(idx % PS.vocab_size, axis=2)
        y = Q.concat([y, y2], axis=2)
        return y, lps

    def new_alive(self, x, lps):
        PS = self.PS
        fs = Q.equal(x[:, :, -1], PS.END)
        lps += Q.cast(fs, Q.floatx()) * self.PS.big_neg
        return self.top_beams([x, lps], lps)

    def new_done(self, i, x, lps):
        PS = self.PS
        y = Q.zeros((PS.batch_size, PS.beam_size, 1), Q.int32)
        y = Q.concat([self.done, y], axis=2)
        ss = lps / self.penalty(i + 1)
        fs = Q.equal(x[:, :, -1], self.PS.END)
        ss += (1. - Q.cast(fs, Q.floatx())) * self.PS.big_neg
        y = Q.concat([self.done, y], axis=1)
        ss = Q.concat([self.scores, ss], axis=1)
        fs = Q.concat([self.flags, fs], axis=1)
        return self.top_beams([y, ss, fs], ss)

    def gather_beams(self, xs, bidx, k):
        PS = self.PS
        idx = Q.range(PS.batch_size * k) // k
        idx = Q.reshape(idx, (PS.batch_size, k))
        idx = Q.stack([idx, bidx], axis=2)
        return nest.map_structure(lambda x: Q.gather_nd(x, idx), xs)

    def top_beams(self, xs, vs):
        k = self.PS.beam_size
        _, idx = Q.top_k(vs, k=k)
        return self.gather_beams(xs, idx, k)

    def penalty(self, n):
        n = Q.cast(n, Q.floatx())
        y = Q.pow(((5. + n) / 6.), self.PS.beam_alpha)
        return y
