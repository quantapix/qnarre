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

from tensorflow.python.util import nest


class Beam(Q.Layer):
    def __init__(self, PS, to_logits, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.to_logits = to_logits

    def call(self, inputs, **kw):
        PS = self.PS
        y = Q.expand_dims(inputs, axis=1)
        y = Q.tile(y, [1, PS.beam_size, 1])
        tgt = Q.reshape(y, (PS.batch_size * PS.beam_size, -1))
        sh = (PS.batch_size, PS.beam_size, 0)
        self.alive = Q.zeros(sh, Q.int32)
        self.done = Q.zeros(sh, Q.int32)
        lps = Q.constant([[0.] + [PS.big_neg] * (PS.beam_size - 1)])
        self.logps = Q.tile(lps, [PS.batch_size, 1])
        self.scores = Q.ones(sh[:-1]) * self.PS.big_neg
        self.flags = Q.zeros(sh[:-1], Q.bool)
        i = 0
        while self.not_done(i):
            y, lps = self.grow_alive(i, tgt)
            self.alive, self.logps = self.new_alive(y, lps)
            self.done, self.scores, self.flags = self.new_done(i, y, lps)
            i += 1
        done = Q.where(Q.reduce_any(self.flags, 1), self.done, self.alive)
        scores = Q.where(Q.reduce_any(self.flags, 1), self.scores, self.logps)
        return done, scores

    def not_done(self, i):
        PS = self.PS
        y = self.scores * Q.cast(self.flags, Q.floatx())
        old = Q.reduce_min(y, axis=1)
        fs = Q.reduce_any(self.flags, axis=1)
        old += (1. - Q.cast(fs, Q.floatx())) * PS.big_neg
        new = self.logps[:, 0] / self.penalty(PS.tgt_len)
        done = Q.reduce_all(Q.greater(old, new))
        return Q.logical_and(Q.less(i, PS.tgt_len), Q.logical_not(done))

    def grow_alive(self, i, tgt):
        PS = self.PS
        y = Q.reshape(self.alive, (PS.batch_size * PS.beam_size, -1))
        y = self.to_logits(i, y, tgt)
        y = Q.reshape(y, (PS.batch_size, PS.beam_size, -1))
        y = y - Q.reduce_logsumexp(y, axis=2, keep_dims=True)
        y += Q.expand_dims(self.logps, axis=2)
        lps = Q.reshape(y, (-1, PS.beam_size * PS.vocab_size))
        k = 2 * PS.beam_size
        lps, idx = Q.top_k(lps, k=k)
        y = self.gather_beams([self.alive], idx // PS.vocab_size, k)
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
