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
import qnarre.neura.layers as L


class Attent(L.Layer):
    def __init__(self, PS, pre, post, comp=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        self.comp = comp or self.dense_comp

    def build(self, input_shape):
        src, tgt, _ = input_shape
        hs = src[2]
        assert hs == tgt[2]
        PS = self.PS
        assert hs == PS.hidden_size
        n = PS.attn_heads
        assert hs % n == 0
        self.q_comp = self.comp(hs, name='Q')
        self.k_size = ks = PS.attn_k_size or hs
        assert ks % n == 0
        self.k_comp = self.comp(ks, name='K')
        vs = PS.attn_v_size or hs
        assert vs % n == 0
        self.v_comp = self.comp(vs, name='V')
        self.out = self.dense_comp(hs, name='out')
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.out.output_shape

    def call(self, inputs, **kw):
        s, t, b = inputs
        s = self.pre(s, **kw)
        q = self.split_heads(self.q_comp(s, **kw))
        k = self.split_heads(self.k_comp(t, **kw))
        v = self.split_heads(self.v_comp(t, **kw))
        y = self.scores(q, k, v, b, **kw)
        y = self.join_heads(y)
        y = self.out(y, **kw)
        return self.post([s, y], **kw)

    def dense_comp(self, size, **kw):
        return L.Dense(size,
                       use_bias=False,
                       kernel_initializer=self.PS.initializer,
                       **kw)

    def split_heads(self, x):
        sh = Q.int_shape(x)
        n = self.PS.attn_heads
        y = Q.reshape(x, (-1, sh[1], n, sh[-1] // n))
        return Q.permute_dimensions(y, [0, 2, 1, 3])

    def scores(self, q, k, v, b, **kw):
        raise NotImplementedError()

    @staticmethod
    def join_heads(x):
        y = Q.permute_dimensions(x, [0, 2, 1, 3])
        sh = Q.int_shape(y)
        return Q.reshape(y, (-1, sh[1], sh[2] * sh[3]))


class ConvComp(L.Layer):
    dilation_rate = (1, 1)
    padding = 'VALID'

    def __init__(self, filters, ksize, dilation_rate=None, padding=None, **kw):
        super().__init__(**kw)
        assert ksize % 2 == 1
        self.ksize = ksize
        if dilation_rate:
            self.dilation_rate = dilation_rate
        if padding:
            self.padding = padding
        kw = dict(dilation_rate=self.dilation_rate, padding='VALID')
        self.conv = L.Conv1D(filters, ksize, **kw)

    def call(self, inputs, **kw):
        x = inputs
        if self.padding == 'LEFT':
            sh = Q.int_shape(x)
            # h = 2 * (self.ksize // 2) * self.dilation_rate[0]
            # w = 0 if sh[2] == 1 else 2 * (ks[1] // 2) * self.dilation_rate[1]
            # p = T.constant([[0, 0], [h, 0], [w, 0], [0, 0]])
            # x = T.pad(x, p)
            # x.set_shape([sh[0], None, None, sh[3]])
        return self.conv(x)


class DotAttent(Attent):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.drop = L.Dropout(self.PS.attn_drop)

    def scores(self, q, k, v, b, **kw):
        y = Q.matmul(q, k, transpose_b=True)
        y *= (self.k_size // self.PS.attn_heads)**-0.5
        y = self.drop(Q.softmax(y + b, **kw), **kw)
        return Q.matmul(y, v)


attents = {
    'dot_attent': DotAttent,
}
