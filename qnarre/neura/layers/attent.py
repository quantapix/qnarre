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


class Attent(Q.Layer):
    def __init__(self, PS, pre, post, comp=None, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post
        self.comp = comp or self.dense_comp

    def build(self, input_shape):
        qry, ctx, _ = input_shape
        qs = qry[2]
        assert qs == ctx[2]
        PS = self.PS
        assert qs == PS.hidden_size
        n = PS.attn_heads
        assert qs % n == 0
        self.k_size = ks = PS.attn_k_size or qs
        assert ks % n == 0
        self.q_comp = self.comp(ks, name='Q')
        self.k_comp = self.comp(ks, name='K')
        vs = PS.attn_v_size or qs
        assert vs % n == 0
        self.v_comp = self.comp(vs, name='V')
        self.dense = self.dense_comp(qs, name='dense')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        qry, ctx, bias = inputs
        x = self.pre([qry, qry], **kw)
        q = self.split_heads(self.q_comp(x, **kw))
        k = self.split_heads(self.k_comp(ctx, **kw))
        v = self.split_heads(self.v_comp(ctx, **kw))
        y = self.scores(q, k, v, bias, **kw)
        y = self.join_heads(y)
        y = self.dense(y, **kw)
        y = self.post([x, y], **kw)
        return y

    def dense_comp(self, size, **kw):
        kw.update(kernel_initializer=self.PS.initializer)
        return Q.Dense(size, use_bias=False, **kw)

    def split_heads(self, x):
        sh = Q.int_shape(x)
        n = self.PS.attn_heads
        y = Q.reshape(x, (-1, sh[1], n, sh[-1] // n))
        y = Q.permute_dimensions(y, [0, 2, 1, 3])
        return y

    def scores(self, q, k, v, bias, **kw):
        raise NotImplementedError()

    @staticmethod
    def join_heads(x):
        y = Q.permute_dimensions(x, [0, 2, 1, 3])
        sh = Q.int_shape(y)
        y = Q.reshape(y, (-1, sh[1], sh[2] * sh[3]))
        return y


class ConvComp(Q.Layer):
    dilation = (1, 1)
    padding = 'VALID'

    def __init__(self, filters, ksize, dilation=None, padding=None, **kw):
        super().__init__(**kw)
        assert ksize % 2 == 1
        self.k_size = ksize
        if dilation:
            self.dilation = dilation
        if padding:
            self.padding = padding
        kw = dict(dilation_rate=self.dilation, padding='VALID')
        self.conv = Q.Conv1D(filters, ksize, **kw)

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


class DotProdAttn(Attent):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        self.drop = Q.Dropout(PS.attn_drop or PS.hidden_drop)

    def scores(self, q, k, v, bias, **kw):
        y = Q.matmul(q, k, transpose_b=True)
        y *= (self.k_size // self.PS.attn_heads)**-0.5
        y = Q.softmax(y + bias, **kw)
        y = self.drop(y, **kw)
        y = Q.matmul(y, v)
        return y


attns = {
    None: DotProdAttn,
    'dot_prod_attn': DotProdAttn,
}
