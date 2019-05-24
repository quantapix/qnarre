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
from qnarre.neura.layers import base

from qnarre.neura.layers.bert import Bert


def adapter(ps, feats, x):
    d = tf.parse_example(x, feats)
    img = tf.to_dense(d['flt_img'])
    # img = tf.cast(d['int_img'], tf.float32) / 255.
    lbl = d['int_lbl']
    return img, lbl


def model(ps):
    seq = tf.Input(shape=(), dtype=tf.float32)
    typ = tf.Input(shape=(), dtype=tf.float32)
    opt = tf.Input(shape=(), dtype=tf.float32)
    beg = tf.Input(shape=(), dtype=tf.float32)
    end = tf.Input(shape=(), dtype=tf.float32)
    uid = tf.Input(shape=(), dtype=tf.float32)
    ins = [seq, typ, opt, beg, end, uid]
    y = Squad(ps)([seq, typ])
    outs = [SquadLoss(ps)([beg, end], y)]
    m = tf.Model(name='SquadModel', inputs=ins, outputs=outs)
    return m


class Squad(base.Layer):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)  # dtype='float32', **kw)
        self.bert = Bert(ps)

    def build(self, input_shape):
        _, slen = input_shape[0]
        cfg = self.cfg
        assert slen == cfg.max_seq_len
        sh = (2, cfg.hidden_size)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        self.bias = self.add_weight(shape=2, initializer='zeros')
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, **kw):
        y = self.bert.transformer([inputs, None], **kw)
        y = tf.bias_add(tf.matmul(y, self.gain, transpose_b=True), self.bias)
        return list(tf.unstack(tf.transpose(y, [2, 0, 1]), axis=0))


class SquadLoss(base.Layer):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)  # dtype='float32', **kw)
        self.slen = self.cfg.max_seq_len

    def build(self, input_shape):
        cfg = self.cfg
        sh = (2, cfg.hidden_size)
        self.gain = self.add_weight(shape=sh, initializer=cfg.initializer)
        self.bias = self.add_weight(shape=2, initializer='zeros')
        return super().build(input_shape)

    @tf.function
    def call(self, inputs, **_):
        span, pred = inputs

        def _loss(i):
            y = tf.log_softmax(pred[i], axis=-1)
            y = tf.one_hot(span[:, i], self.slen) * y
            return -tf.reduce_mean(tf.reduce_sum(y, axis=-1))

        self.add_loss((_loss(0) + _loss(1)) / 2.0)
        return pred
