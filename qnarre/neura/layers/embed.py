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


class TokEmbed(tf.Embedding):
    def __init__(self, PS, **_):
        super().__init__(
            input_dim=PS.vocab_size,
            input_length=PS.ctx_len,
            output_dim=PS.hidden_size,
            embeddings_initializer=PS.initializer,
            embeddings_regularizer=PS.regularizer,
            mask_zero=True,
        )


class TypEmbed(tf.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        tok, typ = input_shape
        _, tlen, hsize = tok
        assert tlen == typ[1]
        PS = self.PS
        sh = (PS.token_types, hsize)
        self.gain = self.add_weight(shape=sh, initializer=PS.initializer)
        return super().build(input_shape)

    def call(self, inputs, mask, **_):
        tok, typ = inputs
        y = typ * tf.cast(mask[0], typ.dtype)
        y = tf.one_hot(y, self.PS.token_types)
        return tok + tf.matmul(y, self.gain)


class PosEmbed(tf.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        _, tlen, hsize = input_shape
        PS = self.PS
        plen = max(PS.max_pos or 0, PS.ctx_len, PS.tgt_len)
        assert tlen <= plen
        sh = (plen, hsize)
        b = self.add_weight(shape=sh, initializer=PS.initializer)
        b = b[:tlen, :]
        self.bias = tf.expand_dims(b, axis=0)
        return super().build(input_shape)

    def call(self, inputs, mask, **_):
        y = tf.cast(mask, self.bias.dtype)
        y = self.bias * tf.expand_dims(y, axis=2)
        return inputs + y


class PosTiming(tf.Layer):
    start = 0
    min_scale = 1.0
    max_scale = 1.0e4

    def __init__(self, _, start=None, min_scale=None, max_scale=None, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        if start:
            self.start = start
        if min_scale:
            self.min_scale = float(min_scale)
        if max_scale:
            self.max_scale = float(max_scale)

    def build(self, input_shape):
        _, tlen, hsize = input_shape
        assert hsize % 2 == 0
        n = hsize // 2
        s = np.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * tf.exp(tf.range(n, dtype=tf.floatx()) * -s)
        p = tf.range(tlen, dtype=tf.floatx()) + self.start
        p = tf.expand_dims(p, axis=1) * tf.expand_dims(s, axis=0)
        p = tf.concat([tf.sin(p), tf.cos(p)], axis=1)
        self.bias = tf.expand_dims(p, axis=0)
        return super().build(input_shape)

    def call(self, inputs, mask, **_):
        y = tf.cast(mask, self.bias.dtype)
        y = self.bias * tf.expand_dims(y, axis=2)
        return inputs + y
