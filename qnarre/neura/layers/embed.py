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

import numpy as N
import tensorflow as T


KS = T.keras
K = KS.backend
KL = KS.layers


class TokEmbed(KL.Embedding):
    def __init__(self, PS, **_):
        super().__init__(
            input_dim=PS.vocab_size,
            input_length=PS.ctx_len,
            output_dim=PS.hidden_size,
            embeddings_initializer=PS.initializer,
            embeddings_regularizer=PS.regularizer,
            mask_zero=True,
        )


class TypEmbed(KL.Layer):
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

    def call(self, inputs, **_):
        tok, typ = inputs
        typ = K.one_hot(typ, self.PS.token_types)
        return tok + K.dot(typ, self.gain)


class PosEmbed(KL.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        _, tlen, hsize = input_shape
        PS = self.PS
        plen = max(PS.max_pos, PS.ctx_len, PS.tgt_len)
        assert tlen <= plen
        sh = (plen, hsize)
        b = self.add_weight(shape=sh, initializer=PS.initializer)
        self.bias = b[:tlen, :]
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + K.expand_dims(self.bias, 0)


class PosTiming(KL.Layer):
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
        s = N.log(self.max_scale / self.min_scale) / max(n - 1, 1)
        s = self.min_scale * K.exp(K.arange(n, dtype=K.floatx()) * -s)
        p = K.arange(tlen, dtype=K.floatx()) + self.start
        p = K.expand_dims(p, 1) * K.expand_dims(s, 0)
        p = K.concatenate([K.sin(p), K.cos(p)], axis=1)
        self.bias = K.expand_dims(p, axis=0)
        return super().build(input_shape)

    def call(self, inputs, **_):
        return inputs + self.bias
