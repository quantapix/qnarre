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


class TokEmbed(tf.Layer):
    emb_g = None

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        PS = self.PS
        self.emb_one_hot = PS.emb_one_hot
        h = PS.dim_hidden
        e = PS.dim_embed or h
        kw = dict(initializer=PS.initializer)
        if e != h:
            self.emb_g = self.add_weight('emb_g', (e, h), **kw)
        kw.update(regularizer=PS.regularizer)
        self.table = self.add_weight('table', (PS.num_tokens, e), **kw)
        return super().build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)

    def call(self, inputs, mask, **_):
        tok, typ = inputs
        if self.emb_one_hot:
            i = tf.one_hot(tok, tf.shape(self.table)[0], axis=-1)
            y = tf.einsum('ne,bin->bie', self.table, i)
        else:
            y = tf.embedding_lookup(self.table, tok)
        if self.emb_g is not None:
            y = tf.einsum('bie,eh->bih', y, self.emb_g)
        y *= tf.shape(y)[-1]**0.5

        y = typ * tf.cast(mask[0], typ.dtype)
        y = tf.one_hot(y, self.PS.token_types)
        return tok + tf.matmul(y, self.gain)

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
          return input_shape + (self.output_dim,)
        else:
          # input_length can be tuple if input is 3D or higher
          if isinstance(self.input_length, (list, tuple)):
            in_lens = list(self.input_length)
          else:
            in_lens = [self.input_length]
          if len(in_lens) != len(input_shape) - 1:
            raise ValueError('"input_length" is %s, '
                             'but received input has shape %s' % (str(
                                 self.input_length), str(input_shape)))
          else:
            for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
              if s1 is not None and s2 is not None and s1 != s2:
                raise ValueError('"input_length" is %s, '
                                 'but received input has shape %s' % (str(
                                     self.input_length), str(input_shape)))
              elif s1 is None:
                in_lens[i] = s2
          return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

class TokEmbed2(tf.Embedding):
    def __init__(self, PS, **_):
        super().__init__(
            input_dim=PS.num_tokens,
            input_length=PS.ctx_len,
            output_dim=PS.dim_hidden,
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
