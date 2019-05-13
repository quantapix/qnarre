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

from qnarre.neura.layers.trafo import Trafo
from qnarre.neura.layers.norm import LayerNorm


class Bert(tf.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        self.PS = PS
        self.trafo = Trafo(PS)
        self.pool = tf.Dense(PS.hidden_size,
                             tf.Tanh,
                             kernel_initializer=PS.initializer)
        self.mlm_dense = tf.Dense(PS.hidden_size, PS.hidden_act, **kw)
        self.norm = LayerNorm()

    def build(self, input_shape):
        PS = self.PS
        sh = (2, PS.hidden_size)
        self.gain = self.add_weight(shape=sh, initializer=PS.initializer)
        self.mlm_bias = self.add_weight(shape=PS.vocab_size,
                                        initializer='zeros')
        self.bias = self.add_weight(shape=2, initializer='zeros')
        return super().build(input_shape)

    def compute_output_shape(self, _):
        return self.mlm_dense.output_shape

    def call(self, inputs, **kw):
        PS = self.PS
        seq, typ, idx, val, fit, mlm = inputs
        seq = y = self.trafo([[seq, typ], None], **kw)
        fit_y = self.pool(tf.squeeze(y[:, 0:1, :], axis=1), **kw)
        y = tf.gather(y, idx, axis=1)
        y = self.norm(self.mlm_dense(y, **kw), **kw)
        e = self.trafo.tok_embed.embeddings
        y = tf.matmul(y, e, transpose_b=True)
        y = tf.log_softmax(tf.bias_add(y, self.mlm_bias), axis=-1)
        mlm_loss = -tf.reduce_sum(y * tf.one_hot(val, PS.vocab_size), axis=-1)
        y = tf.matmul(fit_y, self.gain, transpose_b=True)
        y = tf.log_softmax(tf.bias_add(y, self.bias), axis=-1)
        fit_loss = -tf.reduce_sum(y * tf.one_hot(fit, 2), axis=-1)
        loss = tf.reduce_sum(mlm * mlm_loss)
        loss /= (tf.reduce_sum(mlm) + 1e-5) + tf.reduce_mean(fit_loss)
        return seq, loss
