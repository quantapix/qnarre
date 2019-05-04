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

from qnarre.neura.layers.bert import Bert


class Squad(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        self.PS = PS
        self.bert = Bert(PS)

    def build(self, input_shape):
        _, slen = input_shape[0]
        PS = self.PS
        assert slen == PS.max_seq_len
        sh = (2, PS.hidden_size)
        self.gain = self.add_weight(shape=sh, initializer=PS.initializer)
        self.bias = self.add_weight(shape=2, initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        y = self.bert.transformer([inputs, None], **kw)
        y = Q.bias_add(Q.matmul(y, self.gain, transpose_b=True), self.bias)
        return list(Q.unstack(Q.transpose(y, [2, 0, 1]), axis=0))


class SquadLoss(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(dtype='float32', **kw)
        self.PS = PS
        self.slen = PS.max_seq_len

    def build(self, input_shape):
        PS = self.PS
        sh = (2, PS.hidden_size)
        self.gain = self.add_weight(shape=sh, initializer=PS.initializer)
        self.bias = self.add_weight(shape=2, initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs, **_):
        span, pred = inputs

        def _loss(i):
            y = Q.log_softmax(pred[i], axis=-1)
            y = Q.one_hot(span[:, i], self.slen) * y
            return -Q.reduce_mean(Q.reduce_sum(y, axis=-1))

        self.add_loss((_loss(0) + _loss(1)) / 2.0)
        return pred
