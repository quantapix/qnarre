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


class FFN(Q.Layer):
    conv_pad = 'SAME'

    def __init__(self, PS, pre, post, conv_pad=None, **kw):
        super().__init__(**kw)
        self.PS = PS
        self.pre = pre
        self.post = post
        if conv_pad:
            self.conv_pad = conv_pad


class DenseDense(FFN):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        kw = dict(kernel_initializer=PS.initializer, use_bias=True)
        self.dense1 = Q.Dense(PS.ffn_units, activation=PS.ffn_act, **kw)
        self.drop = Q.Dropout(PS.ffn_drop)
        self.dense2 = Q.Dense(PS.hidden_size, **kw)

    def call(self, inputs, pad_remover=None, **kw):
        x = inputs
        y = self.pre(x, **kw)
        sh = Q.int_shape(y)
        if pad_remover:
            y = Q.reshape(y, Q.concatenate([[-1], sh[2:]], axis=0))
            y = Q.expand_dims(pad_remover.remove(y), axis=0)
        y = self.dense1(y, **kw)
        y = self.drop(y, **kw)
        y = self.dense2(y, **kw)
        if pad_remover:
            y = Q.reshape(pad_remover.restore(Q.squeeze(y, axis=0)), sh)
        return self.post([x, y], **kw)


ffns = {
    None: DenseDense,
    'dense_dense': DenseDense,
}
