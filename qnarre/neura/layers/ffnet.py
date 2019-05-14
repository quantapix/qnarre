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


class FFN(tf.Layer):
    conv_pad = 'SAME'

    def __init__(self, PS, pre, post, conv_pad=None, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.pre = pre
        self.post = post
        if conv_pad:
            self.conv_pad = conv_pad


class DenseDenseFFN(FFN):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        PS = self.PS
        kw = dict(kernel_initializer=PS.initializer, use_bias=True)
        self.dense1 = tf.Dense(PS.ffn_size, activation=PS.ffn_act, **kw)
        self.drop = tf.Dropout(PS.ffn_drop or PS.hidden_drop)
        self.dense2 = tf.Dense(PS.hidden_size, **kw)

    def call(self, inputs, **kw):
        x = inputs
        x = self.pre([x, x], **kw)
        y = self.dense1(x, **kw)
        y = self.drop(y, **kw)
        y = self.dense2(y, **kw)
        y = self.post([x, y], **kw)
        return y


ffns = {
    None: DenseDenseFFN,
    'dense_dense_ffn': DenseDenseFFN,
}
