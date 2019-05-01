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

import tensorflow as T

KS = T.keras
K = KS.backend
KL = KS.layers


class LayerNorm(KL.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True

    def build(self, input_shape):
        kw = dict(shape=input_shape[-1], trainable=True)
        self.gain = self.add_weight(initializer='ones', **kw)
        self.bias = self.add_weight(initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **_):
        x = inputs
        m = K.mean(x, axis=-1, keepdims=True)
        v = K.mean(K.square(x - m), axis=-1, keepdims=True)
        e = K.constant(1e-5, dtype=K.floatx())
        y = (x - m) / K.sqrt(v + e)
        return self.gain * y + self.bias
