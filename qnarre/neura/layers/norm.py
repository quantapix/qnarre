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
import qnarre.neura.layers as L


class LayerNorm(L.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.supports_masking = True

    def build(self, input_shape):
        sh = input_shape[-1]
        self.gain = self.add_weight(shape=sh, initializer='ones')
        self.bias = self.add_weight(shape=sh, initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs, **_):
        x = inputs
        m = Q.mean(x, axis=-1, keepdims=True)
        v = Q.mean(Q.square(x - m), axis=-1, keepdims=True)
        e = Q.constant(1e-5, dtype=Q.floatx())
        y = (x - m) / Q.sqrt(v + e)
        return self.gain * y + self.bias
