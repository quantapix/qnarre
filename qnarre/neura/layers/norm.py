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


def _layer_norm(self, inputs, **_):
    x = inputs
    m = Q.reduce_mean(x, axis=-1, keepdims=True)
    v = Q.reduce_mean(Q.square(x - m), axis=-1, keepdims=True)
    y = (x - m) / Q.sqrt(v + self.PS.norm_epsilon)
    y = self.gain * y + self.bias
    return y


class LayerNorm(Q.Layer):
    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS

    def build(self, input_shape):
        sh = input_shape[-1]
        self.gain = self.add_weight(shape=sh, initializer='ones')
        self.bias = self.add_weight(shape=sh, initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        return _layer_norm(self, inputs, **kw)


class LayerProc(Q.Layer):
    cmd = ''
    batch = None

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.drop = self.dropout()
        if PS.norm_type == 'batch':
            self.batch = Q.BatchNormalization(epsilon=PS.norm_epsilon)

    def build(self, input_shape):
        _, x = input_shape
        self.gain = self.add_weight(shape=x[-1], initializer='ones')
        self.bias = self.add_weight(shape=x[-1], initializer='zeros')
        # self.gamma = self.add_weight(shape=(), initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs, **kw):
        prev, x = inputs
        y = x
        if self.cmd:
            PS = self.PS
            for c in self.cmd:
                if c == 'a':
                    y = prev + x
                elif c == 'z':
                    y = prev + x * self.gamma
                elif c == 'n':
                    if PS.norm_type == 'layer':
                        y = _layer_norm(self, x, **kw)
                    elif PS.norm_type == 'batch':
                        y = self.batch(x, **kw)
                    elif PS.norm_type == 'l2':
                        m = Q.reduce_mean(x, axis=-1, keepdims=True)
                        n = Q.square(x - m)
                        n = Q.reduce_sum(n, axis=-1, keepdims=True)
                        y = (x - m) / Q.sqrt(n + PS.norm_epsilon)
                        y = y * self.gain + self.bias
                    elif PS.norm_type == 'group':
                        sh = Q.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % PS.num_groups == 0
                        gs = (PS.num_groups, sh[-1] // PS.num_groups)
                        x = Q.reshape(x, sh[:-1] + gs)
                        m, v = Q.moments(x, [1, 2, 4], keep_dims=True)
                        y = (x - m) / Q.sqrt(v + PS.group_epsilon)
                        y = Q.reshape(y, sh) * self.gain + self.bias
                    elif PS.norm_type == 'noam':
                        y = Q.cast_to_floatx(Q.int_shape(x)[-1])
                        y = Q.l2_normalize(x, axis=-1) * Q.sqrt(y)
                    else:
                        assert PS.norm_type == 'none'
                else:
                    assert c == 'd'
                    y = self.drop(x, **kw)
                x = y
        return y

    def dropout(self):
        PS = self.PS
        ns, ds = None, [int(i) for i in PS.prepost_bdims.split(',') if i]
        if ds:
            sh = ()
            n = len(sh)
            ds = [d + n if d < 0 else d for d in ds]
            ns = [1 if i in ds else sh[i] for i in range(n)]
        return Q.Dropout(PS.prepost_drop or PS.hidden_drop, noise_shape=ns)


class PreProc(LayerProc):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.pre_cmd
        assert 'a' not in self.cmd
        assert 'z' not in self.cmd


class PostProc(LayerProc):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.post_cmd
