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


class Processor(L.Layer):
    cmd = ''

    @staticmethod
    def _dropout(rate, shape, bdims):
        ns, bds = None, [int(i) for i in bdims.split(',') if i]
        if bds:
            n = len(shape)
            bds = [d + n if d < 0 else d for d in bds]
            ns = [1 if i in bds else shape[i] for i in range(n)]
        return L.Dropout(rate, noise_shape=ns)

    def __init__(self, PS, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        self.PS = PS
        self.drop = self._dropout(PS.prepost_drop, (), PS.prepost_bdims)
        self.batch = L.BatchNormalization(epsilon=PS.norm_epsilon)

    def build(self, input_shape):
        _, x = input_shape
        kw = dict(shape=x[-1], trainable=True)
        self.gain = self.add_weight(initializer='ones', **kw)
        self.bias = self.add_weight(initializer='zeros', **kw)
        # kw.update(shape=())
        self.gamma = self.add_weight(initializer='zeros', **kw)
        return super().build(input_shape)

    def call(self, inputs, **kw):
        prev, x = inputs
        if self.cmd:
            PS = self.PS
            for c in self.cmd:
                if c == 'a':
                    x += prev
                elif c == 'z':
                    x = prev + self.gamma * x
                elif c == 'n':
                    if PS.norm_type == 'layer':
                        m = Q.mean(x, axis=-1, keepdims=True)
                        v = Q.mean(Q.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / Q.sqrt(v + PS.norm_epsilon)
                        x = x * self.gain + self.bias
                    elif PS.norm_type == 'batch':
                        x = self.batch(x, **kw)
                    elif PS.norm_type == 'l2':
                        m = Q.mean(x, axis=-1, keepdims=True)
                        n = Q.sum(Q.square(x - m), axis=-1, keepdims=True)
                        x = (x - m) / Q.sqrt(n + PS.norm_epsilon)
                        x = x * self.gain + self.bias
                    elif PS.norm_type == 'group':
                        sh = Q.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % PS.num_groups == 0
                        gsh = (PS.num_groups, sh[-1] // PS.num_groups)
                        x = Q.reshape(x, sh[:-1] + gsh)
                        m, v = Q.moments(x, [1, 2, 4], keep_dims=True)
                        x = (x - m) / Q.sqrt(v + PS.group_epsilon)
                        x = Q.reshape(x, sh) * self.gain + self.bias
                    elif PS.norm_type == 'noam':
                        d = Q.cast_to_floatx(Q.int_shape(x)[-1])
                        x = Q.l2_normalize(x, axis=-1) * Q.sqrt(d)
                    else:
                        assert PS.norm_type == 'none'
                else:
                    assert c == 'd'
                    x = self.drop(x, **kw)
        return x


class PreProc(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.pre_cmd
        assert 'a' not in self.cmd
        assert 'z' not in self.cmd

    def build(self, input_shape):
        return super().build((None, input_shape))

    def compute_output_shape(self, input_shape):
        return (input_shape, )

    def call(self, inputs, **kw):
        return super().call([None, inputs], **kw)


class PostProc(Processor):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.cmd = self.PS.post_cmd
