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

from qnarre.neura.layers.base import Layer


def _layer_norm(self, inputs):
    x = inputs
    m = tf.reduce_mean(x, axis=-1, keepdims=True)
    v = tf.reduce_mean(tf.square(x - m), axis=-1, keepdims=True)
    y = (x - m) / tf.sqrt(v + self.cfg.norm_epsilon)
    y = y * self.norm_w + self.norm_b
    return y


class Norm(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(ps.cfg_items('norm_epsilon', ))

    def build(self, input_shape):
        s = input_shape[-1]
        self.norm_w = self.add_weight('norm_w', s, initializer='ones')
        self.norm_b = self.add_weight('norm_b', s, initializer='zeros')
        return super().build(input_shape)

    def call(self, inputs):
        return _layer_norm(self, inputs)


class LayerProc(Layer):
    cmd = ''
    batch = None

    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'bdims_prepost',
                'cmd_post',
                'cmd_pre',
                'drop_hidden',
                'drop_prepost',
                'norm_epsilon',
                'norm_type',
            ))

    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        cfg = self.cfg
        if cfg.norm_type == 'batch':
            self.batch = tf.BatchNormalization(epsilon=cfg.norm_epsilon)

    def build(self, input_shape):
        s = input_shape[1][-1]
        self.norm_w = self.add_weight('norm_w', s, initializer='ones')
        self.norm_b = self.add_weight('norm_b', s, initializer='zeros')
        # self.gamma = self.add_weight(shape=(), initializer='zeros')
        return super().build(input_shape)

    @tf.function
    def call(self, inputs):
        prev, x = inputs
        y = x
        if self.cmd:
            cfg = self.cfg
            for c in self.cmd:
                if c == 'a':
                    y = prev + x
                elif c == 'z':
                    y = prev + x * self.gamma
                elif c == 'n':
                    if cfg.norm_type == 'layer':
                        y = _layer_norm(self, x)
                    elif cfg.norm_type == 'batch':
                        y = self.batch(x)
                    elif cfg.norm_type == 'l2':
                        m = tf.reduce_mean(x, axis=-1, keepdims=True)
                        n = tf.square(x - m)
                        n = tf.reduce_sum(n, axis=-1, keepdims=True)
                        y = (x - m) / tf.sqrt(n + cfg.norm_epsilon)
                        y = y * self.gain + self.bias
                    elif cfg.norm_type == 'group':
                        sh = tf.int_shape(x)
                        assert len(sh) == 4 and sh[-1] % cfg.num_groups == 0
                        gs = (cfg.num_groups, sh[-1] // cfg.num_groups)
                        x = tf.reshape(x, sh[:-1] + gs)
                        m, v = tf.moments(x, [1, 2, 4], keep_dims=True)
                        y = (x - m) / tf.sqrt(v + cfg.group_epsilon)
                        y = tf.reshape(y, sh) * self.gain + self.bias
                    elif cfg.norm_type == 'noam':
                        y = tf.cast_to_floatx(tf.int_shape(x)[-1])
                        y = tf.l2_normalize(x, axis=-1) * tf.sqrt(y)
                    else:
                        assert cfg.norm_type == 'none'
                else:
                    assert c == 'd'
                    y = self.drop(y)
                x = y
        return y

    def drop(self, x):
        cfg = self.cfg
        r = cfg.drop_prepost or cfg.drop_hidden
        ns, ds = None, [int(i) for i in cfg.bdims_prepost.split(',') if i]
        if ds:
            sh = ()
            n = len(sh)
            ds = [d + n if d < 0 else d for d in ds]
            ns = [1 if i in ds else sh[i] for i in range(n)]
        return super().drop(x, r, noise_shape=ns)


class PreProc(LayerProc):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        self.cmd = self.cfg.cmd_pre
        assert 'a' not in self.cmd
        assert 'z' not in self.cmd


class PostProc(LayerProc):
    def __init__(self, ps, **kw):
        super().__init__(ps, **kw)
        self.cmd = self.cfg.cmd_post
