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


class FFNet(Layer):
    @staticmethod
    def cfg_items(ps):
        return dict(
            ps.cfg_items(
                'dim_ffnet',
                'dim_hidden',
                'drop_ffnet',
                'drop_hidden',
            ))

    def __init__(self, ps, owner, **kw):
        super().__init__(ps, **kw)
        self.pre = owner.pre
        self.post = owner.post
        cfg = self.cfg
        kw = dict(kernel_initializer=ps.initializer, use_bias=True)
        self.dense2 = tf.Dense(cfg.dim_hidden, **kw)
        kw.update(activation=ps.act_ffnet)
        self.dense1 = tf.Dense(cfg.dim_ffnet, **kw)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.pre([x, x])
        y = self.dense1(x)
        r = self.cfg.drop_ffnet or self.cfg.drop_hidden
        y = self.drop(y, r)
        y = self.dense2(y)
        y = self.post([x, y])
        return y
