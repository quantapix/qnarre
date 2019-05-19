# Copyright 2018 Quantapix Authors. All Rights Reserved.
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


class Config:
    def __init__(self, **kw):
        self._items = kvs = kw.items()
        for k, v in kvs:
            setattr(self, k, v)

    def items(self):
        return self._items


class Layer(tf.Layer):
    @classmethod
    def from_config(cls, cfg):
        return cls(Config(**cfg))

    def __init__(self, ps, **kw):
        super().__init__(**kw)
        self.supports_masking = True
        if isinstance(ps, Config):
            self.cfg = ps
        else:
            self.cfg = Config(**self.cfg_items(ps))

    def cfg_items(self, ps):
        return {}

    def get_config(self):
        s = super().get_config().items()
        c = self.cfg.items()
        return dict(list(s) + list(c))

    def compute_output_shape(self, input_shape):
        return input_shape

    def add_weight(self, name, shape, **kw):
        kw.setdefault('dtype', tf.floatx())
        cfg = self.cfg
        if hasattr(cfg, 'init_stddev'):
            kw.setdefault('initializer',
                          tf.TruncatedNormal(stddev=cfg.init_stddev))
        if hasattr(cfg, 'regular_l1') and hasattr(cfg, 'regular_l2'):
            kw.setdefault('regularizer', tf.L1L2(cfg.regular_l1,
                                                 cfg.regular_l2))
        return super().add_weight(name, shape, **kw)

    def add_bias(self, name, shape, **kw):
        kw.setdefault('dtype', tf.floatx())
        kw.setdefault('initializer', tf.zeros_initializer())
        return super().add_weight(name, shape, **kw)

    def add_resource(self, name, shape, **kw):
        kw.setdefault('dtype', tf.floatx())
        kw.setdefault('trainable', False)
        kw.setdefault('use_resource', True)
        kw.setdefault('initializer', tf.zeros_initializer())
        kw.setdefault('aggregation', tf.VariableAggregation.NONE)
        kw.setdefault('synchronization', tf.VariableSynchronization.NONE)
        return self.add_variable(name, shape, **kw)

    def drop(self, x, rate, **kw):
        if tf.learning_phase():
            return tf.dropout(x, rate, **kw)
        return x
