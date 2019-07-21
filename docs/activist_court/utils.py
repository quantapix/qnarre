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

import re
import numpy as np
import tensorflow as tf

ks = tf.keras


class Params:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def cfg_items(self, *keys):
        for k in keys:
            yield k, getattr(self, k, None)


class Config(Params):
    runtime = Params(is_training=True, print_toks=False)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._items = kw.items()

    def items(self):
        return self._items


class Loss(ks.losses.Loss):
    def __init__(self):
        super().__init__(name='loss')

    def call(self, tgt, out):
        return cross_entropy(tgt, out)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, tgt, out, sample_weight=None):
        vs = cross_entropy(tgt, out)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


def gelu(x):
    # https://arxiv.org/pdf/1606.08415.pdf
    c = np.sqrt(2 / np.pi)
    c *= x + 0.044715 * tf.pow(x, 3)
    c = 0.5 * (1.0 + tf.tanh(c))
    return x * c


def activation(x):
    if x == 'gelu':
        return gelu
    return ks.activations.get(x)


def cross_entropy(tgt, out):
    t = tf.reshape(tf.cast(tgt, tf.int64), [-1])
    s = tf.shape(out)
    y = tf.reshape(out, [-1, s[-1]])
    y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=y)
    y = tf.reshape(y, s[:-1])
    return y


def print_toks(x, via):
    def print_row(r):
        tf.print(
            tf.numpy_function(
                lambda ts: ''.join([via[t][0] for t in ts]),
                [r],
                Tout=[tf.string],
            ))
        return r

    tf.print()
    tf.map_fn(print_row, x)


def to_snake_case(x):
    y = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', x)
    y = re.sub('([a-z])([A-Z])', r'\1_\2', y).lower()
    return 'private' + y if y[0] == '_' else y
