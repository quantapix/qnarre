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
# !pip install -U tf-nightly-2.0-preview

import tensorflow as tf
import advanced_tf.dataset as qd
import advanced_tf.custom as qc

ks = tf.keras
kl = ks.layers


class ToRagged(qc.ToRagged):
    @tf.function(input_signature=[[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int64)
    ] * 3])
    def call(self, x):
        ys = []
        for i in range(3):
            i *= 2
            fv, rs = x[i:i + 2]
            ys.append(tf.RaggedTensor.from_row_splits(fv, rs))
        return ys


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    print(qc.ToRagged(), x)
    tf.autograph.trace(ToRagged(), x)
    y = ToRagged()(x)
    y = qc.Frames(ps)(y)
    embed = qc.Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    m = ks.Model(inputs=x, outputs=y)
    m.compile(
        optimizer=ps.optimizer,
        loss={'debed': ps.loss},
        metrics={'debed': [ps.metric]},
    )
    print(m.summary())
    return m


class Loss(ks.losses.Loss):
    @staticmethod
    def xent(tgt, out):
        tgt = tf.reshape(tf.cast(tgt, tf.int64), [-1])
        s = tf.shape(out)
        out = tf.reshape(out, [-1, s[-1]])
        y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt,
                                                           logits=out)
        return tf.reshape(y, s[:-1])

    def __init__(self):
        super().__init__(name='qloss')

    def call(self, tgt, out):
        return self.xent(tgt, out)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='qmetric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, tgt, out, sample_weight=None):
        vs = Loss.xent(tgt, out)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


params = qc.params
params.update(
    loss=Loss(),
    metric=Metric(),
)

if __name__ == '__main__':
    ps = qd.Params(**params)
    import advanced_tf.masking as qm
    qm.main_graph(ps, qc.dset_for(ps), model_for(ps))
