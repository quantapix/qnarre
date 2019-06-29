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

from datetime import datetime

import advanced_tf.custom as qc
import advanced_tf.dataset as qd

ks = tf.keras
kl = ks.layers


@tf.function
def adapter(d, target):
    enc, dec, tgt = d['enc'], d['dec'], d['tgt']
    target.assign(tgt.to_tensor())
    return (
        (
            enc.flat_values,
            enc.row_splits,
            dec.flat_values,
            dec.row_splits,
            tgt.flat_values,
            tgt.row_splits,
        ),
        tgt.to_tensor()
    )


def dset_for(ps, target):
    ds = tf.data.TFRecordDataset(list(qd.files(ps)))
    ds = ds.take(1000).batch(ps.dim_batch)
    fs = {
        'defs': tf.io.VarLenFeature(tf.int64),
        'op': tf.io.VarLenFeature(tf.int64),
        'res': tf.io.VarLenFeature(tf.int64),
    }
    ds = ds.map(lambda x: tf.io.parse_example(x, fs)).map(qd.caster)
    return ds.map(qc.formatter).map(lambda x: adapter(x, target))


class Loss(ks.losses.Loss):
    @staticmethod
    def xent(y_true, y_pred):
        kw = dict(labels=y_true, logits=y_pred)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(**kw)

    def __init__(self):
        super().__init__(name='loss')

    def call(self, y_true, y_pred):
        return self.xent(y_true, y_pred)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        vs = Loss.xent(y_true, y_pred)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


params = qc.params
params.update(
    loss=Loss(),
    metric=Metric(),
)


def main(_):
    ps = qd.Params(**params)
    with tf.Graph().as_default():
        target = tf.Variable(0, shape=tf.TensorShape(None))
        ds = dset_for(ps, target)
        m = qc.model_for(ps)
        m.compile(
            optimizer=ps.optimizer,
            loss={'debed': ps.loss},
            metrics={'debed': [ps.metric]},
            target_tensors={'debed': target},
        )
        ld = datetime.now().strftime('%Y%m%d-%H%M%S')
        ld = f'/tmp/q/logs/{ld}'
        cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
        m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


if __name__ == '__main__':
    from absl import app
    app.run(main)
