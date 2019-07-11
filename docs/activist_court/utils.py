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

from datetime import datetime

import tensorflow as tf

ks = tf.keras


def cross_entropy(tgt, y):
    tgt = tf.reshape(tf.cast(tgt, tf.int64), [-1])
    s = tf.shape(y)
    y = tf.reshape(y, [-1, s[-1]])
    y = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tgt, logits=y)
    return tf.reshape(y, s[:-1])


class Loss(ks.losses.Loss):
    def __init__(self):
        super().__init__(name='loss')

    def call(self, tgt, y):
        return cross_entropy(tgt, y)


class Metric(ks.metrics.Metric):
    def __init__(self):
        super().__init__(name='metric', dtype=tf.float32)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, tgt, y, sample_weight=None):
        vs = cross_entropy(tgt, y)
        self.total.assign_add(tf.math.reduce_sum(vs))
        return self.count.assign_add(tf.cast(tf.size(vs), dtype=tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


def train_graph(ps, ds, m):
    ld = datetime.now().strftime('%Y%m%d-%H%M%S')
    ld = f'/tmp/q/logs/{ld}'
    cs = [ks.callbacks.TensorBoard(log_dir=ld, histogram_freq=1)]
    m.fit(ds, callbacks=cs, epochs=ps.num_epochs)


def train_eager(ps, ds, m):
    def step(x, y):
        with tf.GradientTape() as tape:
            logits = m(x)
            loss = ps.loss(y, logits)
            loss += sum(m.losses)
            xent = ps.metric(y, logits)
        grads = tape.gradient(loss, m.trainable_variables)
        ps.optimizer.apply_gradients(zip(grads, m.trainable_variables))
        return loss, xent

    @tf.function
    def epoch():
        s, loss, xent = 0, 0.0, 0.0
        for x, y in ds:
            s += 1
            loss, xent = step(x, y)
            if tf.equal(s % 10, 0):
                e = ps.metric.result()
                tf.print('Step:', s, ', loss:', loss, ', xent:', e)
        return loss, xent

    for e in range(ps.num_epochs):
        loss, xent = epoch()
        print(f'Epoch {e} loss:', loss.numpy(), ', xent:', xent.numpy())
