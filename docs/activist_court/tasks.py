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

import pathlib as pth
import tensorflow as tf

from datetime import datetime

ks = tf.keras


def train_eager(ps, ds, m):
    def step(x, t):
        with tf.GradientTape() as tape:
            y = m(x)
            loss = ps.loss(t, y)
            loss += sum(m.losses)
            xent = ps.metric(t, y)
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


def train_graph(ps, ds, m):
    b = pth.Path('/tmp/q')
    b.mkdir(parents=True, exist_ok=True)
    lp = datetime.now().strftime('%Y%m%d-%H%M%S')
    lp = b / f'logs/{lp}'
    c = tf.train.Checkpoint(model=m)
    mp = b / 'model' / f'{m.name}'
    mgr = tf.train.CheckpointManager(c, str(mp), max_to_keep=3)
    # if mgr.latest_checkpoint:
    #     vs = tf.train.list_variables(mgr.latest_checkpoint)
    #     print(f'\n*** checkpoint vars: {vs}')
    c.restore(mgr.latest_checkpoint).expect_partial()

    class CheckpointCB(ks.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mgr.save()

    cbs = [
        CheckpointCB(),
        ks.callbacks.TensorBoard(
            log_dir=str(lp),
            histogram_freq=1,
        ),
    ]
    m.fit(ds, callbacks=cbs, epochs=ps.num_epochs)
    # mgr.save()


def evaluate(ps, ds, m):
    mp = pth.Path('/tmp/q/model')
    if tf.train.get_checkpoint_state(str(mp)):
        m.train_on_batch(ds)
        m.load_weights(str(mp / f'{m.name}'))
        loss, xent = m.evaluate(ds)
        print(f'\nEvaluate loss, xent: {loss}, {xent}')


def predict(ps, ds, m):
    mp = pth.Path('/tmp/q/model')
    if tf.train.get_checkpoint_state(str(mp)):
        m.train_on_batch(ds)
        m.load_weights(str(mp / f'{m.name}'))
        for x, t in ds:
            y = m.predict(x)
            print(y, t.numpy())
