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

from datetime import datetime

import pathlib as pth
import tensorflow as tf

import dataset as qd
import custom as qc

ks = tf.keras
kl = ks.layers


def model_for(ps):
    x = [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    x += [ks.Input(shape=(), dtype='int32'), ks.Input(shape=(), dtype='int64')]
    y = qc.ToRagged()(x)
    y = qc.Frames(ps)(y)
    embed = qc.Embed(ps)
    ye = qc.Encode(ps)(embed(y[:2]))
    yd = qc.Decode(ps)(embed(y[2:]) + [ye[0]])
    y = qc.Debed(ps)(yd)
    m = ks.Model(name='callbacks', inputs=x, outputs=y)
    m.compile(optimizer=ps.optimizer, loss=ps.loss, metrics=[ps.metric])
    print(m.summary())
    return m


def main_graph(ps, ds, m):
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


if __name__ == '__main__':
    ps = qd.Params(**qc.params)
    main_graph(ps, qc.dset_for(ps), model_for(ps))
