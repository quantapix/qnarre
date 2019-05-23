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

import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.neura import tf
from qnarre.feeds.dset.mnist import dset
from qnarre.neura.session import session_for


def dset_for(ps, kind):
    ds_1 = dset(ps, kind)
    ds_2 = dset(ps, kind)
    ds_3 = dset(ps, kind)
    n = 50000
    if kind == 'train':
        ds_1 = ds_1.shuffle(n)
        ds_2 = ds_2.shuffle(n)
        ds_3 = ds_3.shuffle(n)
    ds = tf.Dataset.zip((ds_1, ds_2, ds_3))
    ds = ds.map(lambda s, s2, s3: (
        (s['image'], s2['image'], s2['label'], s3['image'], s3['label']),
        s['label'],
    ))
    ds = ds.batch(ps.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(ps, compiled=False):
    w, h = ps.img_width, ps.img_height
    ins = [
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(1, ), dtype='int32'),
        tf.Input(shape=(w * h, ), dtype='float32'),
        tf.Input(shape=(1, ), dtype='int32'),
    ]
    outs = [L.Mnist(ps)(ins)]
    m = tf.Model(name='MnistModel', inputs=ins, outputs=outs)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
            # target_tensors=[ins[4]],
        )
    print(m.summary())
    return m


_params = dict(
    act_hidden='relu',
    batch_size=64,
    dim_hidden=512,
    drop_hidden=0.2,
    dset='mnist',
    img_height=28,
    img_width=28,
    model='mnist',
    num_classes=10,
    optimizer='sgd',
)


def main(_):
    ps = U.Params(_params).init_comps()
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    U.load_flags()
    from absl import flags
    flags.DEFINE_integer('num_classes', None, '')
    from absl import app
    app.run(main)
