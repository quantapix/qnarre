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

import qnarre.neura as Q
import qnarre.neura.utils as U
import qnarre.neura.layers as L

from qnarre.feeds.dset.mnist import dset
from qnarre.neura.session import session_for


def dset_for(PS, kind):
    ds_1 = dset(PS, kind)
    ds_2 = dset(PS, kind)
    ds_3 = dset(PS, kind)
    n = 50000
    if kind == 'train':
        ds_1 = ds_1.shuffle(n)
        ds_2 = ds_2.shuffle(n)
        ds_3 = ds_3.shuffle(n)
    ds = Q.Dataset.zip((ds_1, ds_2, ds_3))
    ds = ds.map(lambda s, s2, s3: ((s[0], s2[0], s2[1], s3[0], s3[1]), s[1]))
    ds = ds.batch(PS.batch_size)
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(PS, compiled=False):
    w, h = PS.img_width, PS.img_height
    ins = [
        Q.Input(shape=(w * h), dtype='float32'),
        Q.Input(shape=(w * h), dtype='float32'),
        Q.Input(shape=(1, ), dtype='int32'),
        Q.Input(shape=(w * h), dtype='float32'),
        Q.Input(shape=(1, ), dtype='int32'),
    ]
    outs = [L.Mnist(PS)(ins)]
    m = Q.Model(inputs=ins, outputs=outs)
    if compiled:
        m.compile(
            optimizer=PS.optimizer,
            loss=PS.losses,
            metrics=[PS.metrics],
            # target_tensors=[ins[4]],
        )
    print(m.summary())
    return m


params = dict(
    batch_size=64,
    hidden_act='relu',
    hidden_drop=0.2,
    hidden_size=512,
    img_height=28,
    img_width=28,
    model_name='mlp',
    num_classes=10,
    optimizer='sgd',
)

params.update(
    data_dir='.data/mnist',
    log_dir='.model/mnist/logs',
    model_dir='.model/mnist',
    save_dir='.model/mnist/save',
)


def main(_):
    PS = U.Params(params).init_comps()
    session_for(PS)(dset_for, model_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    U.load_flags()
    from absl import flags
    flags.DEFINE_integer('num_classes', None, '')
    from absl import app
    app.run(main)
