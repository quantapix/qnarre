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

import qnarre.neura.utils as utils

# from qnarre.neura import tf
from qnarre.neura.session import session_for
from qnarre.feeds.dset.mnist import dset as mnist_dset
from qnarre.neura.layers.mnist import model as mnist_model
from qnarre.neura.layers.mnist import adapter as mnist_adapter


def dset_for(ps, kind):
    ds, feats = mnist_dset(ps, kind)
    n = 50000
    if kind == 'train':
        ds = ds.shuffle(n)
    ds = ds.batch(1 if ps.eager_mode else ps.batch_size)
    ds = ds.map(lambda d: mnist_adapter(ps, feats, d))
    # ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def model_for(ps, compiled=False):
    m = mnist_model(ps)
    if compiled:
        m.compile(
            optimizer=ps.optimizer,
            loss=ps.losses,
            metrics=[ps.metrics],
            # target_tensors=[ins[4]],
        )
    print(m.summary())
    return m


params = dict(
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
    # eager_mode=True,
)


def main(_):
    ps = utils.Params(params).init_comps()
    session_for(ps)(dset_for, model_for)


if __name__ == '__main__':
    # T.logging.set_verbosity(T.logging.INFO)
    utils.load_flags()
    from absl import flags
    flags.DEFINE_integer('num_classes', None, '')
    from absl import app
    app.run(main)
